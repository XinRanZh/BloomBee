from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from bloombee.utils.cache_compat import make_past_kv_cache, make_empty_kv_cache, read_kv_from_cache

from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextDecoderLayer,
    Gemma4TextConfig,
    Gemma4TextRotaryEmbedding,
)


class WrappedGemma4Block(Gemma4TextDecoderLayer):
    """Wraps Gemma4TextDecoderLayer for BloomBee's distributed block-serving protocol.

    BloomBee calls each block with:
        forward(hidden_states, layer_past=(k,v), attention_mask=..., use_cache=True)
    and expects:
        (hidden_states, present_key_value) where present_key_value is (k_new, v_new)

    Gemma4TextDecoderLayer expects:
        forward(hidden_states, position_embeddings=(cos,sin), attention_mask=4D_mask,
                past_key_values=Cache, shared_kv_states=dict, ...)
    and returns:
        hidden_states (cache updated in-place)

    Key Gemma4 quirks handled here:
    - Heterogeneous layers: sliding_attention (16 KV heads, D=256) vs
      full_attention (4 KV heads, D=512). Q head_dim is always 256.
    - Gemma4 does NOT build causal masks internally — we must provide the
      correct per-layer-type mask (causal for full, sliding-window for sliding).
    - shared_kv_states: cross-layer KV sharing dict. In distributed mode each
      server handles a contiguous block range; we thread a shared dict through
      all local layers on a given forward pass.
    """

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self._config = config
        self._rotary_emb = Gemma4TextRotaryEmbedding(config)
        self._layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else "full_attention"

        # Derive actual per-layer KV dimensions from weights (not config).
        attn = self.self_attn
        kv_groups = getattr(attn, "num_key_value_groups", 1)
        self._num_kv_heads = config.num_attention_heads // kv_groups
        self._kv_head_dim = attn.k_proj.weight.shape[0] // self._num_kv_heads
        # BloomBee's backend.py reads attn.num_heads for cache allocation.
        # Only set if missing — do NOT override native attributes.
        if not hasattr(attn, "num_heads"):
            attn.num_heads = config.num_attention_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0

        # --- Convert BloomBee's layer_past to HF DynamicCache ---
        # IMPORTANT: always use layer_idx=0 for the per-block temporary cache.
        # In tf 5.x, DynamicCache.get_seq_length() defaults to layer 0.
        # If we stored at self.layer_idx (e.g. 30), get_seq_length() would
        # check layer 0 (empty) and return 0, causing create_causal_mask to
        # build a mask with no past context → garbled repetitive output.
        _cache_idx = 0
        past_key_values = None
        if layer_past is not None:
            pk, pv = layer_past
            if pk.dtype != hidden_states.dtype or pk.device != hidden_states.device:
                pk = pk.to(device=hidden_states.device, dtype=hidden_states.dtype)
                pv = pv.to(device=hidden_states.device, dtype=hidden_states.dtype)
            past_key_values_length = pk.shape[2]
            pk, pv = self._reorder_cache_from_bloom((pk, pv), batch_size, past_key_values_length)
            past_key_values = make_past_kv_cache(
                pk, pv, layer_idx=_cache_idx, seen_tokens=past_key_values_length,
            )
            assert past_key_values.get_seq_length() == past_key_values_length, (
                f"Cache seq_length mismatch: {past_key_values.get_seq_length()} != {past_key_values_length}"
            )
        elif use_cache:
            past_key_values = make_empty_kv_cache(_cache_idx)

        # --- Position IDs & cache_position ---
        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + seq_length,
            dtype=torch.long, device=hidden_states.device,
        )
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # --- Rotary position embeddings (per layer type) ---
        position_embeddings = self._rotary_emb(hidden_states, position_ids, self._layer_type)

        # --- Build correct causal mask (Gemma4 does NOT create masks internally) ---
        mask_kwargs = dict(
            config=self._config,
            input_embeds=hidden_states,
            attention_mask=None,  # no padding mask in BloomBee
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        if self._layer_type == "sliding_attention":
            causal_mask = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_mask = create_causal_mask(**mask_kwargs)

        # --- Call native forward ---
        skip_keys = {'position_ids', 'attention_mask', 'use_cache', 'position_embeddings',
                     'past_key_value', 'past_key_values', 'cache_position', 'shared_kv_states'}
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in skip_keys}

        outputs = super().forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            shared_kv_states=kwargs.get("shared_kv_states", {}),
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
            **extra_kwargs,
        )

        # Extract hidden_states (tf 5.x returns tensor; cache updated in-place)
        if isinstance(outputs, torch.Tensor):
            output_hidden = outputs
        elif isinstance(outputs, tuple):
            output_hidden = outputs[0]
        else:
            output_hidden = outputs

        # --- Extract NEW KV tokens from in-place-updated cache ---
        if use_cache and past_key_values is not None:
            pk, pv = read_kv_from_cache(past_key_values, _cache_idx)
            if pk is not None:
                pk = pk[:, :, past_key_values_length:, :]
                pv = pv[:, :, past_key_values_length:, :]
                present_key_value = self._reorder_cache_to_bloom((pk, pv), batch_size, seq_length)
                return (output_hidden, present_key_value)

        return (output_hidden, None)

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        """Convert BloomBee cache [B*H, D, S] / [B*H, S, D] to HF [B, H, S, D]."""
        key_states, value_states = key_value
        if key_states.dim() == 4:
            nkv = self._num_kv_heads
            key_states = key_states[:, :nkv, :, :]
            value_states = value_states[:, :nkv, :, :]
            return (key_states, value_states)
        # 3D bloom format
        nkv = key_states.shape[0] // batch_size
        head_dim = key_states.shape[1]  # key is [B*H, D, S]
        key_states = key_states.permute(0, 2, 1)  # -> [B*H, S, D]
        key_states = key_states.view(batch_size, nkv, seq_length, head_dim)
        value_states = value_states.view(batch_size, nkv, seq_length, head_dim)
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        """Convert HF [B, H, S, D] cache back to BloomBee format."""
        key_states, value_states = key_value
        head_dim = key_states.shape[-1]
        nkv = key_states.shape[1]
        value_states = value_states.reshape(batch_size * nkv, seq_length, head_dim)
        key_states = key_states.reshape(batch_size * nkv, seq_length, head_dim)
        key_states = key_states.permute(0, 2, 1)  # -> [B*H, D, S]
        return (key_states, value_states)
