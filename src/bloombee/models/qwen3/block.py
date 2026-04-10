from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer as _BaseDecoderLayer,
        Qwen3RotaryEmbedding,
    )
    from transformers.models.qwen3 import Qwen3Config as _BaseBlockConfig
    _HAS_QWEN3 = True
except ImportError:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as _BaseDecoderLayer
    from transformers import Qwen2Config as _BaseBlockConfig
    _HAS_QWEN3 = False


class WrappedQwen3Block(_BaseDecoderLayer):
    """Wraps Qwen3DecoderLayer for BloomBee's distributed block-serving protocol.

    BloomBee calls each block with:
        forward(hidden_states, layer_past=(k,v), attention_mask=..., use_cache=True)
    and expects:
        (hidden_states, present_key_value) where present_key_value is (k_new, v_new)

    Qwen3DecoderLayer (tf 5.x) expects:
        forward(hidden_states, position_embeddings=(cos,sin), attention_mask=..., past_key_values=Cache, ...)
    and returns:
        hidden_states (cache updated in-place)
    """

    def __init__(self, config: _BaseBlockConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self._attn_implementation = config._attn_implementation
        self.sliding_window = getattr(config, "sliding_window", None)

        # Create rotary embedding for this block (normally lives in the model)
        if _HAS_QWEN3:
            self._rotary_emb = Qwen3RotaryEmbedding(config)
        else:
            self._rotary_emb = None  # Qwen2 fallback handles RoPE internally

        # BloomBee's backend.py accesses self_attn.num_heads — add it for compatibility
        if not hasattr(self.self_attn, "num_heads"):
            self.self_attn.num_heads = config.num_attention_heads
        if not hasattr(self.self_attn, "num_key_value_heads"):
            self.self_attn.num_key_value_heads = config.num_key_value_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0

        # --- Convert BloomBee's layer_past to HF DynamicCache ---
        past_key_values = None
        if layer_past is not None:
            pk, pv = layer_past
            if pk.dtype != hidden_states.dtype or pk.device != hidden_states.device:
                pk = pk.to(device=hidden_states.device, dtype=hidden_states.dtype)
                pv = pv.to(device=hidden_states.device, dtype=hidden_states.dtype)
            past_key_values_length = pk.shape[2]
            pk, pv = self._reorder_cache_from_bloom((pk, pv), batch_size, past_key_values_length)
            past_key_values = DynamicCache()
            past_key_values.key_cache = [torch.empty(0, device=pk.device, dtype=pk.dtype) for _ in range(self.layer_idx)] + [pk]
            past_key_values.value_cache = [torch.empty(0, device=pv.device, dtype=pv.dtype) for _ in range(self.layer_idx)] + [pv]
            past_key_values._seen_tokens = past_key_values_length
        elif use_cache:
            past_key_values = DynamicCache()
            past_key_values.key_cache = [torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(self.layer_idx)]
            past_key_values.value_cache = [torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype) for _ in range(self.layer_idx)]

        # --- Compute position_ids and position_embeddings ---
        position_ids = torch.arange(
            past_key_values_length, past_key_values_length + seq_length,
            dtype=torch.long, device=hidden_states.device,
        ).unsqueeze(0).expand(batch_size, -1)

        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + seq_length,
            dtype=torch.long, device=hidden_states.device,
        )

        # --- Build causal attention mask ---
        # Qwen3's attention does NOT auto-create causal masks. None means bidirectional.
        # We must provide a proper causal mask for autoregressive generation.
        total_length = past_key_values_length + seq_length
        causal_mask = torch.full(
            (seq_length, total_length), torch.finfo(hidden_states.dtype).min,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(seq_length, total_length, device=hidden_states.device, dtype=torch.bool),
                       diagonal=past_key_values_length + 1).logical_not(),
            0,
        )
        # Expand to [batch, 1, seq, total] for multi-head attention
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # --- Compute position_embeddings (cos, sin) for RoPE ---
        if self._rotary_emb is not None:
            position_embeddings = self._rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        # --- Call native forward ---
        # tf 5.x: Qwen3DecoderLayer returns just hidden_states; cache is updated in-place
        forward_kwargs = dict(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        if position_embeddings is not None:
            forward_kwargs["position_embeddings"] = position_embeddings

        # Filter out any kwargs that would conflict
        skip_keys = set(forward_kwargs.keys()) | {"past_key_value", "layer_past"}
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in skip_keys}

        output_hidden = super().forward(hidden_states, *args, **forward_kwargs, **extra_kwargs)

        # --- Extract updated cache and convert back to BloomBee format ---
        if use_cache and past_key_values is not None:
            pk = past_key_values.key_cache[self.layer_idx]   # [B, H, S_full, D]
            pv = past_key_values.value_cache[self.layer_idx]  # [B, H, S_full, D]
            # Only keep NEW tokens (BloomBee manages cumulative cache externally)
            pk = pk[:, :, past_key_values_length:, :]
            pv = pv[:, :, past_key_values_length:, :]
            present_key_value = self._reorder_cache_to_bloom((pk, pv), batch_size, seq_length)
            return (output_hidden, present_key_value)

        return (output_hidden,)

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor, torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert BloomBee cache format to HF [B, H, S, D] format."""
        key_states, value_states = key_value
        if key_states.dim() == 4:
            nkv = self.self_attn.num_key_value_heads
            key_states = key_states[:, :nkv, :, :]
            value_states = value_states[:, :nkv, :, :]
            return (key_states, value_states)
        # 3D case: key is [B*H, D, S], value is [B*H, S, D]
        key_states = key_states.permute(0, 2, 1)  # [B*H, D, S] -> [B*H, S, D]
        key_states = key_states.reshape(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.reshape(batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim)
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor, torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert HF [B, H, S, D] cache back to BloomBee format."""
        key_states, value_states = key_value
        value_states = value_states.reshape(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.reshape(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)  # [B*H, S, D] -> [B*H, D, S]
        return (key_states, value_states)
