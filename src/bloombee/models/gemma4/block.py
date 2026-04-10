from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

try:
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextDecoderLayer,
        Gemma4TextConfig,
        Gemma4TextRotaryEmbedding,
    )
    _HAS_NATIVE_GEMMA4 = True
except ImportError:
    _HAS_NATIVE_GEMMA4 = False

if not _HAS_NATIVE_GEMMA4:
    raise ImportError(
        "Gemma4 support requires transformers >= 5.0 with native Gemma4 classes. "
        "Install with: pip install 'transformers>=5.0'"
    )


class WrappedGemma4Block(Gemma4TextDecoderLayer):
    """Wraps Gemma4TextDecoderLayer for BloomBee's distributed block-serving protocol.

    BloomBee calls each block with:
        forward(hidden_states, layer_past=(k,v), attention_mask=..., use_cache=True, position_ids=...)
    and expects:
        (hidden_states, present_key_value) where present_key_value is (k_new, v_new)

    Gemma4TextDecoderLayer expects:
        forward(hidden_states, position_embeddings=(cos,sin), attention_mask=..., past_key_values=Cache, ...)
    and returns:
        hidden_states (cache updated in-place)
    """

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        # Create rotary embedding for this block (normally lives in the model)
        self._rotary_emb = Gemma4TextRotaryEmbedding(config)
        # Determine layer type for rotary
        self._layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else "full_attention"
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
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
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
            past_key_values.key_cache = [torch.empty(0) for _ in range(self.layer_idx)] + [pk]
            past_key_values.value_cache = [torch.empty(0) for _ in range(self.layer_idx)] + [pv]
            past_key_values._seen_tokens = past_key_values_length
        elif use_cache:
            past_key_values = DynamicCache()
            past_key_values.key_cache = [None] * self.layer_idx
            past_key_values.value_cache = [None] * self.layer_idx

        # --- Compute position_ids and position_embeddings ---
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length,
            dtype=torch.long, device=hidden_states.device
        ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self._rotary_emb(hidden_states, position_ids, self._layer_type)

        # --- Build attention mask ---
        # Gemma4 native attention handles masking internally via the attention implementation.
        # We pass a causal mask. For simplicity, pass None and let the attention build it.
        # The native Gemma4 attention expects a 4D mask or None.
        causal_mask = None  # Let the native attention handle causal masking

        # --- Call native forward ---
        # Gemma4TextDecoderLayer.forward returns just hidden_states (cache updated in-place)
        output_hidden = super().forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            shared_kv_states={},  # No cross-layer KV sharing in distributed mode
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

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
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        """Convert BloomBee cache format to HF [B, H, S, D] format."""
        key_states, value_states = key_value
        if key_states.dim() == 4:
            # Already [B, H, S, D] — just slice to valid KV heads
            nkv = self.self_attn.config.num_key_value_heads
            if hasattr(self.self_attn, 'num_key_value_heads'):
                nkv = self.self_attn.num_key_value_heads
            elif hasattr(self.self_attn, 'num_key_value_groups'):
                nkv = self.self_attn.config.num_attention_heads // self.self_attn.num_key_value_groups
            key_states = key_states[:, :nkv, :, :]
            value_states = value_states[:, :nkv, :, :]
            return (key_states, value_states)
        # 3D case: key is [B*H, D, S], value is [B*H, S, D]
        head_dim = self.self_attn.head_dim
        nkv = key_states.shape[0] // batch_size
        key_states = key_states.permute(0, 2, 1)  # [B*H, D, S] -> [B*H, S, D]
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
        key_states = key_states.permute(0, 2, 1)  # [B*H, S, D] -> [B*H, D, S]
        return (key_states, value_states)
