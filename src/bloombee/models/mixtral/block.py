from typing import Optional, Tuple

import torch
from transformers import MixtralConfig
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from bloombee.utils.cache_compat import make_past_kv_cache, make_empty_kv_cache, read_kv_from_cache, _IS_TF5


class WrappedMixtralBlock(MixtralDecoderLayer):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self._attn_implementation = config._attn_implementation
        self.sliding_window = config.sliding_window
        self.layer_idx = layer_idx

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

        seq_length_with_past = seq_length
        past_key_values_length = 0

        past_key_value = layer_past

        if past_key_value is not None:
            # Fix dtype and device mismatch (analogous to Falcon fixes #7 and pipeline-parallel fix):
            # Cache may be float16 while hidden_states is bfloat16, and may be on cuda:0
            # while this block lives on a different GPU (pipeline parallelism).
            pk, pv = past_key_value
            if pk.dtype != hidden_states.dtype or pk.device != hidden_states.device:
                pk = pk.to(device=hidden_states.device, dtype=hidden_states.dtype)
                pv = pv.to(device=hidden_states.device, dtype=hidden_states.dtype)
                past_key_value = (pk, pv)
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            _past_key_value = self._reorder_cache_from_bloom(past_key_value, batch_size, past_key_values_length)
            past_key_value = make_past_kv_cache(
                _past_key_value[0], _past_key_value[1],
                layer_idx=self.layer_idx, seen_tokens=past_key_values_length,
            )
        elif use_cache:
            past_key_value = make_empty_kv_cache(self.layer_idx)

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa":
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            # Pass None instead of the backend's 3D float mask: the backend mask has the right
            # causal structure but wrong shape/type for this function (expects 2D binary or None).
            # Passing None causes it to build a correct causal mask from past_key_values_length.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                None,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            # Pass None instead of the backend's 3D float mask (same reason as sdpa branch above).
            attention_mask = _prepare_4d_causal_attention_mask(
                None,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
                sliding_window=self.sliding_window,
            )

        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=hidden_states.device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # tf 5.x renamed past_key_value → past_key_values (plural)
        skip_kw = {'position_ids', 'attention_mask', 'use_cache', 'past_key_value', 'past_key_values'}
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in skip_kw}
        cache_kwarg_name = "past_key_values" if _IS_TF5 else "past_key_value"

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            **{cache_kwarg_name: past_key_value},
            **extra_kwargs,
        )

        if use_cache:
            # tf 5.x: outputs is just hidden_states tensor, cache updated in-place
            # tf 4.x: outputs is (hidden_states, ..., cache) tuple
            if isinstance(outputs, torch.Tensor):
                output_hidden = outputs
                cache_obj = past_key_value  # in-place updated
            else:
                output_hidden = outputs[0]
                cache_obj = outputs[-1]

            pk, pv = read_kv_from_cache(cache_obj, self.layer_idx)
            if pk is not None:
                pk = pk[:, :, past_key_values_length:, :]
                pv = pv[:, :, past_key_values_length:, :]
                present_key_value = self._reorder_cache_to_bloom((pk, pv), batch_size, seq_length)
                return (output_hidden, present_key_value)

            return (output_hidden, None) if isinstance(outputs, torch.Tensor) else outputs

        return outputs

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        # TODO: Move to mixin
        key_states, value_states = key_value
        if key_states.dim() == 4:
            # select_cache() returns [B, H, S, D] where H = num_attention_heads (32 for Mixtral).
            # The cache is allocated for num_attention_heads but only the first num_key_value_heads
            # heads are valid (GQA: 8 for Mixtral-8x7B). Slice to only valid KV heads.
            # (Analogous to Falcon fix #3 for 4D + MQA/GQA.)
            nkv = self.self_attn.num_key_value_heads
            key_states = key_states[:, :nkv, :, :]
            value_states = value_states[:, :nkv, :, :]
            return (key_states, value_states)
        # 3D case: key is [B*H, D, S], value is [B*H, S, D]
        key_states = key_states.permute(0, 2, 1)  # [B*H, D, S] -> [B*H, S, D]
        key_states = key_states.view(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.view(*key_states.shape)
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        # TODO: Move to mixin
        key_states, value_states = key_value
        # Use reshape (not view) since tensors from DynamicCache may be non-contiguous
        value_states = value_states.reshape(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.reshape(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)
