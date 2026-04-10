from typing import Optional

import torch
import torch.nn as nn
from hivemind import DHT
from hivemind.utils.logging import get_logger
from transformers.modeling_outputs import BaseModelOutputWithPast

from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4PreTrainedModel,
    Gemma4TextModel,
)

from bloombee.client.from_pretrained import FromPretrainedMixin
from bloombee.client.lm_head import LMHead
from bloombee.client.ptune import PTuneMixin
from bloombee.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from bloombee.client.remote_sequential import RemoteSequential
from bloombee.models.gemma4.config import DistributedGemma4Config
from bloombee.utils.auto_config import DefaultRevisionMixin

logger = get_logger(__name__)


class DistributedGemma4Model(DefaultRevisionMixin, FromPretrainedMixin, PTuneMixin, Gemma4TextModel):
    """Gemma4TextModel where all transformer layers are hosted by the swarm"""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedGemma4Config

    def __init__(self, config: DistributedGemma4Config, *, dht: Optional[DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent initialization
        super().__init__(config)
        assert len(self.layers) == 0
        config.num_hidden_layers = n_layer

        self.layers = RemoteSequential(config, dht=dht)

        self.requires_grad_(False)
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert (
            attention_mask is None or (attention_mask == 1).all()
        ), f"Custom attention masks are not supported, {attention_mask=}"
        if cache_position is not None:
            assert position_ids is not None and torch.all(torch.eq(cache_position, position_ids)).item()
        assert (
            position_ids is None or (position_ids[:, 1:] - position_ids[:, :-1] == 1).all()
        ), f"Non-consecutive position_ids are not supported, {position_ids=}"
        assert head_mask is None, f"Custom head masks are not supported, {head_mask=}"
        assert use_cache is None or use_cache, f"{use_cache=} is not supported"
        assert not output_attentions, f"{output_attentions=} is not supported"
        assert not output_hidden_states, f"{output_hidden_states=} is not supported"
        assert return_dict is None or return_dict, f"{return_dict=} is not supported"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.h.position == 0
        if use_prompts:
            batch_size = inputs_embeds.shape[0]
            prompts, intermediate_prompts = self.get_prompt(batch_size)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            prompts = intermediate_prompts = None

        hidden_states = inputs_embeds
        output_shape = input_shape + (hidden_states.size(-1),)

        if past_key_values is None:
            past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(hidden_states.size(1))

        hidden_states = self.layers(
            hidden_states,
            prompts=intermediate_prompts,
            hypo_ids=past_key_values.hypo_ids if past_key_values is not None else None,
        )

        if use_prompts:
            hidden_states = hidden_states[:, self.pre_seq_len :]

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    @property
    def word_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    @property
    def word_embeddings_layernorm(self) -> nn.Module:
        return nn.Identity()

    @property
    def h(self) -> RemoteSequential:
        return self.layers

    @property
    def ln_f(self) -> nn.Module:
        return self.norm


class DistributedGemma4ForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, nn.Module):
    _keys_to_ignore_on_load_missing = DistributedGemma4Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedGemma4Model._keys_to_ignore_on_load_unexpected

    config_class = DistributedGemma4Config

    def __init__(self, config: DistributedGemma4Config):
        nn.Module.__init__(self)
        self.config = config
        self.model = DistributedGemma4Model(config)
        self.lm_head = LMHead(config)

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedGemma4Model:
        return self.model


class DistributedGemma4ForSequenceClassification(FromPretrainedMixin, nn.Module):
    _keys_to_ignore_on_load_missing = DistributedGemma4Model._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = DistributedGemma4Model._keys_to_ignore_on_load_unexpected

    config_class = DistributedGemma4Config

    def __init__(self, config: DistributedGemma4Config):
        nn.Module.__init__(self)
        self.config = config
        self.num_labels = config.num_labels
        self.model = DistributedGemma4Model(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

    @property
    def transformer(self) -> DistributedGemma4Model:
        return self.model
