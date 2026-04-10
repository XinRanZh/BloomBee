from transformers import AutoConfig

from bloombee.models.gemma4.block import WrappedGemma4Block
from bloombee.models.gemma4.config import DistributedGemma4Config
from bloombee.models.gemma4.model import (
    DistributedGemma4ForCausalLM,
    DistributedGemma4ForSequenceClassification,
    DistributedGemma4Model,
)
from bloombee.utils.auto_config import register_model_classes

# Register "gemma4" model_type with HuggingFace's AutoConfig so that
# AutoConfig.from_pretrained("google/gemma-4-31b-it") works with transformers
# that only know about "gemma2" natively.
AutoConfig.register("gemma4", DistributedGemma4Config)

register_model_classes(
    config=DistributedGemma4Config,
    model=DistributedGemma4Model,
    model_for_causal_lm=DistributedGemma4ForCausalLM,
    model_for_sequence_classification=DistributedGemma4ForSequenceClassification,
)
