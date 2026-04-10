from transformers import AutoConfig

from bloombee.models.qwen3.block import WrappedQwen3Block
from bloombee.models.qwen3.config import DistributedQwen3Config
from bloombee.models.qwen3.model import (
    DistributedQwen3ForCausalLM,
    DistributedQwen3ForSequenceClassification,
    DistributedQwen3Model,
)
from bloombee.utils.auto_config import register_model_classes

# Register "qwen3" model_type with HuggingFace's AutoConfig.
# Skip if already registered (transformers >= 5.x has native qwen3 support).
try:
    AutoConfig.register("qwen3", DistributedQwen3Config)
except ValueError:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    CONFIG_MAPPING_NAMES["qwen3"] = "DistributedQwen3Config"

register_model_classes(
    config=DistributedQwen3Config,
    model=DistributedQwen3Model,
    model_for_causal_lm=DistributedQwen3ForCausalLM,
    model_for_sequence_classification=DistributedQwen3ForSequenceClassification,
)
