import logging

logger = logging.getLogger(__name__)

try:
    from bloombee.models.gemma4.block import WrappedGemma4Block
    from bloombee.models.gemma4.config import DistributedGemma4Config
    from bloombee.models.gemma4.model import (
        DistributedGemma4ForCausalLM,
        DistributedGemma4ForSequenceClassification,
        DistributedGemma4Model,
    )
    from bloombee.utils.auto_config import register_model_classes
    from transformers import AutoConfig

    # Register "gemma4" model_type. Skip if already registered natively.
    try:
        AutoConfig.register("gemma4", DistributedGemma4Config)
    except ValueError:
        pass

    register_model_classes(
        config=DistributedGemma4Config,
        model=DistributedGemma4Model,
        model_for_causal_lm=DistributedGemma4ForCausalLM,
        model_for_sequence_classification=DistributedGemma4ForSequenceClassification,
    )
except ImportError:
    logger.info("Gemma4 support unavailable (requires transformers >= 5.0 with native Gemma4 classes)")
