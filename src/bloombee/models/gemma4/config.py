import os
from typing import Optional, Union

from transformers.models.gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.gemma4.block import WrappedGemma4Block
from bloombee.utils.hivemind_compat import get_logger

logger = get_logger(__name__)


class DistributedGemma4Config(Gemma4TextConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    model_type = "gemma4"

    block_class = WrappedGemma4Block
    attn_class = Gemma4TextAttention
    block_prefix = "model.layers"

    num_key_value_groups = 1

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")
        from transformers import AutoConfig as _HFAutoConfig
        # Load the raw HF config first to check if it's a multimodal wrapper
        raw_config = _HFAutoConfig.from_pretrained(model_name_or_path, *args, **kwargs)
        raw = raw_config[0] if isinstance(raw_config, tuple) else raw_config

        # If the config has a nested text_config (multimodal model), extract it
        text_cfg = getattr(raw, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
            logger.info(f"Extracting text_config from multimodal Gemma4 config "
                        f"(text layers={text_cfg.num_hidden_layers})")
            # Convert text_config to our DistributedGemma4Config
            config = cls(**text_cfg.to_dict(), dht_prefix=dht_prefix)
            config.model_type = "gemma4"  # Keep as gemma4 for BloomBee routing
        else:
            result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
            config = result[0] if isinstance(result, tuple) else result

        if config.pad_token_id is None:
            config.pad_token_id = 0
        return config
