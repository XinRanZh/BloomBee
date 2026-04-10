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
        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result

        # google/gemma-4-31b-it has nested text_config inside multimodal config.
        # Our Gemma4TextConfig base may load default values (30 layers) instead of
        # the actual text model values (60 layers). Fix by loading the raw HF config
        # and copying text_config fields.
        try:
            from transformers import AutoConfig as _HFAutoConfig
            raw = _HFAutoConfig.from_pretrained(model_name_or_path)
            text_cfg = getattr(raw, "text_config", None)
            if text_cfg is not None and text_cfg.num_hidden_layers != config.num_hidden_layers:
                logger.info(f"Copying text_config fields (layers {text_cfg.num_hidden_layers})")
                # Only copy simple (non-dict, non-object) fields + layer_types
                skip_keys = {"model_type", "generation_config", "architectures", "transformers_version"}
                for key, val in text_cfg.to_dict().items():
                    if key in skip_keys:
                        continue
                    if isinstance(val, dict):
                        continue  # Skip nested configs
                    setattr(config, key, val)
        except Exception as e:
            logger.warning(f"Could not extract text_config: {e}")

        if config.pad_token_id is None:
            config.pad_token_id = 0
        return result
