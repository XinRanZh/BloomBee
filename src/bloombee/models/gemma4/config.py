import os
from typing import Optional, Union

try:
    from transformers.models.gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention
    _HAS_NATIVE_GEMMA4 = True
except ImportError:
    _HAS_NATIVE_GEMMA4 = False

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.gemma4.block import WrappedGemma4Block
from bloombee.utils.hivemind_compat import get_logger

logger = get_logger(__name__)

if not _HAS_NATIVE_GEMMA4:
    raise ImportError(
        "Gemma4 support requires transformers >= 5.0 with native Gemma4 classes. "
        "Install with: pip install 'transformers>=5.0'"
    )


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
        if config.pad_token_id is None:
            config.pad_token_id = 0
        return result
