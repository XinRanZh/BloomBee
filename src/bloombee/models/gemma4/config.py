import os
from typing import Optional, Union

try:
    from transformers.models.gemma4 import Gemma4TextConfig as _BaseConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention as _BaseAttention
except ImportError:
    from transformers.models.gemma2 import Gemma2Config as _BaseConfig
    from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention as _BaseAttention

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.gemma4.block import WrappedGemma4Block
from bloombee.utils.hivemind_compat import get_logger

logger = get_logger(__name__)


class DistributedGemma4Config(_BaseConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    model_type = "gemma4"

    block_class = WrappedGemma4Block
    attn_class = _BaseAttention
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
