"""Compatibility helpers for transformers 4.x / 5.x Cache API differences."""

import inspect
from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache, DynamicCache

# Detect whether Cache.__init__ requires the `layers` keyword (transformers >= 5.x)
_CACHE_INIT_PARAMS = inspect.signature(Cache.__init__).parameters
_CACHE_USES_LAYER_API = "layers" in _CACHE_INIT_PARAMS


def init_cache_base(cache: Cache) -> None:
    """Call Cache.__init__ in a way that works on both tf 4.x and 5.x."""
    if _CACHE_USES_LAYER_API:
        Cache.__init__(cache, layers=[])
    else:
        Cache.__init__(cache)


def make_dynamic_cache() -> DynamicCache:
    """Create an empty DynamicCache compatible with both tf 4.x and 5.x."""
    return DynamicCache()
