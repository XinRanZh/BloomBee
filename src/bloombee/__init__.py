import os
import platform

if platform.system() == "Darwin":
    # Necessary for forks to work properly on macOS, see https://github.com/kevlened/pytest-parallel/issues/93
    os.environ.setdefault("no_proxy", "*")
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import hivemind.compression.base as hivemind_compression_base
import transformers
from packaging import version

# Bypass transformers' strict tokenizers version check to allow newer tokenizers
# (needed for Gemma4 tokenizer format support with transformers 4.43.x)
try:
    import transformers.utils.versions as _tf_versions
    _orig_check = _tf_versions._compare_versions
    def _relaxed_check(op, got_ver, want_ver, requirement, pkg, hint):
        if pkg == "tokenizers":
            return  # Skip tokenizers version check
        return _orig_check(op, got_ver, want_ver, requirement, pkg, hint)
    _tf_versions._compare_versions = _relaxed_check
except Exception:
    pass

from bloombee.client import *
from bloombee.models import *
from bloombee.utils import *
from bloombee.utils.logging import initialize_logs as _initialize_logs

__version__ = "2.3.0.dev2"


if not os.getenv("BLOOMBEE_IGNORE_DEPENDENCY_VERSION"):
    assert (
        version.parse("4.43.1") <= version.parse(transformers.__version__)
    ), "Please install transformers>=4.43.1"


def _override_bfloat16_mode_default():
    if os.getenv("USE_LEGACY_BFLOAT16") is None:
        hivemind_compression_base.USE_LEGACY_BFLOAT16 = False


_initialize_logs()
_override_bfloat16_mode_default()
