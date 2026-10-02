"""CPU tests for the Pearl mining integration (simulate mode and the W7A7 reference math).

The kernel path (``--pearl_mining on/dry_run``) needs an sm90 GPU and the Pearl miner
packages; it is exercised by running a server, not by these tests.
"""

import pytest
import torch
import torch.nn.functional as F

from bloombee.data_structures import ServerInfo, ServerState
from bloombee.mining import pearl
from bloombee.mining.pearl import (
    MAX_INT7,
    PearlMiningConfig,
    PearlMiningMode,
    disable_pearl_mining,
    enable_pearl_mining,
    get_pearl_mining_stats,
    hadamard_block,
    is_mining_shape,
    pearl_linear,
    quantize_weight_int7,
    reference_w7a7_linear,
)


@pytest.fixture(autouse=True)
def _reset_mining():
    disable_pearl_mining()
    pearl._stats = pearl.PearlMiningStats()
    yield
    disable_pearl_mining()


def _rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("size", [2, 16, 128])
def test_hadamard_block_is_orthogonal(size):
    h = hadamard_block(size, dtype=torch.float64)
    torch.testing.assert_close(h @ h.T, torch.eye(size, dtype=torch.float64))
    # Pearl's convention negates the first column so that no row is all-ones.
    assert (h[0] < 0).sum() == 1


def test_hadamard_is_folded_into_weights():
    torch.manual_seed(0)
    x = torch.randn(4, 256, dtype=torch.float64)
    w = torch.randn(64, 256, dtype=torch.float64)
    xh = pearl._apply_hadamard(x, 16)
    wh = pearl._apply_hadamard(w, 16)
    torch.testing.assert_close(xh @ wh.T, x @ w.T)


@pytest.mark.parametrize("hadamard_block_size", [0, 16])
def test_weight_quantization_is_int7_per_channel(hadamard_block_size):
    torch.manual_seed(0)
    w = torch.randn(300, 256) * torch.linspace(0.01, 3.0, 300)[:, None]
    w[7] = 0  # all-zero rows must not divide by zero
    packed = quantize_weight_int7(w, hadamard_block_size)
    assert packed.w_q.dtype == torch.int8 and packed.w_q.shape == (300, 256)
    assert packed.w_q.abs().max().item() == MAX_INT7
    assert packed.w_scale.shape == (300,) and torch.isfinite(packed.w_scale).all()
    assert (packed.w_q[7] == 0).all()
    dequant = packed.w_q.float() * packed.w_scale[:, None]
    # Round-to-nearest: every element is within half a quantization step of its rotated value.
    err = (dequant - pearl._apply_hadamard(w, hadamard_block_size)).abs()
    assert (err <= packed.w_scale[:, None] / 2 + 1e-6).all()


def test_w7a7_reference_tracks_full_precision_and_hadamard_helps_outliers():
    torch.manual_seed(0)
    x = torch.randn(64, 1024)
    x[:, 3] *= 50  # a few outlier channels, as in real LLM activations
    x[:, 700] *= 30
    w = torch.randn(512, 1024) / 32
    exact = x @ w.T
    err_plain = _rel_err(reference_w7a7_linear(x, quantize_weight_int7(w, 0)), exact)
    err_hadamard = _rel_err(reference_w7a7_linear(x, quantize_weight_int7(w, 16)), exact)
    assert err_hadamard < 0.05
    assert err_hadamard < err_plain


def test_mining_shape_follows_consensus_and_token_threshold():
    assert is_mining_shape(1024, 4096, 4096, min_tokens=1024)
    assert not is_mining_shape(1023, 4096, 4096, min_tokens=1024)  # too few tokens: noising would dominate
    assert not is_mining_shape(4096, 4096, 1024, min_tokens=1024)  # k < 16 * noise_rank
    assert not is_mining_shape(4096, 128, 4096, min_tokens=1024)  # hash pattern does not fit in n
    assert is_mining_shape(16, 4096, 4096, min_tokens=16)


def test_config_validation():
    with pytest.raises(ValueError):
        PearlMiningConfig(hadamard_block_size=12)
    with pytest.raises(ValueError):
        PearlMiningConfig(min_tokens=0)
    assert PearlMiningConfig(mode="simulate").mode is PearlMiningMode.SIMULATE


def test_disabled_mining_is_plain_linear():
    x = torch.randn(3, 5, 256, dtype=torch.bfloat16)
    w = torch.randn(128, 256, dtype=torch.bfloat16)
    assert not pearl.is_pearl_mining_enabled()
    assert torch.equal(pearl_linear(x, w), F.linear(x, w))


def test_simulate_mode_runs_w7a7_and_records_stats():
    enable_pearl_mining(PearlMiningConfig(mode="simulate", min_tokens=4, stats_interval=0))
    torch.manual_seed(0)
    x = torch.randn(2, 8, 256, dtype=torch.bfloat16)
    w = torch.randn(512, 256, dtype=torch.bfloat16) / 16
    bias = torch.randn(512, dtype=torch.bfloat16)

    out = pearl_linear(x, w, bias)
    assert out.shape == (2, 8, 512) and out.dtype == torch.bfloat16
    expected = reference_w7a7_linear(x, quantize_weight_int7(w, 16)) + bias
    torch.testing.assert_close(out, expected)
    assert _rel_err(out, F.linear(x, w, bias)) < 0.05

    stats = get_pearl_mining_stats()
    assert stats.gemm_calls == 1 and stats.total_macs == 16 * 512 * 256
    assert stats.mining_gemm_calls == 0  # simulate never mines


def test_ineligible_shapes_fall_back_to_full_precision():
    enable_pearl_mining(PearlMiningConfig(mode="simulate"))
    x = torch.randn(4, 200, dtype=torch.bfloat16)  # in_features not a multiple of the kernel's K tile
    w = torch.randn(256, 200, dtype=torch.bfloat16)
    assert torch.equal(pearl_linear(x, w), F.linear(x, w))
    x32 = torch.randn(4, 256)  # fp32 activations are not handled by the int7 quantizer
    w32 = torch.randn(256, 256)
    assert torch.equal(pearl_linear(x32, w32), F.linear(x32, w32))


def test_packed_weight_cache_is_invalidated_by_in_place_updates():
    enable_pearl_mining(PearlMiningConfig(mode="simulate"))
    torch.manual_seed(0)
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    w = torch.randn(256, 256, dtype=torch.bfloat16)
    first = pearl_linear(x, w)
    assert pearl._get_packed_weight(w) is pearl._get_packed_weight(w)  # cached between calls

    # FlexGen reuses GPU weight buffers across layers when offloading; an in-place copy must re-pack.
    w.copy_(torch.randn(256, 256, dtype=torch.bfloat16))
    second = pearl_linear(x, w)
    torch.testing.assert_close(second, reference_w7a7_linear(x, quantize_weight_int7(w, 16)))
    assert not torch.equal(first, second)


def test_kernel_modes_fail_fast_without_pearl_packages():
    pytest.importorskip("torch")
    try:
        import pearl_gemm  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("Pearl packages are installed")
    with pytest.raises(RuntimeError, match="Pearl miner packages"):
        enable_pearl_mining(PearlMiningConfig(mode="dry_run"))
    assert not pearl.is_pearl_mining_enabled()


def test_server_info_announces_mining_mode_and_stays_backward_compatible():
    info = ServerInfo(state=ServerState.ONLINE, throughput=1.0, pearl_mining="on")
    assert ServerInfo.from_tuple(info.to_tuple()).pearl_mining == "on"

    state, throughput, extra = info.to_tuple()
    del extra["pearl_mining"]  # announcement from a server without this field
    assert ServerInfo.from_tuple((state, throughput, extra)).pearl_mining is None
