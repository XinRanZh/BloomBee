"""Pearl (PRL) proof-of-useful-work mining inside BloomBee's linear layers.

Pearl secures its chain with *NoisyGEMM*: an int7 x int7 matrix multiplication
whose operands are perturbed by low-rank noise derived from the current block
header. The kernel removes the noise again, so the product is still the exact
int7 GEMM result, while hashing tiles of the noisy intermediate as lottery
tickets for the next block. Mining therefore costs nothing beyond the GEMMs a
server already runs for real inference (plus the noising overhead), and an idle
server mines nothing.

This module routes the transformer-block projections that go through
``flexgen_utils/pytorch_backend.py`` (q/k/v/o, gate/up/down) to Pearl kernels
when mining is enabled:

* Every eligible projection is computed as a W7A7 GEMM (int7 per-output-channel
  weights, int7 per-token activations, block-Hadamard rotation for outliers),
  so a server's numerics do not depend on the batch shape.
* GEMMs whose token count reaches ``min_tokens`` (and whose shape satisfies
  Pearl's consensus rules) run as NoisyGEMM and may find a block; the rest run
  as plain int7 GEMM. Noising costs O(n * k * rank) per call, so it only pays
  off once the GEMM itself is large (vllm-miner uses ``m >= 1024``).

Modes:

* ``on``: Pearl kernels + a running ``pearl-gateway`` that submits blocks.
* ``dry_run``: Pearl kernels with Pearl's dummy gateway client; nothing is
  submitted. Use it to measure speed on a Hopper GPU without a node.
* ``simulate``: pure-PyTorch W7A7 reference math, no Pearl packages or Hopper
  GPU needed. Use it to check a model's output quality under mining numerics.

The kernel path requires the Pearl miner packages (``pearl-gemm``,
``miner-base``, ``pearl-gateway`` from https://github.com/pearl-research-labs/pearl)
and an sm90 (H100/H200) GPU. The NoisyGEMM call sequence is ported from Pearl's
``vllm-miner`` (ISC license, Copyright (c) 2025-2026 Pearl Research Labs).
"""

from __future__ import annotations

import dataclasses
import enum
import math
import os
import threading
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from bloombee.utils.hivemind_compat import get_logger

logger = get_logger(__name__)

MAX_INT7 = 63
# Pearl consensus (zk-pow sanity_checks.rs): k >= 1024 and k >= 16 * noise_rank,
# with noise_rank >= 128.
MIN_MINING_K = 2048
# The hash column pattern of the 128x256 tile reaches column 249, so n must exceed it.
MIN_MINING_N = 256
# Pearl GEMM kernels need K % tile_k == 0 and 128-bit aligned bf16 output rows.
GEMM_K_ALIGN = 128
GEMM_N_ALIGN = 8
DEFAULT_HADAMARD_BLOCK_SIZE = 16
# Rows quantized at once when packing weights, bounds the fp32 scratch memory.
_WEIGHT_QUANT_ROW_CHUNK = 1024
_PACKED_ATTR = "_bloombee_pearl_packed"


class PearlMiningMode(str, enum.Enum):
    OFF = "off"
    ON = "on"
    DRY_RUN = "dry_run"
    SIMULATE = "simulate"


@dataclasses.dataclass(frozen=True)
class PearlMiningConfig:
    mode: PearlMiningMode = PearlMiningMode.ON
    # Minimum number of tokens (GEMM rows) for a call to run as NoisyGEMM.
    min_tokens: int = 1024
    # Block-Hadamard width applied to activations and folded into weights; 0 disables it.
    hadamard_block_size: int = DEFAULT_HADAMARD_BLOCK_SIZE
    gateway_socket_path: str = "/tmp/pearlgw.sock"
    # Seconds between mining summaries in the log; 0 disables them.
    stats_interval: float = 300.0

    def __post_init__(self):
        object.__setattr__(self, "mode", PearlMiningMode(self.mode))
        if self.min_tokens < 1:
            raise ValueError(f"min_tokens must be >= 1, got {self.min_tokens}")
        hb = self.hadamard_block_size
        if hb != 0 and (hb < 2 or hb & (hb - 1) != 0 or hb > GEMM_K_ALIGN):
            raise ValueError(f"hadamard_block_size must be 0 or a power of two in [2, 128], got {hb}")

    @property
    def uses_kernels(self) -> bool:
        return self.mode in (PearlMiningMode.ON, PearlMiningMode.DRY_RUN)


@dataclasses.dataclass
class PackedInt7Weight:
    """A linear weight prepared for W7A7 GEMM: ``w_q`` is (n, k) int8 in [-63, 63]."""

    w_q: torch.Tensor
    w_scale: torch.Tensor  # (n,) float32
    hadamard_block_size: int

    @property
    def out_features(self) -> int:
        return self.w_q.shape[0]

    @property
    def in_features(self) -> int:
        return self.w_q.shape[1]


@dataclasses.dataclass
class PearlMiningStats:
    gemm_calls: int = 0
    mining_gemm_calls: int = 0
    total_macs: int = 0
    mining_macs: int = 0
    mining_errors: int = 0
    blocks_found: int = 0
    blocks_submitted: int = 0

    @property
    def mining_mac_fraction(self) -> float:
        return self.mining_macs / self.total_macs if self.total_macs else 0.0


# ---------------------------------------------------------------------------
# Reference math (pure PyTorch; also used to pack weights for the kernels)
# ---------------------------------------------------------------------------


def hadamard_block(size: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Normalized ``size x size`` Hadamard matrix in Pearl's convention (first column negated).

    Matches ``pearl_gemm.quantization.quantize``, which computes ``x_blocks @ H``.
    """
    if size < 1 or size & (size - 1) != 0:
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < size:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h[:, 0] *= -1
    return (h / math.sqrt(size)).to(device=device, dtype=dtype)


def _apply_hadamard(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size == 0:
        return x
    *leading, k = x.shape
    h = hadamard_block(block_size, device=x.device, dtype=x.dtype)
    return (x.reshape(*leading, k // block_size, block_size) @ h).reshape(*leading, k)


def quantize_weight_int7(
    weight: torch.Tensor, hadamard_block_size: int = DEFAULT_HADAMARD_BLOCK_SIZE
) -> PackedInt7Weight:
    """Symmetric per-output-channel int7 quantization of an ``(n, k)`` weight.

    The block-Hadamard rotation is folded into the weight (``W @ H``) so that
    ``(x @ H) @ (W @ H).T == x @ W.T`` when the kernel rotates activations.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight, got shape {tuple(weight.shape)}")
    n, k = weight.shape
    if hadamard_block_size and k % hadamard_block_size != 0:
        raise ValueError(f"in_features={k} is not divisible by hadamard_block_size={hadamard_block_size}")
    w_q = torch.empty((n, k), dtype=torch.int8, device=weight.device)
    w_scale = torch.empty((n,), dtype=torch.float32, device=weight.device)
    for start in range(0, n, _WEIGHT_QUANT_ROW_CHUNK):
        rows = _apply_hadamard(weight[start : start + _WEIGHT_QUANT_ROW_CHUNK].float(), hadamard_block_size)
        scale = rows.abs().amax(dim=1) / MAX_INT7
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        quantized = torch.round(rows / scale[:, None]).clamp_(-MAX_INT7, MAX_INT7)
        w_q[start : start + rows.shape[0]] = quantized.to(torch.int8)
        w_scale[start : start + rows.shape[0]] = scale
    return PackedInt7Weight(w_q=w_q, w_scale=w_scale, hadamard_block_size=hadamard_block_size)


def quantize_activations_int7(x: torch.Tensor, hadamard_block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference for ``pearl_gemm.quantization.quantize``: per-token int7 of an ``(m, k)`` input."""
    xf = _apply_hadamard(x.float(), hadamard_block_size)
    scale = xf.abs().amax(dim=1, keepdim=True) / MAX_INT7
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round(xf / scale).clamp_(-MAX_INT7, MAX_INT7).to(torch.int8), scale


def reference_w7a7_linear(x: torch.Tensor, packed: PackedInt7Weight) -> torch.Tensor:
    """What the Pearl kernels compute, in plain PyTorch: ``x @ W.T`` under W7A7 quantization."""
    x2 = x.reshape(-1, packed.in_features)
    x_q, x_scale = quantize_activations_int7(x2, packed.hadamard_block_size)
    # float64 holds the int32 accumulator of the kernel exactly (|sum| <= 63 * 63 * 2^16).
    acc = x_q.double() @ packed.w_q.double().T
    out = acc * x_scale.double() * packed.w_scale.double()[None, :]
    return out.to(x.dtype).reshape(*x.shape[:-1], packed.out_features)


def is_mining_shape(m: int, n: int, k: int, min_tokens: int) -> bool:
    """Whether a GEMM of this shape should run as NoisyGEMM (consensus rules + overhead threshold)."""
    return m >= min_tokens and n >= MIN_MINING_N and k >= MIN_MINING_K


# ---------------------------------------------------------------------------
# Pearl kernel runtime (lazy: only imported when mode is on/dry_run)
# ---------------------------------------------------------------------------


def _import_pearl():
    try:
        import pearl_gemm  # noqa: F401
        from miner_base.async_loop_manager import AsyncLoopManager  # noqa: F401
        from pearl_gateway.config import MinerRpcConfig  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Pearl mining needs the Pearl miner packages (pearl-gemm, miner-base, pearl-gateway). "
            "See docs/pearl_mining.md for installation, or use --pearl_mining simulate to test "
            f"the numerics without them. Import error: {e}"
        ) from e


class _StatusCheckCallback:
    """Runs on the miner's async loop once a NoisyGEMM finished; submits a block if one was found."""

    def __init__(
        self, runtime: "_PearlKernelRuntime", host_signal_header_pinned, commitment_hash_a, commitment_hash_b, a, b,
        mining_job,
    ):
        self.runtime = runtime
        self.host_signal_header_pinned = host_signal_header_pinned
        self.commitment_hash_a = commitment_hash_a
        self.commitment_hash_b = commitment_hash_b
        self.a = a
        self.b = b
        self.mining_job = mining_job

    def __call__(self, handle_submit_block):
        from pearl_gateway.comm.dataclasses import CommitmentHash, OpenedBlockInfo
        from pearl_gemm import HostSignalStatus, extract_indices, get_host_signal_header

        try:
            header = get_host_signal_header(self.host_signal_header_pinned)
            if header.status == HostSignalStatus.kSignalTriggered:
                logger.info(f"Pearl block found! {header}")
                _stats_add(blocks_found=1)
                indices = extract_indices(header)
                opened_block_info = OpenedBlockInfo(
                    A_row_indices=indices.A_row_indices,
                    B_column_indices=indices.B_column_indices,
                    A=self.a.cpu().detach(),
                    B_t=self.b.cpu().detach(),
                    commitment_hash=CommitmentHash(
                        noise_seed_A=self.commitment_hash_a.cpu().numpy().tobytes(),
                        noise_seed_B=self.commitment_hash_b.cpu().numpy().tobytes(),
                    ),
                    noise_rank=self.runtime.settings.noise_rank,
                )
                handle_submit_block(opened_block_info, self.mining_job)
        finally:
            self.runtime.pinned_pool.release(self.host_signal_header_pinned)
            self.host_signal_header_pinned = self.a = self.b = None
            self.commitment_hash_a = self.commitment_hash_b = None


class _PearlKernelRuntime:
    def __init__(self, config: PearlMiningConfig):
        _import_pearl()
        from miner_base.async_loop_manager import AsyncLoopManager
        from miner_base.settings import MinerSettings
        from pearl_gateway.config import MinerRpcConfig
        from pearl_gemm import HostSignalHeaderPinnedPool

        self.config = config
        # In dry_run, Pearl's dummy client serves a job with target=1, so no block is ever found.
        self.settings = MinerSettings(
            enable_async_cuda_event_processing=True,
            no_gateway=config.mode == PearlMiningMode.DRY_RUN,
        )
        self.manager = AsyncLoopManager(
            MinerRpcConfig(transport="uds", socket_path=config.gateway_socket_path), self.settings
        )
        self.manager.start()
        self.pinned_pool = HostSignalHeaderPinnedPool(self.settings.pinned_pool_size)
        logger.info(
            f"Pearl mining runtime started (mode={config.mode.value}, noise_rank={self.settings.noise_rank}, "
            f"min_tokens={config.min_tokens}, gateway={config.gateway_socket_path})"
        )

    def stop(self):
        self.manager.wait_until_done_submitting_blocks()
        self.manager.stop()

    def linear(self, x2: torch.Tensor, packed: PackedInt7Weight) -> torch.Tensor:
        from pearl_gemm.quantization import quantize

        m, k = x2.shape
        n = packed.out_features
        x_q = torch.empty((m, k), dtype=torch.int8, device=x2.device)
        x_scale = torch.empty((m, 1), dtype=torch.float32, device=x2.device)
        quantize(x2, x_q, x_scale, smooth_scale=None, max_val=MAX_INT7, block_size=packed.hadamard_block_size)
        x_scale = x_scale.view(m)

        c = torch.empty((m, n), dtype=torch.bfloat16, device=x2.device)
        mined = False
        if is_mining_shape(m, n, k, self.config.min_tokens):
            try:
                self._noisy_gemm(x_q, packed.w_q, x_scale, packed.w_scale, c)
                mined = True
            except Exception as e:
                # Mining must never break inference: fall back to the plain int7 GEMM.
                _stats_add(mining_errors=1)
                logger.warning(f"Pearl NoisyGEMM failed for shape m={m} n={n} k={k}, running plain GEMM: {e!r}")
        if not mined:
            self._gemm(x_q, packed.w_q, x_scale, packed.w_scale, c)
        _record_gemm(m, n, k, mined)
        return c

    def _gemm(self, a, b, a_scale, b_scale, c):
        from pearl_gemm import gemm

        s = self.settings
        gemm(A=a, B=b, A_scales=a_scale, B_scales=b_scale, C=c,
             tile_size_m=s.tile_size_m, tile_size_n=s.tile_size_n, tile_size_k=s.tile_size_k)

    def _noisy_gemm(self, a, b, a_scale, b_scale, c):
        from miner_base.commitment_hash import CommitmentHasher
        from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
        from pearl_gemm import (
            commitment_hash_from_merkle_roots,
            get_host_signal_sync_size,
            get_required_scratchpad_bytes,
            make_pow_target_tensor,
            noise_gen,
            noisy_gemm,
            tensor_hash,
        )

        s = self.settings
        device = a.device
        m, k = a.shape
        n = b.shape[0]
        r = s.noise_rank

        matmul_config = GPUMatmulConfigFactory.create(k=k, noise_rank=r)
        mining_job = self.manager.get_mining_job()
        adjusted_target = mining_job.adjust_target(mining_config=matmul_config.mining_config)
        hash_key = CommitmentHasher.get_key(mining_job.incomplete_header_bytes, matmul_config.mining_config)
        key_tensor = torch.frombuffer(bytearray(hash_key), dtype=torch.uint8).to(device)

        scratchpad = torch.empty(get_required_scratchpad_bytes(max(m * k, n * k)), dtype=torch.uint8, device=device)
        a_hash = torch.empty(32, dtype=torch.uint8, device=device)
        b_hash = torch.empty(32, dtype=torch.uint8, device=device)
        tensor_hash(a.to(torch.uint8), key_tensor, a_hash, scratchpad)
        tensor_hash(b.to(torch.uint8), key_tensor, b_hash, scratchpad)

        commitment_hash_a = torch.empty(32, dtype=torch.uint8, device=device)
        commitment_hash_b = torch.empty(32, dtype=torch.uint8, device=device)
        commitment_hash_from_merkle_roots(
            a_hash, b_hash, key_tensor, commitment_hash_a, commitment_hash_b,
            salted_dims=(m, n) if mining_job.cert_version.uses_salted_seeds else None,
        )

        eal = torch.empty((m, r), dtype=torch.int8, device=device)
        ebr = torch.empty((n, r), dtype=torch.int8, device=device)
        ear_r_major = torch.empty((k, r), dtype=torch.int8, device=device)
        ebl_r_major = torch.empty((k, r), dtype=torch.int8, device=device)
        ear_k_major = torch.empty((r, k), dtype=torch.int8, device=device)
        ebl_k_major = torch.empty((r, k), dtype=torch.int8, device=device)
        eal_fp16 = torch.empty((m, r), dtype=torch.float16, device=device)
        ebr_fp16 = torch.empty((n, r), dtype=torch.float16, device=device)
        noise_gen(
            R=r, EAL=eal, EAL_fp16=eal_fp16, EAR_R_major=ear_r_major, EAR_K_major=ear_k_major,
            EBL_R_major=ebl_r_major, EBL_K_major=ebl_k_major, EBR=ebr, EBR_fp16=ebr_fp16,
            key_A=commitment_hash_a, key_B=commitment_hash_b,
        )

        host_signal_sync = torch.zeros((get_host_signal_sync_size(),), dtype=torch.int8, device=device)
        host_signal_header_pinned = self.pinned_pool.acquire()
        try:
            noisy_gemm(
                A=a, B=b, EAL=eal, EAL_fp16=eal_fp16, EBR=ebr, EBR_fp16=ebr_fp16,
                EAR_R_major=ear_r_major, EBL_R_major=ebl_r_major,
                EAR_K_major=ear_k_major, EBL_K_major=ebl_k_major,
                AxEBL_fp16=torch.empty((m, r), dtype=torch.float16, device=device),
                EARxBpEB_fp16=torch.empty((n, r), dtype=torch.float16, device=device),
                ApEA=torch.empty((m, k), dtype=torch.int8, device=device),
                BpEB=torch.empty((n, k), dtype=torch.int8, device=device),
                A_scales=a_scale, B_scales=b_scale, C=c,
                host_signal_header_pinned=host_signal_header_pinned,
                host_signal_sync=host_signal_sync,
                pow_target=make_pow_target_tensor(adjusted_target, device=device),
                pow_key=commitment_hash_a.view(torch.uint32),
                tile_size_m=s.tile_size_m, tile_size_n=s.tile_size_n, tile_size_k=s.tile_size_k,
                run_noising_A=True, run_noising_B=True, skip_reduction=False, skip_denoising=False,
            )
        except BaseException:
            self.pinned_pool.release(host_signal_header_pinned)
            raise

        # The callback owns the pinned header and keeps a/b alive until the kernel finished.
        cuda_event = torch.cuda.Event()
        cuda_event.record(torch.cuda.current_stream(device))
        self.manager.schedule_status_check(
            cuda_event,
            _StatusCheckCallback(
                self, host_signal_header_pinned, commitment_hash_a, commitment_hash_b, a, b, mining_job
            ),
        )


# ---------------------------------------------------------------------------
# Global state and the public entry points
# ---------------------------------------------------------------------------

_config: Optional[PearlMiningConfig] = None
_runtime: Optional[_PearlKernelRuntime] = None
_runtime_pid: Optional[int] = None
_runtime_lock = threading.Lock()
_stats = PearlMiningStats()
_stats_lock = threading.Lock()
_last_stats_log = time.monotonic()


def _stats_add(**deltas):
    with _stats_lock:
        for name, delta in deltas.items():
            setattr(_stats, name, getattr(_stats, name) + delta)


def _record_gemm(m: int, n: int, k: int, mined: bool):
    global _last_stats_log
    macs = m * n * k
    with _stats_lock:
        _stats.gemm_calls += 1
        _stats.total_macs += macs
        if mined:
            _stats.mining_gemm_calls += 1
            _stats.mining_macs += macs
        interval = _config.stats_interval if _config is not None else 0
        now = time.monotonic()
        should_log = interval > 0 and now - _last_stats_log >= interval
        if should_log:
            _last_stats_log = now
    if should_log:
        s = get_pearl_mining_stats()
        logger.info(
            f"Pearl mining: {s.mining_gemm_calls}/{s.gemm_calls} GEMMs mined "
            f"({s.mining_mac_fraction:.1%} of MACs), blocks found={s.blocks_found}, "
            f"submitted={s.blocks_submitted}, errors={s.mining_errors}"
        )


def enable_pearl_mining(config: PearlMiningConfig) -> None:
    """Turn mining on for this process. Fails fast if the kernel path cannot work here."""
    global _config
    if config.mode == PearlMiningMode.OFF:
        disable_pearl_mining()
        return
    if config.uses_kernels:
        _import_pearl()
        if not torch.cuda.is_available():
            raise RuntimeError("Pearl mining kernels need a CUDA GPU (use --pearl_mining simulate to test numerics)")
        for index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(index)
            if major < 9:
                raise RuntimeError(
                    f"Pearl GEMM kernels need compute capability >= 9.0 (H100/H200); "
                    f"cuda:{index} is sm{major}{minor}"
                )
    _config = config
    if config.uses_kernels:
        _get_runtime()  # surface gateway/connection errors at startup, not in the first forward pass
    logger.info(f"Pearl mining enabled: {config}")


def disable_pearl_mining() -> None:
    global _config, _runtime, _runtime_pid
    with _runtime_lock:
        runtime, _runtime, _runtime_pid = _runtime, None, None
        _config = None
    if runtime is not None:
        runtime.stop()


def is_pearl_mining_enabled() -> bool:
    return _config is not None


def get_pearl_mining_stats() -> PearlMiningStats:
    with _stats_lock:
        snapshot = dataclasses.replace(_stats)
    runtime = _runtime
    if runtime is not None:
        snapshot.blocks_submitted = runtime.manager.blocks_submitted
    return snapshot


def _get_runtime() -> _PearlKernelRuntime:
    global _runtime, _runtime_pid
    # Threads (and the miner's async loop) do not survive fork(): rebuild the runtime in a child process.
    if _runtime is not None and _runtime_pid == os.getpid():
        return _runtime
    with _runtime_lock:
        if _runtime is None or _runtime_pid != os.getpid():
            _runtime = _PearlKernelRuntime(_config)
            _runtime_pid = os.getpid()
        return _runtime


def _get_packed_weight(weight: torch.Tensor) -> Optional[PackedInt7Weight]:
    """Int7-packed copy of ``weight``, cached on the tensor and invalidated by in-place writes.

    FlexGen may reuse a GPU buffer for different layers when weights are offloaded;
    those copies bump ``weight._version``, which forces a re-pack.
    """
    cached = getattr(weight, _PACKED_ATTR, None)
    if cached is not None and cached[0] == weight._version:
        return cached[1]
    hb = _config.hadamard_block_size
    n, k = weight.shape
    if k % GEMM_K_ALIGN != 0 or n % GEMM_N_ALIGN != 0 or (hb and k % hb != 0):
        packed = None
    else:
        with torch.no_grad():
            packed = quantize_weight_int7(weight.detach(), hb)
    setattr(weight, _PACKED_ATTR, (weight._version, packed))
    return packed


def pearl_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Drop-in for ``F.linear`` that mines Pearl on eligible projections when mining is enabled."""
    config = _config
    if config is None or weight.dim() != 2 or x.dtype not in (torch.float16, torch.bfloat16):
        return F.linear(x, weight, bias)
    if config.uses_kernels and not x.is_cuda:
        return F.linear(x, weight, bias)
    packed = _get_packed_weight(weight)
    if packed is None or x.numel() == 0:
        return F.linear(x, weight, bias)

    if config.mode == PearlMiningMode.SIMULATE:
        out = reference_w7a7_linear(x, packed)
        _record_gemm(x.numel() // packed.in_features, packed.out_features, packed.in_features, False)
    else:
        x2 = x.reshape(-1, packed.in_features).contiguous()
        with torch.cuda.device(x.device):
            c = _get_runtime().linear(x2, packed)
        out = c.to(x.dtype).view(*x.shape[:-1], packed.out_features)
    if bias is not None:
        out = out + bias
    return out
