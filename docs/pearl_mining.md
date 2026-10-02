# Pearl mining (experimental, opt-in)

[Pearl](https://github.com/pearl-research-labs/pearl) (PRL) is a proof-of-useful-work
chain whose "hash" is a matrix multiplication (NoisyGEMM). A BloomBee server can mine
it with the GEMMs it already runs for inference requests, instead of burning a GPU on
throwaway matrices. An idle server mines nothing, and mining never changes which
requests a server takes.

Mining is **off by default**. Turn it on with `--pearl_mining`.

## How it works

With mining enabled, the Llama-family block projections (q/k/v/o, gate/up/down)
computed in `flexgen_utils/pytorch_backend.py` go through `bloombee.mining.pearl.pearl_linear`:

1. On first use, each weight is quantized to int7 per output channel, with a
   block-Hadamard rotation folded in (`--pearl_hadamard_block_size`, default 16) to spread
   activation outliers. The packed copy is cached on the weight tensor.
2. Activations are rotated and quantized to int7 per token by Pearl's fused kernel.
3. If the GEMM is large enough (below), it runs as **NoisyGEMM**: the operands are noised
   with rank-128 noise seeded by the current block header, multiplied, and denoised.
   The output is the exact int7 product, and hashes of the noisy tiles are lottery tickets.
   A winning ticket is turned into a ZK proof by the local `pearl-gateway` and submitted.
4. Smaller GEMMs run as a plain int7 GEMM, so a server's numerics do not depend on batch shape.

A GEMM mines when all of these hold:

| Dimension | Requirement | Why |
|---|---|---|
| tokens `m` | `>= --pearl_min_tokens` (default 1024) | Noising costs `O(n·k·128)`, i.e. as much as a 128-token GEMM; below ~1k tokens it dominates |
| out features `n` | `>= 256` | Consensus: the hash pattern must fit in a 128x256 tile |
| in features `k` | `>= 2048` | Consensus: `k >= 16 * noise_rank` |

### Decode

Single-token decode GEMMs (`m` = number of sequences in the batch) do not mine by default.
Consensus would accept small `m`, so you can lower `--pearl_min_tokens` (e.g. to 64) to
also mine large decode batches, speculative-decoding verification trees and
microbatches. Each such call pays the full noising cost for a fraction of the tickets, so
measure the slowdown with `dry_run` before lowering it in production (it has not been
benchmarked yet).

### Privacy

Only hashes of the operands, the matrix sizes, the winning tile coordinates and a ZK proof
go on chain. Activations (derived from user prompts) are sent only to the local gateway
process to build the proof.

## Requirements

* **`on` / `dry_run`**: NVIDIA H100/H200 (sm90) on every visible GPU, Python 3.12, and the
  Pearl miner packages `pearl-gemm`, `miner-base` and `pearl-gateway`. They pin their own
  torch/CUDA versions, so use a separate environment:

  ```bash
  git clone https://github.com/pearl-research-labs/pearl && cd pearl
  uv sync --package vllm-miner   # builds pearl-gemm's CUDA kernels; pulls miner-base and pearl-gateway
  uv pip install -e /path/to/BloomBee
  ```

* **`on`** also needs a running `pearl-gateway` connected to a `pearld` node (or a pool) and a
  payout wallet. Follow Pearl's miner README; the gateway listens on `/tmp/pearlgw.sock` by
  default (`--pearl_gateway_socket`).
* **`simulate`**: nothing beyond BloomBee. Works on any device.

## Modes

| Mode | What runs | Use it to |
|---|---|---|
| `simulate` | Pure-PyTorch W7A7 reference, no mining | Check your model's output quality under int7 before buying into mining |
| `dry_run` | Pearl kernels, dummy gateway (no blocks) | Measure speed and the share of GEMMs that would mine on your traffic |
| `on` | Pearl kernels + real gateway | Mine |

```bash
python -m bloombee.cli.run_server meta-llama/Llama-3.1-8B-Instruct \
    --initial_peers $PEERS --torch_dtype bfloat16 --pearl_mining dry_run
```

Every 5 minutes the server logs a summary such as
`Pearl mining: 120/9000 GEMMs mined (61.3% of MACs), blocks found=0, submitted=0, errors=0`.
The MAC share is what you earn on, compared with a dedicated miner running the same GPU at 100%.

## Effects on the swarm

* Mining servers compute their blocks in W7A7, so their outputs differ slightly from fp16/bf16
  servers. They announce `pearl_mining` in their DHT `ServerInfo` so clients can see it.
  Older clients ignore the field.
* GPU memory: the int8 weight copy is kept next to the original weights (+50% weight memory
  for fp16/bf16). Reserve memory accordingly (for example with `--num_blocks`).
* With weight offloading, a reused GPU buffer is re-quantized each time it is refilled.
* FlexGen weight compression is decompressed before the int7 path, so combining the two
  gives no memory saving.
* Only the Llama-family FlexGen path is hooked. Other architectures run unchanged.
