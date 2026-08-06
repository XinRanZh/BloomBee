# 交接文件：EAGLE-2 SD GPU-tree 迁移 + S2S 回归修复 + v3/v4 传输异常分析

**日期：** 2026-08-06 · **分支：** `eagle2-runtime`（本地 `/Users/zhangxinran/research/coop/BloomBee`）
**测试机：** `ubuntu@54.167.236.115`（4× NVIDIA L4, 181G RAM, 466G 空闲），SSH key `~/.ssh/bloombee_iad`

本文档面向接手的 agent，自包含地记录本轮工作的**目标、改动、验证证据、结论、未竟事项**。读完后应能直接继续工作，无需重读对话。

---

## 0. 任务来源（用户原始诉求）

1. PR57 已合并进 bloombee 主线，遗留最大问题：**EAGLE-2 SD 的 tree 结构生成和 accept walk 在 CPU/Python 中进行**，应搬到 GPU 以大幅缩小 compute headroom。
2. v3/v4 结果异常：**为什么 GPU2→1（回传腿）比 1→2（push 腿）传输时间长？**用户心智模型：两腿都传 eagle2 数据，1→2 还额外多带中间 hidden state，所以 2→1 不应更慢。
3. 追问：v4（双腿 int8）测完 return 仍大于 push，为什么？push/return 各包含什么？
4. 追问：支持的设备上用 **fp8 替代 int8** 有好处吗？

## 1. 环境现状（可直接复用）

- 测试机已 provision 完毕：`~/bbvenv`（torch 2.7.1+cu126, transformers 5.14.1, numpy 1.26.4）、模型 `~/models/vicuna-13b-v1.3`（25G）与 `~/models/EAGLE-Vicuna-13B-v1.3`（2.1G）、FlexGen numpy 缓存 `/tmp/data/llama_weights/vicuna-13b-v1.3-np`（已预转换，含 sentinel）。
- 代码：`~/BloomBee` 是本地工作树的 rsync 副本（**不含 .git**；与本地同步截至本轮最后一次 rsync——注意本地若再改需重新 rsync）。
- 验证脚本：`~/validation/`（`remote_setup.sh`、`run_gpu_tree_validation.sh`、`bench_sd.py`、`bench_drafter_build.py`、`bench_drafter_subphase.py`、`prof_nonzero.py`、`prof_stack.py`）。本地源文件在 `/tmp/gpu_tree_validation/`（易失）。
- 复跑验证 campaign：`ssh … 'nohup bash ~/validation/run_gpu_tree_validation.sh > ~/gpu_tree_results.log 2>&1 &'`，约 15 分钟，9 个 cell（b8/b32 × obj/bridge/shadow/native + het 对）。
- 结果已归档本地 `results/gpu_tree_migration_2026-08-06/`（含 README 与逐 cell json/io）。

### 环境坑（接手必读）

- **`huggingface-cli download 已失效**：hub 1.x 改名 `hf download`。`remote_setup.sh` 里的旧命令会静默失败（只打印 help）。
- **双 server 并发首启会竞态损坏 FlexGen np 缓存**：两台 server 同时 `convert_local_llama_weights` 到同一目录会写截断（S1 报 `cannot reshape array of size 4194240 into shape (5120,5120)`）。已按模型单线程预转换解决；若删了 `/tmp/data/llama_weights/*-np` 必须重新预转换再启动 swarm。
- **`pkill -f <pattern>` 会自杀**：ssh 远程命令行本身含 pattern 文本。用 `[x]` 括号技巧（`pkill -f "run_serve[r]"`）。
- **cProfile 在本机会误导**：`torch.tensor(list, device=cuda)` 的 tottime 把 GPU drain 算进去；真实分解要用 torch.profiler 或 enqueue/drain 分离计时。
- 本机 torch 算子启动开销异常高（小 matmul ~92µs、add ~19µs、arange ~8µs），**op 数量本身就是成本**；`aten::nonzero`（布尔掩码索引）每次强制一次 stream sync（~0.4-0.5ms）。

## 2. 代码改动总清单（本地工作树，未提交）

### 2.1 GPU-tree 迁移（本轮主体工作）

| 文件 | 改动 |
|---|---|
| `src/bloombee/models/llama/tensor_tree.py` | ①`greedy_verify_tensorized` 重写：固定步数（新增 `TensorTreeBatch.max_depth_host` 深度上界，10-3 树 6 步而非 12 步）、**逐步 sync 全删**、输出组装全向量化（accepted 前缀性质→切片），整个 verifier 只剩 1 次 host sync（`accept_count.max()` 定输出宽度）。②`tensor_tree_from_eagle_candidate_tensors`：闭包循环去 sync（固定 D+1 迭代，幂等）；最终组装向量化（scatter 替代逐行循环）；**path-key 构造改用 dumpster-column scatter，消除 24 次/轮的布尔索引 nonzero 同步**；删 dead `NEG_INF`。③新增 `local_tree_mask_from_tensor_tree`（GPU 祖先闭包，1 次深度 sync）与 `prepare_incremental_tensor_tree_batch`（local 热路径全 GPU；prefill/dense 走共享 impl 保持字节一致）。 |
| `src/bloombee/models/llama/spe_dec_tree.py` | `prepare_incremental_tree_batch` 重构：线性化/leaf-path 留在 wrapper，mask 装配抽成 `_prepare_incremental_tree_batch_impl`（对象路径与 tensor 路径共用同一份 mask 代码，防分叉）。早退语义与原实现完全一致。 |
| `src/bloombee/models/llama/eagle_drafter.py` | ①新增 `_build_tensor_tree_batched`：扩展候选全程 `[B,C]` GPU 张量（slot==creation_index；fp64 path_logp 累积 + fp32 frontier topk，与对象路径逐位一致）；②`build_trees_parallel` 接线：native 条件（EMIT=1 + TENSOR_TREE=1 + 非 sampling + B>1 + 单 prefix 组全覆盖）满足时返回 `None` 并置 `self.last_tensor_tree`；`shadow` 模式跑两套并逐轮断言相等；③**`build_trees_parallel` 逐行 `.item()` 批量化**（`seq_lengths.tolist()` 等，原来 3B 次 sync/轮）；④root tokens 改 GPU 直取（`prev_last_token.to(device)`），去掉 `torch.tensor(list→cuda)` 的同步页式拷贝。 |
| `src/bloombee/models/llama/speculative_model.py` | `_verify_trees_with_forward` 加 `tt` 参数：native 时跳过对象 prepare 与 bridge，直接 tensor prepare + tensor verify；generate 循环捕获 `last_tensor_tree`、ROUND_PROF 支持 tt 节点计数。sampling 路径完全不动。 |

**开关语义（默认全关，对象路径不变）**：
- `BLOOMBEE_TENSOR_TREE=1`：tensor prepare + tensorized greedy verifier（无 native tt 时走 stage-1 bridge）。
- `BLOOMBEE_TENSOR_TREE_EMIT=shadow`：对象树 + native tt 并行构建并逐轮断言一致（验证用，2× drafter 成本）。
- `BLOOMBEE_TENSOR_TREE_EMIT=1`：native emit（需 TENSOR_TREE=1、greedy、B>1、全批同一 batched prefix 组）；不满足时自动回退对象路径。**异构 batch（各行 seq_len 不等）走逐行对象路径，native 不覆盖**（见 §5 遗留）。

### 2.2 关键回归修复（建议优先合入主线）

`src/bloombee/server/s2s_flow.py`：`S2SLinkTelemetry` 在 8/4 的 upstream 合并（7aebee7，带入 PR61）中**丢了 `@dataclass` 装饰器**。后果：每次 S2S push 抛 `TypeError: S2SLinkTelemetry() takes no arguments`，client verify 每轮 3×60s 重试——**合并后整个 S2S 推理路径是坏的**。7 月 campaign 没踩到是因为跑的是合并前的 c89d2c3。已 +1 行修复并远程验证。

### 2.3 先前已存在、本轮未动的未提交改动

`inference_session.py`、`block_functions.py`、`handler.py`、`s2s_activation_quant.py` 的未提交改动是 v4 campaign 的 tail→client int8 回传量化 + 合成延迟 + ROUND_PROF 仪器化（`BLOOMBEE_TAIL_ACTIVATION_QUANT` 默认关），原样保留并随 rsync 到了测试机。

### 2.4 新增测试

- `tests/test_eagle_native_emit.py`（pytest，12 例）：CPU stub `_step`/`_logits` 下 native==object 逐位一致；EMIT=1 返回 None+置 `last_tensor_tree`；shadow 通过；do_sample 正确回退。
- `scripts/test_tensor_prepare_identity.py`：80/80 随机树 prefill/local/dense 三分支 mask 字节一致。
- `scripts/test_tensor_greedy_identity_randomized.py`：400/400 随机树 verifier 五元组输出完全一致（含重复兄弟 token、root-only 行、first/non-first iteration）。
- 既有门槛保持绿：`test_native_tensor_tree.py` 60/60、`test_tensor_greedy_identity.py` 4/4、pytest 69 passed（spe_dec_tree / eagle_drafter_budget / spec_decoding_* 等）。

## 3. 远程 E2E 验证结果（4×L4，vicuna-13b 0:20/20:40 在 GPU0/1，client+drafter 在 GPU2，greedy，qOFF，无合成延迟）

**Token 一致性（硬门槛）**：b8_obj vs b8_bridge vs b8_native、b32 同组、b8het_obj vs b8het_native —— **全部逐字节 IDENTICAL**（8/8、32/32、8/8 行）。b8_shadow 逐轮断言 native==object 通过。

**性能（稳态每轮 ms / 稳态 tok/s）**：

| cell | draft obj→native | extract obj→native | tps obj→native |
|---|---|---|---|
| b8 同质 | 40.3 → 36.6 | 8.2 → 10.5 | 75.4 → 76.5 (+1.5%) |
| b32 同质 | 66.0 → 57.1 | 10.7 → 11.5 | 104.7 → 107.7 (+2.9%) |

**诚实解读**：树结构的 CPU/Python 工作已清零（28 次 stream sync/轮 → 1-2 次；无 `_CandNode`、无 Python DFS/排序/闭包/绑定、无逐行 mask while 循环），但这台 L4 机器上单轮被 ①verify forward RPC（b8 ~360ms / b32 ~855ms）和 ②EAGLE head 扩展的 GPU GEMM + torch 启动开销主导，净 E2E 收益 modest。extract 在 b8 略高于对象路径（10.5 vs 8.2ms：7 次全 B lm_head 投影 vs 旧路径 ~4-5 次 + 每深度 sync），b32 基本持平（11.5 vs 10.7）。本轮迁移的最大价值是**解锁后续杠杆**（见 §5），且 drafter 输出 TensorTreeBatch 后 active-row compaction 就是一次 `index_select`。

## 4. v3/v4 传输异常——完整结论

### 4.1 两条腿各装什么（代码依据：`block_functions.py:2548-2598`、`handler.py` yield 路径）

- **push（S1→S2）**：int8 hidden `[B, 1+N, H]`（S1 过 block 20 的**中间**激活）+ fp32 per-token scale + spec 元数据（`draft_tokens [B,N]` int64、`tree_attention_mask`、`keep_indices`）。
- **return（S2→client）**：int8 hidden `[B, 1+N, H]`（S2 过 block 40 的**最终**激活，**形状与 push 完全相同**）+ scale + 同类 aux + `kv_cache_position_ids`。
- **S1→client 观察流**：中间 server 每步也向 client yield 同一份响应（v4 为 int8，888 KB）。不被 await、不进合成延迟模型、不打延迟日志，但是真实的 client 方向流量。

用户心智模型需要修正的点：**两条腿都带整树 hidden**——LM head 和 accept walk 在客户端（GPU2），verify 需要每个树位置的 hidden；eagle2 元数据只有几 KB。不存在"只传 eagle2"的腿。

### 4.2 数字结论（`results/sd_arc_decomposition_2026-07-17`，公式 8·bytes/BW+base_lat 逐 env 验证）

- **v3 异常 = dtype 不对称**：push int8（803 KB）vs return fp16（1595 KB）→ 正好 2× 字节 → E5 延迟 428.9 vs 753.2 ms，公式精确吻合。这就是 v4 做 tail int8 量化的动机。
- **v4 稳态双腿已对称**（甚至 return 略小，因为 push 多带 spec 元数据）：

| v4 稳态/轮 | push 1→2 | return 2→1 |
|---|---|---|
| E5 sd_fixed | 1773 KB / 826.2 ms | 1763 KB / 822.0 ms |
| E5 ARC-on | 888 KB / 463.9 ms | 883 KB / 461.7 ms |

E1–E4 同验，全部吻合。
- **"v4 里 return 仍 > push"的两个来源**：(a) 聚合均值被 prefill 污染——return 通道每个 cell 含 2 条一次性 prefill 响应（v4 int8 ~27.7 MB/条），push 通道从不含 prefill（走 `rpc_forward` 另一通道）；(b) 若在客户端/网卡层面量回程总量，还要加上 S1 观察流的 888 KB/轮 → v4 回程方向 ≈ 2× push（883+888 vs 888）。

## 5. fp8 vs int8（完整分析，对话中曾被截断的部分）

- **带宽零收益**：二者都是 8 bit。v4 的减半来自 fp16→int8（16→8 bit）；换 fp8 还是 8 bit，E4/E5 的 WAN 延迟一个字节都省不了。
- **保真要测，且对 argmax 场景 fp8 未必更好**：v4 实测 int8 噪声使 acceptance −6%（3.78→3.53，30/32 行轨迹改变）。int8 per-token absmax 是均匀网格（顶部值附近相对误差 ~0.4%）；fp8 e4m3 是 3 位尾数（各处相对误差 ~3-6%，小值更准、**近 absmax 的大值更粗**）。accept 决策对顶部 logit 敏感，int8 在关键区域反而更细。e5m2 免 scale 但精度更低。结论：**先实测 acceptance 再谈切换；若目标是降 int8 的保真损失，per-channel int8 或 fp16 scale 是更直接的路径**。
- **计算侧**：quant/dequant 是 elementwise，fp8 tensor core 用不上；L4(sm_89)/H100(sm_90) 都能跑 fp8 cast。真正能吃 fp8 红利的是"量化张量直接进 fp8 GEMM"（如客户端 lm_head），那是另一项手术。
- 建议：保留 int8 为默认 WAN codec（roundtrip 已验证），fp8 作为实验变体 A/B acceptance，不要期待吞吐变化。

## 6. 遗留问题 / 建议下一步（按 ROI 排序）

1. **异构 batch 的逐行回退路径**（最大遗留）：seq_len 不等时 `can_batch_prefix=False` → 每行 `_build_tree_from_prefix_cache` + 每深度每 seed 一次 `_advance_cached` → b8het draft 高达 **242ms/轮**（同质 b8 才 ~37ms）。native emit 目前只覆盖全批等长的 batched 路径。方案：变长 prefix 的 batched prefill（padding/共长截断策略）+ 逐行 `prefix_next_pos`（position_ids 已支持 [B,S]），或 compactions 后重对齐。这是对真实异构流量最大的 draft 侧收益。
2. **CUDA graph 化 native 扩展循环**：native 路径形状已静态（candidate arena、frontier F=K、tree_mask 按固定序列增长），可把 5 深度扩展 + 选择收录成图，消除本机 19-92µs/op 的启动开销（预估 draft 再降 5-10ms/轮）。障碍：EAGLE head 的 DynamicCache 会增长/crop——需预分配静态 KV buffer。
3. **verifier 投影成本**：7 次 `[B,H]@[H,V]` 在 b8 略高于旧 Python 路径（+2.3ms），b32 持平。可考虑 lm_head int8/fp8 投影或与首步融合。
4. **`_update_eagle_prefix_hidden_states` 与 `_update_input_ids_with_padding`** 仍是逐行 Python + 多次 `.item()`（GPU-tree 计划里的 Stage 4/5），每轮 4-16ms 量级，是树外下一个 per-round CPU 热点。
5. **默认值决策**：目前全 flag-gated（默认关）。若 E2E 复验通过并决定转正，把 `BLOOMBEE_TENSOR_TREE=1` + `EMIT=1` 设为 greedy 默认；sampling 路径仍走对象树（需要 leaf paths + 概率）。
6. **s2s_flow 的 @dataclass 修复应尽快单独合入主线**（一行，但当前主线 S2S 全坏）。

## 7. 复现/继续工作的最小命令集

```bash
# 本地测试（Mac, CPU）
python3 -m pytest tests/test_eagle_native_emit.py tests/test_eagle_drafter_budget.py tests/test_spe_dec_tree.py -q
PYTHONPATH=src python3 scripts/test_native_tensor_tree.py
PYTHONPATH=src python3 scripts/test_tensor_greedy_identity_randomized.py 400
PYTHONPATH=src python3 scripts/test_tensor_prepare_identity.py

# 同步到测试机
rsync -az --delete -e "ssh -i ~/.ssh/bloombee_iad -o IdentitiesOnly=yes" \
  --exclude '.git' --exclude 'results' --exclude 'bloombee_ASPLOS27*' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude '.DS_Store' --exclude '.claude' --exclude '.bloombee_local' \
  --exclude '*.pdf' --exclude '*.zip' ./ ubuntu@54.167.236.115:~/BloomBee/

# 远程验证 campaign（swarm 自启自灭）
ssh -i ~/.ssh/bloombee_iad -o IdentitiesOnly=yes ubuntu@54.167.236.115 \
  'nohup bash ~/validation/run_gpu_tree_validation.sh > ~/gpu_tree_results.log 2>&1 &'

# drafter 微基准（不需 swarm）
ssh … 'CUDA_VISIBLE_DEVICES=2 BB_BENCH_B=32 ~/bbvenv/bin/python ~/validation/bench_drafter_build.py'
```

## 8. 当前工作树状态（供 commit 参考）

未提交改动 = 本轮 GPU-tree 工作（`tensor_tree.py` / `spe_dec_tree.py` / `eagle_drafter.py` / `speculative_model.py`）+ s2s_flow.py 一行修复 + v4 campaign 遗留仪器化（`inference_session.py` / `block_functions.py` / `handler.py` / `s2s_activation_quant.py`）+ 新增测试文件（`tests/test_eagle_native_emit.py`、`scripts/test_tensor_prepare_identity.py`、`scripts/test_tensor_greedy_identity_randomized.py`）。**尚未 commit**（用户未要求）。建议拆成 ≥2 个 commit：①s2s_flow 回归修复（紧急）；②GPU-tree 迁移（含测试）；仪器化改动是否入库由用户决定。
