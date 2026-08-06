"""Tensorized EAGLE-2 speculative tree (GPU-tree migration, Stage 1+2+3).

This module replaces the per-round Python `_CandNode`/`TreeNode` object graph,
the Python global rerank / parent-closure, the recursive DFS linearization, the
per-row attention-mask build, and the Python child-walk in greedy verification
with fixed `[B, max_nodes]` tensors and vectorized GPU ops.

Pipeline (all flag-gated; Python paths remain as reference/fallback):

* Stage 1 (`BLOOMBEE_TENSOR_TREE=1`): `greedy_verify_tensorized` — sync-free
  GPU accept walk (fixed step count = tree depth, no per-depth host syncs,
  vectorized output assembly; one host sync total).
* Stage 2a-c (`BLOOMBEE_TENSOR_TREE_EMIT=shadow|1`):
  `EAGLEDrafter._build_tensor_tree_batched` keeps expansion candidates in
  `[B, C]` device tensors end-to-end (no `_CandNode` churn, no per-depth
  `.tolist()`) and `tensor_tree_from_eagle_candidate_tensors` reproduces the
  Python `_topm_global` + `_close_under_parents` + `_bind` + DFS-preorder
  pipeline bit-for-bit (no boolean-mask indexing — `aten::nonzero` syncs).
* Stage 3: `prepare_incremental_tensor_tree_batch` + `local_tree_mask_from_tensor_tree`
  build the vLLM-style local tree mask on-device via a depth-bounded pointer
  walk (no per-row Python `while` over parent indices).

HARD CONTRACT (must stay token-identical to the Python path for greedy decode):

* Node 0 of every row is the ROOT. Draft nodes occupy indices 1..n in **DFS
  pre-order** (visit a node, then its children in child-insertion order) — the
  exact order `spe_dec_tree.linearize_tree_with_positions` produces. So a draft
  node at tensor index ``k`` (1-based among draft nodes, i.e. node index ``k``)
  has ``position_in_sequence == k - 1``.
* ``parent_pos[b, k]`` is the DFS position (``position_in_sequence``) of the
  parent draft node, or ``-1`` when the parent is the root. This matches the
  ``parent_indices`` list consumed by ``build_tree_attention_mask_with_root``.
* Greedy child matching picks the FIRST child in insertion order (lowest node
  index) whose token equals the target argmax — identical to the Python
  ``for child in parent.children: if child.token_id == predicted: break``.

Validated by: scripts/test_native_tensor_tree.py (selection/DFS identity),
scripts/test_tensor_greedy_identity.py + _randomized.py (verifier identity),
scripts/test_tensor_prepare_identity.py (mask byte-identity),
tests/test_eagle_native_emit.py (drafter wiring + shadow mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from bloombee.models.llama.spe_dec_tree import (
    SpeculativeTree,
    linearize_tree_with_positions,
)


@dataclass
class TensorTreeBatch:
    """Fixed-shape per-row tree tensors. All `[B, max_nodes]` unless noted.

    Index 0 is the root. Draft nodes are 1..n_nodes[b]-1 in DFS pre-order.
    Padding slots (>= n_nodes[b]) are token=pad, parent_idx=-1, alive=False.

    ``max_depth_host`` is a host-side upper bound on the tree depth (no sync):
    the accept walk never needs more than this many steps.
    """

    token: torch.Tensor          # [B, N] long  (node 0 = root token)
    parent_idx: torch.Tensor     # [B, N] long  parent NODE index (root's = -1; node0 = -1)
    depth: torch.Tensor          # [B, N] long  (root depth 0)
    n_nodes: torch.Tensor        # [B] long     valid node count incl. root
    device: torch.device
    max_nodes: int               # N (incl. root slot)
    max_depth_host: int = 0      # host-side upper bound on max node depth (0 = unknown)

    @property
    def batch_size(self) -> int:
        return int(self.token.shape[0])

    def draft_count(self) -> torch.Tensor:
        """Per-row number of DRAFT nodes (excludes root) = n_nodes - 1."""
        return (self.n_nodes - 1).clamp(min=0)

    def position_in_sequence(self) -> torch.Tensor:
        """DFS position of each node: root -> -1, draft node index k -> k-1.

        Padding slots keep -1. Returned shape [B, N] long."""
        ar = torch.arange(self.max_nodes, device=self.device).unsqueeze(0)  # [1, N]
        pos = ar - 1  # node 0 -> -1, node k -> k-1
        valid = ar < self.n_nodes.unsqueeze(1)
        return torch.where(valid, pos, torch.full_like(pos, -1))


def tensor_tree_from_speculative_trees(
    trees: List[SpeculativeTree],
    device: torch.device,
    pad_token_id: int = 0,
) -> TensorTreeBatch:
    """Stage-1 bridge: build a TensorTreeBatch from existing SpeculativeTrees
    using the SAME DFS pre-order as `linearize_tree_with_positions`, so the
    tensor path is byte-identical to the Python path under validation.

    Layout: node 0 = root; draft nodes 1..n in DFS order. parent_idx for a draft
    node = (root's node index 0) if its parent is the root, else the parent draft
    node's node index (= parent position_in_sequence + 1).
    """
    batch_size = len(trees)
    # Per-row linearized draft nodes (DFS pre-order) + parent positions.
    rows_tokens: List[List[int]] = []
    rows_parent_idx: List[List[int]] = []
    rows_depth: List[List[int]] = []
    for tree in trees:
        root_tok = int(tree.root.token_id)
        toks = [root_tok]          # node 0 = root
        par = [-1]                 # root has no parent
        dep = [0]
        if tree.total_nodes > 1:
            linearized_nodes, parent_indices = linearize_tree_with_positions(tree)
            # linearized_nodes[k] has position_in_sequence == k; node index == k + 1.
            for k, node in enumerate(linearized_nodes):
                toks.append(int(node.token_id))
                ppos = int(parent_indices[k])   # parent's position_in_sequence or -1 (root)
                par.append(0 if ppos < 0 else ppos + 1)  # -> node index
                dep.append(int(node.depth))
        rows_tokens.append(toks)
        rows_parent_idx.append(par)
        rows_depth.append(dep)

    max_nodes = max((len(t) for t in rows_tokens), default=1)
    max_nodes = max(max_nodes, 1)
    max_depth_host = max((max(d) for d in rows_depth if d), default=0)
    token = torch.full((batch_size, max_nodes), pad_token_id, dtype=torch.long, device=device)
    parent_idx = torch.full((batch_size, max_nodes), -1, dtype=torch.long, device=device)
    depth = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=device)
    n_nodes = torch.ones(batch_size, dtype=torch.long, device=device)
    for b in range(batch_size):
        n = len(rows_tokens[b])
        token[b, :n] = torch.tensor(rows_tokens[b], dtype=torch.long, device=device)
        parent_idx[b, :n] = torch.tensor(rows_parent_idx[b], dtype=torch.long, device=device)
        depth[b, :n] = torch.tensor(rows_depth[b], dtype=torch.long, device=device)
        n_nodes[b] = n
    return TensorTreeBatch(token=token, parent_idx=parent_idx, depth=depth,
                           n_nodes=n_nodes, device=device, max_nodes=max_nodes,
                           max_depth_host=max_depth_host)


@torch.no_grad()
def greedy_verify_tensorized(
    *,
    tt: TensorTreeBatch,
    hidden_states: torch.Tensor,     # [B, S, H]
    seq_lengths: torch.Tensor,       # [B] long
    tree_len: int,                   # tree_tokens.shape[1] (max draft nodes across batch)
    is_first_iteration: bool,
    project_rows,                    # callable: [N, H] -> [N, vocab] logits (lm_head)
    logits_processor=None,           # LogitsProcessorList (applied to final bonus token only)
    input_ids: Optional[torch.Tensor] = None,  # [B, *] for logits_processor
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU greedy tree verification. Token-identical to
    ``_extract_greedy_verified_paths_from_hidden``.

    The accept walk runs a FIXED ``tree_len + 1`` steps with no per-depth host
    sync: a row that misses a match (or falls out of the hidden window) masks
    itself out permanently, so extra steps are no-ops for it. Projecting the
    lm-head for those dead rows costs one small [B,H]@[H,V] GEMM per step —
    far cheaper than the per-step CUDA syncs + pipeline stalls they replace
    (the old version synced twice per accepted depth). Exactly one host sync
    remains: ``accept_count.max()`` to size the ragged outputs.

    Returns (verified_tokens[B,Lmax] or None, kv_cache_position_ids[B,Lmax+1],
    llm_generated_tokens[B,1], valid_lengths[B], final_positions[B])."""
    device_h = hidden_states.device
    out_device = seq_lengths.device if input_ids is None else input_ids.device
    B = int(hidden_states.shape[0])
    S = int(hidden_states.shape[1])
    N = tt.max_nodes
    token = tt.token.to(device_h)
    parent_idx = tt.parent_idx.to(device_h)
    n_nodes = tt.n_nodes.to(device_h)
    node_ar = torch.arange(N, device=device_h).unsqueeze(0)          # [1, N]
    alive = node_ar < n_nodes.unsqueeze(1)                           # [B, N] valid node slots
    # tree_root_positions[b] = seq_lengths[b]-1 (absolute position of the root/last committed tok)
    seq_h = seq_lengths.to(device_h).long()
    tree_root_positions = seq_h - 1

    active_node = torch.zeros(B, dtype=torch.long, device=device_h)  # current parent node index (0 = root)
    # A row is active iff its current parent (active_node) has at least one live child.
    def children_exist(node_b):  # node_b: [B] -> [B] bool
        return ((parent_idx == node_b.unsqueeze(1)) & alive).any(dim=1)
    active = children_exist(active_node) & (n_nodes > 1)

    # The accept walk descends exactly one tree depth per step, so a host-known
    # depth bound caps the loop (default EAGLE 10-3 tree: 6 steps instead of
    # tree_len+1 = 11). Extra steps are no-ops; fewer would lose accepts.
    depth_bound = int(getattr(tt, "max_depth_host", 0) or 0)
    max_steps = max(int(tree_len), 0) + 1
    if depth_bound > 0:
        max_steps = min(max_steps, depth_bound)
    accepted_tokens_steps: List[torch.Tensor] = []   # each [B]
    accepted_nodes_steps: List[torch.Tensor] = []     # each [B] node idx (or -1)
    accept_count = torch.zeros(B, dtype=torch.long, device=device_h)
    bidx = torch.arange(B, device=device_h)

    for _step in range(max_steps):
        # Parent logits position per row (mirror _tree_parent_logits_position).
        is_root = active_node == 0
        pos_first = torch.where(is_root, seq_h - 1, seq_h + (active_node - 1))
        pos_rest = torch.where(is_root, torch.zeros_like(active_node), active_node)  # non-root: (k-1)+1 = k
        pos = pos_first if is_first_iteration else pos_rest
        in_window = (pos >= 0) & (pos < S)
        cur = active & in_window
        pos_clamped = pos.clamp(0, max(S - 1, 0))
        parent_hidden = hidden_states[bidx, pos_clamped, :]            # [B, H]
        logits = project_rows(parent_hidden)                          # [B, vocab]
        predicted = logits.argmax(dim=-1)                             # [B]
        # Match: children of active_node whose token == predicted; pick lowest node index.
        is_child = (parent_idx == active_node.unsqueeze(1)) & alive   # [B, N]
        match = is_child & (token == predicted.unsqueeze(1))          # [B, N]
        match = match & cur.unsqueeze(1)
        # lowest matching node index, else -1
        big = N
        idx_or_big = torch.where(match, node_ar.expand(B, N), torch.full((B, N), big, device=device_h, dtype=torch.long))
        matched_idx = idx_or_big.min(dim=1).values                   # [B], == big if none
        matched = matched_idx < big
        matched_idx = torch.where(matched, matched_idx, torch.full_like(matched_idx, -1))
        # Record accepted token/node for matched rows; -1 elsewhere this step.
        step_tok = torch.where(matched, token[bidx, matched_idx.clamp(min=0)], torch.full((B,), -1, device=device_h, dtype=torch.long))
        accepted_tokens_steps.append(step_tok)
        accepted_nodes_steps.append(matched_idx)
        accept_count = accept_count + matched.long()
        # Advance: matched rows move to matched_idx; others go inactive.
        active_node = torch.where(matched, matched_idx.clamp(min=0), active_node)
        active = matched & children_exist(active_node)

    # ---- Output assembly, vectorized. A row's accepted steps are a contiguous
    # PREFIX of its step list (once unmatched it can never match again), so the
    # old per-row Python compaction is exactly a prefix slice. ----
    if accepted_tokens_steps:
        toks_mat = torch.stack(accepted_tokens_steps, dim=1)   # [B, max_steps]
        nodes_mat = torch.stack(accepted_nodes_steps, dim=1)   # [B, max_steps]
    else:
        toks_mat = torch.empty(B, 0, dtype=torch.long, device=device_h)
        nodes_mat = torch.empty(B, 0, dtype=torch.long, device=device_h)

    valid_lengths = accept_count.to(out_device)
    # The single host sync of the verifier: ragged output width.
    Lmax = int(accept_count.max().item()) if accept_count.numel() else 0

    verified_tokens: Optional[torch.Tensor] = None
    if Lmax > 0:
        verified_tokens = toks_mat[:, :Lmax].to(out_device)
    # absolute kv positions: root_pos[b] + matched_node_idx (since pos_in_seq = idx-1, +1 -> idx)
    kv_root = tree_root_positions.unsqueeze(1)                       # [B, 1]
    kv_nodes = torch.where(
        nodes_mat >= 0,
        kv_root + nodes_mat.clamp(min=0),
        torch.full_like(nodes_mat, -1),
    )
    kv_cache_position_ids = torch.cat([kv_root, kv_nodes], dim=1)[:, : Lmax + 1].to(out_device)

    # Final / bonus-token position (mirror the Python path):
    #   accept>0 : pos = last_node (relative), or root_pos + last_node on iter 1
    #   accept=0 : pos = seq_len - 1 (iter 1) or (S - tree_len) - 1 (later iters)
    # clamped to the hidden window. Rows with accept==0 gather a dummy slot from
    # nodes_mat and take the no-accept branch, matching row_last_node's 0 default.
    if nodes_mat.shape[1] > 0:
        last_node = nodes_mat.gather(1, (accept_count - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
    else:
        last_node = torch.zeros(B, dtype=torch.long, device=device_h)
    abs_last = tree_root_positions + last_node
    fallback_pos = max(0, S - int(tree_len))
    if is_first_iteration:
        pos_accept = abs_last
        pos_noaccept = seq_h - 1
    else:
        pos_accept = last_node
        pos_noaccept = torch.full_like(seq_h, fallback_pos - 1)
    fpi = torch.where(accept_count > 0, pos_accept, pos_noaccept)
    fpi = fpi.clamp(0, max(S - 1, 0))

    final_hidden = hidden_states[bidx, fpi, :]
    final_logits = project_rows(final_hidden)
    if logits_processor and len(logits_processor) > 0 and input_ids is not None:
        rows = []
        for b in range(B):
            processed = final_logits[b:b + 1].clone()
            for proc in logits_processor:
                processed = proc(input_ids[b:b + 1], processed)
            rows.append(torch.argmax(processed[0], dim=-1, keepdim=True).to(out_device))
        llm_generated_tokens = torch.stack(rows, dim=0)
    else:
        llm_generated_tokens = final_logits.argmax(dim=-1, keepdim=True).to(out_device)

    final_positions = fpi.to(out_device)
    return verified_tokens, kv_cache_position_ids, llm_generated_tokens, valid_lengths, final_positions


@torch.no_grad()
def tensor_tree_from_eagle_candidate_tensors(
    *,
    root_tokens: torch.Tensor,        # [B] long
    cand_token: torch.Tensor,         # [B, C] long
    cand_parent_cidx: torch.Tensor,   # [B, C] long, parent candidate slot (-1 if parent is root)
    cand_depth: torch.Tensor,         # [B, C] long (>=1 for draft candidates)
    cand_path_logp64: torch.Tensor,   # [B, C] float64 cumulative path log-prob
    cand_valid: torch.Tensor,         # [B, C] bool
    total_token: int,                 # tree budget incl. root (m = total_token - 1 kept draft nodes)
    max_candidate_depth: int,
    pad_token_id: int = 0,
) -> TensorTreeBatch:
    """Reconstruct a TensorTreeBatch directly from per-row EAGLE candidate tensors,
    reproducing the Python `_topm_global` + `_close_under_parents` + `_bind` +
    DFS-preorder pipeline EXACTLY (with creation_index == candidate slot index).

    Determinism contract (must match eagle_drafter.py):
      * top-m kept set: largest m valid candidates by (-path_logp64, depth, slot),
        stable on ties (slot ascending) — matches _topm_global stable sort.
      * parent closure: add ancestors (via cand_parent_cidx) — matches _close_under_parents.
      * final DFS pre-order: children visited in (depth, slot) order — matches _bind
        (sorts kept by (depth, creation_index)) + linearize_tree_with_positions.
    """
    device = cand_token.device
    B, C = cand_token.shape
    m = max(0, int(total_token) - 1)

    # --- 1. top-m selection by (-path_logp64, depth, slot), stable, valid only ---
    slot = torch.arange(C, device=device).unsqueeze(0).expand(B, C)  # [B,C] creation_index
    # Build a single sortable key per candidate that lexicographically orders by
    # (path_logp64 DESC, depth ASC, slot ASC). We sort by composite via successive
    # stable sorts (least-significant key first): slot (already ascending), then
    # depth ascending, then path_logp64 descending. Invalid -> sorted to the end.
    # Use rank assignment rather than float packing to avoid precision loss.
    # Order candidates: primary path_logp64 desc, then depth asc, then slot asc.
    # Implement with a stable argsort chain.
    order = torch.arange(C, device=device).unsqueeze(0).expand(B, C).clone()  # identity (slot asc)
    # stable sort by depth ascending (slot asc already the tie-order)
    dep_keys = torch.gather(cand_depth, 1, order)
    idx = torch.argsort(dep_keys, dim=1, stable=True)
    order = torch.gather(order, 1, idx)
    # stable sort by path_logp64 descending
    lp_keys = torch.gather(cand_path_logp64, 1, order)
    idx = torch.argsort(-lp_keys, dim=1, stable=True)
    order = torch.gather(order, 1, idx)
    # Now `order[b]` lists candidate slots best-first by (-logp, depth, slot)...
    # but invalid candidates must be pushed last. Re-rank with validity as the
    # most-significant key (valid first), stable over the (-logp,depth,slot) order.
    valid_keys = torch.gather(cand_valid.long(), 1, order)  # 1 valid, 0 invalid
    idx = torch.argsort(-valid_keys, dim=1, stable=True)
    order = torch.gather(order, 1, idx)
    # selected = first m slots in `order` that are valid
    rank = torch.arange(C, device=device).unsqueeze(0).expand(B, C)
    n_valid = cand_valid.sum(dim=1, keepdim=True)  # [B,1]
    take = torch.minimum(torch.full_like(n_valid, m), n_valid)  # [B,1]
    sel_in_order = rank < take  # [B,C] over the ordered positions
    selected = torch.zeros(B, C, dtype=torch.bool, device=device)
    selected.scatter_(1, order, sel_in_order)
    selected = selected & cand_valid

    # --- 2. parent closure: add ancestors of selected (depth-bounded) ---
    # Fixed iteration count, no early-exit host syncs: a depth-d candidate has at
    # most d-1 candidate ancestors, so D+1 hops always converges and extra hops
    # are idempotent no-ops.
    closed = selected.clone()
    for _ in range(int(max_candidate_depth) + 1):
        # parent slot of each closed candidate (>=0 means parent is a candidate)
        par = cand_parent_cidx
        has_par = (par >= 0) & closed
        par_safe = par.clamp(min=0)
        add = torch.zeros(B, C, dtype=torch.bool, device=device)
        add.scatter_(1, par_safe, has_par)  # mark parents of closed nodes
        closed = closed | (add & cand_valid)

    # --- 3. final DFS pre-order via path-key lexsort ---
    # path_key[b,c] = [slot at depth1 ancestor, slot at depth2 ancestor, ..., own slot]
    # padded with -1 suffix so a parent sorts before its descendants.
    D = int(max_candidate_depth)
    # Boolean-mask indexing (`t[mask]`) dispatches aten::nonzero, which forces a
    # CUDA sync PER CALL (~0.5ms each on multi-GPU hosts — this loop used to cost
    # ~24 syncs/build). Column D is a dumpster for writes from invalid chains:
    # everything lands via scatter_, and the dumpster is sliced away at the end.
    path_key = torch.full((B, C, D + 1), -1, dtype=torch.long, device=device)
    # walk ancestors: level 0 = own slot at position depth-1; fill from the node up.
    cur = slot.clone()                       # [B,C] current ancestor slot (start: self)
    cur_depth = cand_depth.clamp(min=0)      # [B,C]
    # For each node, its own slot goes at column (depth-1); ancestors fill earlier cols.
    for _level in range(D):
        valid_cur = cur >= 0
        col = torch.where(
            valid_cur,
            (cur_depth - 1).clamp(min=0, max=D - 1),
            torch.full_like(cur_depth, D),   # dumpster column
        )
        val = torch.where(valid_cur, cur, torch.full_like(cur, -1))
        path_key.scatter_(2, col.unsqueeze(2), val.unsqueeze(2))
        # step up to parent
        parent_of_cur = torch.where(
            valid_cur,
            torch.gather(cand_parent_cidx, 1, cur.clamp(min=0)),
            torch.full_like(cur, -1),
        )
        cur = parent_of_cur
        cur_depth = (cur_depth - 1).clamp(min=0)
    path_key = path_key[:, :, :D]
    # closed candidates sort by path_key ascending (lexicographic over columns);
    # unclosed -> large sentinel so they fall to the end.
    BIG = C + 1
    unclosed = ~closed
    sortable = torch.where(unclosed.unsqueeze(2), torch.full_like(path_key, BIG), path_key)
    # lexsort over D columns: stable sorts from last column to first.
    order2 = torch.arange(C, device=device).unsqueeze(0).expand(B, C).clone()
    for col in range(D - 1, -1, -1):
        keys = torch.gather(sortable[:, :, col], 1, order2)
        idx = torch.argsort(keys, dim=1, stable=True)
        order2 = torch.gather(order2, 1, idx)
    # order2[b] now lists closed candidates in DFS pre-order, then unclosed.
    draft_count = closed.sum(dim=1)  # [B]
    max_draft = int(draft_count.max().item()) if B > 0 else 0
    max_nodes = max_draft + 1

    token = torch.full((B, max_nodes), pad_token_id, dtype=torch.long, device=device)
    parent_idx = torch.full((B, max_nodes), -1, dtype=torch.long, device=device)
    depth = torch.zeros((B, max_nodes), dtype=torch.long, device=device)
    n_nodes = draft_count + 1
    token[:, 0] = root_tokens.to(device)

    # cand slot -> node index map (node 0 = root): order2 is a permutation, so
    # scattering ranks 1..draft_count along it is collision-free.
    rank = torch.arange(C, device=device).unsqueeze(0).expand(B, C)  # [B, C]
    node_rank = torch.where(
        rank < draft_count.unsqueeze(1), rank + 1, torch.full_like(rank, -1)
    )
    cand_to_node = torch.full((B, C), -1, dtype=torch.long, device=device)
    cand_to_node.scatter_(1, order2, node_rank)

    if max_draft > 0:
        ordered = order2[:, :max_draft]                              # [B, max_draft]
        valid_rank = rank[:, :max_draft] < draft_count.unsqueeze(1)  # [B, max_draft]
        tok_rows = torch.gather(cand_token, 1, ordered)
        dep_rows = torch.gather(cand_depth, 1, ordered)
        par_slots = torch.gather(cand_parent_cidx, 1, ordered)       # -1 -> root
        par_nodes = torch.where(
            par_slots < 0,
            torch.zeros_like(par_slots),
            torch.gather(cand_to_node, 1, par_slots.clamp(min=0)),
        )
        token[:, 1:] = torch.where(
            valid_rank, tok_rows, torch.full_like(tok_rows, pad_token_id)
        )
        depth[:, 1:] = torch.where(valid_rank, dep_rows, torch.zeros_like(dep_rows))
        parent_idx[:, 1:] = torch.where(
            valid_rank, par_nodes, torch.full_like(par_nodes, -1)
        )

    return TensorTreeBatch(token=token, parent_idx=parent_idx, depth=depth,
                           n_nodes=n_nodes, device=device, max_nodes=max_nodes,
                           max_depth_host=int(max_candidate_depth))


def parent_pos_list_per_row(tt: TensorTreeBatch) -> List[List[int]]:
    """Return, per row, the `parent_indices` list (DFS-position space, root=-1)
    for the DRAFT nodes only — the exact input `build_tree_attention_mask_with_root`
    and the prefill/generation mask code expect. Used by the tensor prepare path."""
    out: List[List[int]] = []
    n_nodes = tt.n_nodes.tolist()
    parent_idx = tt.parent_idx.tolist()
    for b in range(tt.batch_size):
        n = int(n_nodes[b])
        row: List[int] = []
        for k in range(1, n):  # draft nodes (skip root at index 0)
            pidx = int(parent_idx[b][k])     # parent NODE index (0 == root)
            row.append(-1 if pidx == 0 else pidx - 1)  # -> parent position_in_sequence
        out.append(row)
    return out


def local_tree_mask_from_tensor_tree(tt: TensorTreeBatch, device: torch.device) -> torch.Tensor:
    """Vectorized [B, I, I] local root/tree adjacency (I = max_nodes), identical
    to the per-row assembly in ``prepare_incremental_tree_batch(...,
    return_local_tree_mask=True)``: root attends itself; each valid draft node
    attends the root, itself, and its strict draft-space ancestors; padding
    rows/cols stay all-False.

    Replaces the per-row Python `while` parent-walk
    (`build_tree_attention_mask_with_root`) with a depth-bounded pointer walk in
    node space (max depth hops, each one [B, N] gather + scatter). One host sync
    total (max draft depth to bound the hop count).
    """
    B = tt.batch_size
    N = tt.max_nodes
    parent_idx = tt.parent_idx.to(device)
    n_nodes = tt.n_nodes.to(device)
    node_ar = torch.arange(N, device=device).unsqueeze(0)           # [1, N]
    alive = node_ar < n_nodes.unsqueeze(1)                          # [B, N]

    mask = torch.zeros(B, N, N, dtype=torch.bool, device=device)
    mask[:, 0, 0] = True
    if N <= 1:
        return mask

    # Ancestor closure in NODE space: anc[b, i, a] == True iff a is a strict
    # ancestor of i. Chains only traverse valid parent links (padding slots have
    # parent -1 and are never a valid node's parent), so no validity masking is
    # needed on the walk itself.
    par = torch.where(alive, parent_idx, torch.full_like(parent_idx, -1))
    anc = torch.zeros(B, N, N, dtype=torch.bool, device=device)
    cur = par
    max_hops = int(tt.depth.to(device).masked_fill(~alive, 0).max().item())
    for _ in range(max_hops):
        valid = cur >= 0
        # Fresh one-hot per hop (never write False over accumulated True).
        one_hot = torch.zeros(B, N, N, dtype=torch.bool, device=device)
        one_hot.scatter_(2, cur.clamp(min=0).unsqueeze(2), valid.unsqueeze(2))
        anc |= one_hot
        nxt = par.gather(1, cur.clamp(min=0))
        cur = torch.where(valid, nxt, torch.full_like(cur, -1))

    draft_valid = alive[:, 1:]                                      # [B, N-1]
    eye = torch.eye(N - 1, dtype=torch.bool, device=device).unsqueeze(0)
    tree_block = (anc[:, 1:, 1:] | eye) & draft_valid.unsqueeze(2) & draft_valid.unsqueeze(1)
    mask[:, 1:, 0] = draft_valid
    mask[:, 1:, 1:] = tree_block
    return mask


def prepare_incremental_tensor_tree_batch(
    tt: TensorTreeBatch,
    input_ids: torch.LongTensor,
    device: torch.device,
    pad_token_id: int = 0,
    seq_lengths: Optional[torch.LongTensor] = None,
    is_prefill: bool = False,
    kv_cache_position_ids: Optional[torch.Tensor] = None,
    return_local_tree_mask: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List]:
    """`prepare_incremental_tree_batch` fed directly from a TensorTreeBatch —
    no SpeculativeTree objects, no DFS re-linearization.

    * Steady-state generation (`return_local_tree_mask=True`, the hot path) is
      fully vectorized on-device: tokens are a slice of `tt.token` and the mask
      comes from `local_tree_mask_from_tensor_tree`.
    * Prefill and the dense generation mask (rare/one-shot paths) reuse the
      exact per-row branch code via `_prepare_incremental_tree_batch_impl`,
      fed with per-row lists extracted from the tensors (one host transfer).
    """
    from bloombee.models.llama.spe_dec_tree import _prepare_incremental_tree_batch_impl

    B = tt.batch_size
    if B == 0 or tt.max_nodes <= 1:
        # All rows root-only — same early-exit contract as the object path.
        return (
            torch.empty(B, 0, dtype=torch.long, device=device),
            None,
            [[] for _ in range(B)],
        )

    if return_local_tree_mask and not is_prefill:
        tree_tokens = tt.token[:, 1:].to(device).contiguous()
        attention_mask = local_tree_mask_from_tensor_tree(tt, device)
        return tree_tokens, attention_mask, [[] for _ in range(B)]

    n_draft = (tt.n_nodes - 1).tolist()
    toks = tt.token[:, 1:].tolist()
    row_tokens = [toks[b][: int(n_draft[b])] for b in range(B)]
    row_parents = parent_pos_list_per_row(tt)
    tree_tokens, attention_mask, _ = _prepare_incremental_tree_batch_impl(
        row_tokens,
        row_parents,
        [[] for _ in range(B)],
        input_ids,
        device,
        pad_token_id=pad_token_id,
        seq_lengths=seq_lengths,
        is_prefill=is_prefill,
        kv_cache_position_ids=kv_cache_position_ids,
        return_local_tree_mask=return_local_tree_mask,
    )
    return tree_tokens, attention_mask, [[] for _ in range(B)]
