"""Validate prepare_incremental_tensor_tree_batch == prepare_incremental_tree_batch.

Builds random SpeculativeTree forests, bridges them to TensorTreeBatch, and
asserts tree_tokens + attention_mask are byte-identical for all three prepare
branches (prefill, vLLM-style local tree mask, dense generation mask).

Run: PYTHONPATH=src python scripts/test_tensor_prepare_identity.py
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch

from bloombee.models.llama.spe_dec_tree import (
    SpeculativeTree,
    prepare_incremental_tree_batch,
)
from bloombee.models.llama.tensor_tree import (
    tensor_tree_from_speculative_trees,
    prepare_incremental_tensor_tree_batch,
)


def random_tree(rng, max_depth=5, max_children=3, keep_prob=0.75, vocab=1000):
    root_tok = rng.randrange(vocab)
    tree = SpeculativeTree(root_tok, request_id="t")

    def grow(node, depth):
        if depth >= max_depth:
            return
        for _ in range(rng.randrange(0, max_children + 1)):
            if rng.random() > keep_prob:
                continue
            child = node.add_child(rng.randrange(vocab), rng.random())
            tree.total_nodes += 1
            tree.max_depth = max(tree.max_depth, child.depth)
            grow(child, depth + 1)

    grow(tree.root, 1)
    return tree


def run_case(seed, B):
    rng = random.Random(seed)
    device = torch.device("cpu")
    trees = [random_tree(rng) for _ in range(B)]
    # Occasionally make every row root-only (early-exit contract).
    if seed % 7 == 0:
        trees = [SpeculativeTree(rng.randrange(1000), request_id=f"t{b}") for b in range(B)]
    tt = tensor_tree_from_speculative_trees(trees, device)

    seq_lengths = torch.tensor([rng.randrange(6, 40) for _ in range(B)], dtype=torch.long)
    input_ids = torch.randint(0, 1000, (B, int(seq_lengths.max().item())))

    # Random previous-round kv positions: root + accepted prefix, -1 padded.
    kv_rows = []
    for b in range(B):
        root = int(seq_lengths[b].item()) - 1
        acc = rng.randrange(0, 4)
        row = [root] + [root + k + 1 for k in range(acc)]
        kv_rows.append(row)
    max_pos = max(len(r) for r in kv_rows)
    kv_pos = torch.full((B, max_pos), -1, dtype=torch.long)
    for b, r in enumerate(kv_rows):
        kv_pos[b, : len(r)] = torch.tensor(r)

    ok = True
    for branch, kwargs in (
        ("prefill", dict(is_prefill=True, return_local_tree_mask=False)),
        ("local", dict(is_prefill=False, return_local_tree_mask=True)),
        ("dense", dict(is_prefill=False, return_local_tree_mask=False)),
    ):
        kw = dict(kwargs)
        kw["kv_cache_position_ids"] = kv_pos
        kw["seq_lengths"] = seq_lengths
        toks_ref, mask_ref, _ = prepare_incremental_tree_batch(
            trees, input_ids, device, return_node_paths=False, **kw
        )
        toks_new, mask_new, paths_new = prepare_incremental_tensor_tree_batch(
            tt, input_ids, device, **kw
        )
        if not torch.equal(toks_ref, toks_new):
            print(f"[FAIL seed={seed} B={B} branch={branch}] tree_tokens differ")
            ok = False
            continue
        if (mask_ref is None) != (mask_new is None):
            print(f"[FAIL seed={seed} B={B} branch={branch}] mask None mismatch")
            ok = False
            continue
        if mask_ref is not None and not torch.equal(mask_ref, mask_new):
            diff = (mask_ref != mask_new)
            idx = diff.nonzero()[0].tolist()
            print(f"[FAIL seed={seed} B={B} branch={branch}] mask differs at {idx} "
                  f"(of {diff.numel()}), shapes {tuple(mask_ref.shape)} vs {tuple(mask_new.shape)}")
            ok = False
    return ok


if __name__ == "__main__":
    cases = []
    for seed in range(80):
        cases.append(run_case(seed, B=1 + (seed % 4)))
    npass = sum(cases)
    print(f"TENSOR_PREPARE: {npass}/{len(cases)} cases byte-identical")
    print("ALL PASS" if npass == len(cases) else "SOME FAILED")
    sys.exit(0 if npass == len(cases) else 1)
