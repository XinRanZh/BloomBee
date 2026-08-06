"""Randomized token-identity stress: tensorized greedy verifier vs Python path.

Random trees (variable depth/width, duplicate sibling tokens), random hidden
encodings that force accept/miss patterns along random branches, batch 1..4,
first/non-first iteration. Compares all five verifier outputs exactly.

Run: PYTHONPATH=src python scripts/test_tensor_greedy_identity_randomized.py [N]
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torch

from bloombee.models.llama.spe_dec_tree import SpeculativeTree
from bloombee.models.llama.tensor_tree import (
    tensor_tree_from_speculative_trees, greedy_verify_tensorized,
)

VOCAB = 60
HID = 8


def build_tree(spec):
    root_tok, children = spec
    tree = SpeculativeTree(root_tok, request_id="t")

    def add(parent_bb, child_specs):
        for ctok, gchildren in child_specs:
            node = parent_bb.add_child(ctok, 0.5)
            tree.total_nodes += 1
            tree.max_depth = max(tree.max_depth, node.depth)
            add(node, gchildren)
    add(tree.root, children)
    return tree


def random_spec(rng, depth=0, max_depth=5):
    # (token, [child specs]); occasional duplicate sibling token ids.
    tok = rng.randrange(VOCAB)
    if depth >= max_depth:
        return (tok, [])
    n_ch = rng.choice([0, 0, 1, 1, 2, 2, 3])
    children = []
    for _ in range(n_ch):
        if children and rng.random() < 0.25:
            ctok = children[-1][0]  # duplicate sibling token
            child = (ctok, children[-1][1] and rng.choice([children[-1][1][0:0], []]) or [])
            child = (ctok, [])
        else:
            child = random_spec(rng, depth + 1, max_depth)
        children.append(child)
    return (tok, children)


def linearize(tree):
    """DFS pre-order draft nodes (token, parent_pos). Mirrors the verifier contract."""
    nodes = []

    def dfs(node, parent_pos):
        for i, ch in enumerate(node.children):
            pos = len(nodes)
            nodes.append((ch.token_id, parent_pos))
            dfs(ch, pos)
    dfs(tree.root, -1)
    return nodes


def run_python(trees, hidden, seq_lengths, tree_len, is_first, project):
    from bloombee.models.llama import speculative_model as sm

    class Shim:
        _tree_parent_logits_position = sm.DistributedLlamaForSpeculativeGeneration._tree_parent_logits_position
        lm_head = None
    shim = Shim()
    orig = sm._project_lm_head_rows
    sm._project_lm_head_rows = lambda lm_head, rows, drafter=None: project(rows)
    try:
        out = sm.DistributedLlamaForSpeculativeGeneration._extract_greedy_verified_paths_from_hidden(
            shim,
            hidden_states=hidden, trees=trees, input_ids=torch.zeros(len(trees), 1, dtype=torch.long),
            logits_processor=[], tree_len=tree_len, seq_lengths=seq_lengths,
            is_first_iteration=is_first, drafter=None,
        )
    finally:
        sm._project_lm_head_rows = orig
    return out


def project_rows(hidden_rows):
    N = hidden_rows.shape[0]
    logits = torch.zeros(N, VOCAB)
    want = hidden_rows[:, 0].long().clamp(0, VOCAB - 1)
    logits[torch.arange(N), want] = 10.0
    return logits


def eq(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    return bool(torch.equal(a.cpu(), b.cpu()))


def run_case(seed):
    rng = random.Random(seed)
    B = 1 + rng.randrange(4)
    is_first = rng.random() < 0.3
    trees = []
    for b in range(B):
        if rng.random() < 0.1:
            trees.append(SpeculativeTree(rng.randrange(VOCAB), request_id=f"r{b}"))  # root-only row
        else:
            trees.append(build_tree(random_spec(rng)))
    tree_len = max((t.total_nodes - 1) for t in trees)
    seq_lengths = torch.tensor([rng.randrange(3, 30) for _ in range(B)], dtype=torch.long)

    # Hidden window: for non-first iterations positions are 0..tree_len (root at 0),
    # for first iteration root at seq-1 and drafts at seq+k.
    S = (int(seq_lengths.max().item()) + tree_len + 1) if is_first else (tree_len + 2)
    hidden = torch.zeros(B, S, HID)
    # Fill every position the walk could touch with a random token id — sometimes
    # matching a child (deep accepts), sometimes not (early stop).
    for b in range(B):
        nodes = linearize(trees[b])
        for p in range(S):
            if rng.random() < 0.55 and nodes:
                # bias toward tokens present in the tree so accepts happen
                hidden[b, p, 0] = float(nodes[rng.randrange(len(nodes))][0])
            else:
                hidden[b, p, 0] = float(rng.randrange(VOCAB))

    tt = tensor_tree_from_speculative_trees(trees, torch.device("cpu"))
    py = run_python(trees, hidden, seq_lengths, tree_len, is_first, project_rows)
    tn = greedy_verify_tensorized(
        tt=tt, hidden_states=hidden, seq_lengths=seq_lengths, tree_len=tree_len,
        is_first_iteration=is_first, project_rows=project_rows,
        logits_processor=[], input_ids=torch.zeros(B, 1, dtype=torch.long),
    )
    ok = all(eq(a, b) for a, b in zip(py, tn))
    if not ok:
        names = ["verified_tokens", "kv_pos", "llm_gen", "valid_lengths", "final_pos"]
        print(f"[FAIL seed={seed} B={B} first={is_first} tree_len={tree_len}]")
        for n, a, b in zip(names, py, tn):
            if not eq(a, b):
                print(f"  {n}: py={None if a is None else a.tolist()} tn={None if b is None else b.tolist()}")
    return ok


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    results = [run_case(seed) for seed in range(N)]
    npass = sum(results)
    print(f"RANDOMIZED_GREEDY_IDENTITY: {npass}/{N} cases match")
    print("ALL PASS" if npass == N else "SOME FAILED")
    sys.exit(0 if npass == N else 1)
