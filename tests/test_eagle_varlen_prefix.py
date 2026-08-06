"""Varlen (heterogeneous-batch) EAGLE prefix path: structural correctness.

Covers `_prefill_with_prefix_batch_varlen` + the ctx-based batched builders:
- suffix-forward additive mask: cached prefix slots [0, L_b) + causal new tokens
  at [L_max, L_max + c_b) per row, everything else -inf;
- expansion masks: scattered prefix validity + tree columns;
- per-row position_ids (T_b + depth - 1) — RoPE stays exact per row;
- equal-length inputs degenerate to the uniform batched path's semantics;
- native varlen emit wiring (returns None, populates last_tensor_tree).
"""
import torch

from bloombee.models.llama.eagle_drafter import EAGLEDrafter

H = 8
V = 101


def _make_drafter(seed=0):
    drafter = object.__new__(EAGLEDrafter)
    drafter.device = torch.device("cpu")
    drafter.dtype = torch.float32
    drafter._prefix_states = {}
    drafter.last_tensor_tree = None
    drafter.head_cfg = None

    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(H, V, generator=g) / (H ** 0.5)
    calls = []

    class DummyCache(dict):
        """Minimal DynamicCache stand-in: tracks (keys, values) shapes per layer."""

        def __init__(self, seq_len=0):
            super().__init__()
            self.seq_len = seq_len
            self.cropped = []

        def get_seq_length(self, layer_idx=0):
            return self.seq_len

        def crop(self, n):
            self.cropped.append(n)
            self.seq_len = n

    def fake_step(hidden_states, input_ids, position_ids, past_key_values, attention_mask=None):
        calls.append(
            dict(
                hidden=hidden_states, ids=input_ids, pos=position_ids,
                mask=attention_mask, cache=past_key_values,
            )
        )
        out = (
            hidden_states * 0.5
            + (input_ids % 17).unsqueeze(-1).to(hidden_states.dtype) * 0.01
            + (position_ids % 5).unsqueeze(-1).to(hidden_states.dtype) * 0.001
        )
        return out, past_key_values

    def fake_logits(hidden):
        return hidden.to(torch.float32) @ proj

    drafter._step = fake_step
    drafter._logits = fake_logits
    drafter._calls = calls
    drafter._DummyCache = DummyCache
    return drafter


def test_varlen_prefill_mask_and_positions():
    drafter = _make_drafter()
    B = 3
    T = [3, 5, 2]  # per-row prefix lengths (het)
    P = max(T)
    g = torch.Generator().manual_seed(7)
    prefix_hidden = torch.randn(B, P, H, generator=g)
    shifted = torch.arange(B * P).reshape(B, P) % V + 1
    ctx = drafter._prefill_with_prefix_batch_varlen(
        prefix_hidden,
        shifted,
        cache_keys=[0, 1, 2],
        prefix_lens=T,
        root_tokens=[11, 12, 13],
    )
    # Fresh caches: L_b = 0, so W = S_max = 5 and the suffix forward wrote all
    # real content at [0, T_b) per row.
    assert ctx.cache_len_w == P
    assert ctx.valid_mask.shape == (B, P)
    for b in range(B):
        assert ctx.valid_mask[b].tolist() == [s < T[b] for s in range(P)]
    assert ctx.prefix_next_pos == T
    assert ctx.root_hiddens.shape == (B, H)

    # One _step call (the suffix forward), mask [B, 1, S_max, W] additive.
    assert len(drafter._calls) == 1
    call = drafter._calls[0]
    mask = call["mask"]
    assert mask.shape == (B, 1, P, P)
    neg = torch.finfo(torch.float32).min
    for b in range(B):
        m = mask[b, 0]  # [S_max, W]
        for i in range(P):
            for s in range(P):
                # fresh caches: L_max=0, so suffix slot j == s; allow j<=i and j<T_b
                expect = (s <= i) and (s < T[b])
                got = bool(m[i, s].item() != neg)
                assert got == expect, f"row{b} q{i} s{s}: got {got} want {expect}"
    # position_ids = arange(0, T_b) padded
    assert call["pos"].shape == (B, P)


def test_varlen_incremental_second_round():
    drafter = _make_drafter()
    B = 2
    T1 = [3, 5]
    P = 6  # second round prefix hiddens are wider than T
    g = torch.Generator().manual_seed(11)
    prefix_hidden = torch.randn(B, P, H, generator=g)
    shifted = (torch.arange(B * P).reshape(B, P) % V) + 1
    ctx1 = drafter._prefill_with_prefix_batch_varlen(
        prefix_hidden[:, : max(T1)], shifted[:, : max(T1)],
        cache_keys=[0, 1], prefix_lens=T1, root_tokens=[21, 22],
    )
    # second round: rows advanced by different accepts
    T2 = [5, 6]  # row0 +2, row1 +1
    ctx2 = drafter._prefill_with_prefix_batch_varlen(
        prefix_hidden, shifted,
        cache_keys=[0, 1], prefix_lens=T2, root_tokens=[23, 24],
    )
    # L = [3, 5], C = [2, 1], L_max = 5, S_max = 2, W = 7
    assert ctx2.cache_len_w == 7
    v0 = ctx2.valid_mask[0].tolist()
    v1 = ctx2.valid_mask[1].tolist()
    assert v0 == [True] * 3 + [False] * 2 + [True, True]          # [0,3) + [5,7)
    assert v1 == [True] * 5 + [True, False]                       # [0,5) + [5,6)
    # suffix mask: row0 new tokens at slots 5,6 (j=0,1); row1 at slot 5 (j=0)
    call = drafter._calls[-1]
    mask = call["mask"]
    neg = torch.finfo(torch.float32).min
    L_max, S_max = 5, 2
    C = [2, 1]
    L = [3, 5]
    for b in range(B):
        m = mask[b, 0]
        for i in range(S_max):
            for s in range(7):
                if s < L[b]:
                    expect = True
                elif s >= L_max:
                    j = s - L_max
                    expect = (j <= i) and (j < C[b])
                else:
                    expect = False
                got = bool(m[i, s].item() != neg)
                assert got == expect, f"row{b} q{i} s{s}: got {got} want {expect}"
    # position_ids per row: L_b + i for the c_b valid slots; pad slots stay 0
    assert call["pos"][0].tolist() == [3, 4]
    assert call["pos"][1].tolist() == [5, 0]


def test_varlen_build_end_to_end_native(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_TENSOR_TREE", "1")
    monkeypatch.setenv("BLOOMBEE_TENSOR_TREE_EMIT", "1")
    B = 3
    drafter = _make_drafter()
    drafter._merge_prefix_caches_batch = lambda caches: drafter._DummyCache()

    # het rows: seq_lengths differ -> varlen path
    input_ids = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19, 20, 21],
    ])
    seq_lengths = torch.tensor([7, 5, 3])
    prefix_hidden = torch.randn(B, 6, H)
    prev_last_token = torch.tensor([7, 12, 17])

    out = drafter.build_trees_parallel(
        input_ids, seq_lengths, 1, 5,
        prev_last_hidden=torch.randn(B, H),
        prev_last_token=prev_last_token,
        prefix_hidden_states=prefix_hidden,
        tree_budget=10, topk_per_step=3, do_sample=False,
    )
    assert out is None
    tt = drafter.last_tensor_tree
    assert tt is not None and tt.batch_size == B
    assert tt.max_nodes >= 2
    assert int(tt.n_nodes.max().item()) <= 11
    # expansion _step calls used per-row positions (T_b = 6, 4, 2; depth 1 -> T_b)
    exp_calls = [c for c in drafter._calls if c["ids"].shape[1] == 3]  # frontier F=K=3
    assert exp_calls, "no expansion calls recorded"
    first = exp_calls[0]
    assert first["pos"][:, 0].tolist() == [6, 4, 2]


def test_varlen_cache_longer_than_target_is_invalidated():
    """Regression: a cached prefix LONGER than the current target (e.g. warmup
    runs past the main run's first root) must be rebuilt — otherwise the reused
    last_hidden is a hidden from a future position and the drafter conditions on
    garbage (observed: accept collapsed to 1.0 for the first ~6 het rounds)."""
    drafter = _make_drafter()
    B = 2
    P = 6
    g = torch.Generator().manual_seed(13)
    prefix_hidden = torch.randn(B, P, H, generator=g)
    shifted = (torch.arange(B * P).reshape(B, P) % V) + 1
    # round 1: build caches of length 5 and 6
    drafter._prefill_with_prefix_batch_varlen(
        prefix_hidden, shifted, cache_keys=[0, 1], prefix_lens=[5, 6], root_tokens=[31, 32],
    )
    assert drafter._prefix_states[0].cache_len == 5
    assert drafter._prefix_states[1].cache_len == 6
    # round 2: targets SHORTER than the caches (e.g. fresh main run) -> rebuild
    ctx = drafter._prefill_with_prefix_batch_varlen(
        prefix_hidden, shifted, cache_keys=[0, 1], prefix_lens=[2, 4], root_tokens=[33, 34],
    )
    assert ctx.prefix_next_pos == [2, 4]
    # both rows rebuilt from scratch: W = S_max = 4, validity = [0, T_b)
    assert ctx.cache_len_w == 4
    assert ctx.valid_mask[0].tolist() == [True, True, False, False]
    assert ctx.valid_mask[1].tolist() == [True, True, True, True]


def test_varlen_object_mode_returns_trees():
    B = 3
    drafter = _make_drafter()
    drafter._merge_prefix_caches_batch = lambda caches: drafter._DummyCache()
    input_ids = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19, 20, 21],
    ])
    seq_lengths = torch.tensor([7, 5, 3])
    prefix_hidden = torch.randn(B, 6, H)
    trees = drafter.build_trees_parallel(
        input_ids, seq_lengths, 1, 5,
        prev_last_hidden=torch.randn(B, H),
        prev_last_token=torch.tensor([7, 12, 17]),
        prefix_hidden_states=prefix_hidden,
        tree_budget=10, topk_per_step=3, do_sample=False,
    )
    assert trees is not None and len(trees) == B
    assert all(t.total_nodes >= 1 for t in trees)
