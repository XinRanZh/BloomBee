"""Native GPU-tree emit (stage 2b/2c): `_build_tensor_tree_batched` must produce
a TensorTreeBatch bit-identical to the object path
(`_build_trees_from_prefix_caches_batched` + bridge) given identical drafter
internals. Runs on CPU with stubbed `_step`/`_logits`/cache-merge so the whole
expansion + selection pipeline executes deterministically.

Also covers the `build_trees_parallel` wiring: BLOOMBEE_TENSOR_TREE_EMIT=1
returns None and populates `last_tensor_tree`; =shadow validates and keeps
object trees; do_sample stays on the object path.
"""
import os

import pytest
import torch

from bloombee.models.llama import eagle_drafter as ED
from bloombee.models.llama.eagle_drafter import EAGLEDrafter, _PrefixBuildJob
from bloombee.models.llama.tensor_tree import tensor_tree_from_speculative_trees

H = 8
V = 101


def _make_drafter(seed=0):
    drafter = object.__new__(EAGLEDrafter)
    drafter.device = torch.device("cpu")
    drafter.dtype = torch.float32
    drafter._prefix_states = {}
    drafter.last_tensor_tree = None

    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(H, V, generator=g) / (H ** 0.5)

    class DummyCache:
        def __init__(self):
            self.cropped = []

        def crop(self, n):
            self.cropped.append(n)

    def fake_step(hidden_states, input_ids, position_ids, past_key_values, attention_mask=None):
        # Deterministic "head": mixes the conditioning hidden, the token id and
        # the position so different branches diverge.
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
    drafter._merge_prefix_caches_batch = lambda caches: DummyCache()
    return drafter


def _make_jobs(B, prefix_next_pos=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    jobs = []
    for b in range(B):
        jobs.append(
            _PrefixBuildJob(
                batch_index=b,
                root_token=int(10 + b),
                root_hidden=torch.randn(H, generator=g),
                prefix_cache=object(),
                prefix_next_pos=prefix_next_pos,
            )
        )
    return jobs


def _assert_tt_eq(tt_native, tt_bridge):
    assert tt_native.max_nodes == tt_bridge.max_nodes, (
        f"max_nodes {tt_native.max_nodes} vs {tt_bridge.max_nodes}"
    )
    for name in ("n_nodes", "token", "parent_idx", "depth"):
        a = getattr(tt_native, name)
        b = getattr(tt_bridge, name)
        assert torch.equal(a, b), (
            f"{name} differ:\nnative={a.tolist()}\nbridge={b.tolist()}"
        )


@pytest.mark.parametrize("B", [2, 3, 5])
@pytest.mark.parametrize("total_token", [4, 8, 11])
def test_native_emit_matches_object_path(B, total_token):
    drafter = _make_drafter()
    jobs = _make_jobs(B)
    trees = drafter._build_trees_from_prefix_caches_batched(
        jobs=jobs, max_candidate_depth=6, total_token=total_token, expansion_width=3,
    )
    tt_native = drafter._build_tensor_tree_batched(
        jobs=jobs, max_candidate_depth=6, total_token=total_token, expansion_width=3,
    )
    tt_bridge = tensor_tree_from_speculative_trees(trees, torch.device("cpu"))
    _assert_tt_eq(tt_native, tt_bridge)


def test_native_emit_build_trees_parallel_wiring(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_TENSOR_TREE", "1")
    monkeypatch.setenv("BLOOMBEE_TENSOR_TREE_EMIT", "1")
    B = 3
    drafter = _make_drafter()

    def fake_prefill_batch(prefix_hidden_states, shifted_input_ids, *, cache_keys):
        return [
            (torch.randn(H), object(), int(shifted_input_ids.shape[1]))
            for _ in cache_keys
        ]

    drafter._prefill_with_prefix_batch = fake_prefill_batch
    out = drafter.build_trees_parallel(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]] * B),
        seq_lengths=torch.tensor([6] * B),
        prefix_hidden_states=torch.randn(B, 5, H),
        prev_last_token=torch.tensor([6] * B),
        beam_width=1,
        max_depth=5,
        tree_budget=10,
        topk_per_step=3,
    )
    assert out is None
    tt = drafter.last_tensor_tree
    assert tt is not None
    assert tt.batch_size == B
    assert tt.max_nodes >= 2
    assert int(tt.n_nodes.max().item()) <= 11  # total_token budget incl. root


def test_shadow_mode_keeps_object_trees_and_validates(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_TENSOR_TREE_EMIT", "shadow")
    B = 3
    drafter = _make_drafter()

    def fake_prefill_batch(prefix_hidden_states, shifted_input_ids, *, cache_keys):
        return [
            (torch.randn(H), object(), int(shifted_input_ids.shape[1]))
            for _ in cache_keys
        ]

    drafter._prefill_with_prefix_batch = fake_prefill_batch
    trees = drafter.build_trees_parallel(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]] * B),
        seq_lengths=torch.tensor([6] * B),
        prefix_hidden_states=torch.randn(B, 5, H),
        prev_last_token=torch.tensor([6] * B),
        beam_width=1,
        max_depth=5,
        tree_budget=10,
        topk_per_step=3,
    )
    assert trees is not None and len(trees) == B
    assert drafter.last_tensor_tree is None


def test_native_emit_skipped_for_sampling(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_TENSOR_TREE", "1")
    monkeypatch.setenv("BLOOMBEE_TENSOR_TREE_EMIT", "1")
    B = 3
    drafter = _make_drafter()

    def fake_prefill_batch(prefix_hidden_states, shifted_input_ids, *, cache_keys):
        return [
            (torch.randn(H), object(), int(shifted_input_ids.shape[1]))
            for _ in cache_keys
        ]

    drafter._prefill_with_prefix_batch = fake_prefill_batch
    trees = drafter.build_trees_parallel(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]] * B),
        seq_lengths=torch.tensor([6] * B),
        prefix_hidden_states=torch.randn(B, 5, H),
        prev_last_token=torch.tensor([6] * B),
        beam_width=1,
        max_depth=5,
        tree_budget=10,
        topk_per_step=3,
        do_sample=True,
    )
    assert trees is not None and len(trees) == B
    assert drafter.last_tensor_tree is None
