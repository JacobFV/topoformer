"""extended-04 Phase A: the fast path is result-identical to the reference path (hard gate).

(a) bit-identical encoded observation vectors, candidate sets and candidate matrices for d1,
    d1-noapp and d1-noattempt on randomized trajectories (random / dep_reuse / dep_naive_reuse /
    epsilon-mixed policies; events of every kind; foreign records), plus identical observations,
    trace dicts and per-step utilities from environments stepped under either path;
(b) identical parameters and optimizer state after a short P1-style actor-critic tranche (KL to
    tranche start, advantage normalization) and a short supervised tranche, fast vs reference;
(c) identical sealed-style evaluation rows (minus timing fields), greedy and sampled.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import partial
import json
import random
import struct

import pytest

from tensegra import campaign04_fast as fast
from tensegra.campaign02_protocol import execute
from tensegra.campaign03_depworld import (EVENT_KINDS, FEATURE_VERSIONS, DepReference, DepWorkshop, action_catalog,
                                          depworld_executor, encode_public, generate_depworld)

EXEC = partial(depworld_executor, execute_call=execute)

WORLDS = [dict(p_event=1.0, foreign_records=4, event_trigger="progress"),
          dict(p_event=1.0, foreign_records=2, event_trigger="step"),
          dict(p_event=.5, foreign_records=0, event_trigger="progress",
               event_kinds=["edge_closed", "capacity_reduced", "slot_closed"]),
          dict(p_event=1.0, foreign_records=3, event_trigger="progress", event_kinds=["deadline_moved"]),
          dict(p_event=1.0, foreign_records=2, event_trigger="progress", event_kinds=["slot_closed"]),
          dict(p_event=1.0, foreign_records=1, event_trigger="step", event_kinds=["capacity_reduced"],
               categories=2, choices=4, locations=6, slots=5)]
POLICIES = ("random", "dep_reuse", "dep_naive_reuse", "mixed")


def _choose(policy, teacher, rng, o, actions):
    if policy == "random" or (policy == "mixed" and rng.random() < .35):
        # Bias toward stateful actions so attempts/rejections/records accumulate.
        weights = [3.0 if a.kind in ("call", "use_return", "commit_pending", "commit_assignment", "retrieve",
                                     "uncommit", "move", "choose_slot", "choose_item") else 1.0 for a in actions]
        weights = [0.0 if a.kind == "abstain" and o.step < 40 else w for a, w in zip(actions, weights)]
        return rng.choices(actions, weights)[0]
    return teacher.choose(o, actions)


def _trajectory_states(seed, policy, world, limit=90):
    rng = random.Random(f"e04-fast-{seed}-{policy}")
    env = DepWorkshop(generate_depworld(seed, **world), executor=EXEC, address_seed=seed)
    teacher = DepReference("naive_reuse" if policy == "dep_naive_reuse" else "reuse")
    o = env.observe()
    states = [o]
    for _ in range(limit):
        if o.done:
            break
        actions = action_catalog(o)
        o = env.step(_choose(policy, teacher, rng, o, actions))
        states.append(o)
    return states


def _bits(vector):
    """Exact identity: python type and the IEEE-754 bit pattern of every coordinate."""
    return [(type(x).__name__, struct.pack("<d", x)) for x in vector]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("world_index", range(len(WORLDS)))
def test_fast_encoder_is_bit_identical_on_random_trajectories(policy, world_index):
    fast.FALLBACKS["encode_public"] = 0
    covered = {"attempts": 0, "events": 0, "retrieved": 0, "foreign": 0}
    for seed in range(3):
        for o in _trajectory_states(7_000 + 31 * world_index + seed, policy, WORLDS[world_index]):
            covered["attempts"] += bool(o.attempts)
            covered["events"] += bool(o.events)
            covered["retrieved"] += bool(o.retrieved)
            covered["foreign"] += any(r["created_step"] == 0 for r in o.records)
            actions = action_catalog(o)
            assert actions == action_catalog(o)
            for version in FEATURE_VERSIONS:
                ref_obs, ref_rows = encode_public(o, actions, version, fast=False)
                new_obs, new_rows = encode_public(o, actions, version, fast=True)
                assert type(new_obs) is list and all(type(r) is list for r in new_rows)
                assert _bits(new_obs) == _bits(ref_obs), version
                assert len(new_rows) == len(ref_rows) == len(actions)
                for a, x, y in zip(actions, new_rows, ref_rows):
                    assert _bits(x) == _bits(y), (version, a)
    assert fast.FALLBACKS["encode_public"] == 0
    if policy in ("random", "mixed"):
        assert covered["attempts"] and covered["retrieved"]
    if WORLDS[world_index]["foreign_records"]:
        assert covered["foreign"]


def test_trajectories_cover_every_event_kind_and_rejections():
    kinds, reasons = set(), set()
    for world_index, world in enumerate(WORLDS):
        for policy in POLICIES:
            for seed in range(3):
                last = _trajectory_states(7_000 + 31 * world_index + seed, policy, world)[-1]
                kinds |= {e["kind"] for e in last.events}
                reasons |= {a["reason"] for a in last.attempts if a["reason"]}
    assert kinds == set(EVENT_KINDS)
    assert len(reasons) >= 4


def _lockstep(seed, world, policy, env_class=DepWorkshop):
    """Step two identical worlds, one under each path; everything visible must match."""
    rng = random.Random(f"lockstep-{seed}-{policy}")
    envs = {}
    for flag in (False, True):
        with fast.path(flag):
            envs[flag] = env_class(generate_depworld(seed, **world), executor=EXEC, address_seed=seed)
    teacher = DepReference("naive_reuse" if policy == "dep_naive_reuse" else "reuse")
    for _ in range(90):
        views = {}
        for flag, env in envs.items():
            with fast.path(flag):
                o = env.observe()
                views[flag] = (o, o.to_dict(), env.evaluate(), env.current_utility())
        (o_ref, d_ref, e_ref, u_ref), (o_new, d_new, e_new, u_new) = views[False], views[True]
        assert o_new == o_ref and d_new == d_ref
        assert json.dumps(d_new, sort_keys=True) == json.dumps(d_ref, sort_keys=True)
        assert d_new == json.loads(json.dumps(asdict(o_new)))
        assert u_new == u_ref == e_ref["utility"] and _strip(e_new) == _strip(e_ref)
        assert _types(asdict(o_new)) == _types(asdict(o_ref))
        if o_ref.done:
            break
        action = _choose(policy, teacher, rng, o_ref, action_catalog(o_ref))
        for flag, env in envs.items():
            with fast.path(flag):
                env.step(action)


def _types(x):
    if isinstance(x, dict):
        return {k: _types(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [_types(v) for v in x])
    return type(x).__name__


@pytest.mark.parametrize("policy", POLICIES)
def test_environment_observations_trace_dicts_and_utilities_match(policy):
    from tensegra.campaign03_p1_audit import P1AuditedDepWorkshop
    for seed, world in enumerate(WORLDS[:4]):
        _lockstep(9_100 + seed, world, policy)
    _lockstep(9_200, WORLDS[0], policy, P1AuditedDepWorkshop)


def test_json_conversion_matches_json_roundtrip():
    samples = [{"a": (1, 2.5, None, True), 3: [{"x": -0.0}], 2.5: "f", True: 1, None: {}},
               {"nan": float("inf"), "t": ((1, 2), [3, (4,)])}, [(), {}, "s", 1e-300, 12345678901234567890],
               {1: "first", "1": "second"}]
    for value in samples:
        assert fast.jsonable(value) == json.loads(json.dumps(value))
        assert json.dumps(fast.jsonable(value)) == json.dumps(json.loads(json.dumps(value)))
    with pytest.raises(fast._NotJSON):
        fast.jsonable({"x": object()})
    copy = fast.plain_copy(samples[0])
    assert copy == samples[0] and copy is not samples[0] and copy["a"] == samples[0]["a"]
    assert _types(copy) == _types(samples[0])


def test_action_key_memo_matches_reference():
    from tensegra.campaign02_world import Action
    from tensegra.campaign03_depworld import action_key
    for action in (Action("call", {"problem": "problem_0", "budget": 1}), Action("call", {"budget": 1, "problem": "problem_0"}),
                   Action("move", {"destination": 1}), Action("move", {"destination": True}),
                   Action("use_return", {"handle": "r1", "as": "select"}), Action("verify"),
                   Action("add_constraint", {"problem": "p", "constraint": "finish_by", "bound": 3.0})):
        assert fast.action_key(action) == action_key(action)
        assert fast.action_key(action) == action_key(action)


def test_patched_reference_helpers_route_the_encoder_to_the_reference_body(monkeypatch):
    """Audits that monkeypatch reference helpers (the Stage B preflight perturbations) must see
    their patches honoured, so the fast encoder steps aside when any helper is replaced."""
    import tensegra.campaign03_depworld as dep
    o = [s for s in _trajectory_states(7_050, "mixed", WORLDS[0]) if s.records and not s.done][-1]
    actions = action_catalog(o)
    base = encode_public(o, actions, "d1", fast=True)
    original = dep.relations

    def flipped(o, record, primitive=None):
        return {k: (not v if k == "request_match" else v) for k, v in original(o, record, primitive).items()}
    monkeypatch.setattr(dep, "relations", flipped)
    patched_fast, patched_ref = encode_public(o, actions, "d1", fast=True), encode_public(o, actions, "d1", fast=False)
    assert patched_fast == patched_ref and patched_fast != base


def test_reference_path_is_selectable_and_default_is_fast():
    assert fast.enabled()
    with fast.path(False):
        assert not fast.enabled() and fast.describe() == "reference"
    assert fast.enabled() and fast.describe() == "fast"


# ---------------------------------------------------------------------------
# (b)/(c) training and evaluation equivalence (torch)
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:  # the pure-python equivalence tests above still run
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch unavailable")

MIX = [dict(categories=3, choices=3, locations=7, slots=6, compute_price=1e-4, event_trigger="progress",
            p_event=.5, foreign_records=f, event_kinds=["edge_closed", "capacity_reduced", "slot_closed"])
       for f in (0, 2)]


def _factory(env_class=DepWorkshop):
    from tensegra.campaign02_population import mix_index
    from tensegra.campaign02_training import independent_address_seed

    def factory(seed):
        return env_class(generate_depworld(seed, **MIX[mix_index(seed, 2)]), executor=EXEC,
                         address_seed=independent_address_seed(seed, "e04-fast-test"))
    return factory


def _learner(method, version="d1", width=64, **extra):
    from tensegra.campaign02_population import new_policy
    from tensegra.campaign02_training import Learner, TrainConfig, public_frame
    torch.manual_seed(11)
    random.seed(11)
    _, obs, cand = public_frame(_factory()(1).observe(), version)
    model = new_policy({"interface": "legacy", "feature_version": version}, len(obs), len(cand[0]), width, "lightweight")
    cfg = TrainConfig(seed=11, width=width, batch_size=4, method=method, max_steps=48, rollout_mode="batched",
                      training_seed_start=3_440_000_000, **extra)
    return Learner(model, cfg)


def _train(flag, method, updates, version="d1", **extra):
    from tensegra.campaign02_references import make_reference
    with fast.path(flag):
        learner = _learner(method, version, **extra)
        torch.manual_seed(12)
        timing = learner.train_tranche(updates, _factory(), (lambda: make_reference("dep_reuse"))
                                       if method == "supervised" else None)
        rng = torch.get_rng_state()
    return learner, timing, rng


def _same_state(a, b):
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    oa, ob = a.optimizer.state_dict(), b.optimizer.state_dict()
    assert oa["param_groups"] == ob["param_groups"] and oa["state"].keys() == ob["state"].keys()
    for key in oa["state"]:
        for name, value in oa["state"][key].items():
            other = ob["state"][key][name]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    assert (a.updates, a.presentations, a.episodes, a.seed_cursor) == (b.updates, b.presentations, b.episodes, b.seed_cursor)
    # solver_cpu_seconds is measured process time, not a result.
    strip = [[{k: v for k, v in c.items() if k != "solver_cpu_seconds"} for c in x.curves] for x in (a, b)]
    assert strip[0] == strip[1]


@pytest.mark.parametrize("version", ["d1", "d1-noapp"])
@needs_torch
def test_actor_critic_tranche_identical_fast_vs_reference(version):
    extra = dict(kl_weight=.3, advantage_normalization=True, learning_rate=3e-4, entropy_weight=.003)
    ref, ref_timing, ref_rng = _train(False, "actor_critic", 3, version, **extra)
    new, new_timing, new_rng = _train(True, "actor_critic", 3, version, **extra)
    _same_state(ref, new)
    assert torch.equal(ref_rng, new_rng)
    assert ref_timing["throughput_path"] == "reference" and new_timing["throughput_path"] == "fast"


@needs_torch
def test_supervised_tranche_identical_fast_vs_reference():
    ref, _, ref_rng = _train(False, "supervised", 3, "d1-noattempt")
    new, _, new_rng = _train(True, "supervised", 3, "d1-noattempt")
    _same_state(ref, new)
    assert torch.equal(ref_rng, new_rng)
    # Teacher-trace data hash contains no timing, so it is identical too.
    assert ref.data_hash.hexdigest() == new.data_hash.hexdigest()


@needs_torch
def test_actor_critic_trace_hash_identical_once_timings_are_fixed(monkeypatch):
    """The on-policy data hash covers the trace, which carries wall-clock fields; with the clock
    frozen, the hashed bytes are identical fast vs reference (hash semantics unchanged)."""
    import tensegra.campaign02_training as training
    monkeypatch.setattr(training.time, "perf_counter", lambda: 0.0)
    extra = dict(kl_weight=.3, advantage_normalization=True)
    ref, _, _ = _train(False, "actor_critic", 2, **extra)
    new, _, _ = _train(True, "actor_critic", 2, **extra)
    assert ref.data_hash.hexdigest() == new.data_hash.hexdigest()


TIMING = {"episode_process_cpu_seconds", "episode_wall_seconds", "neural_forward_wall_seconds_allocated", "timing",
          "solver_cpu_seconds", "cpu_seconds", "child_cpu_seconds"}


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in TIMING}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


@needs_torch
@pytest.mark.parametrize("mode", ["greedy", "sampled"])
def test_sealed_style_evaluation_rows_identical(mode):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "tools"))
    from campaign02_evaluate import episode_counts
    from tensegra.campaign02_training import batched_episodes, sampling_rng_seed
    from tensegra.campaign03_p1_audit import P1AuditedDepWorkshop
    learner = _learner("actor_critic", width=64)
    policy = learner.model.eval()
    seeds = list(range(120_100_000, 120_100_012))
    rows = {}
    for flag in (False, True):
        factory = _factory(P1AuditedDepWorkshop)
        with fast.path(flag), torch.no_grad():
            samplers = None if mode == "greedy" else [random.Random(sampling_rng_seed(s, 20260926)) for s in seeds]
            out = batched_episodes(policy, [factory(s) for s in seeds], max_steps=64,
                                   **({} if samplers is None else {"samplers": samplers}))
            rows[flag] = [json.dumps(_strip({"seed": s, **r, "counts": episode_counts(r["outcome"])}), sort_keys=True)
                          for s, r in zip(seeds, out)]
    assert rows[True] == rows[False]
    assert any('"verify"' in r for r in rows[True])


@needs_torch
def test_collate_identical():
    from tensegra.campaign02_training import Frame, collate, public_frame
    frames = []
    for seed in range(4):
        for o in [o for o in _trajectory_states(8_000 + seed, "mixed", WORLDS[seed]) if not o.done][::7]:
            _, obs, cand = public_frame(o, "d1")
            frames.append(Frame(obs, cand, len(cand) - 1))
    for chunk in (frames[:1], frames[:5], frames):
        a, b = collate(chunk, fast=False), collate(chunk, fast=True)
        assert len(a) == len(b) == 4
        for x, y in zip(a, b):
            assert x.dtype == y.dtype and x.shape == y.shape and x.device == y.device and torch.equal(x, y)
