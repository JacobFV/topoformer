"""extended-04 Track A2 training options (CPU, tiny widths; mechanical fixtures only).

rehearsal / imitation-only, entropy Lagrangian, critic warm-up, masked-sampling rollouts.
Every option defaults off with a bit-identical historical path.
"""
from dataclasses import replace
from functools import partial
import hashlib
import json
import sys
import types

import pytest
import torch

from tensegra.campaign02_policy import CandidatePolicy, PolicyConfig
from tensegra.campaign02_protocol import execute
from tensegra.campaign02_training import (A2_FIELD_DEFAULTS, ROLLOUT_MASKS, Learner, TrainConfig,
                                          actor_critic_objective, batched_on_policy, critic_warmup_trainable,
                                          load_anchor, policy_batch, prepare_frame, public_frame, score_batch)
from tensegra.campaign03_depworld import DepReference, DepWorkshop, depworld_executor, generate_depworld

EXEC = partial(depworld_executor, execute_call=execute)
DEV = 2_070_000_000  # development seeds only
WORLD = {"categories": 2, "choices": 2, "locations": 5, "slots": 5, "foreign_records": 2, "p_event": 0.5,
         "compute_price": 0.0001, "event_trigger": "progress"}


def factory(seed):
    return DepWorkshop(generate_depworld(seed, **WORLD), executor=EXEC, address_seed=seed + 3)


def teacher():
    return DepReference("reuse")


def new_model(seed):
    torch.manual_seed(seed)
    _, obs, candidates = public_frame(factory(DEV).observe(), "d1")
    return CandidatePolicy(PolicyConfig(len(obs), len(candidates[0]), width=8, feature_version="d1"))


def config(**over):
    base = dict(width=8, method="actor_critic", rollout_mode="batched", batch_size=2, max_steps=10,
                learning_rate=1e-3, entropy_weight=0.003, kl_weight=0.3, advantage_normalization=True,
                training_seed_start=DEV + 100)
    base.update(over)
    return TrainConfig(**base)


def train(cfg, updates=3, model_seed=11, torch_seed=5, teacher_factory=None, learner=None):
    learner = learner or Learner(new_model(model_seed), cfg)
    torch.manual_seed(torch_seed)
    timing = learner.train_tranche(updates, factory, teacher_factory)
    return learner, timing


def save_checkpoint(tmp_path, seed, name="anchor.pt"):
    learner = Learner(new_model(seed), config())
    path = tmp_path / name
    learner.save(path, {"role": "fixture"})
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def same_params(a, b, prefix=None):
    sa, sb = a.state_dict(), b.state_dict()
    keys = [k for k in sa if prefix is None or k.startswith(prefix)]
    return all(torch.equal(sa[k], sb[k]) for k in keys)


def same_optimizer(a, b):
    sa, sb = a.optimizer.state_dict()["state"], b.optimizer.state_dict()["state"]
    return sa.keys() == sb.keys() and all(torch.equal(sa[k][n], sb[k][n]) for k in sa for n in sa[k])


# ---------------------------------------------------------------------------- off = identical

def test_config_validation():
    for bad in ({"rehearsal_weight": -1}, {"rl_loss_weight": -0.1}, {"entropy_target": -1.0},
                {"entropy_dual_lr": 0.1}, {"entropy_target": 1.0, "entropy_dual_init": 2.0},
                {"critic_warmup_updates": -1}, {"rehearsal_weight": 0.5, "rollout_mode": "serial"}):
        with pytest.raises(ValueError):
            config(**bad)
    cfg = config()
    assert {k: getattr(cfg, k) for k in A2_FIELD_DEFAULTS} == A2_FIELD_DEFAULTS
    # A population's train block has no method (set per round): options still construct.
    TrainConfig(rollout_mode="batched", rehearsal_weight=0.5)


def test_options_absent_and_present_but_off_are_bit_identical(tmp_path):
    """Absent vs explicitly-off (every A2 field at its default, a teacher supplied, anchor on)."""
    anchor = save_checkpoint(tmp_path, 99)
    base = dict(anchor_kl_weight=0.3, anchor_checkpoint=anchor)
    a, _ = train(config(**base))
    b, timing = train(config(**base, **A2_FIELD_DEFAULTS), teacher_factory=teacher)
    assert same_params(a.model, b.model) and same_optimizer(a, b)
    assert [x["loss"] for x in a.curves] == [x["loss"] for x in b.curves]
    assert [x["objective_parts"] for x in a.curves] == [x["objective_parts"] for x in b.curves]
    assert all("a2" not in x for x in b.curves) and "a2" not in timing and b.entropy_dual is None
    path = tmp_path / "off.pt"
    b.save(path)
    assert "entropy_dual" not in torch.load(path, weights_only=False)


def test_old_checkpoints_without_a2_fields_import_unchanged(tmp_path):
    learner = Learner(new_model(11), config())
    path = tmp_path / "old.pt"
    learner.save(path)
    saved = torch.load(path, weights_only=False)
    for key in A2_FIELD_DEFAULTS:
        saved["config"].pop(key)
    torch.save(saved, path)
    fresh = Learner(new_model(11), config())
    fresh.load(path)
    assert fresh.resume_history[-1]["config_changes"] == {}
    on = Learner(new_model(11), config(rehearsal_weight=0.5))
    with pytest.raises(ValueError, match="Explicit resume"):
        on.load(path)


# ---------------------------------------------------------------------------- A2-reh / A2-imit

class RecordingTeacher(DepReference):
    def __init__(self, log):
        super().__init__("reuse")
        self.log = log

    def choose(self, observation, catalog=None):
        self.log.append((observation.to_dict(), list(catalog)))
        return super().choose(observation, catalog)


def test_rehearsal_teacher_sees_the_actors_public_state_and_gradients_flow():
    model = new_model(11)
    logs = [[], []]
    torch.manual_seed(1)
    envs = [factory(DEV + i) for i in range(2)]
    aux = {}
    results, terms = batched_on_policy(model, envs, max_steps=8, teachers=[RecordingTeacher(l) for l in logs], aux=aux)
    for result, log, ces in zip(results, logs, aux["rehearsal_ce"]):
        # One teacher query per decision, on exactly the observation the actor acted on, with its catalog.
        assert [s["observation"] for s in result["trace"]] == [obs for obs, _ in log]
        assert [s["candidate_count"] for s in result["trace"]] == [len(c) for _, c in log]
        assert len(ces) == len(result["trace"])  # dep_reuse proposes an in-catalog action at every visited state
    assert aux["rehearsal_out_of_catalog"] == 0 and aux["rehearsal_decisions"] == sum(map(len, terms))
    cfg = config(rehearsal_weight=0.5, kl_weight=0.0)
    loss, parts = actor_critic_objective(terms, cfg, rehearsal_ces=aux["rehearsal_ce"])
    plain, _ = actor_critic_objective(terms, replace(cfg, rehearsal_weight=0.0))
    assert float(parts["rehearsal_ce"].detach()) > 0
    assert float(loss - plain) == pytest.approx(0.5 * float(parts["rehearsal_ce"]), rel=1e-5, abs=1e-7)
    model.zero_grad()
    parts["rehearsal_ce"].backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if n.startswith("scorer."))
    assert all(g is None or g.abs().sum() == 0 for n, g in grads.items() if n.startswith("value."))


def test_rehearsal_training_logs_and_requires_teacher():
    with pytest.raises(ValueError, match="teacher"):
        train(config(rehearsal_weight=0.5), updates=1)
    learner, timing = train(config(rehearsal_weight=0.5), teacher_factory=teacher)
    for row in learner.curves:
        assert row["objective_parts"]["rehearsal_ce"] > 0
        assert row["a2"]["rehearsal_decisions"] + row["a2"]["rehearsal_out_of_catalog"] == row["on_policy_decisions"]
    assert timing["a2"]["teacher"] == "dep_reuse" and timing["a2"]["options"]["rehearsal_weight"] == 0.5
    base, _ = train(config())
    assert not same_params(base.model, learner.model)


def test_out_of_catalog_teacher_proposals_are_counted_not_repaired():
    from tensegra.campaign02_world import Action

    class Bad(DepReference):
        def choose(self, observation, catalog=None):
            return Action("no_such_action")
    aux = {}
    torch.manual_seed(1)
    _, terms = batched_on_policy(new_model(11), [factory(DEV)], max_steps=4, teachers=[Bad("reuse")], aux=aux)
    assert aux["rehearsal_out_of_catalog"] == len(terms[0]) and aux["rehearsal_ce"] == [[]]


def test_imitation_only_keeps_kl_entropy_rehearsal_and_leaves_value_head_untouched(tmp_path):
    anchor = save_checkpoint(tmp_path, 99)
    cfg = config(rehearsal_weight=0.5, rl_loss_weight=0.0, anchor_kl_weight=0.3, anchor_checkpoint=anchor)
    start = new_model(11)
    learner, _ = train(cfg, teacher_factory=teacher)
    assert same_params(start, learner.model, "value.")  # no critic gradient, no AdamW decay
    assert not same_params(start, learner.model, "scorer.")
    for row in learner.curves:
        p = row["objective_parts"]
        expected = -0.003 * p["entropy"] + 0.3 * p["kl_to_round_start"] + 0.3 * p["kl_to_anchor"] + 0.5 * p["rehearsal_ce"]
        assert row["loss"] == pytest.approx(expected, rel=1e-5, abs=1e-6)
    # rl_loss_weight scales the RL part exactly
    torch.manual_seed(1)
    _, terms = batched_on_policy(new_model(11), [factory(DEV + i) for i in range(2)], max_steps=6)
    full, parts = actor_critic_objective(terms, config(kl_weight=0.0))
    half, _ = actor_critic_objective(terms, config(kl_weight=0.0, rl_loss_weight=0.5))
    rl = float(parts["policy_loss"] + 0.5 * parts["critic_decision_mean_loss"])
    assert float(full - half) == pytest.approx(0.5 * rl, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------- A2-ent

@pytest.mark.parametrize("target,direction", [(5.0, 1), (0.0, -1)])
def test_entropy_dual_moves_toward_target_and_persists(tmp_path, target, direction):
    cfg = config(entropy_target=target, entropy_dual_lr=0.05)
    learner, timing = train(cfg, updates=3)
    duals = [row["a2"]["entropy_dual_used"] for row in learner.curves] + [learner.entropy_dual]
    assert duals[0] == 0.0
    assert all(direction * (b - a) > 0 for a, b in zip(duals, duals[1:]))
    for row in learner.curves:
        a2 = row["a2"]
        assert a2["entropy_dual_next"] == pytest.approx(a2["entropy_dual_used"] + 0.05 * (target - a2["entropy_measured"]))
        assert row["objective_parts"]["entropy_dual"] == pytest.approx(a2["entropy_dual_used"])
        assert a2["entropy_measured"] == pytest.approx(row["objective_parts"]["entropy"], rel=1e-5)
    assert timing["a2"]["entropy_dual"] == learner.entropy_dual
    path = tmp_path / "ent.pt"
    learner.save(path)
    resumed = Learner(new_model(11), cfg)
    resumed.load(path)
    assert resumed.entropy_dual == learner.entropy_dual
    # clipping
    clipped, _ = train(config(entropy_target=target, entropy_dual_lr=100.0, entropy_dual_max=0.05), updates=2)
    assert abs(clipped.entropy_dual) == 0.05


def test_entropy_dual_enters_the_objective_as_an_entropy_coefficient():
    torch.manual_seed(1)
    _, terms = batched_on_policy(new_model(11), [factory(DEV + i) for i in range(2)], max_steps=6)
    cfg = config(kl_weight=0.0, entropy_target=1.0, entropy_dual_lr=0.1)
    plain, parts = actor_critic_objective(terms, cfg)
    dual, dparts = actor_critic_objective(terms, cfg, entropy_dual=0.2)
    assert float(dual - plain) == pytest.approx(-0.2 * float(parts["entropy"]), rel=1e-5, abs=1e-7)


# ---------------------------------------------------------------------------- A2-crit

def test_critic_warmup_freezes_the_actor_then_releases_it():
    cfg = config(critic_warmup_updates=2)
    start = new_model(11)
    learner = Learner(new_model(11), cfg)
    names = [n for n, _ in learner.model.named_parameters()]
    train(cfg, updates=1, learner=learner)
    train(cfg, updates=1, torch_seed=6, learner=learner)  # the count continues across tranches
    assert [row["a2"]["critic_warmup"] for row in learner.curves] == [True, True]
    assert all(row["loss"] == pytest.approx(0.5 * row["objective_parts"]["critic_decision_mean_loss"], rel=1e-6)
               for row in learner.curves)
    # Actor head and shared trunk bit-identical; value head trained; optimizer state only for the value head.
    assert same_params(start, learner.model, "scorer.")
    assert all(torch.equal(p, dict(start.named_parameters())[n]) for n, p in learner.model.named_parameters()
               if not n.startswith("value."))
    assert not same_params(start, learner.model, "value.")
    state_ids = set(learner.optimizer.state_dict()["state"])
    assert state_ids == {i for i, n in enumerate(names) if n.startswith("value.")}
    # Policy logits bit-identical after warm-up.
    obs = factory(DEV + 5).observe()
    batch = policy_batch(start, [prepare_frame(obs, start)[1]], "cpu")
    assert torch.equal(score_batch(start, batch)[0], score_batch(learner.model, batch)[0])
    train(cfg, updates=1, torch_seed=7, learner=learner)
    assert learner.curves[-1]["a2"]["critic_warmup"] is False
    assert not same_params(start, learner.model, "scorer.")


def test_critic_warmup_shared_trains_trunk_but_not_actor_head():
    cfg = config(critic_warmup_updates=2, critic_warmup_shared=True)
    start = new_model(11)
    learner, _ = train(cfg, updates=2)
    assert same_params(start, learner.model, "scorer.")
    assert not same_params(start, learner.model, "context.") or not same_params(start, learner.model, "observation.")
    assert critic_warmup_trainable("value.1.weight", False) and not critic_warmup_trainable("context.1.weight", False)
    assert critic_warmup_trainable("context.1.weight", True) and not critic_warmup_trainable("scorer.1.weight", True)


def test_warmup_count_excludes_supervised_bootstrap_rows():
    learner = Learner(new_model(11), config(critic_warmup_updates=2))
    learner.curves = [{"rollout_mode": None}] * 5  # an imported supervised bootstrap's curve rows
    train(learner.config, updates=1, learner=learner)
    assert learner.curves[-1]["a2"]["critic_warmup"] is True


# ---------------------------------------------------------------------------- A2-dep

def _recompute_logits(model, observation):
    batch = policy_batch(model, [prepare_frame(observation, model)[1]], "cpu")
    return score_batch(model, batch)[0][0]


def test_masked_sampling_is_exactly_on_policy_for_the_masked_distribution():
    model = new_model(11)
    seen = []

    def mask_fn(index, observation, candidates, history):
        flags = [i % 2 == 0 for i in range(len(candidates))]  # mask even-indexed candidates
        seen.append((index, observation, flags, len(history)))
        return flags
    aux = {}
    torch.manual_seed(2)
    results, terms = batched_on_policy(model, [factory(DEV + i) for i in range(2)], max_steps=8,
                                       action_mask_fn=mask_fn, aux=aux)
    assert aux["masked_decisions"] > 0 and aux["all_masked_fallbacks"] == 0
    by_env = {0: [], 1: []}
    for index, observation, flags, history_length in seen:
        by_env[index].append((observation, flags, history_length))
    for index, result in enumerate(results):
        assert [h for _, _, h in by_env[index]] == list(range(len(result["trace"])))  # history = steps so far
        for step, (observation, flags, _), term in zip(result["trace"], by_env[index], terms[index]):
            assert not flags[step["action_index"]]  # never a masked action
            logits = _recompute_logits(model, observation)
            masked = logits.masked_fill(torch.tensor(flags), -torch.inf)
            expected = masked.log_softmax(-1)[step["action_index"]]
            assert float(term[0]) == pytest.approx(float(expected), abs=1e-5)
            assert step["probability"] == pytest.approx(float(expected.exp()), abs=1e-6)
            assert step["masked_actions"] == sum(flags)
            p = masked.softmax(-1)
            ent = -(p[~torch.tensor(flags)] * p[~torch.tensor(flags)].log()).sum()
            assert float(term[2]) == pytest.approx(float(ent), abs=1e-5)
    # The masked log-prob carries the policy gradient.
    loss, _ = actor_critic_objective(terms, config(kl_weight=0.0))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.startswith("scorer."))


def test_all_masked_and_none_masks_fall_back_to_the_unmasked_policy():
    model = new_model(11)
    for fn, fallbacks in ((lambda i, o, c, h: [True] * len(c), True), (lambda i, o, c, h: None, False)):
        aux = {}
        torch.manual_seed(3)
        masked_run = batched_on_policy(model, [factory(DEV)], max_steps=6, action_mask_fn=fn, aux=aux)
        torch.manual_seed(3)
        plain = batched_on_policy(model, [factory(DEV)], max_steps=6)
        assert [s["action"] for s in masked_run[0][0]["trace"]] == [s["action"] for s in plain[0][0]["trace"]]
        assert [float(t[0]) for t in masked_run[1][0]] == [float(t[0]) for t in plain[1][0]]
        assert aux["masked_decisions"] == 0 and (aux["all_masked_fallbacks"] > 0) == fallbacks


def test_registered_mask_rule_in_training(monkeypatch):
    calls = []

    def fake(batch_size):
        calls.append(batch_size)
        return lambda i, o, c, h: [a.kind == "abstain" for a in c]
    monkeypatch.setitem(ROLLOUT_MASKS, "fake_test", fake)
    learner, timing = train(config(rollout_mask="fake_test"), updates=2)
    assert calls == [2, 2]
    assert all(row["a2"]["masked_decisions"] > 0 for row in learner.curves)
    assert timing["a2"]["options"]["rollout_mask"] == "fake_test"
    with pytest.raises(ValueError, match="Unknown rollout mask"):
        train(config(rollout_mask="nope"), updates=1)


def test_progress_v1_mask_absent_module_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensegra.campaign04_progress", None)
    with pytest.raises(RuntimeError, match="campaign04_progress"):
        train(config(rollout_mask="progress_v1"), updates=1)


def test_progress_v1_adapter_uses_tracker_flagged_keys(monkeypatch):
    """Stub diagnostic: flags every action key taken at an identical observation before.
    The adapter never masks verify/abstain and feeds each executed step exactly once."""
    from tensegra.campaign03_depworld import action_key
    fed = []

    class StubTracker:
        def __init__(self):
            self.flag = set()

        def update(self, before, action, after):
            fed.append(action_key(action))
            self.flag.add(action_key(action))

        def flagged_keys(self, observation):
            return self.flag
    module = types.ModuleType("tensegra.campaign04_progress")
    module.ProgressTracker = StubTracker
    monkeypatch.setitem(sys.modules, "tensegra.campaign04_progress", module)
    import tensegra
    monkeypatch.setattr(tensegra, "campaign04_progress", module, raising=False)
    fn = ROLLOUT_MASKS["progress_v1"](2)
    aux = {}
    torch.manual_seed(4)
    results, _ = batched_on_policy(new_model(11), [factory(DEV + i) for i in range(2)], max_steps=8,
                                   action_mask_fn=fn, aux=aux)
    assert len(fed) == sum(len(r["trace"]) for r in results) - 2  # the last step of each env is never fed
    from tensegra.campaign02_world import Action
    for r in results:
        keys = [action_key(Action(s["action"]["kind"], s["action"]["arguments"])) for s in r["trace"]]
        for t, step in enumerate(r["trace"]):
            if step["action"]["kind"] not in ("verify", "abstain"):
                assert keys[t] not in keys[:t] or step.get("masked_actions", 0) == 0
    assert aux["masked_decisions"] > 0


# ---------------------------------------------------------------------------- population flow

def test_population_flows_a2_options_teacher_and_dual(tmp_path):
    from tensegra.campaign02_population import PopulationConfig, PopulationRun
    torch.set_num_threads(1)
    bank = save_checkpoint(tmp_path, 99, "bank.pt")
    train_cfg = {"width": 8, "family": "lightweight", "device": "cpu", "batch_size": 2, "learning_rate": 3e-4,
                 "rollout_mode": "batched", "max_steps": 10, "evaluation_batch": 4,
                 "policy_loss_reduction": "decision_mean", "advantage_normalization": True,
                 "rehearsal_weight": 0.5, "entropy_target": 1.0, "entropy_dual_lr": 0.05, "critic_warmup_updates": 1}
    raw = {"mode": "single", "population_seed": 7, "initialization_seeds": [70, 71, 72, 73, 74, 75],
           "optimizer_policy": "inherit", "teacher": "dep_reuse", "threads": 1, "world_family": "depworld",
           "policy": {"interface": "legacy", "feature_version": "d1"}, "world_mix": [WORLD], "train": train_cfg,
           "development_seed_start": DEV + 900_000, "development_examples": 2, "training_seed_stride": 1000,
           "member_hyperparameters": [{"learning_rate": 3e-5, "entropy_weight": 0.003, "kl_weight": 0.3}] * 6,
           "rounds": 1, "updates_per_slot": 1, "methods": ["actor_critic"], "training_seed_start": DEV + 500_000,
           "address_namespace": "a2-test", "initial_checkpoints": [bank] * 6}
    run = PopulationRun(PopulationConfig.from_json(json.loads(json.dumps(raw))), tmp_path / "run", factory, teacher)
    run.run()
    assert run.state["a2"]["options"]["rehearsal_weight"] == 0.5 and run.state["a2"]["rehearsal_teacher"] == "dep_reuse"
    rows = [r["training_timing"]["last"]["a2"] for r in run.state["allocations"]]
    assert rows[0]["critic_warmup"] is True and all(not r["critic_warmup"] for r in rows[1:])
    for previous, current in zip(rows, rows[1:]):  # dual carried through checkpoints between slots
        assert current["entropy_dual_used"] == previous["entropy_dual_next"]
    assert all(r["rehearsal_decisions"] > 0 for r in rows)


def test_population_refuses_an_unbuildable_mask_rule_before_training(tmp_path, monkeypatch):
    from tensegra.campaign02_population import PopulationConfig, PopulationRun
    monkeypatch.setitem(sys.modules, "tensegra.campaign04_progress", None)
    raw = {"mode": "single", "population_seed": 7, "initialization_seeds": [70, 71, 72, 73, 74, 75],
           "teacher": "dep_reuse", "world_family": "depworld", "policy": {"interface": "legacy", "feature_version": "d1"},
           "world_mix": [WORLD], "train": {"width": 8, "device": "cpu", "batch_size": 2, "rollout_mode": "batched",
                                           "max_steps": 10, "rollout_mask": "progress_v1"},
           "development_seed_start": DEV + 900_000, "development_examples": 2, "training_seed_stride": 1000,
           "rounds": 1, "updates_per_slot": 1, "methods": ["actor_critic"], "training_seed_start": DEV + 500_000}
    with pytest.raises(RuntimeError, match="campaign04_progress"):
        PopulationRun(PopulationConfig.from_json(raw), tmp_path / "run", factory, teacher)
    assert not (tmp_path / "run" / "checkpoints").exists()


# ---------------------------------------------------------------------------- configs

def test_a2_config_generator_changes_only_the_intended_fields(tmp_path):
    import importlib.util
    from pathlib import Path
    from tensegra.campaign02_population import PopulationConfig
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("a2cfg", root / "research/tools/campaign04_a2_configs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.write(root / "configs/campaign03", tmp_path, 0.8123, None)
    expected = {"a2-reh": ("c1", {"rehearsal_weight"}), "a2-imit": ("c1", {"rehearsal_weight", "rl_loss_weight"}),
                "a2-crit": ("c1", {"critic_warmup_updates", "critic_warmup_shared"}),
                "a2-dep": ("c1", {"rollout_mask"}),
                "a2-ent": ("c0", {"entropy_target", "entropy_dual_lr", "entropy_dual_init", "entropy_dual_max"})}
    for arm, (base, fields) in expected.items():
        row = manifest["arms"][arm]
        assert row["base"] == f"p2a-{base}-x1-r2.json"
        assert set(row["difference_from_base"]) == {f"train.{f}" for f in fields}
        config = json.loads((tmp_path / f"{arm}-x1-r2.json").read_text())
        base_config = json.loads((root / "configs/campaign03" / row["base"]).read_text())
        assert {k: v for k, v in config.items() if k != "train"} == {k: v for k, v in base_config.items() if k != "train"}
        cfg = PopulationConfig.from_json(config)
        TrainConfig(**cfg.train)
        assert cfg.rounds * 6 * cfg.updates_per_slot == 1800
        assert ("anchor_checkpoint" in config["train"]) == (base == "c1")
    ent = json.loads((tmp_path / "a2-ent-x1-r2.json").read_text())["train"]
    assert ent["entropy_target"] == 0.8123 and "anchor_kl_weight" not in ent
    with pytest.raises(ValueError):
        module.build(root / "configs/campaign03", None)
    # The committed configs are this generator's output for the committed measurement.
    committed = json.loads((root / "configs/campaign04/a2-manifest-x1-r2.json").read_text())
    target = round(committed["entropy_measurement"]["entropy_decision_mean"], 4)
    regenerated = module.build(root / "configs/campaign03", target)
    for arm, (_, config, _) in regenerated.items():
        assert json.loads((root / "configs/campaign04" / f"{arm}-x1-r2.json").read_text()) == config
