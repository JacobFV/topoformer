"""extended-04 Track A2 improvement-screen configs (lineage X1-r2; design.md §2 A2 + v2 revisions 2-4).

Each arm is ONE declared change to a declared base, and is otherwise the P2a base byte for byte: the same seeds,
training streams, bank, 1,800-update horizon (5 rounds x 6 slots x 60), development worlds and namespace.

| arm     | base | change (train block only)                                                    |
|---------|------|------------------------------------------------------------------------------|
| a2-reh  | C1   | rehearsal_weight = 0.5 (public teacher = the config's `teacher`, dep_reuse)  |
| a2-imit | C1   | rehearsal_weight = 0.5, rl_loss_weight = 0 (matched imitation-only control)   |
| a2-crit | C1   | critic_warmup_updates = 300 (value head only; critic_warmup_shared = false)  |
| a2-dep  | C1   | rollout_mask = "progress_v1" (training samples from the masked policy)       |
| a2-ent  | C0   | entropy_target = H_boot, entropy_dual_lr = 0.02 (no anchor)                  |

C0 = configs/campaign03/p2a-c0-x1-r2.json (the exact P1 recipe); C1 = C0 + the fixed bootstrap anchor
(p2a-c1-x1-r2.json; regenerated here from C0 and checked). Every emitted config is checked to differ from
its base only in the intended train fields.

Registered choices:
- **Rehearsal weight, lambda = 0.5.** The CE to the teacher's (deterministic) action is a KL to a
  point-mass teacher policy. lambda = 0.5 puts it on the same order as the anchor it sits beside (0.3 x
  KL to the bootstrap), slightly stronger because it is the arm's mechanism. The RL term (unit-scale
  normalized advantages) still dominates wherever the policy already agrees with the teacher. The bootstrap
  imitated this teacher, so its initial CE is small. One value only: this is a screen, not a sweep.
- **Imitation-only control.** The same lambda, the same visited-state source (sampled on-policy rollouts)
  and the same optimizer exposure. It keeps the entropy bonus, the tranche KL and the anchor KL; only the
  RL policy-gradient and critic losses are off.
- **Critic warm-up, W = 300 updates.** That is 1/6 of the run (5 slots). It counts within the 1,800
  updates, so optimizer exposure and worlds are matched to C1. The policy is bit-identical during warm-up.
- **Entropy target = H_boot.** This is the bootstrap's measured decision-mean entropy on its own sampled
  rollouts over the training world_mix (research/tools/campaign04_bootstrap_entropy.py). It is two-sided:
  lambda moves up when H < H*, and down (penalizing entropy) when H > H*. The step size is eta = 0.02 per
  nat of error per update, with |lambda| <= 1. C0's entropy drift (~.8 -> 1.6 nats over 1,800 updates) is
  slow, so a 0.2-nat gap moves lambda by 0.004 per update: it reaches an entropy coefficient of ~0.1
  (about 30x C0's 0.003) within ~25 updates, without per-batch noise (sd ~0.1 nat) dominating. lambda is
  added to C0's fixed entropy_weight (0.003) and starts at 0.

    python research/tools/campaign04_a2_configs.py --entropy-measurement <results>/a2-bootstrap-entropy-x1-r2.json
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P2A = _load("campaign03_p2a_configs")
LINEAGE = P2A.LINEAGE  # "x1-r2"
REHEARSAL_WEIGHT = 0.5
CRITIC_WARMUP_UPDATES = 300
ENTROPY_DUAL_LR = 0.02
ENTROPY_DUAL_MAX = 1.0
ROLLOUT_MASK = "progress_v1"
HORIZON = 1800

ARMS = {  # arm -> (base, intended train changes; entropy target filled from the measurement)
    "a2-reh": ("c1", {"rehearsal_weight": REHEARSAL_WEIGHT}),
    "a2-imit": ("c1", {"rehearsal_weight": REHEARSAL_WEIGHT, "rl_loss_weight": 0.0}),
    "a2-crit": ("c1", {"critic_warmup_updates": CRITIC_WARMUP_UPDATES, "critic_warmup_shared": False}),
    "a2-dep": ("c1", {"rollout_mask": ROLLOUT_MASK}),
    "a2-ent": ("c0", {"entropy_target": None, "entropy_dual_lr": ENTROPY_DUAL_LR,
                      "entropy_dual_init": 0.0, "entropy_dual_max": ENTROPY_DUAL_MAX}),
}


def bases(configs: Path):
    """C0 and C1 exactly as registered in P2a (C1 regenerated from C0 + banks and compared)."""
    banks = json.loads((configs / "p1-banks.json").read_text())
    c0_text = (configs / f"p2a-c0-{LINEAGE}.json").read_text()
    c0 = P2A.c0_config(c0_text, banks)
    c1 = json.loads((configs / f"p2a-c1-{LINEAGE}.json").read_text())
    if c1 != P2A.c1_config(c0, banks):
        raise ValueError("p2a-c1 differs from C0 + the registered anchor")
    if c0["rounds"] * 6 * c0["updates_per_slot"] != HORIZON or c0["mode"] != "single":
        raise ValueError("Unexpected base horizon")
    return {"c0": c0, "c1": c1}, banks


def difference(base: dict, arm: dict) -> dict:
    return P2A.arm_difference(base, arm)


def build(configs: Path, entropy_target: float):
    if not (entropy_target is not None and 0 < entropy_target < 10):
        raise ValueError("A measured bootstrap entropy target (nats) is required")
    base_configs, banks = bases(configs)
    out = {}
    for arm, (base_name, changes) in ARMS.items():
        changes = dict(changes)
        if "entropy_target" in changes:
            changes["entropy_target"] = entropy_target
        config = json.loads(json.dumps(base_configs[base_name]))
        config["train"] = {**config["train"], **changes}
        diff = difference(base_configs[base_name], config)
        if set(changes) & set(base_configs[base_name]["train"]) or set(changes) - set(_DEFAULTS):
            raise ValueError(f"{arm}: changes must be new A2 train fields")
        if set(diff) != {f"train.{k}" for k in changes}:
            raise ValueError(f"{arm} differs from {base_name} beyond its change: {sorted(diff)}")
        _validate(config)
        out[arm] = (base_name, config, diff)
    return out


_DEFAULTS = {"rehearsal_weight": 0.0, "rl_loss_weight": 1.0, "entropy_target": None, "entropy_dual_lr": 0.0,
             "entropy_dual_init": 0.0, "entropy_dual_max": 1.0, "critic_warmup_updates": 0,
             "critic_warmup_shared": False, "rollout_mask": None}


def _validate(config):
    """Construct the population config when torch is available (remote); JSON-level checks otherwise."""
    try:
        from tensegra.campaign02_population import PopulationConfig
        from tensegra.campaign02_training import A2_FIELD_DEFAULTS
    except ImportError:
        return
    if A2_FIELD_DEFAULTS != _DEFAULTS:
        raise ValueError("A2 field defaults drifted from the generator's copy")
    PopulationConfig.from_json(config)


def write(configs: Path, output: Path, entropy_target: float, measurement: dict | None):
    arms = build(configs, entropy_target)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"lineage": LINEAGE, "horizon_updates": HORIZON, "entropy_measurement": measurement, "arms": {}}
    for arm, (base_name, config, diff) in arms.items():
        path = output / f"{arm}-{LINEAGE}.json"
        path.write_text(json.dumps(config, indent=2))
        shown = path.resolve().relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path
        manifest["arms"][arm] = {"path": str(shown), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "base": f"p2a-{base_name}-{LINEAGE}.json",
                                 "difference_from_base": {k: {"base": v[0], "arm": v[1]} for k, v in sorted(diff.items())}}
    (output / f"a2-manifest-{LINEAGE}.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", type=Path, default=ROOT / "configs/campaign03")
    p.add_argument("--output", type=Path, default=ROOT / "configs/campaign04")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--entropy-measurement", type=Path, help="campaign04_bootstrap_entropy.py output")
    g.add_argument("--entropy-target", type=float)
    a = p.parse_args()
    measurement = None
    target = a.entropy_target
    if a.entropy_measurement:
        measurement = json.loads(a.entropy_measurement.read_text())
        banks = json.loads((a.configs / "p1-banks.json").read_text())
        if measurement["checkpoint"] != banks[LINEAGE]:
            raise SystemExit("The entropy measurement is not of the X1-r2 bootstrap bank checkpoint")
        target = round(measurement["entropy_decision_mean"], 4)
        measurement = {k: measurement[k] for k in ("checkpoint", "entropy_decision_mean", "entropy_decision_sd",
                                                   "decisions", "episodes", "seed_range", "sampling_seed")}
    manifest = write(a.configs, a.output, target, measurement)
    print(json.dumps({arm: v["difference_from_base"] for arm, v in manifest["arms"].items()}, indent=1))


if __name__ == "__main__":
    main()
