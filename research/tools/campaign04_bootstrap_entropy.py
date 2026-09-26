"""extended-04 A2-ent: measure a checkpoint's mean per-decision policy entropy on its own rollouts.

The measurement uses the exact training-rollout distribution of the A2 RL runs:
- sampled (on-policy) batched rollouts: ``batched_on_policy(sample=True)``, the same code path as training;
- worlds from the run config's training ``world_mix``, selected per seed by the population's ``mix_index``;
- the run's ``max_steps`` and neural tariff.

H is the entropy of the sampled distribution over valid candidates at each visited decision, averaged over
decisions. This is the same statistic that the A2-ent Lagrangian measures during training (``entropy_measured``).

The seeds are a fresh measurement range (default 3,970,000,000+), disjoint from the P1/P2a training, development
and screening ranges, with their own address namespace. The run also preflights the rehearsal teacher (design
review F2): the configured public reference is queried on every visited state. The run records whether its proposal
is in the actor's catalog, and whether it agrees with the policy's greedy choice. No gradient, no training.

    python research/tools/campaign04_bootstrap_entropy.py --config configs/campaign03/p2a-c0-x1-r2.json \\
        --output <results>/a2-bootstrap-entropy-x1-r2.json [--examples 256]
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time

DEFAULT_SEED_START = 3_970_000_000
NAMESPACE = "e04a2-entropy"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def summarize(entropies_by_episode, outcomes, aux_rows):
    flat = [h for episode in entropies_by_episode for h in episode]
    n = len(flat)
    mean = sum(flat) / n
    per_episode = [sum(e) / len(e) for e in entropies_by_episode if e]
    decisions = sum(a["rehearsal_decisions"] + a["rehearsal_out_of_catalog"] for a in aux_rows)
    return {"decisions": n, "episodes": len(entropies_by_episode),
            "entropy_decision_mean": mean,
            "entropy_decision_sd": (sum((h - mean) ** 2 for h in flat) / n) ** .5,
            "entropy_episode_mean_of_means": sum(per_episode) / len(per_episode),
            "sampled_success": sum(o["verified_success"] for o in outcomes) / len(outcomes),
            "sampled_utility": sum(o["utility"] for o in outcomes) / len(outcomes),
            "teacher_preflight": {
                "decisions": decisions,
                "out_of_catalog": sum(a["rehearsal_out_of_catalog"] for a in aux_rows),
                "greedy_agreement_rate": sum(a["rehearsal_agreement"] for a in aux_rows) / max(1, decisions)}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="the RL run config (world_mix, bank, train block)")
    p.add_argument("--checkpoint", type=Path, help="default: the config's initial bank checkpoint (verified)")
    p.add_argument("--sha256", help="required with --checkpoint")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--examples", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    p.add_argument("--sampling-seed", type=int, default=20260926)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()

    import torch
    from tensegra.campaign02_population import build_policy, mix_index
    from tensegra.campaign02_protocol import BoundedSolver
    from tensegra.campaign02_references import make_reference
    from tensegra.campaign02_training import (TrainConfig, batched_on_policy, independent_address_seed,
                                              normalized_policy_config)
    from tensegra.campaign03_depworld import DepWorkshop, depworld_executor, generate_depworld

    config = json.loads(a.config.read_text())
    if config.get("world_family") != "depworld":
        raise SystemExit("depworld configs only")
    if a.checkpoint is None:
        banks = {json.dumps(c, sort_keys=True) for c in config["initial_checkpoints"]}
        if len(banks) != 1:
            raise SystemExit("config has several bank checkpoints; pass --checkpoint/--sha256")
        source = json.loads(banks.pop())
    else:
        source = {"path": str(a.checkpoint), "sha256": a.sha256}
    if sha256(source["path"]) != source["sha256"]:
        raise SystemExit(f"checkpoint hash mismatch: {source['path']}")
    train = TrainConfig(**config["train"])
    torch.set_num_threads(a.threads)
    saved = torch.load(source["path"], map_location="cpu", weights_only=False)
    model = build_policy(normalized_policy_config(saved["policy_config"]))
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    mix = config["world_mix"]
    seeds = list(range(a.seed_start, a.seed_start + a.examples))
    cpu0, wall0 = time.process_time(), time.perf_counter()
    entropies, outcomes, aux_rows = [], [], []
    with BoundedSolver() as solver:
        executor = partial(depworld_executor, execute_call=solver.execute)

        def factory(seed):
            return DepWorkshop(generate_depworld(seed, **mix[mix_index(seed, len(mix))]), executor=executor,
                               address_seed=independent_address_seed(seed, NAMESPACE))
        for offset in range(0, len(seeds), a.batch):
            chunk = seeds[offset:offset + a.batch]
            torch.manual_seed(a.sampling_seed + offset)
            aux = {}
            with torch.no_grad():
                results, terms = batched_on_policy(model, [factory(s) for s in chunk], device="cpu",
                    max_steps=train.max_steps, neural_work_per_forward=train.neural_work_per_forward,
                    bptt_steps=train.bptt_steps, sample=True,
                    teachers=[make_reference(config["teacher"]) for _ in chunk], aux=aux)
            aux.pop("rehearsal_ce")
            aux_rows.append(aux)
            entropies.extend([float(t[2]) for t in episode] for episode in terms)
            outcomes.extend(r["outcome"] for r in results)
    result = {"measurement": "mean per-decision entropy (nats) of the sampled policy over valid candidates, "
                             "on its own sampled batched rollouts over the run's training world_mix",
              "checkpoint": source, "config": str(a.config), "config_sha256": sha256(a.config),
              "seed_range": [seeds[0], seeds[-1] + 1], "address_namespace": NAMESPACE,
              "sampling_seed": a.sampling_seed, "batch": a.batch, "max_steps": train.max_steps,
              "teacher": config["teacher"], **summarize(entropies, outcomes, aux_rows),
              "process_cpu_seconds": time.process_time() - cpu0, "wall_seconds": time.perf_counter() - wall0,
              "solver_child_note": "solver worker CPU is outside process_cpu_seconds (metered by the outer runner)"}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("entropy_decision_mean", "decisions", "sampled_success",
                                            "teacher_preflight")}))


if __name__ == "__main__":
    main()
