"""extended-04 Phase A: throughput benchmark / profiler for depworld training and evaluation.

Runs short P1/P2a-style workloads in-process (CPU by default) and reports process CPU per phase:
  tranche     actor-critic batched tranche (C0 recipe: KL .3 to tranche start, advantage
              normalization, lr 3e-5, entropy .003, batch 8, max_steps 96) + a development evaluation
  supervised  supervised bootstrap tranche with the dep_reuse teacher (collect_teacher path)
  evaluate    sealed-style evaluation: batched greedy + sampled episodes on P1AuditedDepWorkshop,
              rows with episode_counts, gzip JSONL export (as research/tools/campaign02_evaluate.py)

--path reference|fast selects the implementation (tensegra.campaign04_fast); --profile writes cProfile
stats. Usage: PYTHONPATH=src python research/tools/campaign04_throughput_bench.py tranche --updates 4 ...
"""
from __future__ import annotations

import argparse
import cProfile
from functools import partial
import gzip
import io
import json
from pathlib import Path
import pstats
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

WORLD_MIX = [dict(categories=3, choices=3, locations=7, slots=6, compute_price=1e-4, event_trigger="progress",
                  p_event=.5, foreign_records=f, event_kinds=["edge_closed", "capacity_reduced", "slot_closed"])
             for f in (0, 2)]


def build(args):
    import torch
    from tensegra.campaign02_population import build_policy, mix_index, new_policy
    from tensegra.campaign02_protocol import execute
    from tensegra.campaign02_training import Learner, TrainConfig, independent_address_seed, public_frame
    from tensegra.campaign03_depworld import DepWorkshop, depworld_executor, generate_depworld
    executor = partial(depworld_executor, execute_call=execute)

    def factory(seed):
        return DepWorkshop(generate_depworld(seed, **WORLD_MIX[mix_index(seed, 2)]), executor=executor,
                           address_seed=independent_address_seed(seed, "e04-throughput-bench"))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = build_policy(saved["policy_config"])
        model.load_state_dict(saved["model"])
    else:
        _, obs, cand = public_frame(factory(1).observe(), args.feature_version)
        model = new_policy({"interface": "legacy", "feature_version": args.feature_version}, len(obs), len(cand[0]),
                           args.width, "lightweight")
    method = "supervised" if args.mode == "supervised" else "actor_critic"
    extra = {} if method == "supervised" else dict(kl_weight=.3, advantage_normalization=True,
                                                    learning_rate=3e-5, entropy_weight=.003)
    cfg = TrainConfig(seed=args.seed, width=model.config.width, batch_size=8, method=method, device=args.device,
                      max_steps=96, rollout_mode="batched", training_seed_start=3_440_000_000, **extra)
    return Learner(model, cfg), factory


def run_tranche(args):
    from tensegra.campaign02_references import make_reference
    learner, factory = build(args)
    out = {}
    cpu = time.process_time()
    timing = learner.train_tranche(args.updates, factory, (lambda: make_reference("dep_reuse"))
                                   if args.mode == "supervised" else None)
    out["train_cpu"] = time.process_time() - cpu
    out["phase_timing"] = {k: round(v["process_cpu_seconds"], 3) for k, v in timing.get("phase_timing", {}).items()}
    if args.dev_examples:
        cpu = time.process_time()
        out["dev"] = learner.evaluate(range(3_902_000_000, 3_902_000_000 + args.dev_examples), factory,
                                      Path(args.scratch) / "dev.jsonl.gz")
        out["dev_cpu"] = time.process_time() - cpu
    out["data_hash_scope"] = "includes wall timings; not comparable across runs"
    return out


def run_evaluate(args):
    import torch
    from campaign02_evaluate import episode_counts, summarize, write_rows
    from tensegra.campaign02_protocol import execute
    from tensegra.campaign02_training import batched_episodes, independent_address_seed, sampling_rng_seed
    from tensegra.campaign03_depworld import depworld_executor, generate_depworld
    from tensegra.campaign03_p1_audit import P1AuditedDepWorkshop
    learner, _ = build(args)
    policy = learner.model.eval()
    executor = partial(depworld_executor, execute_call=execute)
    seeds = list(range(120_100_000, 120_100_000 + args.eval_examples))
    specs = [generate_depworld(s, **WORLD_MIX[1]) for s in seeds]
    out = {}
    for mode in ("greedy", "sampled"):
        cpu = time.process_time()
        rows = []
        with torch.no_grad():
            for offset in range(0, len(seeds), 32):
                idx = list(range(offset, min(len(seeds), offset + 32)))
                samplers = None if mode == "greedy" else [random.Random(sampling_rng_seed(seeds[i], 20260926)) for i in idx]
                envs = [P1AuditedDepWorkshop(specs[i], executor=executor,
                                             address_seed=independent_address_seed(seeds[i], "e04-bench")) for i in idx]
                for i, r in zip(idx, batched_episodes(policy, envs, device=args.device, max_steps=96,
                                                      **({} if samplers is None else {"samplers": samplers}))):
                    rows.append({"seed": seeds[i], **r, "counts": episode_counts(r["outcome"])})
        rollout = time.process_time() - cpu
        summary = summarize(rows)
        write_rows(Path(args.scratch) / f"eval-{mode}.jsonl.gz", rows)
        out[mode] = {"rollout_cpu": rollout, "total_cpu": time.process_time() - cpu, "success": summary["success"],
                     "utility": summary["means"]["utility"]}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("tranche", "supervised", "evaluate"))
    p.add_argument("--path", choices=("reference", "fast"), default=None)
    p.add_argument("--updates", type=int, default=4)
    p.add_argument("--dev-examples", type=int, default=32)
    p.add_argument("--eval-examples", type=int, default=32)
    p.add_argument("--checkpoint")
    p.add_argument("--feature-version", default="d1")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--profile", type=int, default=0, help="print top-N cProfile rows (tottime and cumulative)")
    p.add_argument("--scratch", default="/tmp/e04-bench")
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    Path(args.scratch).mkdir(parents=True, exist_ok=True)
    if args.path is not None:
        from tensegra import campaign04_fast
        campaign04_fast.set_default(args.path == "fast")
    fn = run_evaluate if args.mode == "evaluate" else run_tranche
    wall, cpu = time.perf_counter(), time.process_time()
    if args.profile:
        prof = cProfile.Profile()
        result = prof.runcall(fn, args)
        for key in ("tottime", "cumulative"):
            s = io.StringIO()
            pstats.Stats(prof, stream=s).sort_stats(key).print_stats(args.profile)
            print(s.getvalue())
    else:
        result = fn(args)
    result.update(mode=args.mode, path=args.path or "default", wall=time.perf_counter() - wall,
                  cpu=time.process_time() - cpu)
    print("RESULT " + json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
