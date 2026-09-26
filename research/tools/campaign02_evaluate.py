"""Frozen public-controller evaluation on paired fresh workshop worlds.

No training, teacher forcing, checkpoint selection, or hidden-state actor input.
The configuration lists checkpoint hashes and explicit world-seed conditions.
Reference schedules are labeled supplied controls, never learned policies.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from functools import partial
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time



def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_rows(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True)+"\n")


def episode_counts(outcome):
    """Reconstruct observable return-address contracts, not gold optimal choices."""
    counts = Counter()
    problem_types, returns, retrieved = {}, {}, {}
    for event in outcome["history"]:
        action, feedback = event["action"], event["feedback"]
        kind, args = action["kind"], action.get("arguments") or {}
        counts[f"action/{kind}"] += 1
        counts[f"status/{feedback.get('status', 'missing')}"] += 1
        if kind in {"start_subset", "build_route", "start_assign"} and feedback.get("status") == "success":
            problem_types[args.get("handle")] = {"start_subset": "constrained_subset", "build_route": "shortest_path",
                                             "start_assign": "csp"}[kind]
        if kind == "call" and "return" in feedback:
            returns[feedback["return"]] = {"primitive": problem_types.get(args.get("problem")),
                                          "version": event.get("state_version", event.get("stage"))}
        if kind == "retrieve" and "record" in feedback:
            record = feedback["record"]
            retrieved[args.get("handle")] = record
        if kind == "use_return":
            counts["return_use_attempts"] += 1
            multiple = len(returns) >= 2
            counts["multiple_return_use_attempts"] += int(multiple)
            record = retrieved.get(args.get("handle"), {})
            expected = {"subset": "constrained_subset", "select": "constrained_subset", "route": "shortest_path",
                        "assign": "csp"}.get(args.get("as"))
            # Route execution itself may discover an obstacle and advance version.
            stale = feedback.get("reason") in ("stale_result", "stale_dependency")
            valid = (bool(record) and record.get("primitive") == expected and not stale
                     and record.get("certificate_valid", False)
                     and record.get("status") in {"success", "timeout"})
            counts["return_address_contract_valid"] += int(valid)
            counts["multiple_return_address_contract_valid"] += int(valid and multiple)
            counts["return_application_success"] += int(feedback.get("status") == "success")
            counts["stale_return_use"] += int(stale)
    if "reuse_audit" in outcome:  # depworld-v1: evaluator-side applicability audit of every use
        for event in outcome["history"]:
            reason = event["feedback"].get("reason")
            if event["feedback"].get("status") in ("rejected", "incomplete") and reason:
                counts[f"reason/{reason}"] += 1
        for use in outcome["reuse_audit"]:
            counts["dep_uses"] += 1
            counts["dep_applicable_uses"] += int(use["applicable_hidden"])
            counts["dep_invalid_uses"] += int(not use["applicable_hidden"])
            counts["dep_foreign_uses"] += int(use["foreign"])
            counts["dep_foreign_applicable_uses"] += int(use["foreign"] and use["applicable_hidden"])
        counts["dep_revisions"] = sum(outcome.get("revisions", {}).values())
        counts["dep_revocations"] = len(outcome.get("revocations", []))
        counts["dep_calls"] = outcome.get("calls", 0)
    reductions = outcome.get("reductions", [])
    counts["reductions"] = len(reductions)
    counts["correct_reductions"] = sum(bool(row["correct"]) for row in reductions)
    counts["certificate_valid_reductions"] = sum(bool(row["certificate_valid"]) for row in reductions)
    audits = outcome.get("return_fault_audit", [])
    counts["fault_target_groups"] = len(audits)
    counts["fault_applied_groups"] = sum(r.get("status")=="applied" for r in audits)
    changed = {h for r in audits for h,c in zip(r["handles"],r.get("changed",[])) if c}
    targets = {h for r in audits for h in r["handles"]}
    counts["fault_changed_records"] = len(changed)
    for event in outcome["history"]:
        action,feedback = event["action"],event["feedback"]
        handle = action["arguments"].get("handle")
        if action["kind"]=="retrieve" and handle in changed:
            counts["fault_changed_retrieve_attempts"] += 1
            counts["fault_changed_retrieval_success"] += int("record" in feedback)
        if action["kind"]=="use_return":
            counts["fault_target_use_attempts"] += int(handle in targets)
            counts["fault_changed_use_attempts"] += int(handle in changed)
            if handle in changed:
                counts["fault_changed_application_success"] += int(feedback.get("status")=="success")
                counts["fault_changed_application_rejected"] += int(feedback.get("status") in {"rejected","invalid_input"})
    return dict(counts)


def summarize(rows):
    count = len(rows)
    successes = sum(bool(row["outcome"]["verified_success"]) for row in rows)
    totals = Counter()
    for row in rows:
        totals.update(row["counts"])
    cost = sum(row["outcome"]["cost"] for row in rows)
    fields = ("utility", "cost", "steps", "observations", "work_units", "travel_distance",
              "compute_units", "modeled_compute_cost", "solver_cpu_seconds")
    causal_supports = {}
    for label,predicate in (
        ("target_available",lambda r:r["counts"].get("fault_target_groups",0)>0),
        ("fault_applied",lambda r:r["counts"].get("fault_applied_groups",0)>0),
        ("record_changed",lambda r:r["counts"].get("fault_changed_records",0)>0),
        ("changed_return_used",lambda r:r["counts"].get("fault_changed_use_attempts",0)>0)):
        subset = [r for r in rows if predicate(r)]
        successes_subset = sum(bool(r["outcome"]["verified_success"]) for r in subset)
        causal_supports[label] = {"support":len(subset),"successes":successes_subset,
            "success":successes_subset/len(subset) if subset else None}
    return {"examples": count,"intervention_supports":causal_supports, "success_count": successes, "success": successes/count,
            "means": {field: sum(row["outcome"].get(field, 0) for row in rows)/count for field in fields},
            "cost_per_success_including_failures": cost/successes if successes else None,
            "counts": dict(totals), "truncated": sum(bool(row.get("truncated", False)) for row in rows),
            "neural_forward_wall_seconds": sum(sum(step.get("neural_forward_wall_seconds_allocated", 0)
                                                       for step in row.get("trace", [])) for row in rows)}


def condition_faults(condition):
    """Normalize only explicit config descriptors; never infer fault targets."""
    from topoformer.campaign02_interventions import ReturnFault
    if "fault" in condition and "return_faults" in condition:
        raise ValueError("Use either fault or return_faults, not both")
    descriptors = condition.get("return_faults",[condition["fault"]] if "fault" in condition else [])
    if not isinstance(descriptors,list):
        raise ValueError("return_faults must be a list")
    normalized = []
    for descriptor in descriptors:
        values = dict(descriptor)
        if "payload_path" in values:
            values["payload_path"] = tuple(values["payload_path"])
        normalized.append(ReturnFault(**values))
    return tuple(normalized)


def world_kwargs(world):
    """JSON has no tuples: restore the tuple-typed public call-budget menu."""
    world = dict(world)
    if "call_budgets" in world:
        world["call_budgets"] = tuple(world["call_budgets"])
    return world


def semantic_spec_hash(spec):
    # Public resource interventions may differ; physical facts/goals may not.
    resources = {"step_limit","work_limit","observation_price","action_price","work_price",
                 "travel_price","travel_limit","compute_price","call_budgets","include_remaining_budget"}
    return canonical_hash({k:v for k,v in spec.items() if k not in resources})


def paired_condition_outcomes(control, intervention):
    """Paired total-effect summaries, plus descriptive post-treatment supports."""
    if len(control)!=len(intervention) or not control:
        raise ValueError("Paired conditions require equal nonempty support")
    support = len(control)
    control = {row["seed"]:row for row in control}
    intervention = {row["seed"]:row for row in intervention}
    if len(control)!=support or len(intervention)!=support or set(control)!=set(intervention):
        raise ValueError("Paired condition seeds differ or duplicate")
    transitions = Counter()
    changed = used = correct_then_wrong_changed = 0
    utility = 0.0
    for seed,a in control.items():
        b = intervention[seed]
        if a["semantic_spec_hash"]!=b["semantic_spec_hash"]:
            raise ValueError("Paired conditions change physical semantics")
        before,after = bool(a["outcome"]["verified_success"]),bool(b["outcome"]["verified_success"])
        transitions[f"{int(before)}->{int(after)}"] += 1
        has_change = b["counts"].get("fault_changed_records",0)>0
        changed += has_change
        used += b["counts"].get("fault_changed_use_attempts",0)>0
        correct_then_wrong_changed += bool(before and not after and has_change)
        utility += b["outcome"]["utility"]-a["outcome"]["utility"]
    return {"support":len(control),"success_transitions_control_to_intervention":dict(transitions),
            "mean_utility_difference":utility/len(control),"changed_record_support":changed,
            "changed_return_used_support":used,"control_success_to_failure_with_changed_record":correct_then_wrong_changed,
            "scope":"paired task outcomes; changed/used subsets are descriptive post-treatment supports, not independent causal estimates"}


def light_rows(rows):
    """What within-condition pairing reads; full rows are already on disk."""
    return [{"seed": row["seed"], "outcome": {"verified_success": row["outcome"]["verified_success"],
                                              "utility": row["outcome"]["utility"]}} for row in rows]


def read_rows(path):
    with gzip.open(path,"rt") as stream:
        return [json.loads(line) for line in stream]


def main():
    import torch
    from topoformer.campaign02_policy import CandidatePolicy, PolicyConfig
    from topoformer.campaign02_protocol import BoundedSolver
    from topoformer.campaign02_references import make_reference, run_episode
    from topoformer.campaign02_population import build_policy
    from topoformer.campaign02_training import TrainConfig, batched_episodes, independent_address_seed, sampling_rng_seed
    import random
    from topoformer.campaign02_world import Workshop, generate_world, protocol_executor
    from topoformer.campaign02_interventions import FaultedWorkshop

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--limit", type=int, help="Explicit profile-only episode cap; not full protocol")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Profile cap must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty evaluation directory")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(args.threads)
    sources = {"evaluator": file_hash(__file__)}
    import topoformer.campaign02_training as training
    for name in ("training", "policy", "protocol", "world", "references", "interventions", "population", "memory", "memory_policy"):
        sources[name] = file_hash(Path(training.__file__).with_name(f"campaign02_{name}.py"))
    if any(c.get("world_family", cfg.get("world_family")) == "depworld" for c in cfg["conditions"]):
        sources["depworld"] = file_hash(Path(training.__file__).with_name("campaign03_depworld.py"))
        # Evaluator-only P1 step audit (logging only; dynamics unchanged).
        sources["p1_audit"] = file_hash(Path(training.__file__).with_name("campaign03_p1_audit.py"))
        # extended-04 result-identical fast path (TENSEGRA_REFERENCE_PATH=1 forces the reference path).
        sources["fast"] = file_hash(Path(training.__file__).with_name("campaign04_fast.py"))
    bindings = []
    for binding in cfg["checkpoints"]:
        actual = file_hash(binding["path"])
        if actual != binding["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {binding['name']}")
        bindings.append({**binding, "sha256": actual})
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    summary = {"config": cfg, "config_sha256": file_hash(args.config), "sources": sources,
               "checkpoints": bindings, "profile_limit": args.limit, "results": [], "paired": [], "paired_conditions": [],
               "policy_input_scope": "public observation/candidate encodings only; no teacher at evaluation",
               "address_metric_scope": "claimed role/status/retrieval/provenance validity, not independent validity of faulted payloads or task relevance",
               "intervention_scope": "explicit public-ordinal post-validation return faults; canonical records and actual-world validators unchanged",
               "selection_rule": "frozen supplied checkpoint, no selection", "no_training": True,
               "throughput_path": training._fast.describe()}
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    with BoundedSolver() as solver:
        executor = partial(protocol_executor, execute_call=solver.execute)
        for condition in cfg["conditions"]:
            name = condition["name"]
            n = min(condition.get("examples", 512), args.limit or condition.get("examples", 512))
            if n < 1 or Path(name).name != name:
                raise ValueError("Invalid condition name or support")
            seeds = list(range(condition["seed_start"], condition["seed_start"]+n))
            family = condition.get("world_family", cfg.get("world_family", "workshop"))
            modular = family == "modular"
            depworld = family == "depworld"
            if depworld:
                from topoformer.campaign03_depworld import depworld_executor, generate_depworld
                # Same dynamics as DepWorkshop plus the evaluator-only P1 audit (outcome["p1_audit"]).
                from topoformer.campaign03_p1_audit import P1AuditedDepWorkshop as DepWorkshop
                condition_executor = partial(depworld_executor, execute_call=solver.execute)
                specs = [generate_depworld(seed, **world_kwargs(condition.get("world", {}))) for seed in seeds]
            elif modular:
                from topoformer.campaign02_modular import ModularWorkshop, generate_modular, modular_executor
                condition_executor = partial(modular_executor, execute_call=solver.execute)
                specs = [generate_modular(seed, **world_kwargs(condition.get("world", {}))) for seed in seeds]
            else:
                condition_executor = executor
                specs = [generate_world(seed, **world_kwargs(condition.get("world", {}))) for seed in seeds]
            namespace = condition.get("address_namespace", cfg.get("address_namespace", "extended-02-frozen-eval-v1"))
            # Learned-policy action choice: greedy (historical default) or fixed-seed sampled, one
            # sample per world from a stream seeded by (world seed, sampling_seed) only.
            policy_mode = condition.get("policy_mode", cfg.get("policy_mode", "greedy"))
            sampling_seed = condition.get("sampling_seed", cfg.get("sampling_seed"))
            if policy_mode not in ("greedy", "sampled"):
                raise ValueError(f"Unknown policy_mode {policy_mode!r}")
            if policy_mode == "sampled" and not isinstance(sampling_seed, int):
                raise ValueError("Sampled evaluation needs an explicit integer sampling_seed")
            faults = condition_faults(condition)
            def factory(index):
                kwargs = {"executor":condition_executor,"address_seed":independent_address_seed(seeds[index],namespace)}
                if modular or depworld:
                    if faults:
                        raise ValueError("Return faults are defined for workshop-v1 only")
                    return (DepWorkshop if depworld else ModularWorkshop)(specs[index],**kwargs)
                return FaultedWorkshop(specs[index],**kwargs,faults=faults) if faults else Workshop(specs[index],**kwargs)
            folder = args.output/name
            folder.mkdir()
            world_rows = [{"seed": seed, "spec": asdict(spec), "spec_hash": canonical_hash(asdict(spec)),
                           "address_seed": independent_address_seed(seed, namespace),
                           "semantic_spec_hash":semantic_spec_hash(asdict(spec))} for seed, spec in zip(seeds, specs)]
            write_rows(folder/"worlds.jsonl.gz", world_rows)
            by_arm = {}
            for binding in bindings:
                checkpoint = torch.load(binding["path"], map_location="cpu", weights_only=False)
                train_cfg = TrainConfig(**checkpoint["config"])
                policy = build_policy(checkpoint["policy_config"]).to(args.device)
                policy_cfg = policy.config
                policy.load_state_dict(checkpoint["model"], strict=True)
                policy.eval()
                rows = []
                batch_size = cfg.get("evaluation_batch", 32)
                if batch_size < 1:
                    raise ValueError("Positive evaluation batch required")
                with torch.no_grad():
                    for offset in range(0, n, batch_size):
                        indices = list(range(offset, min(n, offset+batch_size)))
                        samplers = None if policy_mode == "greedy" else [
                            random.Random(sampling_rng_seed(seeds[i], sampling_seed)) for i in indices]
                        outputs = batched_episodes(policy, [factory(i) for i in indices], device=args.device,
                            max_steps=train_cfg.max_steps, neural_work_per_forward=train_cfg.neural_work_per_forward,
                            **({} if samplers is None else {"samplers": samplers}))
                        for i, result in zip(indices, outputs):
                            rows.append({"seed": seeds[i], "spec_hash": world_rows[i]["spec_hash"], "semantic_spec_hash":world_rows[i]["semantic_spec_hash"], **result,
                                         "counts": episode_counts(result["outcome"])})
                            if samplers is not None:
                                rows[-1].update(policy_mode="sampled", sampling_seed=sampling_seed,
                                                sampling_rng_seed=sampling_rng_seed(seeds[i], sampling_seed))
                if not cfg.get("trace_observations", True):
                    # Opt-in size control: drop the per-step public observation copies from learned
                    # traces (the exact action/feedback history and the P1 audit remain in the row).
                    for row in rows:
                        for step in row.get("trace", []):
                            step.pop("observation", None)
                artifact = folder/f"{binding['name']}.jsonl.gz"
                write_rows(artifact, rows)
                summary["results"].append({"condition": name, "arm": binding["name"], "kind": "learned",
                    "checkpoint_binding": {k: v for k, v in binding.items() if k != "path"},
                    **summarize(rows), "artifact": str(artifact.relative_to(args.output)), "artifact_sha256": file_hash(artifact),
                    "policy_config": asdict(policy_cfg), "training_config": asdict(train_cfg),
                    "actual_parameter_count": sum(p.numel() for p in policy.parameters()),
                    "checkpoint_updates": checkpoint["updates"], "checkpoint_presentations": checkpoint["presentations"],
                    **({} if policy_mode == "greedy" else {"policy_mode": policy_mode, "sampling_seed": sampling_seed,
                        "sampling_rule": "one sample per world; random.Random(sampling_rng_seed(world seed, sampling_seed, 0)); "
                                         "one uniform per decision through float64 softmax (inverse CDF)"})})
                by_arm[binding["name"]] = light_rows(rows)
                del policy, checkpoint, rows
            for mode in condition.get("references", cfg.get("references", ["cheap", "always_tool", "cheap_first"])):
                rows = []
                for i, seed in enumerate(seeds):
                    outcome = run_episode(factory(i), make_reference(mode), cfg.get("reference_compute_tariff", 0.0))
                    outcome.pop("trace", None)  # Same information retained once in exact history.
                    rows.append({"seed": seed, "spec_hash": world_rows[i]["spec_hash"], "semantic_spec_hash":world_rows[i]["semantic_spec_hash"], "outcome": outcome,
                                 "counts": episode_counts(outcome), "truncated": False})
                artifact = folder/f"reference-{mode}.jsonl.gz"
                write_rows(artifact, rows)
                summary["results"].append({"condition": name, "arm": f"reference-{mode}", "kind": "supplied_schedule",
                    **summarize(rows), "artifact": str(artifact.relative_to(args.output)), "artifact_sha256": file_hash(artifact),
                    "reference_compute_tariff": cfg.get("reference_compute_tariff", 0.0),
                    "controller_cpu_seconds": sum(r["outcome"]["controller_cpu_seconds"] for r in rows)})
                by_arm[f"reference-{mode}"] = light_rows(rows)
            for binding in bindings:
                a = by_arm[binding["name"]]
                for other_name, b in by_arm.items():
                    if other_name == binding["name"]:
                        continue
                    transitions = Counter(f"{int(x['outcome']['verified_success'])}->{int(y['outcome']['verified_success'])}" for x,y in zip(b,a))
                    summary["paired"].append({"condition": name, "learned": binding["name"], "comparator": other_name,
                        "success_transitions_comparator_to_learned": dict(transitions), "support": n,
                        "mean_utility_difference": sum(x["outcome"]["utility"]-y["outcome"]["utility"] for x,y in zip(a,b))/n})
            (args.output/"summary.partial.json").write_text(json.dumps(summary, indent=2))
        for condition in cfg["conditions"]:
            if "paired_control" not in condition:
                continue
            control_name = condition["paired_control"]
            available = {(r["condition"],r["arm"]):r for r in summary["results"]}
            for result in [r for r in summary["results"] if r["condition"]==condition["name"]]:
                key = (control_name,result["arm"])
                if key not in available:
                    raise ValueError(f"Missing paired control {key}")
                a = read_rows(args.output/available[key]["artifact"])
                b = read_rows(args.output/result["artifact"])
                summary["paired_conditions"].append({"control":control_name,"intervention":condition["name"],
                    "arm":result["arm"],**paired_condition_outcomes(a,b)})
        summary["solver"] = {"executor_version": solver.executor_version,
            "startup_wall_seconds": solver.startup_wall_seconds, "startup_child_cpu_seconds": solver.startup_child_cpu_seconds,
            "call_wall_seconds": solver.call_wall_seconds, "restarts": solver.restarts}
    summary["resources"] = {"wall_seconds": time.perf_counter()-start_wall, "parent_cpu_seconds": time.process_time()-start_cpu,
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)) if torch.device(args.device).type == "cuda" else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.device(args.device)) if torch.device(args.device).type == "cuda" else None,
        "cuda_device_capacity_bytes": torch.cuda.get_device_properties(torch.device(args.device)).total_memory if torch.device(args.device).type == "cuda" else None,
        "accounting_scope": "parent CPU excludes solver process; authoritative inclusive workload ledger is external"}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    (args.output/"summary.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
