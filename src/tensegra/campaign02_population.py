"""Synchronous, resource-recorded PBT / multistart / single-learner search.

No jobs launch on import. The JSON protocol and public environment mix are frozen
before starting. Selection uses development episodes only; this CLI never opens
confirmation episodes. Checkpoints are trusted coordinator-owned files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import resource
import time
from typing import Any

import torch

from .campaign02_policy import CandidatePolicy, PolicyConfig
from .campaign02_training import Learner, TrainConfig, digest, independent_address_seed, public_frame


@dataclass(frozen=True)
class PopulationConfig:
    mode: str = "pbt"
    population_seed: int = 0
    initialization_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105)
    rounds: int = 3
    updates_per_slot: int = 10
    population_size: int = 6
    optimizer_policy: str = "inherit"
    training_seed_start: int = 100_000_000
    training_seed_stride: int = 1_000_000
    development_seed_start: int = 900_000_000
    development_examples: int = 64
    address_namespace: str = "extended-02-population-v1"
    teacher: str = "cheap_first"
    # Each round may use an explicitly registered method, never DEV-derived gold.
    methods: tuple[str, ...] = ()
    train: dict[str, Any] = field(default_factory=lambda: {"width": 1024, "device": "cpu"})
    world_mix: tuple[dict[str, Any], ...] = ({},)
    threads: int = 1
    # Input contract shared by all members: legacy summaries or public-tree memory,
    # with public feature version v1 (historical) or v2 (supplied relations).
    policy: dict[str, Any] = field(default_factory=dict)
    # Optional per-slot initial {learning_rate, entropy_weight}; identical across
    # search modes so PBT/multistart/single start from the same initial bank.
    member_hyperparameters: tuple[dict[str, Any], ...] = ()
    # Optional six {path, sha256} bootstrap checkpoints forming the initial bank.
    # Weights are imported; optimizer state is reset, the per-slot training
    # stream restarts at this run's disjoint interval, hyperparameters are this
    # run's member rows. Bootstrap cost is charged to the producing job.
    initial_checkpoints: tuple[dict[str, Any], ...] = ()
    # Adaptive curriculum (PBT only): each member carries sampling weights over
    # world_mix components for TRAINING streams; children inherit the donor's
    # weights with one component scaled by 0.5 or 2 (separate RNG stream).
    # Development selection always uses the fixed uniform-hash mixture anchor.
    curriculum_mutation: bool = False
    # Optional separate development (selection) mixture. Training streams keep
    # world_mix; only the fitness panel changes. Empty = historical behavior.
    development_world_mix: tuple[dict[str, Any], ...] = ()
    # Successive halving: survivors per round (first must be 6, nonincreasing,
    # each divides 6). A survivor receives 6/k consecutive slots per round, i.e.
    # sequential depth. Optional niche protection keeps the best member of each
    # SUPPLIED behavioral niche (greedy-first rate on public DEV trajectories).
    halving_survivors: tuple[int, ...] = ()
    niche_protection: bool = False
    # World family: historical workshop-v1, the modular staged workshop, or depworld-v1.
    world_family: str = "workshop"

    def __post_init__(self):
        if self.mode not in {"pbt", "multistart", "single", "halving"}:
            raise ValueError("Unknown search mode")
        if self.population_size != 6 or len(self.initialization_seeds) != 6:
            raise ValueError("This registered search uses exactly six initial members")
        if self.optimizer_policy not in {"inherit", "reset"}:
            raise ValueError("Explicit optimizer policy required")
        if min(self.rounds, self.updates_per_slot, self.development_examples, self.training_seed_stride) < 1:
            raise ValueError("Positive search budgets required")
        if self.threads not in {1, 2} or not self.world_mix:
            raise ValueError("Bounded CPU threads and nonempty world mix required")
        if self.methods and (len(self.methods) != self.rounds or set(self.methods) - {"supervised", "actor_critic"}):
            raise ValueError("Provide one declared training method per round")
        train = TrainConfig(**self.train)
        maximum = self.rounds * self.updates_per_slot * train.batch_size * 6
        if maximum >= self.training_seed_stride:
            raise ValueError("Training streams could overlap")
        lo, hi = self.training_seed_start, self.training_seed_start + 6 * self.training_seed_stride
        devlo, devhi = self.development_seed_start, self.development_seed_start + self.development_examples
        if lo < devhi and devlo < hi:
            raise ValueError("Training and development seed ranges overlap")
        if len(set(self.initialization_seeds)) != 6:
            raise ValueError("Independent initializations required")
        if set(self.policy) - {"interface", "feature_version", "zero_memory"}:
            raise ValueError("Unknown policy interface option")
        if self.policy.get("interface", "legacy") not in {"legacy", "memory"}:
            raise ValueError("Unknown policy interface")
        if self.member_hyperparameters and (len(self.member_hyperparameters) != 6 or any(
                set(h) - {"learning_rate", "entropy_weight", "kl_weight"} for h in self.member_hyperparameters)):
            raise ValueError("Member hyperparameters need six {learning_rate, entropy_weight} rows")
        if self.world_family not in {"workshop", "modular", "depworld"}:
            raise ValueError("Unknown world family")
        if self.mode == "halving":
            k = self.halving_survivors
            if len(k) != self.rounds or k[0] != 6 or any(6 % x for x in k) or any(b > a for a, b in zip(k, k[1:])):
                raise ValueError("halving_survivors: one entry per round, start at 6, nonincreasing divisors of 6")
        elif self.halving_survivors or self.niche_protection:
            raise ValueError("halving options require mode='halving'")
        if self.curriculum_mutation and (self.mode != "pbt" or len(self.world_mix) < 2):
            raise ValueError("Curriculum mutation requires PBT and a multi-component mixture")
        if self.initial_checkpoints and (len(self.initial_checkpoints) != 6 or any(
                set(c) != {"path", "sha256"} for c in self.initial_checkpoints)):
            raise ValueError("Initial bank import needs six {path, sha256} rows")
        for kwargs in self.world_mix + self.development_world_mix:
            if "seed" in kwargs:
                raise ValueError("World mixture cannot override episode seed")

    @classmethod
    def from_json(cls, raw):
        raw = dict(raw)
        for key in ("initialization_seeds", "methods", "world_mix", "member_hyperparameters", "initial_checkpoints",
                    "development_world_mix", "halving_survivors"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)


def slot_member(mode: str, slot: int) -> int:
    if mode not in {"pbt", "multistart", "single"} or not 0 <= slot < 6:
        raise ValueError("Invalid allocation")
    return 0 if mode == "single" else slot


def selection_pairs(scores: list[dict]) -> list[tuple[int, int]]:
    """Two highest utility donors -> two lowest; ties resolve by slot index.

    This rule never replaces on a perfect all-member tie. Selection direction is
    utility, not tool usage, and every raw quality/cost score remains in records.
    """
    if len(scores) != 6 or {s["slot"] for s in scores} != set(range(6)):
        raise ValueError("Selection requires a complete six-slot round")
    ordered = sorted(scores, key=lambda s: (-s["utility"], s["slot"]))
    donors, recipients = ordered[:2], list(reversed(ordered[-2:]))
    return [(d["slot"], r["slot"]) for d, r in zip(donors, recipients) if d["utility"] > r["utility"]]


def build_policy(policy_config: dict):
    """Rebuild either registered public-input family from a checkpoint config."""
    if "memory_dim" in policy_config:
        from .campaign02_memory_policy import MemoryCandidatePolicy, MemoryPolicyConfig
        return MemoryCandidatePolicy(MemoryPolicyConfig(**policy_config))
    return CandidatePolicy(PolicyConfig(**policy_config))


def new_policy(options: dict, obs_dim: int, candidate_dim: int, width: int, family: str):
    version = options.get("feature_version", "v1")
    if options.get("interface", "legacy") == "memory":
        from .campaign02_memory import ROW_DIM
        from .campaign02_memory_policy import MemoryCandidatePolicy, MemoryPolicyConfig
        return MemoryCandidatePolicy(MemoryPolicyConfig(obs_dim, candidate_dim, width=width, family=family,
            feature_version=version, memory_dim=ROW_DIM, zero_memory=bool(options.get("zero_memory", False))))
    if options.get("zero_memory"):
        raise ValueError("zero_memory applies only to the memory interface")
    return CandidatePolicy(PolicyConfig(obs_dim, candidate_dim, width=width, family=family, feature_version=version))


def weighted_index(seed: int, weights) -> int:
    """Deterministic weighted component choice from a public-free seed hash."""
    total = float(sum(weights))
    if not weights or total <= 0 or min(weights) < 0:
        raise ValueError("Curriculum weights must be nonnegative with positive mass")
    u = int(digest({"world_seed": seed, "role": "curriculum-component"})[:16], 16) / 2**64 * total
    acc = 0.0
    for index, w in enumerate(weights):
        acc += w
        if u < acc:
            return index
    return len(weights) - 1


def mutate_curriculum(weights, rng: random.Random, floor: float = .02):
    index, factor = rng.randrange(len(weights)), rng.choice((.5, 2.))
    changed = [w*factor if i == index else w for i, w in enumerate(weights)]
    total = sum(changed)
    changed = [max(floor, w/total) for w in changed]
    total = sum(changed)
    return [w/total for w in changed], {"component": index, "factor": factor}


def greedy_first_rate(rows) -> float:
    """Supplied behavioral descriptor: fraction of DEV episodes whose first
    selection-relevant action is a direct item choice/commit, not a solver path."""
    def first(trace):
        for step in trace:
            kind = step["action"]["kind"]
            if kind in ("choose_item", "commit_pending"):
                return 1
            if kind in ("call", "start_subset"):
                return 0
        return 0
    rows = list(rows)
    return sum(first(r["trace"]) for r in rows)/len(rows) if rows else 0.0


def halving_keep(scores: list[dict], k: int, niche: bool) -> list[int]:
    """Keep k members by last-slot DEV utility (member-index tie break); with
    niche protection, first keep the best of each greedy-first niche."""
    ordered = sorted(scores, key=lambda x: (-x["utility"], x["member"]))
    kept = []
    if niche and k >= 2:  # with one survivor, protection would become a hard mode preference
        for label in (True, False):
            best = next((x for x in ordered if (x["behavior_greedy_first"] >= .5) == label), None)
            if best is not None and len(kept) < k:
                kept.append(best["member"])
    for x in ordered:
        if len(kept) >= k:
            break
        if x["member"] not in kept:
            kept.append(x["member"])
    return sorted(kept)


def mix_index(seed: int, size: int) -> int:
    # Independent of opaque return-handle spelling and initializer seed.
    return int(digest({"world_seed": seed, "role": "public-world-mixture"})[:16], 16) % size


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def _atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def _atomic_torch(path: Path, value):
    if path.exists():
        raise FileExistsError(f"Immutable checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def usage() -> dict:
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"wall_seconds": time.perf_counter(), "process_cpu_seconds": own.ru_utime + own.ru_stime,
            "reaped_child_cpu_seconds": children.ru_utime + children.ru_stime}


def elapsed(before: dict) -> dict:
    after = usage()
    return {k: after[k] - before[k] for k in before}


def inherit_state(parent: dict, recipient: dict, *, optimizer_policy: str,
                  learning_rate: float, entropy_weight: float, event: dict,
                  kl_weight: float | None = None) -> dict:
    """Weight ancestry changes; recipient's disjoint data/compute ledger does not."""
    if parent["policy_config"] != recipient["policy_config"]:
        raise ValueError("Cannot cross incompatible policy architectures")
    if optimizer_policy not in {"inherit", "reset"}:
        raise ValueError("Explicit optimizer policy required")
    result = cpu_tree(recipient)
    result["model"] = cpu_tree(parent["model"])
    if optimizer_policy == "inherit":
        result["optimizer"] = cpu_tree(parent["optimizer"])
    else:
        result["optimizer"]["state"] = {}
    for group in result["optimizer"]["param_groups"]:
        group["lr"] = learning_rate
    result["config"]["learning_rate"] = learning_rate
    result["config"]["entropy_weight"] = entropy_weight
    if kl_weight is not None:
        result["config"]["kl_weight"] = kl_weight
    result["resume_history"] = list(result.get("resume_history", [])) + [event]
    # Recipient RNG is retained: siblings do not consume the same future stream.
    return result


class PopulationRun:
    def __init__(self, config: PopulationConfig, output: Path, factory, teacher_factory, component_factory=None,
                 development_factory=None):
        self.config, self.output = config, output
        self.factory, self.teacher_factory = factory, teacher_factory
        if config.development_world_mix and development_factory is None:
            raise ValueError("A separate development mixture needs its own world factory")
        self.development_factory = development_factory or factory
        self.component_factory = component_factory
        if config.curriculum_mutation and component_factory is None:
            raise ValueError("Curriculum mutation needs a component-indexed world factory")
        self.curriculum_rng = random.Random(f"curriculum-{config.population_seed}")
        output.mkdir(parents=True, exist_ok=True)
        self.state_path = output / "state.json"
        self.protocol = asdict(config)
        self.protocol_hash = digest(self.protocol)
        self.mutation_rng = random.Random(config.population_seed)
        self.sources = {name: sha256(Path(__file__).with_name(name)) for name in (
            "campaign02_population.py", "campaign02_training.py", "campaign02_policy.py",
            "campaign02_world.py", "campaign02_protocol.py", "campaign02_references.py",
            "campaign02_memory.py", "campaign02_memory_policy.py", "campaign02_modular.py")}
        if config.world_family == "depworld":  # only then, so historical resume hashes are unchanged
            self.sources["campaign03_depworld.py"] = sha256(Path(__file__).with_name("campaign03_depworld.py"))
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state["source_hashes"] != self.sources:
                raise ValueError("Resume source changed; fork a separately registered run")
            if self.state["protocol_hash"] != self.protocol_hash:
                raise ValueError("Resume protocol differs from frozen protocol")
            def tuples(x):
                return tuple(tuples(v) for v in x) if isinstance(x, list) else x
            self.mutation_rng.setstate(tuples(self.state["mutation_rng"]))
            if "curriculum_rng" in self.state:
                self.curriculum_rng.setstate(tuples(self.state["curriculum_rng"]))
            for attempt in self.state["attempts"]:
                if attempt["status"] == "running":
                    attempt["status"] = "interrupted"
                    attempt["cost"] = None
                    attempt["accounting_note"] = "Hard interruption: outer process ledger must charge unrecovered attempt"
            if self.state["status"] == "initializing":
                self._initialize()
                self.state["status"] = "ready"
            self._save_state()
        else:
            if any(output.iterdir()):
                raise ValueError("New run needs an empty directory")
            _atomic_json(output / "protocol.json", self.protocol)
            self.state = {"version": 1, "protocol_hash": self.protocol_hash, "round": 0,
                          "slot": 0, "round_scores": [], "members": [], "lineage": [],
                          "allocations": [], "attempts": [], "status": "initializing", "source_hashes": self.sources}
            self._save_state()
            self._initialize()
            self.state["status"] = "ready"
            self._save_state()

    def _save_state(self):
        self.state["mutation_rng"] = self.mutation_rng.getstate()
        if self.config.curriculum_mutation:
            self.state["curriculum_rng"] = self.curriculum_rng.getstate()
        self.state["queue"] = [{"round": r, "slot": s, "member": self._slot_member(r, s)}
            for r in range(self.state["round"], self.config.rounds)
            for s in range(self.state["slot"] if r == self.state["round"] else 0, 6)]
        _atomic_json(self.state_path, self.state)
        _atomic_json(self.output / "population_lineage.json", self.state["lineage"])

    def _slot_member(self, round_index: int, slot: int) -> int:
        if self.config.mode != "halving":
            return slot_member(self.config.mode, slot)
        survivors = self.state.get("survivors", list(range(6)))
        # Future rounds are planned with current survivors; the plan is refreshed after each cut.
        return survivors[slot // (6 // len(survivors))]

    def _initialize(self):
        cfg = self.config
        _, obs, candidates = public_frame(self.factory(cfg.training_seed_start).observe(),
                                          cfg.policy.get("feature_version", "v1"))
        self.state["feature_dimensions"] = [len(obs), len(candidates[0])]
        start = usage()
        for slot, seed in enumerate(cfg.initialization_seeds):
            if slot < len(self.state["members"]):
                continue
            torch.manual_seed(seed)
            random.seed(seed)
            hyper = cfg.member_hyperparameters[slot] if cfg.member_hyperparameters else {}
            train = TrainConfig(**{**cfg.train, **hyper, "seed": seed, "device": "cpu",
                "training_seed_start": cfg.training_seed_start + slot * cfg.training_seed_stride})
            imported = None
            if cfg.initial_checkpoints:
                source = cfg.initial_checkpoints[slot]
                if sha256(Path(source["path"])) != source["sha256"]:
                    raise ValueError("Initial bank checkpoint hash mismatch")
                saved = torch.load(source["path"], map_location="cpu", weights_only=False)
                model = build_policy(saved["policy_config"])
                if asdict(model.config) != asdict(new_policy(cfg.policy, len(obs), len(candidates[0]),
                                                             train.width, train.family).config):
                    raise ValueError("Imported bank member does not match the declared interface")
                learner = Learner(model, train)
                learner.load(Path(source["path"]), allow_config_changes=True)
                learner.config = train
                learner.optimizer = torch.optim.AdamW(model.parameters(), lr=train.learning_rate)
                learner.seed_cursor = train.training_seed_start
                torch.manual_seed(seed)
                random.seed(seed)
                imported = {"source": source, "source_updates": saved["updates"],
                            "source_presentations": saved["presentations"], "optimizer": "reset at import"}
                learner.resume_history.append({"kind": "bank_import", **imported})
            else:
                model = new_policy(cfg.policy, len(obs), len(candidates[0]), train.width, train.family)
                learner = Learner(model, train)
            path = self.output / "initial_bank" / f"member-{slot}.pt"
            if path.exists():
                # A previous initialization may have written this deterministic bank
                # entry before committing its metadata. Verify instead of overwrite.
                old = torch.load(path, map_location="cpu", weights_only=False)
                if old["policy_config"] != asdict(model.config) or any(
                    not torch.equal(old["model"][k], v) for k, v in model.state_dict().items()):
                    raise ValueError("Conflicting initial-bank checkpoint")
            else:
                learner.save(path, {"role": "initial bank; no training in this run", "slot": slot, "imported": imported})
            entry = {"slot": slot, "individual_id": f"root-{slot}", "lineage_id": f"root-{slot}",
                "parent_id": None, "generation": 0, "checkpoint": str(path.relative_to(self.output)),
                "checkpoint_sha256": sha256(path), "initialization_seed": seed,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "learning_rate": train.learning_rate, "entropy_weight": train.entropy_weight,
                "kl_weight": train.kl_weight, "imported": imported}
            if cfg.curriculum_mutation:
                entry["curriculum"] = [1/len(cfg.world_mix)]*len(cfg.world_mix)
            self.state["members"].append(entry)
            self.state["lineage"].append({"kind": "initialization", **entry})
            self._save_state()
            del learner, model
        self.state.setdefault("initialization_attempt_costs", []).append(elapsed(start))
        self._record_anchor()
        self._record_a2()

    def _record_a2(self):
        """extended-04 A2 options (only when any is on, so historical state is unchanged)."""
        from .campaign02_training import A2_FIELD_DEFAULTS, A2_OPTION_NOTES, ROLLOUT_MASKS, a2_options_active
        train = TrainConfig(**self.config.train)
        if not a2_options_active(train):
            return
        if train.rollout_mask is not None:  # fail before any training if the rule cannot be built
            if train.rollout_mask not in ROLLOUT_MASKS:
                raise ValueError(f"Unknown rollout mask rule: {train.rollout_mask}")
            ROLLOUT_MASKS[train.rollout_mask](1)
        self.state["a2"] = {"options": {k: getattr(train, k) for k in A2_FIELD_DEFAULTS},
                            "rehearsal_teacher": self.config.teacher if train.rehearsal_weight > 0 else None,
                            "semantics": A2_OPTION_NOTES}

    def _record_anchor(self):
        """P2a bootstrap anchor provenance (only when configured, so historical state is unchanged).

        The anchor file is hash-verified here, before any training, and again by
        every Learner that loads it. The anchor is fixed for the whole run."""
        train = TrainConfig(**self.config.train)  # anchor fields are train-level (identical for all members)
        checkpoint = train.anchor_checkpoint
        if checkpoint is None:
            return
        if sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
            raise ValueError("Anchor checkpoint hash mismatch")
        self.state["anchor"] = {"checkpoint": checkpoint, "sha256_verified_at_initialization": True,
                                "anchor_kl_weight": train.anchor_kl_weight,
                                "objective": "anchor_kl_weight * mean KL(pi_anchor || pi_current) over rollout-visited states",
                                "replaced": "never"}

    def _checkpoint(self, member):
        path = self.output / member["checkpoint"]
        if sha256(path) != member["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        return torch.load(path, map_location="cpu", weights_only=False)

    def _learner(self, member, round_index):
        saved = self._checkpoint(member)
        method = self.config.methods[round_index] if self.config.methods else self.config.train.get("method", "supervised")
        train = TrainConfig(**{**saved["config"], "device": self.config.train.get("device", "cpu"), "method": method})
        model = build_policy(saved["policy_config"])
        learner = Learner(model, train)
        learner.load(self.output / member["checkpoint"], allow_config_changes=True)
        return learner

    def _run_slot(self):
        cfg, state = self.config, self.state
        r, slot = state["round"], state["slot"]
        member_index = self._slot_member(r, slot)
        member = state["members"][member_index]
        attempt_id = len(state["attempts"])
        attempt = {"id": attempt_id, "round": r, "slot": slot, "member_id": member["individual_id"],
                   "status": "running", "input_checkpoint_sha256": member["checkpoint_sha256"]}
        state["attempts"].append(attempt)
        self._save_state()
        start = usage()
        learner = None
        try:
            learner = self._learner(member, r)
            cursor_before = learner.seed_cursor
            train_factory = self.factory
            if cfg.curriculum_mutation:
                weights = list(member["curriculum"])
                train_factory = lambda seed: self.component_factory(seed, weighted_index(seed, weights))
            # The public teacher also serves A2 rehearsal (extended-04) when that option is on.
            timing = learner.train_tranche(cfg.updates_per_slot, train_factory,
                self.teacher_factory if learner.config.method == "supervised" or learner.config.rehearsal_weight > 0
                else None)
            output = self.output / "development" / f"round-{r}-slot-{slot}-attempt-{attempt_id}.jsonl.gz"
            development_clock = time.perf_counter(), time.process_time()
            metrics = learner.evaluate(range(cfg.development_seed_start,
                cfg.development_seed_start + cfg.development_examples), self.development_factory, output)
            if "phase_timing" in timing:  # observational only
                timing["phase_timing"]["development_evaluation"] = {
                    "wall_seconds": time.perf_counter() - development_clock[0],
                    "process_cpu_seconds": time.process_time() - development_clock[1], "calls": 1}
            import gzip
            with gzip.open(output, "rt") as stream:
                metrics["behavior_greedy_first"] = greedy_first_rate(json.loads(line) for line in stream)
            # Move every tensor to CPU before writing/parking, then release GPU.
            learner.model.to("cpu")
            for key in list(learner.optimizer.state):
                learner.optimizer.state[key] = cpu_tree(learner.optimizer.state[key])
            path = self.output / "checkpoints" / f"round-{r}-slot-{slot}-attempt-{attempt_id}.pt"
            save_clock = time.perf_counter(), time.process_time()
            learner.save(path, {"round": r, "slot": slot, "metrics": metrics,
                "protocol_hash": self.protocol_hash, "individual_id": member["individual_id"]})
            if "phase_timing" in timing:
                timing["phase_timing"]["checkpoint_save"] = {
                    "wall_seconds": time.perf_counter() - save_clock[0],
                    "process_cpu_seconds": time.process_time() - save_clock[1], "calls": 1}
            member["checkpoint"], member["checkpoint_sha256"] = str(path.relative_to(self.output)), sha256(path)
            row = {"round": r, "slot": slot, "member": member_index, "member_id": member["individual_id"],
                "training_seed_interval": [cursor_before, learner.seed_cursor],
                "updates": cfg.updates_per_slot, "cumulative_slot_updates": learner.updates,
                "cumulative_decision_presentations": learner.presentations, "training_timing": timing,
                "checkpoint": member["checkpoint"], "checkpoint_sha256": member["checkpoint_sha256"],
                "development_predictions": str(output.relative_to(self.output)),
                "development_sha256": sha256(output), **metrics}
            state["allocations"].append(row)
            state["round_scores"].append(row)
            attempt.update(status="completed", cost=elapsed(start))
            state["slot"] += 1
            self._save_state()
        except BaseException as error:
            attempt.update(status="failed", error=f"{type(error).__name__}: {error}", cost=elapsed(start))
            self._save_state()
            raise
        finally:
            if learner is not None:
                learner.model.to("cpu")
                del learner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _selection(self):
        cfg, state = self.config, self.state
        r = state["round"]
        if cfg.mode == "halving":
            if r + 1 < cfg.rounds:
                last = {}
                for row in state["round_scores"]:
                    last[row["member"]] = row
                k = cfg.halving_survivors[r + 1]
                kept = halving_keep(list(last.values()), k, cfg.niche_protection)
                state["lineage"].append({"kind": "halving", "round": r, "kept": kept,
                    "dropped": sorted(set(last) - set(kept)), "niche_protection": cfg.niche_protection,
                    "scores": [{k2: v[k2] for k2 in ("member", "member_id", "utility", "success", "behavior_greedy_first",
                                                     "cumulative_slot_updates")} for v in last.values()]})
                state["survivors"] = kept
            state["round"] += 1
            state["slot"] = 0
            state["round_scores"] = []
            self._save_state()
            return
        pairs = selection_pairs(state["round_scores"]) if cfg.mode == "pbt" and r + 1 < cfg.rounds else []
        for donor_slot, recipient_slot in pairs:
            donor, recipient = state["members"][donor_slot], state["members"][recipient_slot]
            parent, previous = self._checkpoint(donor), self._checkpoint(recipient)
            lr_factor, entropy_factor = self.mutation_rng.choice((.8, 1.2)), self.mutation_rng.choice((.8, 1.2))
            lr = min(1e-2, max(1e-6, donor["learning_rate"] * lr_factor))
            entropy = min(1., max(1e-6, donor["entropy_weight"] * entropy_factor))
            # KL-anchor mutation only when the registered protocol uses it; the
            # extra RNG draw would otherwise change historical mutation streams.
            kl = kl_factor = None
            if donor.get("kl_weight", 0.0) > 0:
                kl_factor = self.mutation_rng.choice((.8, 1.2))
                kl = min(10., max(1e-4, donor["kl_weight"] * kl_factor))
            child_id = f"round-{r + 1}-slot-{recipient_slot}"
            event = {"kind": "replacement", "round": r, "individual_id": child_id,
                "parent_id": donor["individual_id"], "replaced_id": recipient["individual_id"],
                "recipient_slot": recipient_slot, "donor_slot": donor_slot,
                "parent_checkpoint": donor["checkpoint"], "parent_sha256": donor["checkpoint_sha256"],
                "recipient_checkpoint": recipient["checkpoint"], "recipient_sha256": recipient["checkpoint_sha256"],
                "optimizer_policy": cfg.optimizer_policy, "learning_rate_factor": lr_factor,
                "entropy_factor": entropy_factor, "learning_rate": lr, "entropy_weight": entropy,
                "kl_factor": kl_factor, "kl_weight": kl,
                "retained_recipient_seed_cursor": previous["seed_cursor"],
                "curriculum_parent": donor.get("curriculum"),
                "retained_recipient_updates": previous["updates"], "scores": state["round_scores"]}
            if cfg.curriculum_mutation:
                event["curriculum_child"], event["curriculum_mutation"] = mutate_curriculum(donor["curriculum"], self.curriculum_rng)
            child = inherit_state(parent, previous, optimizer_policy=cfg.optimizer_policy,
                learning_rate=lr, entropy_weight=entropy, event=event, kl_weight=kl)
            path = self.output / "checkpoints" / f"replacement-round-{r}-slot-{recipient_slot}.pt"
            # Safe retry after crash between immutable replacement and state commit.
            if path.exists():
                existing = torch.load(path, map_location="cpu", weights_only=False)
                if existing["resume_history"][-1] != event:
                    raise ValueError("Conflicting replacement checkpoint")
            else:
                _atomic_torch(path, child)
            recipient.update(individual_id=child_id, parent_id=donor["individual_id"],
                lineage_id=donor["lineage_id"], generation=donor["generation"] + 1,
                learning_rate=lr, entropy_weight=entropy, **({"kl_weight": kl} if kl is not None else {}),
                **({"curriculum": event["curriculum_child"]} if cfg.curriculum_mutation else {}),
                checkpoint=str(path.relative_to(self.output)), checkpoint_sha256=sha256(path))
            state["lineage"].append(event)
        state["round"] += 1
        state["slot"] = 0
        state["round_scores"] = []
        self._save_state()

    def run(self):
        while self.state["round"] < self.config.rounds:
            while self.state["slot"] < 6:
                self._run_slot()
            self._selection()
        # Best CURRENT final population only. No historical checkpoint cherry-pick.
        final = [r for r in self.state["allocations"] if r["round"] == self.config.rounds - 1]
        eligible = final[-1:] if self.config.mode == "single" else final
        if self.config.mode == "halving":
            latest = {}
            for row in final:
                latest[row["member"]] = row  # each survivor's last sequential slot only
            eligible = list(latest.values())
        winner = sorted(eligible, key=lambda x: (-x["utility"], x["slot"]))[0]
        self.state["status"] = "completed"
        self.state["finalist"] = winner
        self.state["selection_rule"] = "maximum final-round development utility, slot-index tie break; single latest only"
        self._save_state()
        return self.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = PopulationConfig.from_json(json.loads(args.config.read_text()))
    torch.set_num_threads(config.threads)
    from .campaign02_protocol import BoundedSolver
    from .campaign02_references import make_reference
    from .campaign02_world import Workshop, generate_world, protocol_executor
    with BoundedSolver() as solver:
        executor = partial(protocol_executor, execute_call=solver.execute)
        make_world, make_env = generate_world, Workshop
        if config.world_family == "modular":
            from .campaign02_modular import ModularWorkshop, generate_modular, modular_executor
            make_world, make_env = generate_modular, ModularWorkshop
            executor = partial(modular_executor, execute_call=solver.execute)
        elif config.world_family == "depworld":
            from .campaign03_depworld import DepWorkshop, depworld_executor, generate_depworld
            make_world, make_env = generate_depworld, DepWorkshop
            executor = partial(depworld_executor, execute_call=solver.execute)
        def component_factory(seed, index):
            return make_env(make_world(seed, **config.world_mix[index]), executor=executor,
                address_seed=independent_address_seed(seed, config.address_namespace))
        def factory(seed):
            return component_factory(seed, mix_index(seed, len(config.world_mix)))
        development_factory = None
        if config.development_world_mix:
            def development_factory(seed):
                kwargs = config.development_world_mix[mix_index(seed, len(config.development_world_mix))]
                return make_env(make_world(seed, **kwargs), executor=executor,
                    address_seed=independent_address_seed(seed, config.address_namespace))
        run = PopulationRun(config, args.output, factory, lambda: make_reference(config.teacher),
                            component_factory=component_factory, development_factory=development_factory)
        run.run()
        _atomic_json(args.output / "solver-process-accounting.json", {
            "startup_wall_seconds": solver.startup_wall_seconds,
            "startup_child_cpu_seconds": solver.startup_child_cpu_seconds,
            "call_wall_seconds": solver.call_wall_seconds, "restarts": solver.restarts,
            "note": "Process occupancy and solver CPU are separate; episode solver CPU is in raw outcomes. Reaped child CPU may omit live persistent worker until exit."})


if __name__ == "__main__":
    main()
