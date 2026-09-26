"""depworld-v1: a changing dependency graph of computations (extended-03, Stage A).

A workshop episode chains three data-dependent computations over the frozen
extended-02 typed protocol (``campaign02_protocol``; unchanged):

  selection  (constrained_subset over inspected items + requirement constraints)
  assignment (csp over the *committed selection's* start slots; single machine,
              no two selected jobs overlap in time; optional finish_by bound)
  route      (shortest_path from the current position; the deadline check
              finish_time + travel <= deadline is the agent's job)

No stage order is scripted beyond data preconditions (an assignment needs a
committed selection; a route draft needs a committed assignment). Public
change events alter one requirement mid-episode, may revoke committed pieces,
and bump public requirement versions. Every computation record carries public
applicability metadata (problem snapshot + depends_on). Foreign problems and
records are *registered* at t=0, some still applicable, some not.

The hidden spec (``DepSpec``) is evaluator-only. Item facts, requirements and
the map become public only after inspection. Actors receive ``DepObservation``
objects only. ``audit_record`` / reduction audits read hidden state and are
logging-only; they never feed an actor, encoder, catalogue or reference.
"""
from __future__ import annotations

from copy import deepcopy
from copy import deepcopy as _deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import heapq
import json
import math
import random
import time
from typing import Any, Callable

from . import campaign04_fast as _fast
from .campaign02_world import Action

VERSION = "depworld-v1"
MAX_DURATION = 4
PRIMITIVES = ("constrained_subset", "csp", "shortest_path")
USES = ("select", "assign", "route")
USE_OF_PRIMITIVE = {"constrained_subset": "select", "csp": "assign", "shortest_path": "route"}
PRIMITIVE_OF_USE = {v: k for k, v in USE_OF_PRIMITIVE.items()}
REQUIREMENTS = ("capacity", "funds", "incompatible", "slots", "deadline", "map")
# Requirement names each primitive's *instance* is built from (dependency metadata).
READS = {"constrained_subset": ("capacity", "funds", "incompatible"), "csp": ("slots",), "shortest_path": ("map",)}
EVENT_KINDS = ("edge_closed", "capacity_reduced", "slot_closed", "deadline_moved")
EVENT_TRIGGERS = ("step", "progress")
# Progress trigger: kinds that can invalidate what is committed at the k-th completion
# (k=1: a selection is committed; k=2: normally the assignment too).
PROGRESS_KINDS = {1: ("capacity_reduced", "edge_closed"), 2: ("slot_closed", "deadline_moved", "edge_closed")}
EVENT_AFFECTS = {"edge_closed": "map", "capacity_reduced": "capacity", "slot_closed": "slots",
                 "deadline_moved": "deadline"}
SUBSET_CONSTRAINTS = ("capacity", "funds", "incompatibility")
CSP_CONSTRAINTS = ("conflicts",)
CONSTRAINT_NAMES = ("capacity", "funds", "incompatibility", "exclude", "conflicts", "finish_by")
# Declared public rejection reasons (spec list first, then declared extras).
REASONS = ("capacity", "funds", "incompatibility", "slot_conflict", "slot_unavailable", "deadline",
           "stale_dependency", "type_mismatch", "not_applicable", "missing_edge",
           "category_coverage", "missing_dependency", "incomplete", "already_committed",
           "retrieve_required", "unknown_item", "travel_budget", "work_budget")
USABLE_STATUSES = ("success", "timeout")
ATTEMPT_KINDS = ("commit_pending", "commit_assignment", "use_return", "move", "call", "uncommit", "verify")
KINDS = ("inspect", "start_subset", "start_assign", "build_route", "add_constraint", "call", "retrieve",
         "use_return", "choose_item", "choose_slot", "commit_pending", "commit_assignment", "uncommit",
         "move", "verify", "think", "abstain")
OUTCOMES = ("success", "rejected", "invalid_input", "timeout", "infeasible", "incomplete",
            "unavailable_resource", "other")
RECORD_STATUSES = ("success", "timeout", "infeasible", "invalid", "unknown")
# Trailing d1 candidate block: attempted, count, last outcome one-hot, last reason one-hot, deps changed.
ATTEMPT_BLOCK = 2 + len(OUTCOMES) + len(REASONS) + 1
FOREIGN_KINDS = ("subset_current", "subset_earlier", "subset_version_mismatch", "route_current",
                 "route_other_goal", "route_earlier", "csp_current", "csp_other_selection")


# ---------------------------------------------------------------------------
# Hidden specification and generator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DepItem:
    handle: str
    category: int
    weight: int
    price: int
    duration: int
    slots: tuple[int, ...]


@dataclass(frozen=True)
class DepSpec:
    """Evaluator-only world. Values are the *initial* (pre-event) requirements."""
    items: tuple[DepItem, ...]
    categories: tuple[int, ...]
    slots: int
    capacity: int
    funds: int
    incompatible: tuple[tuple[str, str], ...]
    deadline: int
    edges: tuple[tuple[int, int, int], ...]
    locations: int
    start: int = 0
    destination: int = -1
    slot_capacity: int = 1
    closed_slots: tuple[int, ...] = ()
    # (kind, when, argument) or None. argument: int, or (u, v) for edge_closed.
    # event_trigger "step": fires at the end of step `when`; "progress": fires right
    # after the world's `when`-th successful completion commit (selection or
    # assignment commit, directly or via use_return). Never after delivery.
    event: tuple | None = None
    event_trigger: str = "step"
    # Foreign registered problems/records, each a dict with evaluator-only "label".
    foreign: tuple = ()
    # Planted plan (selection, ((handle, start), ...), route) valid after the event.
    planted: tuple = ()
    step_limit: int = 96
    work_limit: int = 8192
    observation_price: float = .001
    action_price: float = .001
    work_price: float = 2e-4
    travel_price: float = .001
    travel_limit: int = 64
    compute_price: float = 0.0
    call_budgets: tuple[int, ...] = (16, 128, 1024)
    include_remaining_budget: bool = True

    def __post_init__(self):
        if self.event_trigger not in EVENT_TRIGGERS:
            raise ValueError("event_trigger must be 'step' or 'progress'")
        if self.slot_capacity != 1:
            raise ValueError("depworld-v1 fixes slot_capacity=1 (binary CSP lowering)")
        if not isinstance(self.call_budgets, tuple) or not self.call_budgets or any(
                isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in self.call_budgets):
            raise ValueError("call_budgets must be a nonempty tuple of positive integers")
        if not 0 <= self.destination < self.locations:
            raise ValueError("destination must be a location")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def selection_id(handles) -> str:
    """Content-addressed selection identity (equality only; never a feature)."""
    return "sel-" + _digest(sorted(handles))[:12]


def assignment_id(sel_id: str, assignment: dict) -> str:
    return "asg-" + _digest([sel_id, sorted([h, s] for h, s in assignment.items())])[:12]


def action_key(action: Action) -> str:
    return _digest({"kind": action.kind, "arguments": dict(action.arguments)})[:16]


def overlaps(s: int, d: int, t: int, e: int) -> bool:
    return s < t + e and t < s + d


def _dijkstra(edges, n, start, goal):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    heap, best = [(0, start, (start,))], {start: 0}
    while heap:
        d, u, path = heapq.heappop(heap)
        if d != best.get(u):
            continue
        if u == goal:
            return path, d
        for v, w in graph[u]:
            if v not in best or d + w < best[v]:
                best[v] = d + w
                heapq.heappush(heap, (d + w, v, path + (v,)))
    return None, None


# Pure public instance builders (shared by the environment, public requests and audits).

def subset_problem(rows, requirements, constraints=(), excluded=()):
    """rows: item fact dicts in inventory order (base domain, before exclusions)."""
    excluded = [r["handle"] for r in rows if r["handle"] in set(excluded)]
    kept = [r for r in rows if r["handle"] not in set(excluded)]
    handles = [r["handle"] for r in kept]
    p = {"handles": handles,
         "items": [[r["category"], r["weight"], r["price"], MAX_DURATION + 1 - r["duration"]] for r in kept],
         "constraints": sorted(set(constraints)), "excluded": excluded}
    if "capacity" in constraints:
        p["capacity"] = requirements["capacity"]
    if "funds" in constraints:
        p["max_cost"] = requirements["funds"]
    if "incompatibility" in constraints:
        index = {h: i for i, h in enumerate(handles)}
        p["forbidden_pairs"] = sorted([sorted([index[a], index[b]]) for a, b in requirements["incompatible"]
                                       if a in index and b in index])
    return {"primitive": "constrained_subset", "problem": p}


def csp_problem(rows, closed_slots, constraints=(), finish_by=None):
    """rows: the selected items' fact dicts in category order."""
    closed = set(closed_slots)
    domains = [[s for s in sorted(r["slots"]) if s not in closed and (finish_by is None or s + r["duration"] <= finish_by)]
               for r in rows]
    forbidden = []
    if "conflicts" in constraints:
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                for x in domains[i]:
                    for y in domains[j]:
                        if overlaps(x, rows[i]["duration"], y, rows[j]["duration"]):
                            forbidden.append([i, x, j, y])
    return {"primitive": "csp", "problem": {
        "items": [r["handle"] for r in rows], "durations": [r["duration"] for r in rows],
        "domains": domains, "forbidden": forbidden, "constraints": sorted(set(constraints)),
        "finish_by": finish_by}}


def route_problem(edges, locations, start, goal):
    return {"primitive": "shortest_path", "problem": {
        "n": locations, "edges": [list(e) for e in sorted(tuple(e) for e in edges)], "start": start, "goal": goal}}


def snapshot_options(snapshot) -> dict:
    p = (snapshot or {}).get("problem", {})
    primitive = (snapshot or {}).get("primitive")
    if primitive == "constrained_subset":
        return {"excluded": tuple(p.get("excluded", ()))}
    if primitive == "csp":
        return {"finish_by": p.get("finish_by")}
    return {}


def _item_row(item: DepItem) -> dict:
    return {"handle": item.handle, "category": item.category, "weight": item.weight, "price": item.price,
            "duration": item.duration, "slots": list(item.slots)}


def _validate_selection(rows_by_handle, categories, requirements, handles):
    if not isinstance(handles, (list, tuple)) or any(not isinstance(h, str) for h in handles):
        return False, "unknown_item"
    if any(h not in rows_by_handle for h in handles):
        return False, "unknown_item"
    rows = [rows_by_handle[h] for h in handles]
    if len(set(handles)) != len(handles) or sorted(r["category"] for r in rows) != sorted(categories):
        return False, "category_coverage"
    if sum(r["weight"] for r in rows) > requirements["capacity"]:
        return False, "capacity"
    if sum(r["price"] for r in rows) > requirements["funds"]:
        return False, "funds"
    chosen = set(handles)
    if any(a in chosen and b in chosen for a, b in requirements["incompatible"]):
        return False, "incompatibility"
    return True, "valid"


def _validate_assignment(rows_by_handle, selected, closed_slots, assignment):
    if set(assignment) != set(selected):
        return False, "incomplete"
    closed = set(closed_slots)
    for h in selected:
        s = assignment[h]
        if type(s) is not int or s not in rows_by_handle[h]["slots"] or s in closed:
            return False, "slot_unavailable"
    hs = list(selected)
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            a, b = rows_by_handle[hs[i]], rows_by_handle[hs[j]]
            if overlaps(assignment[hs[i]], a["duration"], assignment[hs[j]], b["duration"]):
                return False, "slot_conflict"
    return True, "valid"


def _finish(rows_by_handle, assignment) -> int:
    return max((s + rows_by_handle[h]["duration"] for h, s in assignment.items()), default=0)


def _solve_local(snapshot):
    """Generator-only exact solve of a public snapshot via the frozen protocol."""
    from .campaign02_protocol import execute
    return depworld_executor(snapshot["primitive"], deepcopy(snapshot["problem"]), 100000, execute_call=execute)


def generate_depworld(seed: int, categories: int = 3, choices: int = 3, locations: int = 7, slots: int = 6,
                      deadline_slack: int = 1, p_event: float = 0.0, foreign_records: int = 0,
                      event_kinds=EVENT_KINDS, event_steps=None, slot_choices=(2, 4),
                      shortcut: float = .25, event_trigger: str = "progress", **overrides: Any) -> DepSpec:
    """Planted-feasible world; the plan stays feasible after the (single) event.

    event_trigger="step" reproduces the original step-scheduled worlds exactly;
    "progress" (default for new calls) fires after the k-th completion commit,
    k in {1, 2}, with a kind that can invalidate what is committed by then."""
    if event_trigger not in EVENT_TRIGGERS:
        raise ValueError("event_trigger must be 'step' or 'progress'")
    if min(categories, choices) < 1 or locations < 3 or slots < 2:
        raise ValueError("positive sizes, >=3 locations, >=2 slots required")
    rng = random.Random(f"depworld-v1-{seed}")
    lo, hi = min(slot_choices[0], slots), min(slot_choices[1], slots)
    handles = rng.sample(range(10_000, 999_999), categories * choices)
    groups = []
    for c in range(categories):
        group = [{"handle": f"i{handles[c * choices + j]}", "category": c, "weight": rng.randint(1, 7),
                  "price": rng.randint(1, 9), "duration": rng.randint(1, MAX_DURATION),
                  "slots": set(rng.sample(range(slots), rng.randint(lo, hi)))} for j in range(choices)]
        groups.append(group)
    planted = [rng.choice(g) for g in groups]
    order = planted[:]
    rng.shuffle(order)
    while sum(x["duration"] for x in order[:-1]) > slots - 1:
        max(order[:-1], key=lambda x: x["duration"])["duration"] -= 1
    start, starts = 0, {}
    for x in order:
        x["slots"].add(start)
        starts[x["handle"]] = start
        start += x["duration"]
    finish = start
    items = [x for g in groups for x in g]
    rng.shuffle(items)
    planted_handles = {x["handle"] for x in planted}
    incompatible = tuple((x["handle"], y["handle"]) for i, x in enumerate(items) for y in items[i + 1:]
                         if x["category"] != y["category"] and not {x["handle"], y["handle"]} <= planted_handles
                         and rng.random() < .18)
    capacity = sum(x["weight"] for x in planted) + rng.randint(0, 2)
    funds = sum(x["price"] for x in planted) + rng.randint(0, 3)
    edges = {(i, i + 1): rng.randint(1, 4) for i in range(locations - 1)}
    if rng.random() < .75:
        edges[(0, locations - 1)] = rng.randint(2, 18)
    for i in range(locations):
        for j in range(i + 2, locations):
            if rng.random() < shortcut:
                edges.setdefault((i, j), rng.randint(1, 7))
    dest = locations - 1
    edge_list = [(a, b, w) for (a, b), w in sorted(edges.items())]
    post_edges = list(edge_list)
    event = None
    initial_capacity, closed = capacity, ()
    deadline_bump = 0
    def canonical_plan(cap):
        """The canonical public pipeline's likely commitments (selection X and its
        first csp assignment) under pre-event requirements; used only to aim the
        fixed event at plans agents typically hold. Agent-independent."""
        rows = [{**x, "slots": sorted(x["slots"])} for x in items]
        sub = subset_problem(rows, {"capacity": cap, "funds": funds,
                                    "incompatible": [list(q) for q in incompatible]}, SUBSET_CONSTRAINTS)
        res = _solve_local(sub)
        if res["status"] != "success":
            return None, None
        chosen = [sub["problem"]["handles"][i] for i in res["payload"][0]]
        by = {r["handle"]: r for r in rows}
        csp = csp_problem(sorted((by[h] for h in chosen), key=lambda r: r["category"]), (), CSP_CONSTRAINTS)
        res = _solve_local(csp)
        return chosen, (dict(zip(csp["problem"]["items"], res["payload"])) if res["status"] == "success" else None)

    if event_steps is None:
        n = categories * choices
        event_steps = (n + 9, n + 19)
    if rng.random() < p_event:
        if event_trigger == "step":
            step = rng.randint(*event_steps)
            kinds = list(event_kinds)
        else:
            step = rng.randint(1, 2)  # completion ordinal k
            kinds = [k for k in PROGRESS_KINDS[step] if k in event_kinds] or list(event_kinds)
        rng.shuffle(kinds)
        for kind in kinds:
            if kind == "edge_closed":
                path, _ = _dijkstra(edge_list, locations, 0, dest)
                options = [(u, v) for u, v in zip(path, path[1:])
                           if _dijkstra([e for e in edge_list if (e[0], e[1]) != (u, v)], locations, 0, dest)[0]]
                if not options:
                    continue
                u, v = rng.choice(options)
                post_edges = [e for e in edge_list if (e[0], e[1]) != (u, v)]
                event = ("edge_closed", step, (u, v))
            elif kind == "capacity_reduced":
                initial_capacity = capacity + rng.randint(1, 3)
                chosen, _ = canonical_plan(initial_capacity)
                planted_weight = sum(x["weight"] for x in planted)
                weight = sum(x["weight"] for x in items if chosen and x["handle"] in chosen)
                post = min(capacity, weight - 1) if chosen and weight > planted_weight else capacity
                event = ("capacity_reduced", step, post)
            elif kind == "slot_closed":
                options = [s for s in range(slots) if s not in starts.values()]
                if not options:
                    continue
                _, plan = canonical_plan(capacity)
                aimed = sorted({s for s in (plan or {}).values() if s in options})
                if aimed:
                    event = ("slot_closed", step, rng.choice(aimed))
                else:
                    used = {s: sum(s in x["slots"] for x in items) for s in options}
                    top = max(used.values())
                    event = ("slot_closed", step, rng.choice([s for s in options if used[s] == top]))
            else:
                deadline_bump = rng.randint(1, 3)
                event = ("deadline_moved", step, None)
            break
    route, distance = _dijkstra(post_edges, locations, 0, dest)
    slack = rng.randint(0, deadline_slack)
    if event is not None and event[0] == "deadline_moved":
        # Aim the moved deadline below the canonical plan's arrival when the
        # planted plan allows it (planted arrival = finish + distance stays feasible).
        _, plan = canonical_plan(capacity)
        if plan:
            by = {x["handle"]: x for x in items}
            canonical_finish = max(s0 + by[h]["duration"] for h, s0 in plan.items())
            slack = min(slack, max(0, canonical_finish - finish - 1))
    deadline = finish + distance + slack
    initial_deadline = deadline + deadline_bump
    if event is not None and event[0] == "deadline_moved":
        event = ("deadline_moved", event[1], deadline)
    item_objs = tuple(DepItem(x["handle"], x["category"], x["weight"], x["price"], x["duration"],
                              tuple(sorted(x["slots"]))) for x in items)
    params = dict(items=item_objs, categories=tuple(range(categories)), slots=slots, capacity=initial_capacity,
                  funds=funds, incompatible=incompatible, deadline=initial_deadline, edges=tuple(edge_list),
                  locations=locations, start=0, destination=dest, closed_slots=closed, event=event,
                  event_trigger=event_trigger,
                  planted=(tuple(sorted(planted_handles)), tuple(sorted(starts.items())), tuple(route)))
    params.update(overrides)
    for key in ("call_budgets", "categories", "closed_slots"):
        if key in params and isinstance(params[key], list):
            params[key] = tuple(params[key])
    spec = DepSpec(**params)
    if foreign_records > 0 and "foreign" not in overrides:
        spec = _with_foreign(spec, seed, foreign_records)
    return spec


def _with_foreign(spec: DepSpec, seed: int, count: int) -> DepSpec:
    from dataclasses import replace
    rng = random.Random(f"depworld-v1-foreign-{seed}")
    rows = [_item_row(x) for x in spec.items]
    req = {"capacity": spec.capacity, "funds": spec.funds, "incompatible": [list(p) for p in spec.incompatible]}
    by_handle = {r["handle"]: r for r in rows}
    current_subset = subset_problem(rows, req, SUBSET_CONSTRAINTS)
    solved = _solve_local(current_subset)
    x_sel = None
    if solved["status"] == "success":
        x_sel = [current_subset["problem"]["handles"][i] for i in solved["payload"][0]]

    def selection_rows(handles):
        return sorted((by_handle[h] for h in handles), key=lambda r: r["category"])

    out = []
    for _ in range(count):
        kind = rng.choice(FOREIGN_KINDS)
        deps = None
        if kind == "subset_current":
            snap = current_subset
            deps = {"requirements_version": {n: 1 for n in READS["constrained_subset"]}}
        elif kind in ("subset_earlier", "subset_version_mismatch"):
            other = dict(req)
            if kind == "subset_earlier":
                if rng.random() < .5:
                    other["capacity"] = req["capacity"] + rng.randint(1, 4)
                else:
                    other["funds"] = req["funds"] + rng.randint(1, 4)
            snap = subset_problem(rows, other, SUBSET_CONSTRAINTS)
            deps = {"requirements_version": {n: 0 for n in READS["constrained_subset"]}}
        elif kind == "route_current":
            snap = route_problem(spec.edges, spec.locations, spec.start, spec.destination)
            deps = {"requirements_version": {"map": 1}}
        elif kind == "route_other_goal":
            goal = rng.randrange(1, spec.destination)
            snap = route_problem(spec.edges, spec.locations, spec.start, goal)
            deps = {"requirements_version": {"map": 1}}
        elif kind == "route_earlier":
            edges = [list(e) for e in spec.edges]
            present = {(u, v) for u, v, _ in edges}
            missing = [(u, v) for u in range(spec.locations) for v in range(u + 2, spec.locations) if (u, v) not in present]
            if missing and rng.random() < .5:
                u, v = rng.choice(missing)
                edges.append([u, v, rng.randint(1, 3)])
            else:
                e = rng.choice(edges)
                e[2] = max(1, e[2] - rng.randint(1, 3))
            snap = route_problem(edges, spec.locations, spec.start, spec.destination)
            deps = {"requirements_version": {"map": 0}}
        else:
            if kind == "csp_current" and x_sel is not None:
                handles = x_sel
            else:
                handles = [rng.choice([r["handle"] for r in rows if r["category"] == c]) for c in spec.categories]
                if x_sel is not None and sorted(handles) == sorted(x_sel):
                    kind = "csp_current"
            snap = csp_problem(selection_rows(handles), spec.closed_slots, CSP_CONSTRAINTS)
            deps = {"requirements_version": {"slots": 1}, "selection_id": selection_id(handles)}
        result = _solve_local(snap)
        out.append({"label": kind, "primitive": snap["primitive"], "snapshot": snap, "depends_on": deps,
                    "status": result["status"], "payload": result["payload"],
                    "certificate_valid": bool(result["certificate_valid"]), "work_units": int(result["work_units"])})
    return replace(spec, foreign=tuple(out))


# ---------------------------------------------------------------------------
# Executor lowering
# ---------------------------------------------------------------------------

def depworld_executor(primitive: str, problem: dict[str, Any], max_work: int, *, execute_call=None) -> dict[str, Any]:
    """Supplied deterministic lowering of public drafts into frozen protocol calls.

    constrained_subset / shortest_path reuse the extended-02 lowering; csp drafts
    lower to (domains, forbidden) as in the modular workshop. Missing constraints
    stay unconstrained; no world access, no completion, no solving here.
    """
    if primitive != "csp":
        from .campaign02_world import protocol_executor
        return protocol_executor(primitive, problem, max_work, execute_call=execute_call)
    from .campaign02_protocol import Budget, Call, execute_isolated, validate_result
    args = (tuple(tuple(d) for d in problem["domains"]), tuple(tuple(r) for r in problem.get("forbidden", [])))
    call = Call("csp", args, budget=Budget(max_work))
    result = (execute_isolated if execute_call is None else execute_call)(call)
    return {"status": result.status, "payload": result.payload, "work_units": result.work_units,
            "cpu_seconds": result.cpu_seconds, "child_cpu_seconds": result.cpu_seconds,
            "certificate": result.certificate, "certificate_valid": validate_result(call, result)}


# ---------------------------------------------------------------------------
# Public observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DepObservation:
    version: str
    step: int
    goal: dict[str, Any]
    requirements: dict[str, Any] | None
    requirements_version: int
    requirement_versions: dict[str, int]
    events: tuple[dict[str, Any], ...]
    item_inventory: tuple[dict[str, Any], ...]
    known_items: dict[str, dict[str, Any]]
    known_edges: tuple[tuple[int, int, int], ...] | None
    problems: dict[str, dict[str, Any]]
    records: tuple[dict[str, Any], ...]
    retrieved: dict[str, dict[str, Any]]
    pending: tuple[str, ...]
    selected: tuple[str, ...]
    selection_id: str | None
    pending_assignment: dict[str, int]
    assignment: dict[str, int] | None
    assignment_id: str | None
    finish_time: int | None
    position: int
    travel: int
    verified: bool
    done: bool
    remaining_steps: int
    remaining_travel: int
    remaining_work: int
    prices: dict[str, float]
    feedback: dict[str, Any]
    attempts: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        if _fast.enabled():  # identical dict, one conversion pass (extended-04 Phase A)
            return _fast.observation_dict(self)
        return json.loads(json.dumps(asdict(self)))


class DepWorkshop:
    """Actor surface observe()/step(); the hidden spec is evaluator-only.

    Executor signature: executor(primitive, submitted_problem, max_work) -> dict.
    """

    def __init__(self, spec: DepSpec, executor: Callable | None = None, address_seed: int = 0):
        self._spec, self._executor = spec, executor
        self._rows = {x.handle: _item_row(x) for x in spec.items}
        self._edges = {(a, b): w for a, b, w in spec.edges}
        self._req = {"capacity": spec.capacity, "funds": spec.funds,
                     "incompatible": [list(p) for p in spec.incompatible], "deadline": spec.deadline,
                     "slot_capacity": spec.slot_capacity, "closed_slots": sorted(spec.closed_slots)}
        self._versions = {n: 1 for n in REQUIREMENTS}
        self._requirements_version = 1
        self._known: dict[str, dict[str, Any]] = {}
        self._req_known = self._map_known = False
        self._problems: dict[str, dict[str, Any]] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._retrieved: dict[str, dict[str, Any]] = {}
        self._pending: dict[int, str] = {}
        self._selected: tuple[str, ...] = ()
        self._selection_id: str | None = None
        self._pending_assignment: dict[str, int] = {}
        self._assignment: dict[str, int] | None = None
        self._assignment_id: str | None = None
        self._position, self._travel = spec.start, 0
        self._verified = self._done = False
        self._steps = self._work = self._observations = self._calls = 0
        self._compute_units = self._solver_cpu = 0.0
        self._events: list[dict[str, Any]] = []
        self._attempts: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._reductions: list[dict[str, Any]] = []
        self._uses: list[dict[str, Any]] = []
        self._revisions = {"select": 0, "assign": 0}
        self._revocations: list[str] = []
        self._completions = 0
        self._feedback: dict[str, Any] = {"status": "ready"}
        self._handle_rng = random.Random(address_seed)
        for i, f in enumerate(spec.foreign):
            name = f"problem_{i}"
            self._problems[name] = {"primitive": f["primitive"], "problem": deepcopy(f["snapshot"]["problem"]),
                                    "depends_on": deepcopy(f["depends_on"]), "created_step": 0}
            self._record(f["primitive"], name, f["status"], f["payload"], f["certificate_valid"],
                         f["work_units"], 0, deepcopy(f["snapshot"]), deepcopy(f["depends_on"]), label=f["label"])
        self._cache = self.observe()

    # --- accounting -------------------------------------------------------
    def charge_compute(self, units: float) -> None:
        if isinstance(units, bool) or not isinstance(units, (int, float)) or not math.isfinite(units) or units < 0:
            raise ValueError("compute units must be finite nonnegative numbers")
        self._compute_units += units

    # --- public view ------------------------------------------------------
    def observe(self) -> DepObservation:
        s = self._spec
        # Value-identical copies either way; the fast copier skips deepcopy's memo machinery.
        deepcopy = _fast.plain_copy if _fast.enabled() else _deepcopy
        req = None
        if self._req_known:
            req = {k: deepcopy(v) for k, v in self._req.items()}
        records = tuple({k: deepcopy(v) for k, v in r.items() if k != "payload" and not k.startswith("_")}
                        for r in self._records.values())
        retrieved = {h: {k: deepcopy(v) for k, v in r.items() if not k.startswith("_")} for h, r in self._retrieved.items()}
        finish = _finish(self._rows, self._assignment) if self._assignment else None
        return DepObservation(
            VERSION, self._steps,
            {"categories": list(s.categories), "destination": s.destination, "locations": s.locations,
             "slots": s.slots, "max_duration": MAX_DURATION, "call_budgets": list(s.call_budgets),
             "include_remaining_budget": s.include_remaining_budget,
             "finish_by_bounds": list(range(1, s.slots + MAX_DURATION))},
            req, self._requirements_version, dict(self._versions), tuple(deepcopy(self._events)),
            tuple({"handle": x.handle, "category": x.category} for x in s.items), deepcopy(self._known),
            tuple((a, b, w) for (a, b), w in sorted(self._edges.items())) if self._map_known else None,
            deepcopy(self._problems), records, retrieved, tuple(self._pending.values()), self._selected,
            self._selection_id, dict(self._pending_assignment),
            dict(self._assignment) if self._assignment is not None else None, self._assignment_id, finish,
            self._position, self._travel, self._verified, self._done,
            max(0, s.step_limit - self._steps), max(0, s.travel_limit - self._travel), max(0, s.work_limit - self._work),
            {"observation": s.observation_price, "action": s.action_price, "work": s.work_price,
             "travel": s.travel_price, "compute": s.compute_price},
            deepcopy(self._feedback), tuple(deepcopy(self._attempts)))

    def _deps(self, primitive, sel_id=None):
        out = {"requirements_version": {n: self._versions[n] for n in READS[primitive]}}
        if primitive == "csp":
            out["selection_id"] = sel_id
        return out

    def _record(self, primitive, problem, status, payload, certificate_valid, work_units, budget, snapshot,
                depends_on, label=None) -> str:
        handle = f"r{self._handle_rng.getrandbits(64):016x}"
        self._records[handle] = {"handle": handle, "kind": "computation", "primitive": primitive,
                                 "problem": problem, "status": status, "certificate_valid": bool(certificate_valid),
                                 "work_units": work_units, "budget": budget, "problem_snapshot": snapshot,
                                 "depends_on": depends_on, "created_step": self._steps,
                                 "payload": deepcopy(payload), "_label": label}
        return handle

    # --- dynamics ----------------------------------------------------------
    def step(self, action: Action) -> DepObservation:
        if self._done:
            raise RuntimeError("episode finished")
        before = self._cache
        self._steps += 1
        try:
            self._feedback = self._apply(action)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            self._feedback = {"status": "invalid_input", "reason": str(exc)}
        if action.kind in ATTEMPT_KINDS:
            reason = self._feedback.get("reason")
            self._attempts.append({"action_kind": action.kind, "action_key": action_key(action),
                                   "arguments": json.loads(json.dumps(dict(action.arguments))),
                                   "outcome_status": self._feedback.get("status"),
                                   "reason": reason if reason in REASONS else None,
                                   "dependency_versions_at_attempt": relevant_dependencies(before, action),
                                   "step": self._steps})
        if self._feedback.get("status") == "success" and (
                "selection_id" in self._feedback or "assignment_id" in self._feedback):
            self._completions += 1  # successful selection/assignment commit (direct or via use_return)
        ev = self._spec.event
        delivered = self._assignment is not None and self._position == self._spec.destination
        due = ev is not None and (self._completions >= ev[1] if self._spec.event_trigger == "progress"
                                  else self._steps >= ev[1])
        # A scheduled event is moot once the job is delivered (it would only punish speed).
        if due and not self._events and not self._done and not delivered:
            self._feedback = {**self._feedback, "event": self._fire(ev)}
        self._history.append({"action": {"kind": action.kind, "arguments": json.loads(json.dumps(dict(action.arguments)))},
                              "feedback": deepcopy(self._feedback), "step": self._steps,
                              "state_version": self._requirements_version, "work": self._work})
        if self._steps >= self._spec.step_limit:
            self._done = True
        self._cache = self.observe()
        return self._cache

    def _fire(self, ev) -> dict[str, Any]:
        kind, _, arg = ev
        affected = EVENT_AFFECTS[kind]
        self._versions[affected] += 1
        self._requirements_version += 1
        revoked = []
        if kind == "edge_closed":
            self._edges.pop(tuple(arg), None)
        elif kind == "capacity_reduced":
            self._req["capacity"] = arg
            if self._selected and sum(self._rows[h]["weight"] for h in self._selected) > arg:
                revoked = ["selection"] + (["assignment"] if self._assignment is not None else [])
        elif kind == "slot_closed":
            self._req["closed_slots"] = sorted(set(self._req["closed_slots"]) | {arg})
            if self._assignment is not None and arg in self._assignment.values():
                revoked = ["assignment"]
        else:
            self._req["deadline"] = arg
        if "selection" in revoked:
            self._selected, self._selection_id = (), None
            self._pending_assignment = {}
        if "assignment" in revoked:
            self._assignment = self._assignment_id = None
        self._revocations += revoked
        entry = {"kind": kind, "argument": list(arg) if isinstance(arg, tuple) else arg, "step": self._steps,
                 "requirements_version": self._requirements_version, "affected": affected, "revoked": revoked}
        self._events.append(entry)
        return deepcopy(entry)

    def _apply(self, action: Action) -> dict[str, Any]:
        a, kind, s = action.arguments, action.kind, self._spec
        if kind == "inspect":
            target = a["target"]
            if target == "requirements":
                self._req_known = True
            elif target == "map":
                self._map_known = True
            elif target in self._rows:
                self._known[target] = deepcopy(self._rows[target])
            else:
                raise ValueError("unknown inspection target")
            self._observations += 1
            return {"status": "success", "target": target}
        if kind == "start_subset":
            name = self._fresh(a["handle"])
            if not self._known:
                raise ValueError("inspected item required")
            rows = [self._known[x.handle] for x in s.items if x.handle in self._known]
            self._problems[name] = {**subset_problem(rows, self._req), "depends_on": self._deps("constrained_subset"),
                                    "created_step": self._steps}
            return {"status": "success", "problem": name}
        if kind == "start_assign":
            name = self._fresh(a["handle"])
            if not self._selected or not self._req_known:
                return {"status": "rejected", "reason": "missing_dependency"}
            if any(h not in self._known for h in self._selected):
                raise ValueError("selected items must be inspected")
            rows = [self._known[h] for h in self._selected]
            self._problems[name] = {**csp_problem(rows, self._req["closed_slots"]),
                                    "depends_on": self._deps("csp", self._selection_id), "created_step": self._steps}
            return {"status": "success", "problem": name}
        if kind == "build_route":
            name = self._fresh(a["handle"])
            if not self._map_known or self._assignment is None:
                return {"status": "rejected", "reason": "missing_dependency"}
            self._problems[name] = {**route_problem([(u, v, w) for (u, v), w in self._edges.items()], s.locations,
                                                    self._position, s.destination),
                                    "depends_on": self._deps("shortest_path"), "created_step": self._steps}
            return {"status": "success", "problem": name}
        if kind == "add_constraint":
            return self._add_constraint(a)
        if kind == "call":
            return self._call(a)
        if kind == "retrieve":
            record = self._records[a["handle"]]
            self._retrieved[a["handle"]] = deepcopy(record)
            return {"status": "success", "record": {k: deepcopy(v) for k, v in record.items() if not k.startswith("_")}}
        if kind == "use_return":
            return self._use_return(a)
        if kind == "choose_item":
            if a["item"] not in self._rows:
                raise ValueError("unknown item")
            self._pending[self._rows[a["item"]]["category"]] = a["item"]
            return {"status": "success", "pending": list(self._pending.values())}
        if kind == "choose_slot":
            item, slot = a["item"], a["slot"]
            if item not in self._selected:
                return {"status": "rejected", "reason": "missing_dependency"}
            if type(slot) is not int:
                raise ValueError("slot must be an integer")
            self._pending_assignment[item] = slot
            return {"status": "success"}
        if kind == "commit_pending":
            return self._commit_selection(list(self._pending.values()))
        if kind == "commit_assignment":
            if not self._selected:
                return {"status": "rejected", "reason": "missing_dependency"}
            if any(h not in self._pending_assignment for h in self._selected):
                return {"status": "rejected", "reason": "incomplete"}
            return self._commit_assignment({h: self._pending_assignment[h] for h in self._selected})
        if kind == "uncommit":
            target = a["target"]
            if target == "select":
                if not self._selected:
                    return {"status": "rejected", "reason": "missing_dependency"}
                self._selected, self._selection_id = (), None
                self._assignment = self._assignment_id = None
                self._pending_assignment = {}
                self._revisions["select"] += 1
                return {"status": "success", "uncommitted": ["selection", "assignment"]}
            if target == "assign":
                if self._assignment is None:
                    return {"status": "rejected", "reason": "missing_dependency"}
                self._assignment = self._assignment_id = None
                self._revisions["assign"] += 1
                return {"status": "success", "uncommitted": ["assignment"]}
            raise ValueError("unknown uncommit target")
        if kind == "move":
            return self._walk([self._position, a["destination"]])
        if kind == "verify":
            ok, reason = self._goal_check()
            if ok:
                self._verified = self._done = True
                return {"status": "success", "verified": True}
            return {"status": "incomplete", "reason": reason, "verified": False}
        if kind == "think":
            return {"status": "success"}
        if kind == "abstain":
            self._done = True
            return {"status": "unknown"}
        raise ValueError("unknown action")

    def _fresh(self, name):
        if not isinstance(name, str) or name in self._problems:
            raise ValueError("problem handle must be fresh")
        return name

    def _add_constraint(self, a):
        name, c = a["problem"], a["constraint"]
        entry = self._problems[name]
        p, primitive = entry["problem"], entry["primitive"]
        if primitive == "constrained_subset":
            if c not in SUBSET_CONSTRAINTS + ("exclude",):
                raise ValueError("unknown subset constraint")
            if c != "exclude" and not self._req_known:
                return {"status": "rejected", "reason": "missing_dependency"}
            base = set(p["handles"]) | set(p["excluded"])
            rows = [self._rows[x.handle] for x in self._spec.items if x.handle in base]
            constraints = set(p["constraints"]) | ({c} if c != "exclude" else set())
            excluded = set(p["excluded"])
            if c == "exclude":
                if a["item"] not in base:
                    raise ValueError("excluded item must be in the draft")
                excluded.add(a["item"])
            entry.update(subset_problem(rows, self._req, tuple(constraints), tuple(excluded)))
            entry["depends_on"] = self._deps("constrained_subset")
        elif primitive == "csp":
            if c not in ("conflicts", "finish_by"):
                raise ValueError("unknown assignment constraint")
            if not self._req_known:
                return {"status": "rejected", "reason": "missing_dependency"}
            rows = [self._rows[h] for h in p["items"]]
            constraints = set(p["constraints"]) | ({"conflicts"} if c == "conflicts" else set())
            bound = p["finish_by"]
            if c == "finish_by":
                bound = a["bound"]
                if type(bound) is not int or not 1 <= bound < self._spec.slots + MAX_DURATION:
                    raise ValueError("finish_by bound out of declared range")
            entry.update(csp_problem(rows, self._req["closed_slots"], tuple(constraints), bound))
            entry["depends_on"] = {**self._deps("csp"), "selection_id": entry["depends_on"].get("selection_id")}
        else:
            raise ValueError("route drafts take no constraints")
        return {"status": "success", "problem": name, "constraint": c}

    def _call(self, a):
        entry = self._problems[a["problem"]]
        budget = a["budget"]
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError("budget must be positive integer")
        if budget > self._spec.work_limit - self._work:
            return {"status": "unavailable_resource", "reason": "work_budget"}
        if self._executor is None:
            return {"status": "unavailable_resource"}
        start = time.process_time()
        result = self._executor(entry["primitive"], deepcopy(entry["problem"]), budget)
        self._solver_cpu += time.process_time() - start + float(result.get("child_cpu_seconds", 0))
        used = result["work_units"]
        if not isinstance(used, int) or used < 0 or used > budget:
            raise RuntimeError("executor violated work contract")
        self._work += used
        self._calls += 1
        snapshot = {"primitive": entry["primitive"], "problem": deepcopy(entry["problem"])}
        hidden = _hidden_relations(self, snapshot, entry["depends_on"], entry["primitive"])
        self._reductions.append({"primitive": entry["primitive"], "problem": a["problem"],
                                 "certificate_valid": result.get("certificate_valid"), "work_budget": budget,
                                 "correct": hidden["request_match"] and hidden["dependency_match"],
                                 "request_match": hidden["request_match"],
                                 "dependency_match": hidden["dependency_match"]})
        handle = self._record(entry["primitive"], a["problem"], result["status"], result.get("payload"),
                              result.get("certificate_valid", False), used, budget, snapshot,
                              deepcopy(entry["depends_on"]))
        return {"status": result["status"], "return": handle}

    def _use_return(self, a):
        record = self._records[a["handle"]]
        use = a["as"]
        if use not in USES:
            raise ValueError("unknown return use")
        if PRIMITIVE_OF_USE[use] != record["primitive"]:
            return {"status": "rejected", "reason": "type_mismatch"}
        if record["status"] not in USABLE_STATUSES or not record["certificate_valid"] or record["payload"] is None:
            return {"status": "rejected", "reason": "not_applicable"}
        if a["handle"] not in self._retrieved:
            return {"status": "rejected", "reason": "retrieve_required"}
        audit = _hidden_relations(self, record["problem_snapshot"], record["depends_on"], record["primitive"])
        entry = {"handle": a["handle"], "as": use, "step": self._steps, "foreign": record["_label"] is not None,
                 "label": record["_label"], "applicable_hidden": audit["applicable"],
                 "request_match_hidden": audit["request_match"], "dependency_match_hidden": audit["dependency_match"]}
        payload, snap = record["payload"], record["problem_snapshot"]["problem"]
        if use == "select":
            out = self._commit_selection([snap["handles"][i] for i in payload[0]])
        elif use == "assign":
            if not self._selected:
                out = {"status": "rejected", "reason": "missing_dependency"}
            elif sorted(snap["items"]) != sorted(self._selected):
                out = {"status": "rejected", "reason": "stale_dependency"}
            else:
                out = self._commit_assignment(dict(zip(snap["items"], payload)))
        else:
            path = list(payload[0])
            if self._assignment is None:
                out = {"status": "rejected", "reason": "missing_dependency"}
            elif path[0] != self._position:
                out = {"status": "rejected", "reason": "stale_dependency"}
            else:
                out = self._walk(path)
        entry["outcome"] = out.get("status")
        entry["reason"] = out.get("reason")
        self._uses.append(entry)
        return out

    def _commit_selection(self, handles):
        if self._selected:
            return {"status": "rejected", "reason": "already_committed"}
        if not self._req_known:
            return {"status": "rejected", "reason": "missing_dependency"}
        ok, reason = _validate_selection(self._rows, self._spec.categories, self._req, handles)
        if not ok:
            return {"status": "rejected", "reason": reason}
        self._selected = tuple(sorted(handles, key=lambda h: self._rows[h]["category"]))
        self._selection_id = selection_id(handles)
        self._pending_assignment = {}
        return {"status": "success", "selection_id": self._selection_id}

    def _commit_assignment(self, assignment):
        if not self._selected:
            return {"status": "rejected", "reason": "missing_dependency"}
        if self._assignment is not None:
            return {"status": "rejected", "reason": "already_committed"}
        ok, reason = _validate_assignment(self._rows, self._selected, self._req["closed_slots"], assignment)
        if not ok:
            return {"status": "rejected", "reason": reason}
        finish = _finish(self._rows, assignment)
        if finish + self._travel > self._req["deadline"]:
            return {"status": "rejected", "reason": "deadline"}
        self._assignment = dict(assignment)
        self._assignment_id = assignment_id(self._selection_id, assignment)
        return {"status": "success", "assignment_id": self._assignment_id, "finish_time": finish}

    def _walk(self, path):
        if not self._selected or self._assignment is None:
            return {"status": "rejected", "reason": "missing_dependency"}
        if not isinstance(path, list) or len(path) < 2 or path[0] != self._position:
            return {"status": "rejected", "reason": "stale_dependency"}
        if any((u, v) not in self._edges for u, v in zip(path, path[1:])):
            return {"status": "rejected", "reason": "missing_edge"}
        total = sum(self._edges[(u, v)] for u, v in zip(path, path[1:]))
        if _finish(self._rows, self._assignment) + self._travel + total > self._req["deadline"]:
            return {"status": "rejected", "reason": "deadline"}
        if self._travel + total > self._spec.travel_limit:
            return {"status": "rejected", "reason": "travel_budget"}
        self._travel += total
        self._position = path[-1]
        return {"status": "success", "position": self._position,
                "delivered": self._position == self._spec.destination}

    def _goal_check(self):
        if not self._selected:
            return False, "missing_dependency"
        ok, reason = _validate_selection(self._rows, self._spec.categories, self._req, list(self._selected))
        if not ok:
            return False, reason
        if self._assignment is None:
            return False, "missing_dependency"
        ok, reason = _validate_assignment(self._rows, self._selected, self._req["closed_slots"], self._assignment)
        if not ok:
            return False, reason
        if self._position != self._spec.destination:
            return False, "incomplete"
        if _finish(self._rows, self._assignment) + self._travel > self._req["deadline"]:
            return False, "deadline"
        return True, "valid"

    # --- evaluator ----------------------------------------------------------
    def _cost(self) -> float:
        s = self._spec
        return (self._steps * s.action_price + self._travel * s.travel_price + self._observations * s.observation_price
                + self._work * s.work_price + self._compute_units * s.compute_price)

    def current_utility(self) -> float:
        """Exactly evaluate()["utility"] (same expression), without copying the history."""
        return float(self._verified) - self._cost()

    def evaluate(self) -> dict[str, Any]:
        s = self._spec
        cost = self._cost()
        uses = self._uses
        return {"verified_success": self._verified, "utility": float(self._verified) - cost, "cost": cost,
                "steps": self._steps, "observations": self._observations, "work_units": self._work,
                "calls": self._calls, "solver_cpu_seconds": self._solver_cpu, "travel_distance": self._travel,
                "compute_units": self._compute_units, "modeled_compute_cost": self._compute_units * s.compute_price,
                "revisions": dict(self._revisions), "revocations": list(self._revocations),
                "events": deepcopy(self._events),
                "reuse": {"uses": len(uses), "applicable_uses": sum(u["applicable_hidden"] for u in uses),
                          "invalid_uses": sum(not u["applicable_hidden"] for u in uses),
                          "foreign_uses": sum(u["foreign"] for u in uses),
                          "foreign_applicable_uses": sum(u["foreign"] and u["applicable_hidden"] for u in uses),
                          "stale_own_uses": sum(not u["applicable_hidden"] and not u["foreign"] for u in uses)},
                "reuse_audit": deepcopy(uses), "reductions": deepcopy(self._reductions),
                "history": deepcopy(self._history)}


# ---------------------------------------------------------------------------
# Applicability (public rule) and evaluator audit (hidden; logging only)
# ---------------------------------------------------------------------------

def current_request(o: DepObservation, primitive: str, options: dict | None = None):
    """The instance the current *public* dependencies would produce (or None if
    the public state does not determine it: uninspected inputs, no selection)."""
    options = options or {}
    if primitive == "constrained_subset":
        if o.requirements is None or len(o.known_items) < len(o.item_inventory):
            return None
        rows = [o.known_items[r["handle"]] for r in o.item_inventory]
        excluded = tuple(options.get("excluded", ()))
        if any(h not in o.known_items for h in excluded):
            return None
        return subset_problem(rows, o.requirements, SUBSET_CONSTRAINTS, excluded)
    if primitive == "csp":
        if o.requirements is None or not o.selected or any(h not in o.known_items for h in o.selected):
            return None
        rows = [o.known_items[h] for h in o.selected]
        return csp_problem(rows, o.requirements["closed_slots"], CSP_CONSTRAINTS, options.get("finish_by"))
    if primitive == "shortest_path":
        if o.known_edges is None:
            return None
        return route_problem(o.known_edges, o.goal["locations"], o.position, o.goal["destination"])
    return None


def current_dependencies(o: DepObservation, primitive: str) -> dict:
    out = {"requirements_version": {n: o.requirement_versions[n] for n in READS[primitive]}}
    if primitive == "csp":
        out["selection_id"] = o.selection_id
    return out


def record_usable(record) -> bool:
    return record.get("status") in USABLE_STATUSES and bool(record.get("certificate_valid"))


def relations(o: DepObservation, record: dict, primitive: str | None = None) -> dict[str, bool]:
    """Component applicability relations, all computed from public state."""
    p = primitive or record.get("primitive")
    type_match = record.get("primitive") == p
    snap = record.get("problem_snapshot")
    req = current_request(o, p, snapshot_options(snap)) if type_match else None
    canonical = current_request(o, p) if type_match else None
    deps = record.get("depends_on") or {}
    cur = current_dependencies(o, p) if p in READS else {}
    return {"type_match": type_match,
            "request_match": req is not None and snap == req,
            "canonical_match": canonical is not None and snap == canonical,
            "dependency_match": type_match and deps == cur,
            "requirements_match": type_match and deps.get("requirements_version") == cur.get("requirements_version"),
            "selection_match": type_match and deps.get("selection_id") == cur.get("selection_id"),
            "usable": record_usable(record)}


def applicable(observation: DepObservation, record: dict, primitive: str | None = None) -> bool:
    """Evaluator/teacher applicability rule, from PUBLIC observation only:
    type match AND request match AND dependency match AND usable status+certificate.
    Request match compares the snapshot with the instance the current public
    dependencies would produce under the snapshot's own declared options
    (subset exclusions, csp finish_by bound)."""
    r = relations(observation, record, primitive)
    return r["type_match"] and r["request_match"] and r["dependency_match"] and r["usable"]


def _hidden_request(env: DepWorkshop, primitive, options):
    s = env._spec
    if primitive == "constrained_subset":
        rows = [env._rows[x.handle] for x in s.items]
        if any(h not in env._rows for h in options.get("excluded", ())):
            return None
        return subset_problem(rows, env._req, SUBSET_CONSTRAINTS, tuple(options.get("excluded", ())))
    if primitive == "csp":
        if not env._selected:
            return None
        return csp_problem([env._rows[h] for h in env._selected], env._req["closed_slots"], CSP_CONSTRAINTS,
                           options.get("finish_by"))
    return route_problem([(u, v, w) for (u, v), w in env._edges.items()], s.locations, env._position, s.destination)


def _hidden_relations(env: DepWorkshop, snapshot, depends_on, primitive):
    req = _hidden_request(env, primitive, snapshot_options(snapshot))
    deps = {"requirements_version": {n: env._versions[n] for n in READS[primitive]}}
    if primitive == "csp":
        deps["selection_id"] = env._selection_id
    rm = req is not None and snapshot == req
    dm = depends_on == deps
    return {"request_match": rm, "dependency_match": dm, "applicable": rm and dm}


def audit_record(env: DepWorkshop, handle: str, primitive: str | None = None) -> dict[str, Any]:
    """EVALUATOR-ONLY: recompute applicability and payload correctness from the
    hidden spec/state. For logging and tests; never exposed to actors."""
    r = env._records[handle]
    p = primitive or r["primitive"]
    type_match = r["primitive"] == p
    rel = _hidden_relations(env, r["problem_snapshot"], r["depends_on"], p) if type_match else \
        {"request_match": False, "dependency_match": False, "applicable": False}
    usable = record_usable(r) and r["payload"] is not None
    return {"type_match": type_match, **{k: rel[k] for k in ("request_match", "dependency_match")},
            "usable": usable, "applicable_hidden": type_match and rel["applicable"] and usable,
            "payload_valid_now": type_match and usable and _payload_valid_now(env, r),
            "label": r["_label"]}


def _payload_valid_now(env: DepWorkshop, r) -> bool:
    try:
        snap, payload = r["problem_snapshot"]["problem"], r["payload"]
        if r["primitive"] == "constrained_subset":
            handles = [snap["handles"][i] for i in payload[0]]
            return _validate_selection(env._rows, env._spec.categories, env._req, handles)[0]
        if r["primitive"] == "csp":
            if sorted(snap["items"]) != sorted(env._selected):
                return False
            return _validate_assignment(env._rows, env._selected, env._req["closed_slots"],
                                        dict(zip(snap["items"], payload)))[0]
        path, dist = payload
        if path[0] != env._position or path[-1] != env._spec.destination:
            return False
        if any((u, v) not in env._edges for u, v in zip(path, path[1:])):
            return False
        best = _dijkstra([(u, v, w) for (u, v), w in env._edges.items()], env._spec.locations, env._position,
                         env._spec.destination)[1]
        return sum(env._edges[(u, v)] for u, v in zip(path, path[1:])) == best
    except (KeyError, TypeError, ValueError, IndexError):
        return False


# ---------------------------------------------------------------------------
# Attempted-action record: relevant dependency versions (public)
# ---------------------------------------------------------------------------

def relevant_dependencies(o: DepObservation, action: Action) -> dict[str, Any]:
    """Public state an attempt of this action depends on. The environment stores
    it at attempt time; encoders recompute it to answer "changed since?"."""
    k, a = action.kind, action.arguments
    rv = o.requirement_versions

    def req(*names):
        return {n: rv[n] for n in names}
    if k == "commit_pending":
        return {"pending": sorted(o.pending), "requirements": req("capacity", "funds", "incompatible"),
                "selection_id": o.selection_id}
    if k == "commit_assignment":
        return {"pending_assignment": sorted([h, s] for h, s in o.pending_assignment.items()),
                "requirements": req("slots", "deadline"), "selection_id": o.selection_id,
                "assignment_id": o.assignment_id, "travel": o.travel}
    if k == "use_return":
        base = {"retrieved": a.get("handle") in o.retrieved}
        use = a.get("as")
        if use == "select":
            return {**base, "requirements": req("capacity", "funds", "incompatible"), "selection_id": o.selection_id}
        if use == "assign":
            return {**base, "requirements": req("slots", "deadline"), "selection_id": o.selection_id,
                    "assignment_id": o.assignment_id, "travel": o.travel}
        return {**base, "requirements": req("map", "deadline"), "assignment_id": o.assignment_id,
                "position": o.position, "travel": o.travel}
    if k == "move":
        return {"requirements": req("map", "deadline"), "selection_id": o.selection_id,
                "assignment_id": o.assignment_id, "position": o.position, "travel": o.travel}
    if k == "call":
        entry = o.problems.get(a.get("problem"), {})
        return {"draft": _digest([entry.get("problem"), entry.get("depends_on")])[:16],
                "remaining_work": o.remaining_work >= a.get("budget", 0)}
    if k == "uncommit":
        return {"selection_id": o.selection_id, "assignment_id": o.assignment_id}
    if k == "verify":
        return {"selection_id": o.selection_id, "assignment_id": o.assignment_id, "position": o.position,
                "travel": o.travel, "requirements_version": o.requirements_version}
    return {}


# ---------------------------------------------------------------------------
# Public catalogue and d1 encoder
# ---------------------------------------------------------------------------

def call_budgets(o: DepObservation) -> tuple[int, ...]:
    extra = (o.remaining_work,) if o.goal.get("include_remaining_budget") and o.remaining_work > 0 else ()
    return tuple(b for b in dict.fromkeys(tuple(o.goal["call_budgets"]) + extra) if b <= o.remaining_work)


def action_catalog(o: DepObservation) -> list[Action]:
    """Syntactic availability only (information/data preconditions), no hidden masks."""
    if o.done:
        return []
    acts = [Action("verify"), Action("think"), Action("abstain"), Action("commit_pending"),
            Action("commit_assignment"), Action("uncommit", {"target": "select"}),
            Action("uncommit", {"target": "assign"}),
            Action("inspect", {"target": "requirements"}), Action("inspect", {"target": "map"})]
    acts += [Action("inspect", {"target": r["handle"]}) for r in o.item_inventory]
    acts += [Action("choose_item", {"item": r["handle"]}) for r in o.item_inventory]
    name = f"problem_{len(o.problems)}"
    if o.known_items:
        acts.append(Action("start_subset", {"handle": name}))
    if o.selected:
        acts.append(Action("start_assign", {"handle": name}))
        acts += [Action("choose_slot", {"item": h, "slot": s}) for h in o.selected
                 for s in o.known_items.get(h, {}).get("slots", [])]
    if o.known_edges is not None:
        acts += [Action("move", {"destination": v}) for u, v, w in o.known_edges if u == o.position]
        if o.assignment is not None:
            acts.append(Action("build_route", {"handle": name}))
    budgets = call_budgets(o)
    for handle, p in o.problems.items():
        if p["primitive"] == "constrained_subset":
            acts += [Action("add_constraint", {"problem": handle, "constraint": c}) for c in SUBSET_CONSTRAINTS]
            acts += [Action("add_constraint", {"problem": handle, "constraint": "exclude", "item": h})
                     for h in p["problem"]["handles"]]
        elif p["primitive"] == "csp":
            acts.append(Action("add_constraint", {"problem": handle, "constraint": "conflicts"}))
            acts += [Action("add_constraint", {"problem": handle, "constraint": "finish_by", "bound": b})
                     for b in o.goal["finish_by_bounds"]]
        acts += [Action("call", {"problem": handle, "budget": b}) for b in budgets]
    for r in o.records:
        acts.append(Action("retrieve", {"handle": r["handle"]}))
        acts += [Action("use_return", {"handle": r["handle"], "as": u}) for u in USES]
    return acts


def _log(x, top=12.0):
    return math.log2(1 + max(0.0, float(x))) / top


class _Context:
    """Per-observation public derived quantities shared by candidate features."""

    def __init__(self, o: DepObservation, applicability: bool = True):
        self.o = o
        req = o.requirements
        self.D = req["deadline"] if req else None
        self.cap = req["capacity"] if req else None
        self.funds = req["funds"] if req else None
        self.closed = set(req["closed_slots"]) if req else set()
        self.T = o.finish_time
        self.records = {r["handle"]: r for r in o.records}
        self.rel_cache: dict = {}
        self.last_attempt: dict = {}
        self.attempt_count: dict = {}
        for at in o.attempts:
            self.last_attempt[at["action_key"]] = at
            self.attempt_count[at["action_key"]] = self.attempt_count.get(at["action_key"], 0) + 1
        dists = [o.retrieved[h]["payload"][1] for h, r in self.records.items()
                 if h in o.retrieved and r["primitive"] == "shortest_path" and o.retrieved[h].get("payload")
                 and self.rel(r)["request_match"] and self.rel(r)["dependency_match"] and self.rel(r)["usable"]]
        # d1-noapp: the route bound is selected by request/dependency match, so the
        # ablation drops it (need_bound falls back to deadline - travel downstream).
        self.best_route = min(dists) if dists and applicability else None
        self.need_bound = (self.D - o.travel - self.best_route) if (self.best_route is not None and self.D is not None) else None
        pend = [o.known_items.get(h, {}) for h in o.pending]
        self.pending_weight = sum(r.get("weight", 0) for r in pend)
        self.pending_price = sum(r.get("price", 0) for r in pend)

    def rel(self, record, primitive=None):
        key = (record["handle"], primitive)
        if key not in self.rel_cache:
            self.rel_cache[key] = relations(self.o, record, primitive)
        return self.rel_cache[key]


def encode_observation_d1(o: DepObservation, ctx: _Context | None = None) -> list[float]:
    c = ctx or _Context(o)
    req = o.requirements
    known = req is not None
    D = c.D or 0
    T = o.finish_time or 0
    last = o.events[-1] if o.events else None
    fb = o.feedback
    rejected = sum(a["outcome_status"] == "rejected" for a in o.attempts)
    uncommits = sum(a["action_kind"] == "uncommit" and a["outcome_status"] == "success" for a in o.attempts)
    return ([float(known), (c.cap or 0) / 64, (c.funds or 0) / 64, D / 32, (req["slot_capacity"] if known else 0) / 4,
             len(c.closed) / 8, (len(req["incompatible"]) if known else 0) / 16, o.requirements_version / 8]
            + [float(o.requirement_versions[n] > 1) for n in REQUIREMENTS]
            + [float(o.selection_id is not None), float(o.assignment_id is not None), T / 16,
               o.position / 32, o.goal["destination"] / 32, float(o.position == o.goal["destination"]), o.travel / 32,
               (D - T - o.travel) / 16 if known and o.assignment is not None else 0.0,
               float(c.best_route is not None), (c.best_route or 0) / 32,
               float(c.best_route is not None and o.assignment is not None and T + o.travel + c.best_route <= D),
               (c.need_bound or 0) / 16,
               len(o.known_items) / max(1, len(o.item_inventory)), float(o.known_edges is not None),
               len(o.problems) / 16, len(o.records) / 16, len(o.retrieved) / 16, len(o.pending) / 8,
               len(o.pending_assignment) / 8, c.pending_weight / 64, c.pending_price / 64,
               ((c.cap or 0) - c.pending_weight) / 64, ((c.funds or 0) - c.pending_price) / 64]
            + [float(last is not None and last["kind"] == k) for k in EVENT_KINDS]
            + [float(last is None), float(last is not None and "selection" in last["revoked"]),
               float(last is not None and "assignment" in last["revoked"]), float("event" in fb),
               _log(o.remaining_work, 14), _log(o.remaining_steps, 7), o.remaining_travel / 64,
               o.prices["action"] * 100, o.prices["observation"] * 100, o.prices["work"] * 1000,
               o.prices["travel"] * 100, o.prices["compute"] * 100]
            + [float(fb.get("status") == s) for s in OUTCOMES[:-1]]
            + [float(fb.get("status") not in OUTCOMES[:-1])]
            + [float(fb.get("reason") == r) for r in REASONS]
            + [len(o.attempts) / 32, rejected / 16, uncommits / 4])


def _action_primitive(o, c, action):
    a, k = action.arguments, action.kind
    if k in ("call", "add_constraint"):
        return o.problems.get(a.get("problem"), {}).get("primitive")
    if k in ("retrieve", "use_return"):
        return c.records.get(a.get("handle"), {}).get("primitive")
    return {"start_subset": "constrained_subset", "start_assign": "csp", "build_route": "shortest_path"}.get(k)


def _payload_facts(o, c, record):
    """Facts of a retrieved payload relative to current public requirements (8)."""
    got = o.retrieved.get(record["handle"])
    if not got or got.get("payload") is None:
        return [0.0] * 8
    payload, snap, p = got["payload"], got["problem_snapshot"]["problem"], record["primitive"]
    try:
        if p == "shortest_path":
            path, dist = payload
            meets = (o.assignment is not None and c.D is not None and (o.finish_time or 0) + o.travel + dist <= c.D)
            return [1.0, dist / 32, float(path[0] == o.position), float(path[-1] == o.goal["destination"]),
                    float(meets), 0.0, 0.0, 0.0]
        if p == "csp":
            rows = {h: d for h, d in zip(snap["items"], snap["durations"])}
            finish = max(s + rows[h] for h, s in zip(snap["items"], payload))
            limit = c.need_bound if c.need_bound is not None else ((c.D - o.travel) if c.D is not None else None)
            return [1.0, 0.0, 0.0, 0.0, 0.0, finish / 16, float(limit is not None and finish <= limit),
                    float(sorted(snap["items"]) == sorted(o.selected))]
        handles = [snap["handles"][i] for i in payload[0]]
        rows = [o.known_items.get(h) for h in handles]
        if any(r is None for r in rows) or o.requirements is None:
            return [1.0, 0, 0, 0, 0, 0, 0, 0]
        ok = _validate_selection({r["handle"]: r for r in rows}, o.goal["categories"], o.requirements, handles)[0]
        return [1.0, 0.0, 0.0, 0.0, 0.0, sum(r["duration"] for r in rows) / 16, float(ok), 0.0]
    except (KeyError, TypeError, ValueError, IndexError):
        return [1.0] + [0.0] * 7


def _draft_match(o, entry):
    snap = {"primitive": entry["primitive"], "problem": entry["problem"]}
    req = current_request(o, entry["primitive"], snapshot_options(snap))
    return float(req is not None and snap == req), float(entry.get("depends_on") == current_dependencies(o, entry["primitive"]))


def encode_action_d1(o: DepObservation, action: Action, ctx: _Context | None = None) -> list[float]:
    """d1 candidate features: relations and public facts only. No handle spelling,
    no hidden labels, no gold actions, no single 'applicable' bit (components only)."""
    c = ctx or _Context(o)
    a, k = action.arguments, action.kind
    prim = _action_primitive(o, c, action)
    use = a.get("as") if k == "use_return" else (a.get("target") if k == "uncommit" else None)
    tgt = a.get("target") if k == "inspect" else None
    out = [float(k == x) for x in KINDS]
    out += [float(use == u) for u in USES]
    out += [float(tgt is not None and tgt not in ("map", "requirements")), float(tgt == "map"), float(tgt == "requirements")]
    out += [float(prim == p) for p in PRIMITIVES]
    D, T = c.D, o.finish_time
    # --- record block (retrieve / use_return) : 24
    rec = c.records.get(a.get("handle")) if k in ("retrieve", "use_return") else None
    if rec is not None:
        want = PRIMITIVE_OF_USE.get(a.get("as")) if k == "use_return" else rec["primitive"]
        rel = c.rel(rec, want)
        opts = snapshot_options(rec.get("problem_snapshot"))
        out += [1.0, float(rel["type_match"]), float(rel["request_match"]), float(rel["canonical_match"]),
                float(rel["dependency_match"]), float(rel["requirements_match"]), float(rel["selection_match"]),
                float(rel["usable"])]
        out += [float(rec.get("status") == s) for s in RECORD_STATUSES]
        out += [float(rec.get("certificate_valid", False)), float(rec["handle"] in o.retrieved),
                float(opts.get("finish_by") is not None), (opts.get("finish_by") or 0) / 16,
                len(opts.get("excluded", ())) / 4]
        out += _payload_facts(o, c, rec)
    else:
        out += [0.0] * 26
    # --- draft block (call / add_constraint) : 24
    entry = o.problems.get(a.get("problem")) if k in ("call", "add_constraint") else None
    if entry is not None:
        p = entry["problem"]
        rm, dm = _draft_match(o, entry)
        cons = p.get("constraints", [])
        snap = {"primitive": entry["primitive"], "problem": p}
        same = [r for r in o.records if r.get("problem_snapshot") == snap]
        cur = [r for r in same if r.get("depends_on") == entry.get("depends_on")]
        budget = a.get("budget", 0)
        most = max((r.get("work_units", 0) for r in same), default=0)
        size = len(p.get("handles", p.get("items", p.get("edges", []))))
        out += [1.0, rm, dm] + [float(x in cons) for x in ("capacity", "funds", "incompatibility", "conflicts")]
        out += [float(p.get("finish_by") is not None), (p.get("finish_by") or 0) / 16, len(p.get("excluded", [])) / 4,
                _log(size, 5)]
        if k == "call":
            out += [_log(budget, 14), float(budget > most),
                    float(any(r.get("status") == "timeout" and r.get("budget", 0) >= budget for r in same)),
                    sum(record_usable(r) for r in cur) / 4, float(any(r.get("status") == "infeasible" for r in cur)),
                    float(budget == o.remaining_work)]
        else:
            out += [0.0] * 6
        if k == "add_constraint":
            cname = a.get("constraint")
            bound = a.get("bound")
            item = o.known_items.get(a.get("item"), {}) if cname == "exclude" else {}
            alternatives = sum(1 for r in o.item_inventory if item and r["category"] == item.get("category")
                               and r["handle"] not in p.get("excluded", []) and r["handle"] != a.get("item"))
            out += [float(cname == x) for x in CONSTRAINT_NAMES]
            out += [float(cname in cons or (cname == "finish_by" and p.get("finish_by") == bound)),
                    (bound or 0) / 16, float(bound is not None and c.need_bound is not None and bound <= c.need_bound),
                    item.get("duration", 0) / 4, alternatives / 4]
        else:
            out += [0.0] * 11
    else:
        out += [0.0] * 28
    # --- start block : 3
    if k in ("start_subset", "start_assign", "build_route"):
        drafts = [e for e in o.problems.values() if e["primitive"] == prim]
        matching = [e for e in drafts if _draft_match(o, e) == (1.0, 1.0)]
        live = [r for r in o.records if r["primitive"] == prim and c.rel(r)["request_match"]
                and c.rel(r)["dependency_match"] and c.rel(r)["usable"]]
        out += [len(drafts) / 8, len(matching) / 4, len(live) / 4]
    else:
        out += [0.0] * 3
    # --- item block (inspect item / choose_item) : 14
    h = a.get("item", a.get("target")) if k in ("inspect", "choose_item") else None
    row = next((r for r in o.item_inventory if r["handle"] == h), None) if h is not None else None
    if row is not None:
        known = o.known_items.get(h, {})
        peers = [o.known_items[r["handle"]] for r in o.item_inventory
                 if r["category"] == row["category"] and r["handle"] in o.known_items]
        drank = sum(p["duration"] < known["duration"] for p in peers) if known else 0
        crank = sum(p["weight"] + p["price"] < known["weight"] + known["price"] for p in peers) if known else 0
        conflicts = [y if x == h else x for x, y in (o.requirements or {}).get("incompatible", []) if h in (x, y)]
        pend_cats = {o.known_items.get(x, {}).get("category") for x in o.pending}
        out += [1.0, float(bool(known)), known.get("weight", 0) / 16, known.get("price", 0) / 16,
                known.get("duration", 0) / 4, len(known.get("slots", [])) / 8, float(h in o.pending),
                float(h in o.selected), float(row["category"] in pend_cats),
                float(bool(known) and c.cap is not None and known["weight"] <= c.cap - c.pending_weight),
                float(bool(known) and c.funds is not None and known["price"] <= c.funds - c.pending_price),
                float(any(x in o.pending for x in conflicts)), drank / 4, crank / 4]
    else:
        out += [0.0] * 14
    # --- slot block (choose_slot) : 10
    if k == "choose_slot" and a.get("item") in o.known_items:
        it = o.known_items[a["item"]]
        s, d = a["slot"], it["duration"]
        clash = any(overlaps(s, d, t, o.known_items[x]["duration"]) for x, t in o.pending_assignment.items()
                    if x != a["item"] and x in o.selected and x in o.known_items)
        finish = s + d
        limit = c.need_bound if c.need_bound is not None else ((D - o.travel) if D is not None else None)
        out += [1.0, d / 4, s / 16, float(s in it["slots"]), float(s in c.closed), float(clash), finish / 16,
                float(limit is not None and finish <= limit), float(o.pending_assignment.get(a["item"]) == s),
                float(a["item"] in o.pending_assignment)]
    else:
        out += [0.0] * 10
    # --- move block : 7
    if k == "move" and o.known_edges is not None:
        v = a["destination"]
        w = next((w for x, y, w in o.known_edges if x == o.position and y == v), 0)
        ahead = [w2 for x, y, w2 in o.known_edges if x == v]
        direct = any(x == v and y == o.goal["destination"] for x, y, _ in o.known_edges)
        meets = T is not None and D is not None and T + o.travel + w <= D
        out += [1.0, w / 32, float(v == o.goal["destination"]), float(direct), min(ahead, default=0) / 32,
                float(meets), (v - o.position) / 8]
    else:
        out += [0.0] * 7
    # --- commit / uncommit block : 10
    if k == "commit_pending":
        rows = [o.known_items.get(x) for x in o.pending]
        full = all(r is not None for r in rows) and o.requirements is not None
        ok_w = full and sum(r["weight"] for r in rows) <= c.cap
        ok_p = full and sum(r["price"] for r in rows) <= c.funds
        inc = full and any(x in o.pending and y in o.pending for x, y in o.requirements["incompatible"])
        out += [len(o.pending) / max(1, len(o.goal["categories"])), float(len(o.pending) == len(o.goal["categories"])),
                float(ok_w), float(ok_p), float(inc), 0.0, 0.0, 0.0, 0.0, 0.0]
    elif k == "commit_assignment":
        sel = [x for x in o.selected if x in o.known_items]
        cover = all(x in o.pending_assignment for x in o.selected) and bool(o.selected)
        allowed = cover and all(o.pending_assignment[x] in o.known_items[x]["slots"] and
                                o.pending_assignment[x] not in c.closed for x in sel)
        clash = cover and any(overlaps(o.pending_assignment[x], o.known_items[x]["duration"],
                                       o.pending_assignment[y], o.known_items[y]["duration"])
                              for i, x in enumerate(sel) for y in sel[i + 1:])
        finish = max((o.pending_assignment[x] + o.known_items[x]["duration"] for x in sel if x in o.pending_assignment),
                     default=0)
        limit = c.need_bound if c.need_bound is not None else ((D - o.travel) if D is not None else None)
        out += [0.0, 0.0, 0.0, 0.0, 0.0, float(cover), float(allowed), float(clash), finish / 16,
                float(cover and limit is not None and finish <= limit)]
    elif k == "uncommit":
        committed = o.selection_id is not None if a.get("target") == "select" else o.assignment_id is not None
        out += [float(committed)] + [0.0] * 9
    else:
        out += [0.0] * 10
    # --- attempted-action block : 2 + outcomes + reasons + 1
    if k in ATTEMPT_KINDS:
        key = action_key(action)
        last = c.last_attempt.get(key)
        if last is not None:
            status = last["outcome_status"]
            out += [1.0, min(c.attempt_count[key], 8) / 4]
            out += [float(status == s) for s in OUTCOMES[:-1]] + [float(status not in OUTCOMES[:-1])]
            out += [float(last["reason"] == r) for r in REASONS]
            out += [float(relevant_dependencies(o, action) != last["dependency_versions_at_attempt"])]
        else:
            out += [0.0] * ATTEMPT_BLOCK
    else:
        out += [0.0] * ATTEMPT_BLOCK
    return out


# Semantic coordinate names, in encoder order (documentation, ablation masks, preflight audit).
OBSERVATION_NAMES_D1 = (
    ("req.known", "req.capacity", "req.funds", "req.deadline", "req.slot_capacity", "req.n_closed_slots",
     "req.n_incompatible", "req.version")
    + tuple(f"req.changed.{n}" for n in REQUIREMENTS)
    + ("plan.selection_committed", "plan.assignment_committed", "plan.finish_time", "plan.position",
       "plan.destination", "plan.arrived", "plan.travel", "plan.deadline_slack",
       "route.applicable_known", "route.distance", "route.plan_meets_deadline", "route.need_bound",
       "progress.items_inspected", "progress.map_known",
       "count.problems", "count.records", "count.retrieved", "count.pending", "count.pending_assignment",
       "pending.weight", "pending.price", "pending.capacity_left", "pending.funds_left")
    + tuple(f"event.last.{k}" for k in EVENT_KINDS)
    + ("event.none", "event.selection_revoked", "event.assignment_revoked", "event.this_step",
       "res.work", "res.steps", "res.travel", "price.action", "price.observation", "price.work", "price.travel",
       "price.compute")
    + tuple(f"feedback.status.{s}" for s in OUTCOMES[:-1]) + ("feedback.status.other",)
    + tuple(f"feedback.reason.{r}" for r in REASONS)
    + ("attempts.count", "attempts.rejected", "attempts.uncommits"))
CANDIDATE_NAMES_D1 = (
    tuple(f"kind.{k}" for k in KINDS) + tuple(f"use.{u}" for u in USES)
    + ("inspect.item", "inspect.map", "inspect.requirements") + tuple(f"primitive.{p}" for p in PRIMITIVES)
    + ("record.present", "record.type_match", "record.request_match", "record.canonical_match",
       "record.dependency_match", "record.requirements_match", "record.selection_match", "record.usable")
    + tuple(f"record.status.{s}" for s in RECORD_STATUSES)
    + ("record.certificate_valid", "record.retrieved", "record.finish_by_option", "record.finish_by_value",
       "record.n_excluded",
       "payload.present", "payload.route_distance", "payload.route_starts_at_position",
       "payload.route_ends_at_destination", "payload.route_meets_deadline", "payload.finish_or_total_duration",
       "payload.meets_bound_or_selection_valid", "payload.csp_items_equal_selection",
       "draft.present", "draft.request_match", "draft.dependency_match", "draft.has_capacity", "draft.has_funds",
       "draft.has_incompatibility", "draft.has_conflicts", "draft.finish_by_option", "draft.finish_by_value",
       "draft.n_excluded", "draft.size",
       "call.budget", "call.budget_exceeds_most_work", "call.timed_out_at_budget", "call.usable_same_records",
       "call.infeasible_same_record", "call.budget_is_remaining")
    + tuple(f"constraint.{c}" for c in CONSTRAINT_NAMES)
    + ("constraint.already_present", "constraint.bound", "constraint.bound_within_need",
       "constraint.excluded_duration", "constraint.category_alternatives",
       "start.n_drafts", "start.matching_drafts", "start.live_applicable_records",
       "item.present", "item.known", "item.weight", "item.price", "item.duration", "item.n_slots", "item.pending",
       "item.selected", "item.category_pending", "item.fits_capacity", "item.fits_funds", "item.conflicts_pending",
       "item.duration_rank", "item.cost_rank",
       "slot.present", "slot.duration", "slot.slot", "slot.allowed", "slot.closed", "slot.clash", "slot.finish",
       "slot.meets_bound", "slot.same_as_pending", "slot.item_has_pending",
       "move.present", "move.edge_weight", "move.is_destination", "move.direct_to_destination",
       "move.min_outgoing", "move.meets_deadline", "move.step",
       # commit_pending fills the first five, commit_assignment the last five; uncommit fills only the
       # first ("is the uncommit target committed").
       "commit.coverage_or_uncommit_target_committed", "commit.complete", "commit.weight_ok", "commit.price_ok",
       "commit.incompatibility", "commit.assign_cover", "commit.assign_allowed", "commit.assign_clash",
       "commit.assign_finish", "commit.assign_meets_bound",
       "attempt.attempted", "attempt.count")
    + tuple(f"attempt.last_outcome.{s}" for s in OUTCOMES[:-1]) + ("attempt.last_outcome.other",)
    + tuple(f"attempt.last_reason.{r}" for r in REASONS) + ("attempt.dependencies_changed",))

# P1 input ablations (protocol-P1 arms X3/X4). Same dimensions as d1; masked coordinates are zero.
# d1-noapp (X3): every request-match / dependency-match relation and every feature derived from one:
#   the record and draft relations (request, canonical, dependency = requirements + selection), the
#   route payload's request components (starts at the current position, ends at the destination), the
#   csp payload's selection dependency (items == committed selection), the start-block counts of
#   matching drafts / live applicable records, and the observation's route bound, which is selected by
#   applicability (the context also drops it, so need_bound falls back to deadline - travel in the
#   candidate features). Kept: type match, usable status, certificate, retrieved flag, snapshot
#   options and payload feasibility facts (distance, meets-deadline, finish/duration, selection validity).
NOAPP_CANDIDATE = ("record.request_match", "record.canonical_match", "record.dependency_match",
                   "record.requirements_match", "record.selection_match",
                   "payload.route_starts_at_position", "payload.route_ends_at_destination",
                   "payload.csp_items_equal_selection",
                   "draft.request_match", "draft.dependency_match",
                   "start.matching_drafts", "start.live_applicable_records")
NOAPP_OBSERVATION = ("route.applicable_known", "route.distance", "route.plan_meets_deadline", "route.need_bound")
# d1-noattempt (X4): the per-candidate attempted-action block and the observation's attempt counters,
# i.e. everything computed from observation.attempts. The one-step feedback status/reason is kept.
NOATTEMPT_CANDIDATE = tuple(n for n in CANDIDATE_NAMES_D1 if n.startswith("attempt."))
NOATTEMPT_OBSERVATION = ("attempts.count", "attempts.rejected", "attempts.uncommits")
FEATURE_MASKS = {
    "d1": ((), ()),
    "d1-noapp": (tuple(OBSERVATION_NAMES_D1.index(n) for n in NOAPP_OBSERVATION),
                 tuple(CANDIDATE_NAMES_D1.index(n) for n in NOAPP_CANDIDATE)),
    "d1-noattempt": (tuple(OBSERVATION_NAMES_D1.index(n) for n in NOATTEMPT_OBSERVATION),
                     tuple(CANDIDATE_NAMES_D1.index(n) for n in NOATTEMPT_CANDIDATE)),
}
FEATURE_VERSIONS = tuple(FEATURE_MASKS)


def _masked(vector: list[float], indices) -> list[float]:
    for i in indices:
        vector[i] = 0.0
    return vector


# The fast encoder re-implements (or memoizes) these reference helpers. If any is replaced at run
# time (e.g. the Stage B preflight's perturbation audits), encode_public uses the reference body.
_FAST_HELPERS = ("relations", "_draft_match", "current_request", "current_dependencies", "record_usable",
                 "snapshot_options", "_payload_facts", "relevant_dependencies", "action_key", "_Context",
                 "encode_observation_d1", "encode_action_d1", "_action_primitive", "_log", "overlaps")


def _unpatched() -> bool:
    g = globals()
    return all(g[name] is original for name, original in _FAST_ORIGINALS.items())


def encode_public(o: DepObservation, actions: list[Action], version: str = "d1", fast: bool | None = None):
    """(observation vector, candidate matrix). fast=None follows campaign04_fast.enabled(); the fast
    encoder is bit-identical (tests/test_campaign04_fast.py) and falls back to this reference body on
    any exception, so errors are the reference's."""
    if (_fast.enabled() if fast is None else fast) and version in FEATURE_MASKS and _unpatched():
        try:
            return _fast.encode_public_d1(o, actions, version)
        except Exception:
            _fast.FALLBACKS["encode_public"] += 1
    if version not in FEATURE_MASKS:
        raise ValueError(f"unknown depworld feature version {version}")
    ctx = _Context(o, applicability=version != "d1-noapp")
    obs_mask, cand_mask = FEATURE_MASKS[version]
    return (_masked(encode_observation_d1(o, ctx), obs_mask),
            [_masked(encode_action_d1(o, x, ctx), cand_mask) for x in actions])


_FAST_ORIGINALS = {name: globals()[name] for name in _FAST_HELPERS}


# ---------------------------------------------------------------------------
# References (public observations only)
# ---------------------------------------------------------------------------

REFERENCE_MODES = ("greedy", "recompute", "reuse", "naive_reuse", "reuse_norevise")


def _stage_open(o: DepObservation, primitive: str) -> int:
    """Public step after which a result for this primitive's stage counts as fresh."""
    def ok(at):
        return at["outcome_status"] == "success"
    sel = [at["step"] for at in o.attempts if ok(at) and at["action_kind"] == "uncommit"
           and at["arguments"].get("target") == "select"]
    sel += [e["step"] for e in o.events if "selection" in e["revoked"]]
    sel += [e["step"] for e in o.events if e["affected"] in READS["constrained_subset"]]
    open_select = max(sel, default=0)
    if primitive == "constrained_subset":
        return open_select
    asg = [open_select]
    asg += [at["step"] for at in o.attempts if ok(at) and (at["action_kind"] == "commit_pending" or
            (at["action_kind"] == "use_return" and at["arguments"].get("as") == "select"))]
    asg += [at["step"] for at in o.attempts if ok(at) and at["action_kind"] == "uncommit"]
    asg += [e["step"] for e in o.events if "assignment" in e["revoked"] or e["affected"] == "slots"]
    open_assign = max(asg)
    if primitive == "csp":
        return open_assign
    route = [open_assign]
    route += [at["step"] for at in o.attempts if ok(at) and (at["action_kind"] == "commit_assignment" or
              (at["action_kind"] == "use_return" and at["arguments"].get("as") == "assign"))]
    route += [e["step"] for e in o.events if e["affected"] in ("map", "deadline")]
    return max(route)


class DepReference:
    """Supplied public schedules over the exact d1 catalogue (not learned planning).

    greedy: direct paths only. recompute: fresh drafts + solver calls for every
    needed result, never uses pre-existing records. reuse: validate-and-reuse by
    the public applicability rule, revising upstream when downstream is
    infeasible. naive_reuse: most recent same-type usable record, no
    applicability check (control). reuse_norevise: reuse without voluntary
    upstream revision (never uncommits).
    """

    def __init__(self, mode: str = "reuse", initial_budget: int = 128):
        if mode not in REFERENCE_MODES:
            raise ValueError("unknown depworld reference mode")
        self.mode, self.initial_budget = mode, initial_budget
        self.reference_name = "dep_" + mode

    def choose_index(self, observation, candidates):
        return candidates.index(self.choose(observation, candidates))

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _last_attempt(o, action):
        key = action_key(action)
        return next((at for at in reversed(o.attempts) if at["action_key"] == key), None)

    def _repeat_failure(self, o, action):
        """Attempted before with a non-success outcome and unchanged dependencies."""
        last = self._last_attempt(o, action)
        return (last is not None and last["outcome_status"] != "success"
                and last["dependency_versions_at_attempt"] == relevant_dependencies(o, action))

    def choose(self, o: DepObservation, catalog=None) -> Action:
        if o.done:
            raise ValueError("cannot act after episode end")
        if o.requirements is None:
            return Action("inspect", {"target": "requirements"})
        for r in o.item_inventory:
            if r["handle"] not in o.known_items:
                return Action("inspect", {"target": r["handle"]})
        if o.known_edges is None:
            return Action("inspect", {"target": "map"})
        if o.selection_id is None:
            return self._select(o)
        if o.assignment_id is None:
            return self._assign(o)
        if o.position == o.goal["destination"]:
            verify = Action("verify")
            return Action("abstain") if self._repeat_failure(o, verify) else verify
        return self._route(o)

    # -- greedy --------------------------------------------------------------
    def _greedy_select(self, o):
        req, chosen, w, p = o.requirements, [], 0, 0
        conflicts = {frozenset(x) for x in req["incompatible"]}
        order = {r["handle"]: i for i, r in enumerate(o.item_inventory)}
        for cat in o.goal["categories"]:
            cands = sorted((h for h, r in o.known_items.items() if r["category"] == cat),
                           key=lambda h: (o.known_items[h]["duration"], o.known_items[h]["weight"] + o.known_items[h]["price"], order[h]))
            pick = next((h for h in cands if w + o.known_items[h]["weight"] <= req["capacity"]
                         and p + o.known_items[h]["price"] <= req["funds"]
                         and all(frozenset((h, q)) not in conflicts for q in chosen)), None)
            if pick is None:
                return Action("abstain")
            chosen.append(pick)
            w += o.known_items[pick]["weight"]
            p += o.known_items[pick]["price"]
        for h in chosen:
            if h not in o.pending:
                return Action("choose_item", {"item": h})
        commit = Action("commit_pending")
        return Action("abstain") if self._repeat_failure(o, commit) else commit

    def _greedy_assign(self, o):
        closed = set(o.requirements["closed_slots"])
        items = sorted(o.selected, key=lambda h: (len([s for s in o.known_items[h]["slots"] if s not in closed]),
                                                  -o.known_items[h]["duration"], o.selected.index(h)))
        chosen = {}
        for h in items:
            d = o.known_items[h]["duration"]
            s = next((s for s in sorted(o.known_items[h]["slots"]) if s not in closed and all(
                not overlaps(s, d, t, o.known_items[x]["duration"]) for x, t in chosen.items())), None)
            if s is None:
                return Action("abstain")
            chosen[h] = s
        for h in o.selected:
            if o.pending_assignment.get(h) != chosen[h]:
                return Action("choose_slot", {"item": h, "slot": chosen[h]})
        commit = Action("commit_assignment")
        return Action("abstain") if self._repeat_failure(o, commit) else commit

    def _greedy_route(self, o):
        out = [(w, -v, v) for u, v, w in o.known_edges if u == o.position and v > o.position]
        if not out:
            return Action("abstain")
        _, _, v = min(out)
        move = Action("move", {"destination": v})
        return Action("abstain") if self._repeat_failure(o, move) else move

    # -- solver-backed stages --------------------------------------------------
    def _select(self, o):
        if self.mode == "greedy":
            return self._greedy_select(o)
        act = self._obtain(o, "constrained_subset", {"excluded": self._exclusions(o)})
        return act if isinstance(act, Action) else Action("abstain")

    def _assign(self, o):
        if self.mode == "greedy":
            return self._greedy_assign(o)
        bound = self._bound(o)
        if bound is not None and bound < 1:
            return Action("abstain")  # even an immediate finish cannot meet the deadline
        act = self._obtain(o, "csp", {"finish_by": bound})
        if isinstance(act, Action):
            return act
        if act == "infeasible" and self.mode != "reuse_norevise":
            return Action("uncommit", {"target": "select"})
        return Action("abstain")

    def _route(self, o):
        if self.mode == "greedy":
            return self._greedy_route(o)
        act = self._obtain(o, "shortest_path", {})
        if isinstance(act, Action):
            return act
        if isinstance(act, tuple):
            record = act[1]
            dist = o.retrieved[record["handle"]]["payload"][1]
            if o.finish_time + o.travel + dist <= o.requirements["deadline"]:
                return Action("use_return", {"handle": record["handle"], "as": "route"})
            if self.mode == "reuse_norevise":
                return Action("abstain")
            return Action("uncommit", {"target": "assign"})
        return Action("abstain")

    def _route_distance(self, o):
        dists = []
        for r in o.records:
            got = o.retrieved.get(r["handle"])
            if r["primitive"] != "shortest_path" or not got or got.get("payload") is None:
                continue
            if self.mode == "naive_reuse" or applicable(o, r, "shortest_path"):
                dists.append((r["created_step"], got["payload"][1]))
        if not dists:
            return None
        return dists[-1][1] if self.mode == "naive_reuse" else min(d for _, d in dists)

    def _bound(self, o):
        """Needed finish-time bound from public facts: deadline - travel - best
        known applicable route distance (0 if no route is known yet)."""
        dist = self._route_distance(o)
        b = o.requirements["deadline"] - o.travel - (dist or 0)
        return None if b >= max(o.goal["finish_by_bounds"]) else b

    def _exclusions(self, o):
        """Items excluded after public evidence that a selection cannot meet the
        needed finish bound (an exhaustive infeasible csp record, current slots)."""
        bound = self._bound(o)
        order = {r["handle"]: i for i, r in enumerate(o.item_inventory)}
        excluded: list[str] = []
        closed = o.requirements["closed_slots"]
        for r in o.records:
            if r["primitive"] != "csp" or r["status"] != "infeasible":
                continue
            snap = r["problem_snapshot"]["problem"]
            if any(h not in o.known_items for h in snap["items"]):
                continue
            rows = [o.known_items[h] for h in snap["items"]]
            b = snap.get("finish_by")
            if csp_problem(rows, closed, CSP_CONSTRAINTS, b) != r["problem_snapshot"]:
                continue  # evidence no longer current (slots changed) or not a full request
            if b is not None and bound is not None and b < bound:
                continue
            if b is not None and bound is None:
                continue
            if set(snap["items"]) & set(excluded):
                continue
            def alternatives(h):
                cat = o.known_items[h]["category"]
                return sum(1 for x in o.item_inventory if o.known_items[x["handle"]]["category"] == cat
                           and x["handle"] not in excluded and x["handle"] != h)
            options = [h for h in snap["items"] if alternatives(h) >= 1]
            if not options:
                continue
            excluded.append(max(options, key=lambda h: (o.known_items[h]["duration"],
                                                        -len(o.known_items[h]["slots"]), -order[h])))
        return tuple(sorted(excluded, key=lambda h: order[h]))

    def _accepts(self, o, record, primitive, want, options):
        snap = record["problem_snapshot"]
        if snap == want:
            return True
        so = snapshot_options(snap)
        if primitive == "csp" and options.get("finish_by") is not None:
            return so.get("finish_by") is not None and so["finish_by"] <= options["finish_by"]
        if primitive == "csp":
            return True  # any bound: payload is a valid assignment for the current selection
        return False

    def _obtain(self, o, primitive, options):
        """Return an Action, ("ready", record) for a retrieved usable record,
        "infeasible" (exhaustive proof for the wanted request) or "exhausted"."""
        use = USE_OF_PRIMITIVE[primitive]
        want = current_request(o, primitive, options)
        if want is None:
            return "exhausted"
        deps = current_dependencies(o, primitive)
        opened = _stage_open(o, primitive)
        tried = {at["action_key"] for at in o.attempts if at["action_kind"] == "use_return" and at["step"] > opened}
        mode = self.mode
        if mode == "naive_reuse":
            # No applicability check; it only avoids re-applying a record it already
            # applied in this episode (otherwise it trivially loops after a revision).
            used = {at["action_key"] for at in o.attempts if at["action_kind"] == "use_return"}
            cands = [r for r in o.records if r["primitive"] == primitive and record_usable(r)
                     and action_key(Action("use_return", {"handle": r["handle"], "as": use})) not in used]
            cands.sort(key=lambda r: r["created_step"])
        else:
            cands = [r for r in o.records if r["status"] == "success" and applicable(o, r, primitive)
                     and self._accepts(o, r, primitive, want, options)
                     and action_key(Action("use_return", {"handle": r["handle"], "as": use})) not in tried]
            if mode == "recompute":
                cands = [r for r in cands if r["created_step"] > opened]
        if not cands and any(r["status"] == "success" and r["problem_snapshot"] == want and r["depends_on"] == deps
                             and action_key(Action("use_return", {"handle": r["handle"], "as": use})) in tried
                             for r in o.records):
            # The exact wanted result exists and was rejected on use: recomputing it
            # cannot help. For an assignment this is evidence to revise the selection.
            return "infeasible" if primitive == "csp" else "exhausted"
        if cands:
            r = cands[-1]
            if r["handle"] not in o.retrieved:
                return Action("retrieve", {"handle": r["handle"]})
            if primitive == "shortest_path":
                return ("ready", r)
            return Action("use_return", {"handle": r["handle"], "as": use})
        # compute: find a draft on the way to `want`
        drafts = [(h, e) for h, e in o.problems.items() if e["primitive"] == primitive and e["created_step"] > 0
                  and (mode != "recompute" or e["created_step"] > opened)]
        for h, e in reversed(drafts):
            nxt = self._draft_next(o, h, e, want, deps, primitive, options)
            if nxt is not None:
                return nxt
        start = {"constrained_subset": "start_subset", "csp": "start_assign", "shortest_path": "build_route"}[primitive]
        return Action(start, {"handle": f"problem_{len(o.problems)}"})

    def _draft_next(self, o, handle, entry, want, deps, primitive, options):
        p, wp = entry["problem"], want["problem"]
        if primitive == "constrained_subset":
            base = set(p["handles"]) | set(p["excluded"])
            if base != {r["handle"] for r in o.item_inventory} or not set(p["excluded"]) <= set(wp["excluded"]):
                return None
            for c in SUBSET_CONSTRAINTS:
                if c not in p["constraints"]:
                    return Action("add_constraint", {"problem": handle, "constraint": c})
            for h in wp["excluded"]:
                if h not in p["excluded"]:
                    return Action("add_constraint", {"problem": handle, "constraint": "exclude", "item": h})
            if p != wp or entry["depends_on"] != deps:
                return Action("add_constraint", {"problem": handle, "constraint": "capacity"})
        elif primitive == "csp":
            if sorted(p["items"]) != sorted(wp["items"]) or entry["depends_on"].get("selection_id") != deps["selection_id"]:
                return None
            if p["finish_by"] is not None and p["finish_by"] != wp["finish_by"]:
                return None
            if "conflicts" not in p["constraints"]:
                return Action("add_constraint", {"problem": handle, "constraint": "conflicts"})
            if wp["finish_by"] is not None and p["finish_by"] != wp["finish_by"]:
                return Action("add_constraint", {"problem": handle, "constraint": "finish_by", "bound": wp["finish_by"]})
            if p != wp or entry["depends_on"] != deps:
                return Action("add_constraint", {"problem": handle, "constraint": "conflicts"})
        else:
            if p != wp or entry["depends_on"] != deps:
                return None
        # the draft equals the wanted request: call with escalation
        snap = {"primitive": primitive, "problem": p}
        mine = [r for r in o.records if r["problem"] == handle and r["problem_snapshot"] == snap
                and r["depends_on"] == entry["depends_on"]]
        budgets = sorted(call_budgets(o))
        if mine:
            last = mine[-1]
            if last["status"] == "infeasible":
                return "infeasible"
            if last["status"] == "success":
                return None  # a success record exists but was rejected on use; try another draft
            bigger = [b for b in budgets if b > last["budget"]]
            if not bigger:
                if record_usable(last):
                    if last["handle"] not in o.retrieved:
                        return Action("retrieve", {"handle": last["handle"]})
                    if primitive == "shortest_path":
                        return ("ready", last)
                    return Action("use_return", {"handle": last["handle"], "as": USE_OF_PRIMITIVE[primitive]})
                return "exhausted"
            return Action("call", {"problem": handle, "budget": bigger[0]})
        first = [b for b in budgets if b >= self.initial_budget]
        if not budgets:
            return "exhausted"
        return Action("call", {"problem": handle, "budget": first[0] if first else budgets[-1]})
