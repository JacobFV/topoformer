"""extended-04 Phase A: result-identical fast paths for depworld training and evaluation.

Every function here has a reference implementation that stays in place and selectable:

  observation copies   DepWorkshop.observe            copy.deepcopy          -> plain_copy
  trace observations   DepObservation.to_dict         json(asdict(o))        -> observation_dict
  d1 encoder           campaign03_depworld.encode_public (reference body)    -> encode_public_d1
  collation            campaign02_training.collate    nested torch.tensor    -> collate (numpy buffers)
  per-step utility     env.evaluate()["utility"]      full evaluate()        -> env.current_utility()

The fast path is the default. Force the reference path with the environment variable
``TENSEGRA_REFERENCE_PATH=1`` (read at import), ``campaign04_fast.set_default(False)``, or the
``campaign04_fast.path(False)`` context manager; ``encode_public(..., fast=False)`` and
``collate(..., fast=False)`` select per call. Equivalence (bit-identical encodings, candidate sets
and matrices; identical parameters/optimizer state after training tranches; identical evaluation
rows) is tested in tests/test_campaign04_fast.py.

Caching policy (provenance safety): nothing is cached across observations except values that are
pure functions of their key (action_key of an action's kind+typed arguments; one-hot prefixes of
(kind, use, inspect-target class, primitive)). Everything that depends on requirements versions,
record status, the retrieved set, attempts, events or any other public state is memoized only
inside one encoder context, i.e. one observation, and discarded with it.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
import os

_ENABLED = os.environ.get("TENSEGRA_REFERENCE_PATH", "") not in ("1", "true", "yes")


def enabled() -> bool:
    return _ENABLED


def set_default(fast: bool) -> None:
    global _ENABLED
    _ENABLED = bool(fast)


@contextmanager
def path(fast: bool):
    """Temporarily select the fast (True) or reference (False) implementation."""
    global _ENABLED
    before = _ENABLED
    _ENABLED = bool(fast)
    try:
        yield
    finally:
        _ENABLED = before


# Fast-path exceptions answered by the reference implementation (tests assert these stay 0).
FALLBACKS = {"encode_public": 0}


def describe() -> str:
    return "fast" if _ENABLED else "reference"


# ---------------------------------------------------------------------------
# Plain-data copies and JSON-equivalent conversion
# ---------------------------------------------------------------------------

_ATOMIC = (str, int, float, bool, type(None))
_ATOMS = frozenset(_ATOMIC)


def plain_copy(x):
    """Value-identical replacement for copy.deepcopy on JSON-like data (dict/list/tuple of atoms).

    Container types are preserved exactly (tuple stays tuple). Anything else is deep-copied by
    copy.deepcopy. Only object identity/aliasing inside the copy may differ, which no consumer reads.
    """
    t = type(x)
    atoms = _ATOMS
    if t is dict:
        return {k: v if type(v) in atoms else plain_copy(v) for k, v in x.items()}
    if t is list:
        return [v if type(v) in atoms else plain_copy(v) for v in x]
    if t is tuple:
        return tuple([v if type(v) in atoms else plain_copy(v) for v in x])
    if t in atoms:
        return x
    return deepcopy(x)


class _NotJSON(Exception):
    pass


def _json_key(k):
    t = type(k)
    if t is str:
        return k
    if t is bool:
        return "true" if k else "false"
    if t is int:
        return int.__repr__(k)
    if t is float:
        if k != k:
            return "NaN"
        if k == math.inf:
            return "Infinity"
        if k == -math.inf:
            return "-Infinity"
        return float.__repr__(k)
    if k is None:
        return "null"
    raise _NotJSON


def jsonable(x):
    """Exactly json.loads(json.dumps(x)) for plain data (tuples -> lists, keys -> JSON strings,
    duplicate keys keep the first position and last value); raises _NotJSON otherwise."""
    t = type(x)
    atoms = _ATOMS
    if t in atoms:
        return x
    if t is dict:
        out = {}
        for k, v in x.items():
            out[k if type(k) is str else _json_key(k)] = v if type(v) in atoms else jsonable(v)
        return out
    if t is list or t is tuple:
        return [v if type(v) in atoms else jsonable(v) for v in x]
    raise _NotJSON


def observation_dict(observation):
    """Fast DepObservation.to_dict(): identical to json.loads(json.dumps(asdict(observation)))."""
    from dataclasses import asdict, fields
    import json
    try:
        return {f.name: jsonable(getattr(observation, f.name)) for f in fields(observation)}
    except _NotJSON:
        return json.loads(json.dumps(asdict(observation)))


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate(frames, device="cpu"):
    """Same tensors as campaign02_training.collate (float32 from python floats, bool mask, int64
    targets), built through preallocated numpy buffers instead of nested torch.tensor calls."""
    import numpy as np
    import torch
    if not frames or any(not f.candidates or not 0 <= f.target < len(f.candidates) for f in frames):
        raise ValueError("Every frame needs a valid candidate target")
    count, dim = max(len(f.candidates) for f in frames), len(frames[0].candidates[0])
    buffer = np.zeros((len(frames), count, dim), dtype=np.float32)
    mask = np.zeros((len(frames), count), dtype=np.bool_)
    for i, frame in enumerate(frames):
        n = len(frame.candidates)
        buffer[i, :n] = np.asarray(frame.candidates, dtype=np.float64)
        mask[i, :n] = True
    observations = np.asarray([f.observation for f in frames], dtype=np.float64).astype(np.float32)
    targets = torch.tensor([f.target for f in frames], device=device)
    return (torch.from_numpy(observations).to(device), torch.from_numpy(buffer).to(device),
            torch.from_numpy(mask).to(device), targets)


# ---------------------------------------------------------------------------
# depworld d1 encoder (fast, bit-identical to campaign03_depworld reference)
# ---------------------------------------------------------------------------

_ACTION_KEYS: dict = {}
_KEYABLE = (str, int, bool, type(None))


def action_key(action):
    """Memoized campaign03_depworld.action_key: a pure function of (kind, typed arguments)."""
    from .campaign03_depworld import action_key as reference
    try:
        items = action.arguments.items()
        if any(type(v) not in _KEYABLE or type(k) is not str for k, v in items):
            return reference(action)
        key = (action.kind, tuple(sorted((k, type(v), v) for k, v in items)))
    except (AttributeError, TypeError):
        return reference(action)
    value = _ACTION_KEYS.get(key)
    if value is None:
        if len(_ACTION_KEYS) > 200_000:
            _ACTION_KEYS.clear()
        value = _ACTION_KEYS[key] = reference(action)
    return value


_LAYOUT = None


def _layout():
    """Block offsets of the d1 candidate vector, checked against CANDIDATE_NAMES_D1."""
    global _LAYOUT
    if _LAYOUT is None:
        from . import campaign03_depworld as d
        names = d.CANDIDATE_NAMES_D1
        lay = {"dim": len(names), "kind": {k: i for i, k in enumerate(d.KINDS)},
               "use": {u: len(d.KINDS) + i for i, u in enumerate(d.USES)},
               "prim": {p: names.index("primitive." + p) for p in d.PRIMITIVES},
               "tgt_item": names.index("inspect.item"), "tgt_map": names.index("inspect.map"),
               "tgt_req": names.index("inspect.requirements"),
               "rec": names.index("record.present"), "draft": names.index("draft.present"),
               "call": names.index("call.budget"), "cons": names.index("constraint.capacity"),
               "start": names.index("start.n_drafts"), "item": names.index("item.present"),
               "slot": names.index("slot.present"), "move": names.index("move.present"),
               "commit": names.index("commit.coverage_or_uncommit_target_committed"),
               "attempt": names.index("attempt.attempted")}
        assert lay["rec"] == len(d.KINDS) + len(d.USES) + 3 + len(d.PRIMITIVES)
        assert (lay["draft"] - lay["rec"], lay["call"] - lay["draft"], lay["cons"] - lay["call"],
                lay["start"] - lay["draft"], lay["item"] - lay["start"], lay["slot"] - lay["item"],
                lay["move"] - lay["slot"], lay["commit"] - lay["move"], lay["attempt"] - lay["commit"],
                lay["dim"] - lay["attempt"]) == (26, 11, 6, 28, 3, 14, 10, 7, 10, d.ATTEMPT_BLOCK)
        _LAYOUT = lay
    return _LAYOUT


def _context_class():
    from . import campaign03_depworld as d

    class FastContext(d._Context):
        """d1 _Context plus per-observation memo tables (discarded with the observation)."""

        def __init__(self, o, applicability=True):
            self.requests: dict = {}
            self.drafts: dict = {}
            self.payloads: dict = {}
            self.records_block: dict = {}
            self.items: dict = {}
            self.starts: dict = {}
            self.inventory: dict = {}
            super().__init__(o, applicability)
            for r in o.item_inventory:
                self.inventory.setdefault(r["handle"], r)

        def request(self, primitive, options=None):
            """current_request(o, primitive, options), memoized within this observation."""
            if not options:
                key = (primitive,)
            else:
                key = (primitive, tuple(sorted(options.items())))
            try:
                return self.requests[key]
            except KeyError:
                value = self.requests[key] = d.current_request(self.o, primitive, options)
                return value
            except TypeError:
                return d.current_request(self.o, primitive, options)

        def rel(self, record, primitive=None):
            key = (record["handle"], primitive)
            cached = self.rel_cache.get(key)
            if cached is None:
                cached = self.rel_cache[key] = self._relations(record, primitive)
            return cached

        def _relations(self, record, primitive):
            # Body of campaign03_depworld.relations with memoized current_request.
            o = self.o
            p = primitive or record.get("primitive")
            type_match = record.get("primitive") == p
            snap = record.get("problem_snapshot")
            req = self.request(p, d.snapshot_options(snap)) if type_match else None
            canonical = self.request(p) if type_match else None
            deps = record.get("depends_on") or {}
            cur = d.current_dependencies(o, p) if p in d.READS else {}
            return {"type_match": type_match,
                    "request_match": req is not None and snap == req,
                    "canonical_match": canonical is not None and snap == canonical,
                    "dependency_match": type_match and deps == cur,
                    "requirements_match": type_match and deps.get("requirements_version") == cur.get("requirements_version"),
                    "selection_match": type_match and deps.get("selection_id") == cur.get("selection_id"),
                    "usable": d.record_usable(record)}

        def draft_match(self, entry):
            snap = {"primitive": entry["primitive"], "problem": entry["problem"]}
            req = self.request(entry["primitive"], d.snapshot_options(snap))
            return (float(req is not None and snap == req),
                    float(entry.get("depends_on") == d.current_dependencies(self.o, entry["primitive"])))

    return FastContext


_FAST_CONTEXT = None


def _draft_info(o, c, name, entry):
    info = c.drafts.get(name)
    if info is None:
        p = entry["problem"]
        rm, dm = c.draft_match(entry)
        cons = p.get("constraints", [])
        snap = {"primitive": entry["primitive"], "problem": p}
        same = [r for r in o.records if r.get("problem_snapshot") == snap]
        cur = [r for r in same if r.get("depends_on") == entry.get("depends_on")]
        most = max((r.get("work_units", 0) for r in same), default=0)
        size = len(p.get("handles", p.get("items", p.get("edges", []))))
        head = [1.0, rm, dm] + [float(x in cons) for x in ("capacity", "funds", "incompatibility", "conflicts")]
        head += [float(p.get("finish_by") is not None), (p.get("finish_by") or 0) / 16, len(p.get("excluded", [])) / 4,
                 _dlog(size, 5)]
        from .campaign03_depworld import record_usable
        info = c.drafts[name] = (head, same, most, sum(record_usable(r) for r in cur) / 4,
                                 float(any(r.get("status") == "infeasible" for r in cur)), cons, p)
    return info


def _dlog(x, top=12.0):
    from .campaign03_depworld import _log
    return _log(x, top)


def encode_public_d1(o, actions, version="d1"):
    """Bit-identical to the reference campaign03_depworld.encode_public (d1 and its ablations)."""
    from . import campaign03_depworld as d
    global _FAST_CONTEXT
    if version not in d.FEATURE_MASKS:
        raise ValueError(f"unknown depworld feature version {version}")
    if _FAST_CONTEXT is None:
        _FAST_CONTEXT = _context_class()
    lay = _layout()
    ctx = _FAST_CONTEXT(o, applicability=version != "d1-noapp")
    obs_mask, cand_mask = d.FEATURE_MASKS[version]
    observation = d._masked(d.encode_observation_d1(o, ctx), obs_mask)
    rows = []
    for action in actions:
        row = _encode_action(o, action, ctx, lay, d)
        for i in cand_mask:
            row[i] = 0.0
        rows.append(row)
    return observation, rows


_TEMPLATES: dict = {}
_START_PRIMITIVE = {"start_subset": "constrained_subset", "start_assign": "csp", "build_route": "shortest_path"}


def _template(key, lay):
    """Zero row with the one-hot prefix of (kind, use, inspect-target class, primitive) set.
    A pure function of the key (no observation content), so it is cached across observations."""
    row = _TEMPLATES.get(key)
    if row is None:
        k, use, tclass, prim = key
        row = [0.0] * lay["dim"]
        for i in (lay["kind"].get(k), lay["use"].get(use) if use is not None else None,
                  lay[tclass] if tclass is not None else None, lay["prim"].get(prim) if prim is not None else None):
            if i is not None:
                row[i] = 1.0
        if len(_TEMPLATES) > 10_000:
            _TEMPLATES.clear()
        _TEMPLATES[key] = row
    return row


def _encode_action(o, action, c, lay, d):
    a, k = action.arguments, action.kind
    use = a.get("as") if k == "use_return" else (a.get("target") if k == "uncommit" else None)
    tclass = None
    if k == "inspect":
        tgt = a.get("target")
        if tgt is not None:
            tclass = "tgt_map" if tgt == "map" else ("tgt_req" if tgt == "requirements" else "tgt_item")
    # primitive (as campaign03_depworld._action_primitive)
    if k == "call" or k == "add_constraint":
        prim = o.problems.get(a.get("problem"), {}).get("primitive")
    elif k == "retrieve" or k == "use_return":
        prim = c.records.get(a.get("handle"), {}).get("primitive")
    else:
        prim = _START_PRIMITIVE.get(k)
    # Only str-valued (or None) categorical parts are template keys; anything else -> reference.
    if type(k) is not str or (use is not None and type(use) is not str) or (prim is not None and type(prim) is not str):
        raise TypeError("non-catalogue action")
    out = _template((k, use, tclass, prim), lay).copy()
    D, T = c.D, o.finish_time
    # --- record block (retrieve / use_return)
    if k == "retrieve" or k == "use_return":
        rec = c.records.get(a.get("handle"))
        if rec is not None:
            want = d.PRIMITIVE_OF_USE.get(a.get("as")) if k == "use_return" else rec["primitive"]
            bkey = (rec["handle"], want)
            block = c.records_block.get(bkey)
            if block is None:
                rel = c.rel(rec, want)
                opts = d.snapshot_options(rec.get("problem_snapshot"))
                block = [1.0, float(rel["type_match"]), float(rel["request_match"]), float(rel["canonical_match"]),
                         float(rel["dependency_match"]), float(rel["requirements_match"]),
                         float(rel["selection_match"]), float(rel["usable"])]
                block += [float(rec.get("status") == s) for s in d.RECORD_STATUSES]
                block += [float(rec.get("certificate_valid", False)), float(rec["handle"] in o.retrieved),
                          float(opts.get("finish_by") is not None), (opts.get("finish_by") or 0) / 16,
                          len(opts.get("excluded", ())) / 4]
                facts = c.payloads.get(rec["handle"])
                if facts is None:
                    facts = c.payloads[rec["handle"]] = d._payload_facts(o, c, rec)
                block += facts
                c.records_block[bkey] = block
            s = lay["rec"]
            out[s:s + 26] = block
    # --- draft block (call / add_constraint)
    elif k == "call" or k == "add_constraint":
        name = a.get("problem")
        entry = o.problems.get(name)
        if entry is not None:
            head, same, most, usable_cur, infeasible_cur, cons, p = _draft_info(o, c, name, entry)
            s = lay["draft"]
            out[s:s + 11] = head
            if k == "call":
                budget = a.get("budget", 0)
                s = lay["call"]
                out[s:s + 6] = [d._log(budget, 14), float(budget > most),
                                float(any(r.get("status") == "timeout" and r.get("budget", 0) >= budget for r in same)),
                                usable_cur, infeasible_cur, float(budget == o.remaining_work)]
            else:
                cname = a.get("constraint")
                bound = a.get("bound")
                item = o.known_items.get(a.get("item"), {}) if cname == "exclude" else {}
                alternatives = sum(1 for r in o.item_inventory if item and r["category"] == item.get("category")
                                   and r["handle"] not in p.get("excluded", []) and r["handle"] != a.get("item"))
                s = lay["cons"]
                out[s:s + 11] = ([float(cname == x) for x in d.CONSTRAINT_NAMES]
                                 + [float(cname in cons or (cname == "finish_by" and p.get("finish_by") == bound)),
                                    (bound or 0) / 16,
                                    float(bound is not None and c.need_bound is not None and bound <= c.need_bound),
                                    item.get("duration", 0) / 4, alternatives / 4])
    # --- start block
    elif k in ("start_subset", "start_assign", "build_route"):
        block = c.starts.get(prim)
        if block is None:
            drafts = [(n, e) for n, e in o.problems.items() if e["primitive"] == prim]
            matching = [e for n, e in drafts if c.draft_match(e) == (1.0, 1.0)]
            live = [r for r in o.records if r["primitive"] == prim and c.rel(r)["request_match"]
                    and c.rel(r)["dependency_match"] and c.rel(r)["usable"]]
            block = c.starts[prim] = [len(drafts) / 8, len(matching) / 4, len(live) / 4]
        s = lay["start"]
        out[s:s + 3] = block
    # --- item block (inspect item / choose_item)
    elif k == "inspect" or k == "choose_item":
        h = a.get("item", a.get("target"))
        if h is not None:
            block = c.items.get(h)
            if block is None:
                row = c.inventory.get(h)
                block = c.items[h] = _item_block(o, c, h, row) if row is not None else False
            if block:
                s = lay["item"]
                out[s:s + 14] = block
    # --- slot block
    elif k == "choose_slot":
        if a.get("item") in o.known_items:
            it = o.known_items[a["item"]]
            sl, du = a["slot"], it["duration"]
            clash = any(d.overlaps(sl, du, t, o.known_items[x]["duration"]) for x, t in o.pending_assignment.items()
                        if x != a["item"] and x in o.selected and x in o.known_items)
            finish = sl + du
            limit = c.need_bound if c.need_bound is not None else ((D - o.travel) if D is not None else None)
            s = lay["slot"]
            out[s:s + 10] = [1.0, du / 4, sl / 16, float(sl in it["slots"]), float(sl in c.closed), float(clash),
                             finish / 16, float(limit is not None and finish <= limit),
                             float(o.pending_assignment.get(a["item"]) == sl), float(a["item"] in o.pending_assignment)]
    # --- move block
    elif k == "move":
        if o.known_edges is not None:
            v = a["destination"]
            w = next((w for x, y, w in o.known_edges if x == o.position and y == v), 0)
            ahead = [w2 for x, y, w2 in o.known_edges if x == v]
            direct = any(x == v and y == o.goal["destination"] for x, y, _ in o.known_edges)
            meets = T is not None and D is not None and T + o.travel + w <= D
            s = lay["move"]
            out[s:s + 7] = [1.0, w / 32, float(v == o.goal["destination"]), float(direct), min(ahead, default=0) / 32,
                            float(meets), (v - o.position) / 8]
    # --- commit / uncommit block
    elif k == "commit_pending":
        rows = [o.known_items.get(x) for x in o.pending]
        full = all(r is not None for r in rows) and o.requirements is not None
        ok_w = full and sum(r["weight"] for r in rows) <= c.cap
        ok_p = full and sum(r["price"] for r in rows) <= c.funds
        inc = full and any(x in o.pending and y in o.pending for x, y in o.requirements["incompatible"])
        s = lay["commit"]
        out[s:s + 5] = [len(o.pending) / max(1, len(o.goal["categories"])),
                        float(len(o.pending) == len(o.goal["categories"])), float(ok_w), float(ok_p), float(inc)]
    elif k == "commit_assignment":
        sel = [x for x in o.selected if x in o.known_items]
        cover = all(x in o.pending_assignment for x in o.selected) and bool(o.selected)
        allowed = cover and all(o.pending_assignment[x] in o.known_items[x]["slots"] and
                                o.pending_assignment[x] not in c.closed for x in sel)
        clash = cover and any(d.overlaps(o.pending_assignment[x], o.known_items[x]["duration"],
                                         o.pending_assignment[y], o.known_items[y]["duration"])
                              for i, x in enumerate(sel) for y in sel[i + 1:])
        finish = max((o.pending_assignment[x] + o.known_items[x]["duration"] for x in sel if x in o.pending_assignment),
                     default=0)
        limit = c.need_bound if c.need_bound is not None else ((D - o.travel) if D is not None else None)
        s = lay["commit"] + 5
        out[s:s + 5] = [float(cover), float(allowed), float(clash), finish / 16,
                        float(cover and limit is not None and finish <= limit)]
    elif k == "uncommit":
        committed = o.selection_id is not None if a.get("target") == "select" else o.assignment_id is not None
        out[lay["commit"]] = float(committed)
    # --- attempted-action block (only nonzero if this exact action was attempted before)
    if c.last_attempt and k in d.ATTEMPT_KINDS:
        key = action_key(action)
        last = c.last_attempt.get(key)
        if last is not None:
            status = last["outcome_status"]
            block = [1.0, min(c.attempt_count[key], 8) / 4]
            block += [float(status == s) for s in d.OUTCOMES[:-1]] + [float(status not in d.OUTCOMES[:-1])]
            block += [float(last["reason"] == r) for r in d.REASONS]
            block += [float(d.relevant_dependencies(o, action) != last["dependency_versions_at_attempt"])]
            s = lay["attempt"]
            out[s:s + d.ATTEMPT_BLOCK] = block
    return out


def _item_block(o, c, h, row):
    known = o.known_items.get(h, {})
    peers = [o.known_items[r["handle"]] for r in o.item_inventory
             if r["category"] == row["category"] and r["handle"] in o.known_items]
    drank = sum(p["duration"] < known["duration"] for p in peers) if known else 0
    crank = sum(p["weight"] + p["price"] < known["weight"] + known["price"] for p in peers) if known else 0
    conflicts = [y if x == h else x for x, y in (o.requirements or {}).get("incompatible", []) if h in (x, y)]
    pend_cats = {o.known_items.get(x, {}).get("category") for x in o.pending}
    return [1.0, float(bool(known)), known.get("weight", 0) / 16, known.get("price", 0) / 16,
            known.get("duration", 0) / 4, len(known.get("slots", [])) / 8, float(h in o.pending),
            float(h in o.selected), float(row["category"] in pend_cats),
            float(bool(known) and c.cap is not None and known["weight"] <= c.cap - c.pending_weight),
            float(bool(known) and c.funds is not None and known["price"] <= c.funds - c.pending_price),
            float(any(x in o.pending for x in conflicts)), drank / 4, crank / 4]
