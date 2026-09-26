# Extended-04 Phase A: depworld throughput (fast path)

**Status:** done on `campaign/e04-throughput`. The fast path is the default because every equivalence test passes. The reference path is kept and can be selected (see "How to force the reference path").

## What changed

The fast path lives in `src/tensegra/campaign04_fast.py`. Each piece sits behind a dispatch point in the reference modules, and the reference code is untouched behind that switch.

| Hot spot | Reference | Fast path | Dispatch point |
|---|---|---|---|
| Observation copies in `DepWorkshop.observe` | `copy.deepcopy` of every record, problem, attempt and event, every step | `plain_copy`: a value-identical copier for dict/list/tuple/atoms that keeps container types. Anything else still goes to `deepcopy` | `DepWorkshop.observe` |
| Trace observations (`DepObservation.to_dict`) | `json.loads(json.dumps(asdict(o)))`, i.e. `asdict` deep-copies and then a JSON round trip | `observation_dict`: one pass that reproduces the JSON round trip exactly (tuples become lists; keys become JSON strings, including bool/int/float/None keys; duplicate keys keep the first position and the last value). It falls back to the reference on any non-JSON type | `DepObservation.to_dict` |
| d1 encoder (and d1-noapp, d1-noattempt) | `encode_action_d1` per candidate. It rebuilds public requests (`subset_problem`/`csp_problem`/`route_problem`) and relations for every candidate, and computes item/record/draft facts once per candidate | `encode_public_d1`: rows start from a cached one-hot template and blocks are written at fixed offsets (checked against `CANDIDATE_NAMES_D1`). Public requests, relations, draft facts, payload facts, record blocks, item blocks and start-block counts are memoized per observation | `campaign03_depworld.encode_public(..., fast=None)` |
| Collation | `torch.tensor` on nested lists, plus one host-to-device copy per frame | Preallocated numpy float32/bool buffers and one transfer per tensor. Values are the same float64→float32 casts | `campaign02_training.collate(..., fast=None)` |
| Per-step utility in batched actor-critic | `env.evaluate()["utility"]`, which deep-copies the whole history, audits and reductions on every step | `env.current_utility()`: the same expression, `float(verified) - _cost()`, where `evaluate()` now also calls `_cost()` | `batched_on_policy` |

- **Hashes and formats are unchanged.** The data-hash definition, trace formats, evaluation-row formats and the gzip settings are all the same.
- **`trace_hash_export` stays as it was.** It was already small (about 2% of a tranche) once `to_dict` was fixed.
- **No hash option was needed.** No alternative hash definition was introduced, so there is no option to gate one.

## What is cached, and why it is safe

The rule: nothing that depends on public state outlives one observation.

- **Across observations**, only three kinds of value are cached, and each is a pure function of its key:
  - `action_key`: sha256 of `{kind, arguments}`. The memo is keyed on `(kind, sorted (name, type, value))` and is used only when every argument value is str/int/bool/None. Types are part of the key, so `1`, `True` and `1.0` never collide. Any other action goes to the reference function.
  - One-hot row templates, keyed on `(kind, use, inspect-target class, primitive)`. A template holds only these one-hots and zeros.
  - Block offsets, computed once from `CANDIDATE_NAMES_D1` and asserted against the reference block sizes.
- **Within one observation only**, a context object holds the memo tables. The context is created per `encode_public` call and discarded with it. The tables cover:
  - `current_request(o, primitive, options)`;
  - relations per (record handle, wanted primitive), as the reference `_Context` already does;
  - draft match and the same-snapshot record lists per problem handle;
  - payload facts per record;
  - record blocks per (handle, wanted primitive);
  - item blocks per item handle;
  - start-block counts per primitive.

  Everything that reads requirement versions, record status, the retrieved set, attempts, events, position or commitments therefore sees exactly the observation being encoded. There is no cross-step cache, so stale caching cannot happen.
- **The attempt block** is computed only when the observation has attempts. With no attempts the reference result is all zeros, and so is the fast result.
- **If reference helpers are patched, the fast encoder steps aside.** If any helper it re-implements or memoizes is replaced at run time (`relations`, `_draft_match`, `current_request`, `action_key`, `_Context`, …), `encode_public` uses the reference body. The Stage B preflight's perturbation audits patch these helpers and still pass. The preflight's distinction audit itself runs on the fast encoder and passes.
- **Exceptions fall back to the reference.** Any exception inside the fast encoder is answered by the reference body, so errors are the reference's. Fallbacks are counted in `campaign04_fast.FALLBACKS`, and the tests assert the count stays 0 on every tested trajectory.
- **`plain_copy` can differ from `deepcopy` in only one way:** aliasing inside a single copy (deepcopy's memo). No consumer reads object identity; values and container types are identical. This is tested on lockstep environments under both paths: observations equal, `asdict` type trees equal, `to_dict` equal, and `evaluate()` equal apart from measured CPU seconds.

## Equivalence evidence (hard gate)

`tests/test_campaign04_fast.py` has 40 tests, all passing on the pro6000. The full `tests/test_campaign0*.py` suite gives **287 passed** (247 existing + 40 new). It passes both on the default fast path and with `TENSEGRA_REFERENCE_PATH=1`.

**(a) Encoder.** Encoded observation vectors, candidate sets and candidate matrices are bit-identical: the same Python type and the same IEEE-754 bit pattern for every coordinate. This holds for d1, d1-noapp and d1-noattempt:
- Trajectories cover 6 world families × 4 policies (random, dep_reuse, dep_naive_reuse, and ε-mixed reuse/random) × 3 seeds.
- The worlds include events of all four kinds (step and progress triggers), foreign records (0–4), rejections with at least 4 distinct reasons, retrieved records and attempt logs.
- Lockstep environments give identical observations, trace dicts, utilities and evaluation outcomes. The same holds for `P1AuditedDepWorkshop`.
- The JSON-conversion and `action_key` memo are checked against the reference.
- Patched helpers route to the reference, and the reference path stays selectable.

**(b) Training.** A 3-update actor-critic tranche (P1/C0 recipe: KL .3 to tranche start, advantage normalization, batched rollouts) was run for d1 and d1-noapp. A 3-update supervised dep_reuse tranche was run for d1-noattempt.
- Model state, optimizer state (every Adam moment and step), counters, curves and the torch RNG state after training are **identical** fast vs reference.
- The supervised data hash (it has no timing fields) is identical.
- The actor-critic data hash is identical once the wall clock is frozen (see "Data-hash semantics" below).

**(c) Evaluation.** A sealed-style evaluation (`batched_episodes` on `P1AuditedDepWorkshop` + `episode_counts`), in greedy and sampled (fixed-seed inverse-CDF) modes, gives identical rows once timing fields are stripped.

**On GPU (pro6000, CUDA), from the real X1-r2 bootstrap, 12 updates:**
- **Tranche:** parameter sha256 `d50bb69b…` and optimizer-state sha256 `975e3d41…` are identical for reference and fast. Development results are identical (success .9375, utility .85972, same seeds hash).
- **Sealed-style evaluation (64 worlds):** row hashes with timing stripped are identical: greedy `94a85812…`, sampled `e2128cdb…`.

## Profile and speedup (pro6000, measured under `metered.sh`)

Tool: `research/tools/campaign04_throughput_bench.py`. All runs used 1 thread and the X1-r2 bootstrap checkpoint, with process CPU seconds per phase (`PhaseClock`). The tranche is the C0 recipe with batch 8 and max_steps 96, on the P1 world mix.

### Before profile (reference, cProfile, CPU, 6 updates + 32 dev worlds)

| Rank | Hot spot | Share of the Python side | Detail |
|---|---|---|---|
| 1 | `copy.deepcopy` | ~4.4 s of ~12 s | 4.0 M calls, from `observe()` and `evaluate()` |
| 2 | `encode_action_d1` | 3.9 s | 188 k calls; `current_request`/`subset_problem`/`_draft_match` rebuilt for every candidate |
| 3 | `asdict` + JSON round trip in `to_dict` | 2.3 s | |
| 4 | Nested `torch.tensor` in `collate` | 1.6 s | |
| 5 | Per-step `env.evaluate()` | 1.5 s | Called only to read the utility |

### GPU tranche (12 updates, CUDA; the production setting)

| Phase (CPU s) | Reference | Fast | Speedup |
|---|---:|---:|---:|
| encode (catalogue + d1 encoder) | 4.603 | 2.551 | 1.80× |
| collate | 3.358 | 1.530 | 2.19× |
| environment_step | 2.190 | 0.779 | 2.81× |
| trace_export | 1.756 | 0.785 | 2.24× |
| trace_hash_export | 0.330 | 0.301 | 1.10× |
| world_construction | 0.051 | 0.040 | |
| **Python side (sum of the six rows above)** | **12.288** | **5.986** | **2.05×** |
| forward_policy + forward_frozen | 3.178 | 2.569 | (fewer host-to-device copies) |
| objective + backward + optimizer | 2.180 | 2.007 | (neural; unchanged code) |
| **Tranche total (train_tranche)** | **17.96** | **10.84** | **1.66×** |
| Development evaluation (64 worlds) | 6.07 | 3.40 | 1.79× |
| **Tranche + development evaluation** | **24.04** | **14.24** | **1.69×** |
| Whole process (including imports and checkpoint load) | 24.92 | 15.10 | 1.65× |

### GPU sealed-style evaluation (64 worlds, P1-audited, foreign-2 condition)

| Mode | Reference rollout | Fast rollout | Speedup | Including rows, counts and gzip export |
|---|---:|---:|---:|---:|
| greedy | 6.20 | 3.34 | 1.86× | 6.55 → 3.65 (1.79×) |
| sampled | 7.19 | 3.62 | 1.99× | 7.62 → 4.00 (1.90×) |

### CPU-only runs (the width-1024 actor on CPU dominates, so gains are diluted)

| Workload | Reference | Fast |
|---|---:|---:|
| Tranche, 6 updates | 19.08 | 16.48 |
| Development evaluation, 32 worlds | 4.69 | 3.46 |
| Evaluation, greedy + sampled × 32 | 11.37 | 9.34 |
| Supervised tranche, 4 updates | 11.47 | 10.47 |

A no-torch local simulation of the supervised Python side (encode + teacher + observation hash + step) went from 1.06 s to 0.44 s (2.4×).

**What is left:**
- About 1 ms per step is spent converting ~10⁵ Python floats per batch to float32. That is the floor for dense per-candidate rows.
- The per-candidate encoder costs about 3–4 µs.
- The actor-critic objective builds per-decision 0-d tensors, which means many small GPU ops. Vectorizing it could save up to ~10% of a GPU tranche, but it changes the autograd graph. It would need its own bit-identity study, so it is not done here.

## How to force the reference path

- **Environment variable:** `TENSEGRA_REFERENCE_PATH=1`, read at import. It was verified to pass through `metered.sh`; check that any other launcher forwards it. It applies to training (`python -m tensegra.campaign02_population`) and evaluation (`research/tools/campaign02_evaluate.py`).
- **In code:**
  - `tensegra.campaign04_fast.set_default(False)`;
  - `with tensegra.campaign04_fast.path(False): ...`;
  - per call: `encode_public(o, actions, version, fast=False)` or `collate(frames, device, fast=False)`.
- **Which path a run used** is recorded in `train_tranche(...)["throughput_path"]`, which lands in the population state rows as `training_timing.throughput_path`, and in the evaluation summary as `throughput_path`.

## Data-hash semantics and provenance

- **The data-hash definition is unchanged.** `Learner.data_hash` is sha256-chained over `json.dumps(trace, sort_keys=True)`:
  - Supervised tranches hash the teacher trace (observation digest, action, index). It has no timing fields, so it is reproducible and identical fast vs reference.
  - Batched actor-critic tranches hash the learned trace. That trace has always included the wall-clock field `neural_forward_wall_seconds_allocated`, so the hash has never been reproducible across runs. This is pre-existing and was not changed. With the clock frozen, the hashed bytes are identical fast vs reference (tested).
- **Source hashes now cover the fast path.**
  - Depworld population runs add `campaign04_fast.py` to `source_hashes`.
  - The evaluator adds `sources["fast"]`.
  - Because `campaign03_depworld.py` and `campaign02_training.py` also changed, a run started on the old sources cannot be *resumed* on this branch. The population refuses on a source-hash mismatch, as it would for any source change. Finish such runs on their original source.
- **Historical ext-03 results, configs and hashes are untouched.** The reference path reproduces them: the same code paths run when the fast path is disabled.
