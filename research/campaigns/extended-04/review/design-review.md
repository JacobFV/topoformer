# Extended-04 design review (internal, pre-implementation)

**Reviewer:** internal design-review subagent. It reviewed and did not implement.
**Scope:** design.md v1 (2026-09-26T18:55Z), checked against the brief's guardrails (campaign.md), the extended-03 P1/P2a reports and audit, and the code:
- `campaign03_depworld.py`;
- `campaign02_training.py`;
- `campaign02_protocol.py` (BoundedSolver);
- `research/tools/campaign03_p2a_analysis.py`.

No code was run.

**Severity:**
- **BLOCKER:** fix before the affected component is built or launched.
- **MAJOR:** fix before the affected protocol is registered.
- **MINOR:** fix when convenient.

**Overall.** The vocabulary (§1), the temperature statement, the named continuation and the telemetry/appraisal/intervention separation are sound, and match the brief. The largest risks are:
- A2-dep, as written, cannot be run on the existing on-policy code (F4).
- The promotion rule and the Track C default can be passed or failed for reasons that have nothing to do with learning (F1, F9).
- Environment copying breaks on the live solver process (F7).
- Hindsight could enter Track C through the way the branch labels are used (F8).

---

## Track A

**F1 [BLOCKER]: The promotion comparator and the deployment mode are mismatched.**
- **Problem.** An arm is promoted when utility "under its registered deployment mode" ≥ *bootstrap-greedy* + .01. The bootstrap is never evaluated under the arm's mode. A1 exists precisely because R-mask may already lift the bootstrap, and it would certainly lift loop-prone policies.
- **Consequences.**
  - An arm could be promoted purely because R-mask rescued its collapsed argmax path. That is a deployment effect credited to training.
  - Each arm gets three modes to try against one fixed bar, a multiple-comparisons advantage.
- **Change.**
  - Compare each arm with the **bootstrap under the same deployment mode**, and require the arm to beat the best of {bootstrap × every A1 mode}. Phase F already uses that comparator, so make promotion consistent with it.
  - Register one primary mode per arm **before** training.
  - Report the full 2×2 (training arm × deployment mode) so that training and deployment effects stay separable.

**F2 [MAJOR]: A2-reh is confounded with DAgger distillation of a stronger supplied teacher.**
- **The teacher itself is fine.** dep_reuse reads only the `DepObservation` and the public `applicable()` rule (checked in `DepReference.choose`/`_obtain`), so it is not privileged-state supervision.
- **The problem is the baseline gap.** The teacher's screening utility is .905, against the bootstrap's .872. CE on the policy's own visited states is DAgger, which can close part of that gap with **no RL contribution**. A promotion would then read as "RL improved on imitation" when it is "on-policy imitation improved on off-policy imitation".
- **Change.**
  - Add a matched **DAgger-only control**: same λ·CE and same visited-state source, RL loss off. Budget it at about the cost of one arm, or shorten it.
  - Promote A2-reh only if it beats both the bootstrap and DAgger-only.
  - Report dep_reuse as the supplied ceiling in every table.
  - Two further points:
    - The rehearsal states are **sampled** rollout states, but the collapse lives on the policy's **own greedy** states (P2a Part D: KL rises only on its own states). State which is used, and consider adding greedy-rollout states.
    - Preflight that `teacher.choose` always returns an in-catalog action on RL-visited (loop) states. `collect_teacher` raises rather than repairing, which is correct, but it is untested off-distribution.

**F3 [MAJOR]: Without the safety rail, the improvement arms are likely to collapse, which confounds them.**
- **Why the base matters.** Every arm changes one thing relative to C0, which collapses under greedy after about attempt 25. A2-crit and A2-reh without the anchor can fail for the known reason (greedy argmax drift) before any improvement signal is visible. P2a's own next step was "keep the anchor or a self-state trust region as the safety rail, and add one mechanism aimed at (b)".
- **Change.** Keep A2-ent relative to C0; that is the right base for asking "is the anchor just entropy control?", and it should be compared with both C0 and C1. Run the improvement arms (reh, crit and a repaired dep; see F4) relative to **C1** (anchor 0.3). Each arm is then still a single-change arm, relative to a declared base.
- **If only four arms fit the budget,** a self-state trust region (KL to the bootstrap measured on the current policy's greedy-rollout states) targets the localized P2a mechanism more directly than A2-dep does.

**F4 [BLOCKER]: A2-dep's off-policy correction is not feasible as specified.**
- **What the code does.** `batched_on_policy` plus `actor_critic_objective` is on-policy REINFORCE with a Monte Carlo return and a learned baseline. It stores only the current-policy `logp`, with no behaviour log-probability, ratio or truncation machinery.
- **Why the behaviour policy breaks it.** R-mask is greedy with a mask, so the behaviour policy is **deterministic**: μ(a|h) ∈ {0, 1}.
  - The ratio π/μ is undefined for unchosen actions.
  - There is no exploration, so there is no policy-gradient signal about the alternatives.
- **Change: make it exactly on-policy.** Define the masked policy π_M(a|h) ∝ π(a|h)·1[a ∉ M(h)]. M(h) is a public function of the history, so π_M is a legitimate policy.
  - **Train** by sampling from π_M and using log π_M, the renormalized log-probability. No importance weights are needed; it is a one-line change to the distribution in `batched_on_policy`.
  - **Deploy** greedy + R-mask.
  - **Separability.** This changes the training distribution and the deployment rule together. Also evaluate the trained model under plain greedy and plain sampled, and report whether the gain survives without the mask.

**F5 [MAJOR]: The A1 recovery rules are public-only but under-defined.**
- **R-mask, as written,** masks "the flagged action_key" and releases the mask "when the relevant state changes". The flag, however, fires **after** the no-progress step is taken. Registration needs a precise ex-ante rule.
- **Change: key masks by `(signature, action_key)`.**
  - After step t is flagged (pre-state signature s, key a), mask a whenever the current signature equals s.
  - No separate release rule is then needed: events, new information and calls change the signature and release the mask automatically.
  - With this rule the P1 2-cycles are handled: the finish_by 8/9 alternation is masked at each endpoint, and the r2 repeated `retrieve` is masked at its state.
- **Also register:**
  - a cap on masked keys per signature;
  - the fallback when every non-terminal key is masked, e.g. the unmasked argmax or abstain, declared in advance;
  - that `verify`/`abstain` are never masked;
  - how the diagnostic’s own compute is charged (F16).
- **R-sample** needs a paired sampling stream per world: reuse `sampling_rng_seed`, with a new, registered `sampling_seed`.
- **Label.** Both rules are **supplied recovery computations** written after seeing the P1/P2a loops. Any A1 or Track C gain from them goes in the "supplied" column.

**F6 [MINOR]: A-track details to fix.**
1. **C0/C1 must still be re-evaluated.** "C0 and C1 exist from P2a and are not re-run" holds for training. They must still be **re-evaluated** on the new A2 screening worlds, under R-mask and with diagnostic v1. P2a's numbers use other worlds and the P2a definition.
2. **Deployment selection.** A2 must register a checkpoint-selection rule, and report **final and selected separately**. The P2a audit found the .02 margin on 128 dev worlds too coarse: C0's deployed checkpoint lost .049 on new worlds. Use ≥ 256 dev worlds and include no-progress/step-to-cap in the gate.
3. **Statistics.** The utility margin of +.01 on 512 IID worlds is about 5 worlds of success. Report paired (world-matched) bootstrap CIs. Label promotions as screening, and treat Phase F as the only claim.
4. **A2-crit.** The value head shares the encoder, so "actor frozen" must mean the shared trunk is frozen too, with only value-head parameters trainable. Verify that the policy logits are bit-identical after warm-up. Count the W warm-up updates and their rollouts in optimizer exposure and CPU.
5. **A2-ent.** Specify a two-sided target (C0's entropy rises .82 → 1.60), the states on which entropy is measured, and the multiplier's learning rate. Log the multiplier trace.
6. **§0 fact 2 and Track C bases.** P1 X1-r1 did **not** collapse. "P1 RL finals r0–r2 … loop-prone under greedy" is true of r0 and r2 only. Treat r1 as a separate stratum.

## Track C

**F7 [BLOCKER]: Environment copy fails on the executor, and branch determinism has two gaps.**
- **Why `deepcopy(env)` fails.** Training and evaluation use `executor = partial(depworld_executor, execute_call=solver.execute)`, where `solver` is a `BoundedSolver` holding a live `multiprocessing` Process and Pipe. `deepcopy(env)` deep-copies the partial's bound method and therefore the solver instance: it either raises or duplicates or orphans a worker.
- **Everything else copies correctly:**
  - `_handle_rng` (`random.Random`) keeps its state, so branch record handles equal the main line's;
  - the frozen `DepSpec`, the dicts/lists and `_cache`.
- **Determinism gaps:**
  - (a) Solver calls carry a **2 s wall deadline**, and a missed deadline returns `timeout`/`unknown`. Under 24-core contention, a branch can therefore diverge from the main line nondeterministically.
  - (b) `_solver_cpu` is process-time, used for logging only; utility uses work units. That is fine, but it must be excluded from label equality tests.
- **Change.**
  - Add a `clone()` that uses `deepcopy(env, memo={id(env._executor): env._executor})`, or that re-attaches a shared executor.
  - During label generation, wrap the executor in a **pure memo cache** keyed by (primitive, canonical problem, budget). The solver is a deterministic function of work units, so the cache is exact. It removes wall-clock nondeterminism and cuts branch CPU sharply.
  - Test that clone + replay reproduces the main-line trajectory bit-for-bit.
  - Branching must also copy the actor's recurrent state (the lightweight family's `score()` takes `hidden`), the R-mask/diagnostic state and the appraisal GRU state. Test this.

**F8 [BLOCKER]: Hindsight guard for branch labels.**
- **Why the labels are hindsight.** Each branch label is a **single-world** Monte Carlo return. Given the history, it is a hindsight quantity: it conditions on the true hidden world. Regressing Q̂ on these returns with MSE is legitimate; its target is E[return | I_t, u], the belief-conditioned value, because label states are drawn along rollouts on prior-sampled worlds.
- **What is not legitimate.**
  - Turning per-world branch comparisons into **class labels** ("u* = argmax of this world's branch returns").
  - Training the controller by imitation of those per-world argmaxes.
  - Both are hindsight, and identical visible histories would carry incompatible labels.
- **Change.**
  - Register Q̂ training as **regression with a proper scoring rule** on paired branch returns, with common worlds across the u branches. Consider regressing the advantage Q(u) − Q(u₀) to cut variance.
  - Register that no classification target is derived from per-world outcomes.
  - For Track C, restate the "no incompatible labels" test as: *model inputs are a function of visible history only, and targets are samples of a declared conditional expectation*.
  - Put a **stop-gradient** between control utility and the appraisal heads, so appraisal can never be pushed optimistic.

**F9 [BLOCKER]: The default action and the declared continuation are inconsistent.**
- **The inconsistency.** Q̂ labels use the continuation *greedy + R-mask*, but the controller's default is u = *greedy*. At flagged states, the default then does worse than the continuation it was labelled under. This leads to two failure modes:
  - **Trivially false:** learned control loses to the fixed R-mask rule whenever the margin gate keeps it at "greedy".
  - **Trivially true on paper:** learned control merely re-derives R-mask and is credited as "learned".
- **Change.**
  - Define u₀ = **follow the declared continuation D₀ (greedy + R-mask)**. The other u values are deviations from D₀.
  - The appraisal-only arm then behaves as D₀.
  - The success criterion becomes "beats D₀ and every other fixed rule".
- **Consequence.** Deployment re-applies argmax Q̂ at every step, but labels are "one deviation, then D₀". The controller is therefore a one-step rollout improvement over D₀. That is well-defined and should be named as such in the report.
  - Label states come from base-policy rollouts, while the controller visits its own states. Evaluate calibration on **controller-induced** states too, and allow one registered DAgger-style relabelling round if the budget permits.

**F10 [MAJOR]: The Track C comparison can be decided by the setup rather than by learning.**
- **Ways it could be trivially true:**
  - margin or c_meta tuned on held-out worlds;
  - label worlds overlapping evaluation worlds;
  - comparing against greedy only on collapsed bases (P1 r0/r2 greedy ≈ 0);
  - the metacontroller's compute left uncharged while R-mask's diagnostic is charged, or the reverse.
- **Ways it could be trivially false:**
  - **ceiling:** X1 bootstraps are at .95 greedy, and dep_reuse is .979/.905, so utility headroom over the best fixed rule is about .02–.03, below what 512 worlds resolves in ≥ 2/3 lineages;
  - c_meta charged at a width-1024 actor forward for a ≤ 64-unit GRU;
  - the default mismatch in F9.
- **Change.**
  - Register the success criterion **per base stratum** (competent X1 boots; loop-prone P1 r0/r2; stable P1 r1).
  - Run a power calculation from A1's paired variances **before** registering.
  - Pick the margin and c_meta on dev worlds only.
  - Charge the metacontroller in proportion to its FLOPs, using the same unit as `neural_work_per_forward`.
- **Add a matched-rate random-intervention control:** intervene at the same rate and with the same u mix as learned control, at random states. This is the key causal test that *where* it intervenes matters.
- **Add an "automatic" appraisal-threshold rule:** intervene (sample or mask) when predicted P(success) < τ, with τ tuned on dev. This is the brief's "automatic control" comparator for the deliberate-regulation deliverable.

**F11 [MAJOR]: The causal tests need to be specified.**
- **Telemetry removal.** "Remove each telemetry group" must say whether it means retraining without the group (necessity), zeroing it at test (reliance), or both. Report both.
- **Shuffling.** Add a test-time **shuffled-telemetry** condition (permute across episodes) as well as the shuffled-target training control.
- **Calibration.** "Reliability error ≤ .05" needs a binning scheme, a minimum count per bin and a held-out set of ≥ 512 episodes per stratum.
- **Predicted vs actual effect.** On held-out states, use fresh branch labels as the check. This is legitimate as evaluation, and must be charged.

**F12 [MAJOR]: The label budget is too small at full scale.**
- **Estimate.** P2a screening cost is about .23 core-s per episode. Branching 4 interventions (K = 4 draws for sample-step, about 7 branches) at every state of about 30-step episodes, each run to the end, costs about 100 episode-equivalents per labelled episode, or about 20–25 core-s. The 15k line then buys only about 600 labelled episodes across 6 bases.
- **Change.**
  - Use the solver memo (F7).
  - Subsample 3–5 states per episode, stratified towards flagged or low-margin states, with weights recorded.
  - Batch the branches as a vectorized environment batch through `batched_episodes`.
  - Smoke-measure core-s per label before registering the count.

## Track B (probeworld)

**F13 [MAJOR]: Exact DP is tractable only if the family stays very small; case labels and splits need definitions.**
- **Minimal family.** The family below still tests every required feature, with exact DP by memoized recursion over (belief, query index, structure flag, remaining budget):
  - |Θ| = 4: heuristic-solvable; feasible-hard (solves only at the large budget); infeasible; trap (the heuristic returns a plausible wrong answer).
  - Discrete public features set the prior.
  - Per query, at most about 5 decisions: probe (failure detected with prob. q); exact at budget b₁ or b₂, where timeout at b₁ is possible for "feasible-hard", so **timeout = unknown**; inspect (a binary noisy signal); commit (irreversible, loss L if wrong); abstain.
  - Episodes of k ∈ {1, 2, 4, 8} iid queries.
  - "Build" pays C_build once, then lowers the per-query exact cost. The only cross-query coupling is the structure flag, plus one binary shared latent in the "correlated heuristic failure" condition.
  - Beliefs are finite: the posteriors reachable from the discrete signals.
  - Drop "partial constraint propagation" from v1, or model it as removing one type from the support. Otherwise the instance state space grows.
- **Required tests.**
  - Posterior after `timeout` keeps mass on feasible-hard: V* differs from the posterior where infeasibility is certified.
  - Labels are a function of the visible history.
  - DP values match brute-force enumeration on tiny cases.
- **Case types must be functions of (I_t, a_t, I_{t+1}) with a registered ε in price units:**
  - (a) Q*(I_t, a_t) < V*(I_t) − ε;
  - (b) a_t ε-optimal at I_t and its outcome is a failure;
  - (c) a_t ε-optimal, and an exogenous event or new evidence (not a_t's own failure outcome) makes the planned next action non-optimal at I_{t+1}. "Strategy" is undefined under ties, so use the optimal-set change;
  - (d) ε-optimal and succeeded.
- **Splits.** The "held-out combinations of case types" split must be defined by **generator parameters** (event × unreliable detection × k × price region). Case types depend on the policy's behaviour, so they cannot define a split.
- **Define ρ_k** explicitly, e.g. (C_build + k·c_exec_structured) / (k·c_exact).

**F14 [MINOR]: The factorization ladder must be matched and budgeted.**
- **Capacity.** Every rung carries all auxiliary heads, with zero loss weight where unused, so capacity is identical.
- **Exposure.** Every rung gets identical environment episodes and optimizer steps. Oracle DP label compute is charged as offline cost, not as episode utility (F16).
- **L0.** Make L0 advantage actor-critic with the same value head, not bare REINFORCE, so the rung measures supervision, not variance.
- **Budget.** On a vectorized numpy environment with a small torch model, about 1e5–1e6 episodes in ≤ 600 core-s per run is realistic, so 15 runs fit in about 9k. Report L0 learning curves and whether it plateaued. If it did not, say "L0 under-trained at matched exposure", not "outcome-only fails".

## Progress diagnostic v1

**F15 [MAJOR]: The signature choices and the P2a equivalence claim.**
- **Applicability values.** They are functions of components already in the signature: requirements known, known items, selection, known edges, position, versions (`relations()` reads only these). Including them therefore does not break cycle matching, but it is redundant and costs CPU per step. Keep them out of the hash, and assert the redundancy in a test.
- **Records by handle.** Every `call` mints a new handle from `_handle_rng`, so keying by handle makes an **identical deterministic re-call** look like progress.
  - Canonicalize records by (primitive, snapshot, depends_on, budget, status).
  - Classify a repeat call with unchanged inputs as no-progress under the "retry" clause.
  - This is a documented divergence from P2a, where any call breaks a cycle.
- **`think`.** The latent (think) class diverges from P2a. `think` is in the d1 catalog, and P2a counts an accepted `think` with no state change as idempotent or short-cycle. A policy could then loop on `think` undetected.
  - Report a separate "latent-stall" count, e.g. more than 2 consecutive thinks with an unchanged signature.
  - List this among the documented differences from P2a.
  - Show the per-class deltas on the P2a rows.
- **Classes are not mutually exclusive.** A triggered uncommit can return to a prior signature. Register a precedence order, e.g. rejected > short-cycle/idempotent > solver-work > useful-inspection > triggered-revision > other.
- **Terminology.** "Justified revision" is definable from public data (event/rejection/infeasible record within a registered window), but it is a *trigger*, not a rationality judgement. Rename it "triggered revision", and reserve "justified" for Track B, where Q* exists.
- **The time field.** "requirement versions it was built against" and `created_step` must not leak step counts into the hash. Drafts carry `created_step`, so exclude it explicitly.
- **Travel.** Keep excluding travel, as P2a did, so that an A→B→A move is a no-progress cycle. Say so.

## Accounting, budget, schedule

**F16 [MAJOR]: Charging rules need to be written down.**
- **Two ledgers.** Oracle label compute (Track B DP, Track C branching) is an offline training cost in the CPU ledger. It must never be subtracted from episode utility. Keep in-episode utility (task costs + neural + metacontrol compute) and offline cost as two ledgers, and report both.
- **c_meta vs Q̂.** When Q̂ targets already include the downstream charges of u (they do, since branches run to the end under `evaluate()`), c_meta(u) in the decision rule must contain only charges **not** already in the label. The metacontroller forward is u-independent, so it drops out of the argmax. Otherwise costs are counted twice.
- **R-mask's diagnostic compute** is charged identically in A1, A2 and Track C.

**F17 [MAJOR]: Budget and GPU realism.**
- **CPU.** The binding constraint is 172.8k core-s, which averages about 2 cores over 24 h, not the 24 cores.
  - The dev/test line (6k) is low: 5–6 subagents running the torch test suite (247+ tests), several times each, plus builds. Set it to about 10–12k, and require targeted test selection.
  - Phase F at 30k covers about one promoted arm: 3 bootstraps at about 0.6k each, 3 RL runs at 4–8k each, and a sealed evaluation at ≥ 512/cell. It does not cover a Track C confirmation. State that Track C main claims either fit inside Phase F's 30k or remain screening.
- **GPU.** Occupancy is the union of wall intervals; the ceiling is 12 h. P1's evaluation alone ran 2.8 h of wall. If small Track B/C models or evaluations run `device: cuda` scattered over the day, the union can exceed 12 h.
  - Run Tracks B/C and the evaluations CPU-only where possible. The jobs are CPU-bound anyway: neural work is 16–20% of CPU.
  - Cluster the GPU jobs into overlapping windows, and track the running union in `budget.json`.

**F18 [MAJOR]: Schedule critical path.**
- **A2 gated on the optimizer.** Gating A2 on the Phase A optimizer puts an unproven engineering task on the critical path. The optimizer is defined to be bit-identical, so it only saves CPU.
  - Launch A2 on the current code as soon as the arm code and tests exist.
  - Adopt the optimizer for Phase F only if it has passed equivalence by then.
- **Phase F latest start.** Bootstraps (~10 min) + RL (1–2.5 h concurrent) + sealed evaluation (P1: 2.8 h) + audit + report (≥ 3 h) mean Phase F must launch by about T+15 h (≈ 09:45Z on 09-27). A2 must be analysed by then; register that cutoff now.
- **Track B.** Track B builder + DP + tests + ladder is the item most likely to slip. Timebox it: generator + DP + tests by T+8 h, or descope to k ∈ {1, 4} and no assumption-violation conditions.

**F19 [MINOR]: Deliverables and guardrails not yet covered (brief §24).**
1. **"Measured response to prices and reuse."** In this design it comes only from Track B, because the depworld version is Phase E, "if warranted". Say so explicitly, or add a cheap A1 price sweep over the frozen bootstraps: work_price/compute_price ×{0.5, 2}, evaluation only.
2. **"Deliberate regulation over automatic control."** This needs the appraisal-threshold arm (F10); Phase D is otherwise deferred.
3. **"Sharply localized reason for the absence of improvement."** If no A2 arm is promoted, pre-register the Part D-style diagnostics on the A2 finals: own-state KL, argmax disagreement and critic calibration, as the fallback deliverable.
4. **Supplied/learned/generalized ledger.** Start it now. It should list, for example: R-mask, R-sample, diagnostic v1, the dep_reuse rehearsal teacher, the oracle branching labels and the Track B DP labels as **supplied**.
5. **Transfer floor.** Set absolute floors for foreign4 and events: for example, success ≥ .85 and not below bootstrap − .05 on the same worlds.
6. **Seeds.** Register fresh, disjoint seed ranges for A1, A2 screening, Track C labels, dev and evaluation, Track B and the Phase F sealed worlds. They must not overlap 110M/120M/1.99e9/2.05e9/3.0–3.9e9.
7. **Telemetry immutability.** Enforce it with a test: the metacontroller receives a read-only copy, and `evaluate()` output is identical with or without the controller attached when u = u₀.
8. **"Do not claim temperature repair".** Covered. Also avoid "entropy repair" language if A2-ent passes only under sampled deployment.

---

## Top findings in brief
1. **F4 (BLOCKER).** A2-dep's off-policy correction is infeasible: the code is on-policy MC REINFORCE with no behaviour log-probabilities, and R-mask is deterministic. Train on-policy on the masked distribution π_M instead.
2. **F7 (BLOCKER).** `deepcopy(env)` copies the executor's live BoundedSolver process. Clone with a shared executor, memoize solver results (this also removes the 2 s wall-deadline nondeterminism), and copy the actor, diagnostic and GRU state.
3. **F9 (BLOCKER).** The controller's default (greedy) differs from the labels' continuation (greedy + R-mask). Make u₀ = D₀.
4. **F8 (BLOCKER).** Per-world branch returns are hindsight. Use them only as regression targets for E[· | I_t], never as class labels, and put a stop-gradient on the appraisal heads.
5. **F1 (BLOCKER).** The promotion comparator must be the bootstrap under the same deployment mode, with the mode registered before training.
6. **F2/F3 (MAJOR).** A2-reh needs a DAgger-only control, since the teacher beats the bootstrap. The improvement arms should run on the C1 anchor base, or they will collapse for the known reason.
7. **F10/F12/F17 (MAJOR).**
   - Track C headroom on the X1 bases is about .02–.03, so stratify the success criterion and run a power calculation first.
   - Add matched-rate random-intervention and appraisal-threshold controls.
   - The label and dev budgets are too small.
   - GPU union-wall occupancy needs a plan.
