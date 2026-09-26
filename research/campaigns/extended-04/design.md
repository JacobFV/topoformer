# Extended-04 design: metacognitive control, rational backtracking, and the value of structure

**Status: DESIGN v1**, written 2026-09-26 before any implementation. It is under internal review (`review/design-review.md`). Numeric protocol details are registered per experiment in `protocol-*.md` before each main run. Autonomy, ceilings and scope come from the campaign brief ([campaign.md](campaign.md)).

## 0. Carried-forward facts (extended-03, audited)

1. **Imitation works.** It acquired depworld competence: X1 bootstraps reach ≈ .95 greedy IID success.
2. **P1's collapse is a greedy-deployment collapse.** Sampling from the final RL policies keeps ≈ .80–.93 success. It is deterministic, reproduced bit-exactly.
3. **P2a's anchor.** The fixed bootstrap anchor prevented the collapse on one screening lineage by holding the policy near the bootstrap. It raised work per success by 22% and failed the improvement objective. Its *sampled* utility was .827 vs the sampled bootstrap's .815: a small, unconfirmed screening increase. This is kept visible.
4. **Early checkpoint selection** avoided the worst endpoint, but the selected checkpoint was weaker than the bootstrap on fresh worlds.
5. **The critic-gradient explanation is withdrawn.** Critic warm-up remains a hypothesis.
6. **Applicability relations are supplied computations** over public information.
7. **Not established:** crystallization, affect, or any programmable-attention advantage.

## 1. Vocabulary (kept distinct everywhere)

| Object | Meaning | Who may change it |
|---|---|---|
| logits ℓ, policy probabilities π | the base actor's scores and distribution | training only |
| deployment procedure D | the rule mapping (ℓ, history) → action: greedy, sampled(T), recovery rule, metacontroller | a registered experimental variable |
| telemetry d_t | measured, immutable facts (work, entropy, margins, repeat/cycle evidence, budget, feedback) | nobody; the recorded facts are append-only |
| appraisal m_t | learned *predictions*, e.g. P(eventual verified success), expected remaining cost, Q̂(I_t, u) | fitted to outcomes; never rewarded for being optimistic |
| intervention u_t | a deliberate change to *computation*, e.g. sample instead of argmax, mask a stagnant candidate, allocate another step, stop | the metacontroller |
| external utility U | verified success − (action + observation + travel + work + compute + metacontrol) costs, each charged once | the environment/evaluator only; outside the agent |

- **Temperature.** argmax(ℓ/T) = argmax(ℓ) for every T > 0, so temperature acts only through sampling or another distribution-sensitive mechanism. No temperature-plus-argmax "repair" is ever claimed.
- **Policy probability is not action value.** Q̂ always names its **continuation policy**.

## 2. Tracks and what each can establish

### Track A: deployment-aware preservation and improvement (depworld)

**A0. Frozen references.** These are identified by sha256 and never retrained:
- X1 bootstraps r0–r2;
- P1 RL finals X1 r0–r2;
- P2a C1 (anchored, r2);
- the references dep_reuse, dep_recompute and dep_greedy.

**A1. Deployment matrix (evaluation only, exploratory).** Every A0 learned policy under:
- greedy;
- sampled at T = 1 (one fixed-seed trajectory per world; worlds are the unit of independence);
- **R-mask**, a public-information recovery rule: greedy, but when the progress diagnostic (§4) flags an idempotent repeat or short cycle, the flagged action_key is masked and the argmax is taken over the rest. The mask is released when the relevant state changes.
- **R-sample**: greedy, but a single sampled action at flagged states.

Conditions: iid_f0, iid_f2, events_train_kinds_p1, foreign4. There are 256 fresh worlds per condition, from a new seed range. Output: success, utility, total cost, work per success, no-progress episodes and cycle lengths, per lineage.

**A2. Improvement screens (training; one lineage, X1-r2, which is known to collapse; full 1,800 updates; greedy + sampled + R-mask evaluation on fresh screening worlds).** Each arm changes *one* thing relative to C0, the exact P1 recipe:

| Arm | Change | Question |
|---|---|---|
| A2-reh | supervised rehearsal: add λ·CE(teacher action) on the public states visited in each RL batch. The teacher is dep_reuse on the same public observation. | Can competence-critical supervision keep imitation while RL still moves utility? |
| A2-ent | entropy regulation: hold policy entropy at the bootstrap's level (a Lagrangian on entropy) with no anchor | Is the P2a anchor effect just entropy control? |
| A2-crit | critic warm-up: fit the value head for W updates (actor frozen) before RL | Tests the withdrawn hypothesis directly. |
| A2-dep | deployment-aligned RL: collect training rollouts under R-mask (off-policy correction via the behaviour-policy ratio, clipped), so training sees the deployed procedure | Does training through the actual deployment procedure help? |

C0 and C1 exist from P2a and are reused as controls; they are not re-run.
- **Promotion rule, registered per protocol:** an arm is promoted if, under its registered deployment mode, IID-group utility ≥ bootstrap-greedy utility + .01, or total cost ≤ .9× at success within .02, with no-progress episodes not above the bootstrap's + .02.
- **Promoted arms are confirmed in Phase F** on 3 fresh lineages (new bootstraps) with sealed worlds, against the strongest matched baseline: the bootstrap under its best fixed deployment.

### Track B: synthetic rational probing, switching and recovery (probeworld)

A new small environment where **belief-conditioned optimal values are exactly computable**.

- **Instance.** A hidden instance type θ from a small finite set. The public features give a declared prior P(θ | features). Public prices: c_probe, c_exact, c_inspect, c_prop, error loss L, and the reuse horizon k.
- **Actions:**
  - **cheap heuristic probe:** solves iff θ ∈ H. Its failure is detected reliably, or with probability q in the unreliable-detection condition.
  - **exact solver:** budgeted. Statuses solved / infeasible / **timeout = unknown**, not infeasible.
  - **inspect:** a noisy or partial reveal of θ.
  - **partial constraint propagation:** shrinks the instance and changes the posterior.
  - **commit answer:** irreversible; a wrong answer costs L.
  - **abstain.**
  - **build reusable structure:** C_build, then a cheaper per-query execute + return + verify for the remaining queries in a k-query episode.
- **Events.** A requirement change invalidates a declared subset of computed results (case c).
- **Labels.** Q*(I_t, a) is computed by exact dynamic programming over the finite belief space under the declared prior. Ties produce **sets of optimal actions**. The continuation is always the optimal policy π*, named in the label. The case type is assigned per step from belief-conditioned values:
  - (a) an unjustified choice: Q < V* − ε at the visible history;
  - (b) a rational probe that failed;
  - (c) a strategy invalidated by an event or new evidence;
  - (d) direct success, continue or terminate.
- **Direct-success instances are included** in every split, so "fail first" cannot be a shortcut.
- **No two identical visible histories carry incompatible labels.** This is checked by test: labels are a function of visible history.
- **Assumption violations as separate conditions:** unreliable failure detection, irreversible probe side effects, correlated heuristic failure.
- **Splits:**
  - price combinations held out by region, with ρ_k sweeps;
  - held-out reuse horizons k;
  - held-out combinations of case types, a new composition split never inspected before.
- **Metrics:**
  - utility regret against V*;
  - absolute success and total cost;
  - justified vs unjustified switches;
  - calibration of the value head;
  - cost-sensitivity curves of strategy choice against ρ_k.
- **Model.** A small recurrent or attention policy with a value head over the visible history plus public prices; parameters are disclosed.
- **Factorization ladder (§20 of the brief), 3 seeds each, matched inputs and optimizer exposure:**
  - L0: outcome-only (REINFORCE on U);
  - L1: + optimal-action-set imitation;
  - L2: + dependency/stage-transition supervision;
  - L3: + switch/rollback supervision;
  - L4: + counterfactual Q supervision for all actions.
- **Reporting.** Teacher-forced and free-running results are reported separately. Label-generation compute is charged.

### Track C: learned metacognitive appraisal and control (over a frozen competent depworld policy)

- **Telemetry d_t** (measured, versioned):
  - top-1 probability, entropy, top-1/top-2 margin;
  - steps used and remaining;
  - cumulative work and cost;
  - progress-diagnostic flags and counts (§4);
  - last feedback status and reason;
  - new-information flag;
  - event flags;
  - stage commitment state.
- **Appraisal m_t:** a small GRU (≤ 64 hidden) updated every step automatically (u = 0 path). It predicts:
  - P(eventual verified success | continue under the default deployment);
  - the expected remaining cost;
  - **Q̂(I_t, u)** for u ∈ {greedy-step, sample-step, mask-top-step, stop}, each followed by the **declared continuation** (the default greedy deployment with R-mask). The continuation is named in every label.
- **Labels.** Counterfactual branching over copied environment states along base-policy rollouts: each intervention is applied, then the declared continuation is run to the end.
  - Sampled interventions are averaged over K draws, and label variance is recorded.
  - The environment copy is an **oracle simulation privilege of label generation only**. It is charged, and never available at evaluation.
- **Intervention pathway:** u_t = argmax_u Q̂(I_t, u) − c_meta(u), with the **default u = greedy** unless the predicted gain exceeds a registered margin. The metacontroller's own forward is charged as compute.
- **Arms**, all with the same inputs:
  - fixed greedy;
  - fixed sampled;
  - R-mask and R-sample (public rules);
  - **appraisal-only** (predictions logged, no control authority; behaviourally greedy): calibration only;
  - **learned control**.
- **Causal tests:**
  - remove each telemetry group;
  - a shuffled-target training control;
  - disable the control pathway;
  - clamp u;
  - compare the predicted with the actual intervention effect on held-out states.
- **Bases:** X1 bootstraps r0–r2, which are competent, and P1 RL finals r0–r2, which are loop-prone under greedy. Three metacontroller seeds per base.
- **Success criterion:** learned control beats the best of {greedy, sampled, R-mask, R-sample} on held-out worlds in utility, at matched or better success, in ≥ 2/3 base lineages. Appraisal is calibrated (reliability error ≤ .05) and predictive of failure and of the value of switching.

### Later phases (only if earlier evidence warrants; they stay on the roadmap otherwise)

- **Phase D:** automatic appraisal alone vs deliberate regulation, adding factorization and counterfactual supervision selectively (Track B ladder results feed this).
- **Phase E:** ρ_k and reuse-horizon shifts in depworld (price-condition sweeps using the existing work_price/compute_price); new attempted-strategy compositions; local crystallization/release. The last is restricted to candidate-local commit/release in probeworld, with transactional rollback vs irreversible commit contracts.
- **Phase F:** confirmation of promoted claims. At least 3 fresh lineages, sealed worlds, ≥ 512 examples per required cell, with final and selected checkpoints reported separately. Transfer claims need an absolute competence floor.

## 3. Throughput first (Phase A)

- **Profile P2a again, by phase:** encode ~22%, collate ~18%, environment step ~15%, development evaluation ~11%, export ~11%, neural ~16–20%.
- **Optimize** observation/action encoding (vectorized feature construction and caching of immutable per-world content only, keyed by requirement version and record status so provenance cannot go stale), collation, and trace export (compact lossless records).
- **Keep the reference encoder.** An equivalence test requires bit-identical encoded tensors on randomized trajectories, including the d1-noapp/d1-noattempt masks. A replay of a short P1-style tranche must produce identical parameters.
- **Width stays 1024** for the actor. Metacontrollers are small and disclosed.

## 4. Progress diagnostic v1 (versioned, shared by Tracks A and C)

- **Relevant-state signature.** A canonical hash of the public decision state:
  - draft contents and pending choices, with the requirement versions they were built against;
  - commitments and verification status;
  - the retrieved-record set and its applicability relation values;
  - position;
  - requirement versions and fired events;
  - the information-availability set (what has been inspected);
  - outstanding computation state (record statuses by handle).

  It **excludes** step counters, log lengths and cumulative cost, so that repeated states can match. The equivalence relation is documented in code.
- **Per-step classification:**

| Class | Meaning |
|---|---|
| rejected | environment rejection, with its reason |
| idempotent-success | accepted, signature unchanged |
| short-cycle | accepted, signature equals one seen in the last W = 6 steps, with no intervening solver call, new information or event |
| useful-inspection | new information appears in the availability set |
| solver-work | a call on a new problem, or with a larger budget, or after a dependency change |
| justified-revision | uncommit/redraft after an event, a rejection-driven infeasibility, or a dependency change |
| latent (think) | recorded separately; work charged; never counted as no-progress by itself |
| other-progress | anything else that changes the signature |

- **Retry handling.** Suppression applies only when the relevant inputs are unchanged. A retry after changed evidence, budget, dependencies or a stochastic failure is classified by what changed.
- **Validation.** It is validated against P2a's registered no-progress operationalization. The two are identical on the P2a screening rows, apart from documented extensions, and the references score 0 no-progress.

## 5. Integrity and accounting

- **Inputs.** Each Track C/B model's full encoded input tensors pass a counterfactual-pair preflight before training: telemetry groups present or absent; identical visible histories give identical inputs; no hidden state.
- **Separation.** External utility, validators, held-out targets and budget accounting are outside every agent. Metacontroller telemetry is read-only. Interventions act on action selection or computation, never on recorded facts, costs or certificates.
- **Metering.** All remote work, including subagent development, tests and audits, runs through `~/tensegra-campaign04/bin/metered.sh` or the detached job wrapper, so it leaves receipts. Local no-torch work is estimated and disclosed.
- **Claims.**
  - Promoted claims need an independent raw-metric reconstruction.
  - Exploratory screens are labelled as such.
  - Every failed or capped run is recorded.
  - Historical ext-03 results and hashes are never modified.

## 6. Budget plan (CPU core-s; ceiling 172.8k; ~138k usable before the reserve)

| Item | Estimate |
|---|---:|
| Phase A optimization + equivalence dev/tests | 6k |
| A1 deployment matrix | 8k |
| A2 screens (4 arms) | 12–16k |
| Track C labels (branching) | 15k |
| Track C training + evaluation + ablations | 15k |
| Track B generator/labels + ladder (5 × 3 seeds) + evaluation | 15k |
| Phase F confirmation (3 fresh lineages; boots + best arm; sealed evaluation) | 30k |
| Audits | 10k |
| **Total** | **≈ 115–120k** |

GPU occupancy is tracked as the union of main-job wall intervals.

## 7. Scheduling (24 h wall, deadline 2026-09-27T18:43Z)

Parallel subagents work in isolated worktrees; root is the only launcher of main jobs.

- **Now:** design review; Phase A optimizer; progress diagnostic v1.
- **After review:** the Track B builder, and the Track C label/telemetry builder.
- **Then:**
  - A1 as soon as the diagnostic lands;
  - the A2 screens (concurrent) as soon as the optimizer lands;
  - Track C training once labels exist;
  - the Track B ladder once the benchmark is checked.
- **Last:** Phase F on whatever is promoted, then independent audits and the report.

## 8. Explicit non-goals (this campaign)

- Brain-derived connectivity, brain-state prediction, pretrained-model integration, paid services.
- Affect labels as targets.
- Any claim that machine variables are feelings or consciousness.
- Intrinsic distress, self-preservation or positive self-report objectives.
- A large new architecture before simple controls are established.
