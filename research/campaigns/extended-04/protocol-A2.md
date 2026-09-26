# Protocol A2: improvement screens on lineage X1-r2 (exploratory)

Registered 2026-09-26T19:15Z. The four screens were launched at 19:11Z; **no A2 output has been inspected**. A2-dep is launched later, under the same protocol. See design.md Track A and v2 revisions 1–4.

## Arms (one change each; configs in configs/campaign04/a2-*-x1-r2.json)

| Arm | Base | Change | Hypothesis |
|---|---|---|---|
| a2-reh | C1 (anchor 0.3) | rehearsal λ = 0.5: CE toward the dep_reuse teacher on sampled RL states | Competence-critical supervision keeps imitation-level behaviour while RL moves utility or cost. |
| a2-imit | C1 | rehearsal λ = 0.5, **RL loss off** | Control. Any a2-reh gain attributed to RL must exceed this arm's. |
| a2-crit | C1 | critic warm-up: the first 300 AC updates train the value head only | Withdrawn P2a hypothesis: a miscalibrated early critic harms the actor. |
| a2-ent | C0 (no anchor) | entropy Lagrangian, target 0.8732 nats (the bootstrap's measured entropy) | The P2a anchor effect is (mostly) entropy control. |
| a2-dep | C1 | rollouts sample from the progress-masked policy (on-policy) | Training through the deployed recovery procedure helps. |

The controls C0 and C1 come from P2a on the same seeds, streams and horizon, and are not re-run.

- **Registered deployment mode, fixed before training:** greedy for reh, imit, crit and ent; **R-mask** for dep.
- **Full table:** every arm is evaluated under greedy, sampled (T = 1) and R-mask, and the full arm × mode table is reported.
- **Supplied supervision:** dep_reuse rehearsal is supplied, public-only supervision (ledger).

## Evaluation (screening; non-sealed)

- **Worlds:** fresh screening seeds **140,000,000 + 100,000·i**, 256 worlds per condition, over iid_f0, iid_f2, events_train_kinds_p1 and foreign4. Worlds are identical across policies and modes.
- **Sampling:** sampled mode uses one fixed-seed trajectory per world.
- **Policies:**
  - each arm's **final raw checkpoint**;
  - each arm's **deployed checkpoint** under the P2a rule (the latest tranche checkpoint with development success ≥ bootstrap − .02 and development utility ≥ bootstrap − .02, else rollback);
  - the X1-r2 bootstrap;
  - C0 final and C1 final;
  - the dep_reuse reference.
- **Metrics:** success, utility, total cost, work per success, no-progress rate and episodes (progress diagnostic v1), steps to cap. IID group = {iid_f0, iid_f2}.

## Promotion rule (per arm, final raw checkpoint, IID group)

The arm is promoted to Phase F only if **all** of the following hold:
1. **vs bootstrap under the same mode:** utility ≥ bootstrap + .01, **or** total cost ≤ .90 × bootstrap with success ≥ bootstrap − .02;
2. **vs bootstrap under its best fixed mode** (the best of greedy, sampled and R-mask by utility): utility ≥ that + .005;
3. no-progress episodes ≤ the bootstrap's (same mode) + .02;
4. for a2-reh only: utility ≥ a2-imit (same mode) + .01. Otherwise the gain is attributed to imitation, not RL.

Deployed-checkpoint results are reported, never promoted in place of the final. A2 is one lineage and exploratory. **Claims require Phase F** (3 fresh lineages, sealed worlds, ≥ 512 per cell).

## If no arm is promoted
The Track A deliverable becomes a localized account (design v2 revision 12). It combines:
- the arm × mode table;
- per-arm development curves (greedy vs sampled divergence, as in Part D);
- the entropy trajectories;
- the anchor-vs-entropy comparison (a2-ent vs C0/C1).

## Budget
- **Training:** ≈ 4k core-s per run, 5 runs.
- **Screening evaluation:** ≈ 12 policies × 4 conditions × 3 modes × 256 worlds, ≈ 7k core-s.
