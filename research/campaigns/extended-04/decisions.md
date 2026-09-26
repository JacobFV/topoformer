# Decisions (extended-04)

- 2026-09-26T18:43Z: The user's campaign brief was adopted. The new metered window runs to 2026-09-27T18:43Z (48 CPU core-h, 12 GPU-h, 20% reserve).
  - **State reconciled:** main = 7e9d44f9; no other active coordinator (no worktrees or new branches); the pro6000 is idle.
  - **WSL keep-alive:** the task had stopped (status Ready; WSL uptime 0 min at check). It was restarted and the 63 extended-03 result directories are intact. The keep-alive status is now checked before long jobs.
  - **New remote root:** ~/tensegra-campaign04. `bin/metered.sh` meters all remote dev/test/audit work, closing the extended-03 unmetered gap. The env is reused from ~/tensegra-campaign03/env (torch 2.14.0+cu130).
  - Branch campaign/extended-04 from main.
- 2026-09-26T18:55Z: Design v1 written ([design.md](design.md)). An internal review was commissioned. The Phase A throughput optimizer starts in parallel, since it is design-independent and gated by bit-identity tests. The Track B/C builders and the progress diagnostic start after the review.
- 2026-09-26T19:15Z: **Internal design review** (review/design-review.md: 5 BLOCKER, 11 MAJOR, 3 MINOR). All adopted as **design v2** (design.md, v2 revisions 1–12):
  - promotion compares against the bootstrap under the same mode and under its best mode;
  - an imitation-only control is added for rehearsal;
  - the A2 arms run on the C1 base, except entropy (on C0);
  - A2-dep uses on-policy masked sampling;
  - the environment gets a clone() with a shared solver and an exact solver cache;
  - Track C's default is greedy + R-mask, equal to the label continuation;
  - branch labels are used only as regression targets, with a stop-gradient;
  - Track C criteria are split by base type, with random-rate and automatic-threshold controls;
  - diagnostic fixes: content-keyed records, think counted, class precedence, "triggered revision";
  - probeworld splits are defined by generator parameters;
  - Tracks B/C run CPU-only;
  - the Phase F launch deadline is ~2026-09-27T09:45Z;
  - a fallback "localized reason" deliverable and a supplied/learned ledger are added.
- 2026-09-26T19:20Z: Builders launched in isolated worktrees:
  - **e04-infra:** diagnostic v1, deployment procedures (R-mask, R-sample, masked sampling), clone and cache, A1 tooling;
  - **e04-a2arms:** rehearsal and imitation-only, entropy Lagrangian, critic warm-up, masked-rollout hook, A2 configs;
  - **e04-probeworld:** Track B.

  The Phase A throughput work continues (e04-throughput). All remote dev work is metered.
- 2026-09-26T19:11Z: A2 options merged (6cb0ecce; 268 tests on snapshot 71109534, metered). **Four A2 screens launched:** a2-reh, a2-imit, a2-crit, a2-ent. a2-dep waits for the infra progress tracker. [protocol-A2.md](protocol-A2.md) registered at 19:15Z, before any A2 output: registered modes, the promotion rule against same-mode and best-mode bootstraps, the imitation control, and screening seeds from 140M.
  - Metering note: metered.sh runs commands without a shell, so globs must go through `bash -c`.
