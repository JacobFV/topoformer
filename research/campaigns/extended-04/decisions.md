# Decisions (extended-04)

- 2026-09-26T18:43Z: The user's campaign brief was adopted. The new metered window runs to 2026-09-27T18:43Z (48 CPU core-h, 12 GPU-h, 20% reserve).
  - **State reconciled:** main = 7e9d44f9; no other active coordinator (no worktrees or new branches); the pro6000 is idle.
  - **WSL keep-alive:** the task had stopped (status Ready; WSL uptime 0 min at check). It was restarted and the 63 extended-03 result directories are intact. The keep-alive status is now checked before long jobs.
  - **New remote root:** ~/tensegra-campaign04. `bin/metered.sh` meters all remote dev/test/audit work, closing the extended-03 unmetered gap. The env is reused from ~/tensegra-campaign03/env (torch 2.14.0+cu130).
  - Branch campaign/extended-04 from main.
- 2026-09-26T18:55Z: Design v1 written ([design.md](design.md)). An internal review was commissioned. The Phase A throughput optimizer starts in parallel, since it is design-independent and gated by bit-identity tests. The Track B/C builders and the progress diagnostic start after the review.
