# Extended-04: metacognitive control, rational backtracking, and the value of structure

**Status: ACTIVE**, started 2026-09-26T18:43Z on the pro6000. Design: [design.md](design.md). Decisions: [decisions.md](decisions.md). Ledger: [budget.json](budget.json).

## Mission (from the user's campaign brief, 2026-09-26)
Extend Tensegra from competent but fragile computational orchestration toward a system that can **deliberately regulate its own computation**. It should use an accurate representation of its computational state to choose between these options:
- continue;
- gather information;
- explore another candidate;
- formalize a subproblem;
- crystallize a candidate into exact execution;
- allocate or extend a solver budget;
- release a commitment;
- revise dependencies;
- act or answer.

The choice should improve externally verified utility net of total cost.

> Regulating how information influences behaviour is not the same as altering the information or its apparent certainty.

**Tracks:**
- A: deployment-aware preservation and improvement.
- B: synthetic rational exploration, switching and recovery.
- C: learned metacognitive appraisal and control.

**Phases A–F** follow the brief's execution plan. **Deliverables** follow brief §24: improvement over imitation or a sharply localized reason for its absence; a rational-probe benchmark; a stagnation-recognizing controller; calibrated metacognitive predictions; causal evidence for deliberate regulation over automatic control and heuristics; a measured response to prices and reuse; and an audited supplied/learned/generalized account.

## Authority and limits
- **Autonomy.** Routine experimental decisions are delegated; no per-result approval is needed.
- **New metered window:** ≤ 48 CPU core-h, ≤ 12 GPU device-occupancy h, ≤ 24 h elapsed, ~20% reserved for confirmation, audit and closure. Extended-03's ledger stays separate and unchanged.
- **Hosts.** The pro6000 only. **No training on either GB10** without separate authorization. Do not disturb unrelated desktop or research workloads.
- **Metering.** All remote work is metered, including subagent development, tests and audits (`bin/metered.sh`). Estimates are disclosed where exact metering is impossible.
- **Not in scope:** paid services, purchases, large pretrained-model or brain-model integration, unrelated infrastructure.
- **Coordination.** One coordinator (root) owns shared resources and all main launches. Parallel work happens in isolated worktrees.

## Standing guardrails (brief §§4, 9, 14, 16, 22, 23)
- **Temperature.** argmax(ℓ/T) = argmax(ℓ); temperature acts only through sampling or other distribution-sensitive selection.
- **Values.** Q/V predict outcomes under **named** continuation policies. The agent is never rewarded for optimistic estimates. External utility and accounting sit outside the agent.
- **Labels respect the agent's information:** belief-conditioned over worlds consistent with the visible history; no hindsight labels.
- **Claims.** Main claims need ≥ 3 independent lineages, fresh confirmation worlds and ≥ 512 examples per required cell. Transfer needs an absolute competence floor. Final and selected checkpoints are reported separately.
- **Causal influence is shown by intervention** (remove, shuffle, disable, clamp), not by probes or mutual information.
- **Preserve disagreement across seeds.** Record every failed or capped run. Leave historical results untouched.

## Starting evidence
Extended-03 [report-P1.md](../extended-03/report-P1.md) and [report-P2a.md](../extended-03/report-P2a.md) (audited). Key facts are carried in [design.md](design.md) §0.
