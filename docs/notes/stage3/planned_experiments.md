# Stage 3 / Planned Experiments

**Status:** design-only. Experiments are described but not yet executed. When results arrive, fill in the empty result sections below.

## Domain rollout plan

| Tier | Domains | Rationale |
|---|---|---|
| **Start now** | Doors | Continuity with Stage 1/2; anchor baseline |
| | Gripper | PG3 reports perfect policy; symmetric pick/move/drop |
| | Ferry | PG3 reports perfect policy; board/sail/debark transport |
| **Second wave** | Elevator | More ordering pressure than Ferry/Gripper |
| | Blocks | Richer subgoal interference; stress-tests binding |
| | Travel / TSP | Navigation-style generalized rules |
| **Defer** | Sokoban | Dead-end-heavy; needs search, not direct policy |
| | Depot | Compound logistics + stacking; compositional stress |
| | Towers of Hanoi | Needs recursion/memory (beyond Stage 3 expressiveness) |
| | Probabilistic (River, Triangle Tireworld) | Defer to reactive stage |

Tier 1 provides continuity + two validation domains where PG3 gives ground-truth evidence that lifted decision lists work. All environments are custom Gymnasium implementations (no PDDLGym dependency).

---

## Experiment 1 — Tiny Exact Enumeration

### Goal

Understand the combinatorial landscape of the Stage 3 grammar before training. Mirror the Stage 2 exhaustive analysis ([stage2/exp3](../stage2/exp3_exhaustive_reward_ast.md)) at small bounds across multiple domains.

### Protocol

1. Fix small structural bounds, e.g. $L_{\max} = 2$, $K_s = 1$, $K_g = 1$, $V_{\max} = 2$.
2. For each Tier 1 domain (Doors, Gripper, Ferry), enumerate **all** derivations in the grammar.
3. Compile and evaluate every program on a fixed small instance.

### Metrics

| Metric | Purpose |
|---|---|
| Total syntax count | Raw grammar size |
| Unique normalized policies | After alpha-renaming canonicalization |
| Unique behavior classes | Programs with identical state → action maps |
| Reward histogram | Sparsity of the reward landscape |
| Solve count and density | Feasibility of random search |

### Expected outputs

Three plots per domain, analogous to Stage 2's `stage1_vs_stage2_{reward_histograms, unique_asts, solve_density}.png`:
- `stage3_{domain}_grammar_size.png`
- `stage3_{domain}_reward_histogram.png`
- `stage3_{domain}_unique_ast.png`

### Results

*(Fill in once enumeration runs complete. Include tables of grammar size, canonicalized-policy count, behavior-class count, and solve density per domain.)*

---

## Experiment 2 — Cross-Instance Generalization

### Goal

Test whether lifted policies trained on small instances transfer to larger instances within the same domain.

### Protocol

1. **Train / search** on small instances — Ferry with 2–3 cars, Gripper with 2–3 balls, Doors at $D=3$.
2. Extract the best policy found.
3. **Evaluate** the same lifted policy on held-out larger instances (5, 10, 20 objects).
4. Report solve rate by instance size.

### Expected outcome

Based on PG3 precedent:

| Domain | Train size | Test: 5 obj | Test: 10 obj | Test: 20 obj |
|---|---|---|---|---|
| Ferry | 2–3 cars | pass | pass | pass |
| Gripper | 2–3 balls | pass | pass | pass |
| Doors | $D = 3$ | $D = 4$? | unlikely | no |

**Doors is structurally different.** Ferry/Gripper have per-object goals (`goal:at(c, d)` selects which car to serve). Doors has a single goal literal. A flat decision list might need $\sim 2K$ rules for Doors (one pick + one move per room), in which case $L_{\max}$ must scale with $D$ and the policy does not transfer across sizes. Exp 2 will determine whether a sufficiently rich state-guard vocabulary lets a flat list transfer or whether `ForEach(τ, rules)` is required (see [01_lifted_grammar.md](01_lifted_grammar.md)).

### Results

*(Fill in once training runs complete. Include per-domain solve-rate tables by train and test size, and any failure-case analysis for Doors.)*

---

## Experiment 3 — Cross-Domain Software Reuse

### Goal

Verify that the same grammar/interpreter code runs unchanged across domains — the *framework* side of the generalization claim.

### Protocol

1. Run the identical grammar schema + interpreter + training pipeline on Doors, Gripper, and Ferry.
2. Only the **adapter file** differs between runs.
3. Confirm each domain produces solving policies at small instance sizes.

### Acceptance criteria

| # | Criterion |
|---:|---|
| 1 | Grammar generation is independent of domain-specific masks |
| 2 | Variable typing is enforced by the generic engine |
| 3 | Goal literals are parsed from environment observations |
| 4 | First-applicable-rule semantics are deterministic |
| 5 | The same core module is used unchanged across all 3 domains |
| 6 | At least 2/3 domains produce solvers at small instance sizes |

### Results

*(Fill in once adapter implementations are complete. Note which domains solved, which adapter files were required, and any abstraction leaks discovered.)*

---

## Known limitations of the Stage 3 grammar

The lifted decision-list CFG **cannot express**:

| Limitation | Implication |
|---|---|
| No numeric comparisons | Cannot express "count > 3" or distance-based guards |
| No transitive closure | Cannot express reachability ("$x$ connected to $y$ via path") |
| No universal quantification | Cannot express "for all objects of type $\tau$…" |
| No memory / state machine | Cannot track history across steps; purely reactive |
| No recursion or loops | Control flow is a flat scan of rules |
| **Single-goal domains** | Domains without per-object goals (e.g. Doors) cannot use goal guards to select which object to act on; $L_{\max}$ may need to scale with a flat decision list. `ForEach` is the planned fallback if Exp 2 shows transfer fails |

PG3 (Yang et al. 2022) explicitly notes these same limitations for lifted decision lists. **Stage 3 is a strong first portable grammar, not the final universal policy language.** Domains like Towers of Hanoi and Sokoban likely require richer representations.

## What comes next (beyond Stage 3)

1. **Programmatic state-machine policies.** Add finite controller memory to the decision list; rules can read/write a small state register, enabling history-dependent behavior.
2. **Richer guards.** Numeric predicates, universal/existential quantification, transitive closure — expanding the guard language while keeping the decision-list outer structure.
3. **Hierarchical / macro actions.** Allow an action to invoke a sub-policy, enabling temporal abstraction.
4. **Grammar-guided neural hybrid.** Use the CFG to structure the search space, but let a neural network score candidate programs (cf. PROPEL, Verma et al. 2019).

Stage 3 establishes the portable grammar foundation. Later stages extend expressiveness without abandoning the schema.

## Source slides

Frames 31–33 (Ferry concrete policy, step-by-step execution), frame 36 (domain rollout plan), frames 37–39 (three experiments), frames 40–41 (known limitations, what comes next).
