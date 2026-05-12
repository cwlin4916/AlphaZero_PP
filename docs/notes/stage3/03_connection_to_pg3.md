# Stage 3 / Design — Connection to PG3: What Transfers, What Doesn't

## 1. Doors as a PDDL-shaped domain

Before comparing Stage 3's representation and scorer to PG3, pin down the *object* we are planning over. The Doors domain of [overview_project.md §1](../overview_project.md) has a clean STRIPS description:

```pddl
(:types room loc key)
(:predicates
  (at ?l - loc) (open ?r - room) (key_avail ?k - key)
  (in_room ?l - loc ?r - room) (key_at ?k - key ?l - loc) (unlocks ?k - key ?r - room))

(:action move
  :parameters (?l1 ?l2 - loc ?r - room)
  :precondition (and (at ?l1) (in_room ?l2 ?r) (open ?r))
  :effect (and (at ?l2) (not (at ?l1))))

(:action pick
  :parameters (?k - key ?l - loc ?r - room)
  :precondition (and (at ?l) (key_at ?k ?l) (key_avail ?k) (unlocks ?k ?r))
  :effect (and (not (key_avail ?k)) (open ?r)))
```

**Instance (D=3).**

- **Objects:** `loc0..loc5`, `room0..room2`, `key0`, `key1`.
- **Static facts** (fix the geometry, one instance): `in_room`, `key_at`, `unlocks`.
- **Initial state** $I$: `{ at(loc0), open(room0), key_avail(key0), key_avail(key1) }` (plus the static atoms).
- **Goal** $G$: `{ at(loc5) }`.

**Implementation note.** The *current* codebase does not ship a `.pddl` file. Dynamics live in a Gymnasium env called `DoorsPDDLLite` (spec [§12](../../../spec/2026-03-20-report_doors_synthesis_full_context.md)): PDDL-shaped semantics, but as a Python `step()` function rather than a declarative add/delete model. The name acknowledges the mismatch. This is why PG3's A\*-over-add/delete-effects cannot be applied to our env as-is (revisited in the "What we have vs. what PG3 assumes" section below).

## 2. Baseline: direct AlphaZero on the ground Doors MDP

Before discussing program synthesis or lifted policies, record what AlphaZero already does on Doors *without* any synthesis layer.

1. **It works.** Direct RL — an MLP policy on the raw $(M + 2D - 1)$-dimensional observation, AlphaZero self-play on the `DoorsPDDLLite` env — solves $D=10$ in 5 iterations (spec [§8.5](../../../spec/2026-03-20-report_doors_synthesis_full_context.md); algorithm details in [`docs/specs/direct_play_algorithm_comparison.md`](../../specs/direct_play_algorithm_comparison.md); entry points [`scripts/run/run_doors_direct.py`](../../../scripts/run/run_doors_direct.py) and [`scripts/run/run_surface_vs_direct.py`](../../../scripts/run/run_surface_vs_direct.py)). The ground env has dense per-step shaping (step penalty, unlock bonus, goal reward), so the value head gets gradient at every step.
2. **Why we don't stop there.** The direct-play policy is (i) **per-instance** — weights trained at $D=k$ do not transfer to $D \neq k$; (ii) **opaque** — the MLP is not a human-readable program; (iii) **not lifted** — no notion of "for each key $k$, pick it at its location". Stages 1–3 exist to recover transfer and interpretability, accepting the sparse terminal reward of the synthesis MDP (spec [§11.1](../../../spec/2026-03-20-report_doors_synthesis_full_context.md)).
3. **So the bar for synthesis is *not* solving $D=k$** — that is the direct baseline. The bar is cross-instance / cross-domain generalization, which is precisely what lifted decision lists (Stage 3) target.

## 3. Classical planning, formally

Let a classical planning domain be defined as a tuple $\mathcal{D} = (\mathcal{P}, \mathcal{A})$, where $\mathcal{P}$ is a finite set of predicates and $\mathcal{A}$ is a finite set of action schemas (lifted actions). Each $a \in \mathcal{A}$ acts as an operator equipped with parameters $\mathrm{PAR}(a)$, preconditions $\mathrm{PRE}(a)$, add effects $\mathrm{ADD}(a)$, and delete effects $\mathrm{DEL}(a)$. Given a finite set of objects $\mathcal{O}$, let $\mathcal{X}(\mathcal{P}, \mathcal{O})$ denote the set of all grounded atoms (propositions). The state space is

$$
\mathcal{S} \;=\; 2^{\mathcal{X}(\mathcal{P}, \mathcal{O})}.
$$

A *planning instance* fixes $(\mathcal{O}, I \subseteq \mathcal{X}, G \subseteq \mathcal{X})$ on top of a domain $\mathcal{D}$. A *grounded policy* is a map $\pi_g : \mathcal{S} \to \mathcal{A}_{\mathrm{ground}}$; a *lifted policy* — the kind $G_3$ generates — is a decision list whose rules quantify over typed variables and bind existentially at execution time. The lifted → grounded step (find a type-respecting binding $\theta$ whose preconditions hold in $s$, execute the grounded action $\mathrm{ACT}(\rho)[\theta]$) is domain-independent; Stage 3 inherits it verbatim from PG3.

**Where this sits in the grammar literature.** Stage 3 is an instance of *syntax-guided synthesis* (SyGuS): the typed CFG defines the search space, the Doors / Ferry / Gripper environment defines the semantic objective (rollout reward), and AlphaZero MCTS performs the search. PG3's lifted decision-list representation slots in as the SyGuS grammar; PG3's policy-guided $A^*$ scorer is replaced by environment reward (see §"Why replacing the scorer matters"). Note also that typing does not raise the grammar above context-free — the CFG generates abstract syntax and the type system rejects ill-formed trees; the right framing is *typed DSL grammar*, not context-sensitive grammar. See [appendix/sygus_background.md](../appendix/sygus_background.md) for the full SyGuS background.

## 4. Why lift? The motivation behind the formalism

Classical planners like $A^*$ are instance-specific: solving a 10-room maze provides zero computational advantage on a 100-room maze. By defining a domain through predicates $\mathcal{P}$ and lifted action schemas $\mathcal{A}$ — independent of the object set $\mathcal{O}$ — the formalism produces solutions that are invariant to the size or specific configuration of the environment. The computational burden is shifted: pay heavy compute *upfront* to find a generalized structural rule (the policy), then apply it reactively in $O(1)$ per step on arbitrarily large, unseen instances of the same domain.

### Transitions as set-algebraic updates

From a structural view, the state space is not a physical geometry but a dynamic set of logical truths.

- **Problem instance** $\Pi = (\mathcal{O}, I, G)$: $\mathcal{O}$ are the raw elements, $I \subseteq \mathcal{X}$ is the set of facts true at $t = 0$, and $G \subseteq \mathcal{X}$ is the subset that must eventually become true.
- **Transition** $T_{\underline{a}}(S) = (S \setminus \mathrm{DEL}(\underline{a})) \cup \mathrm{ADD}(\underline{a})$: an action $\underline{a}$ is a local operator; when its preconditions $\mathrm{PRE}(\underline{a})$ hold in $S$, it deletes a specific subset of facts and unions in a new subset. The "world" simply updates its set of true statements.

This makes the state space $\mathcal{S} = 2^{\mathcal{X}(\mathcal{P}, \mathcal{O})}$ from §3 unexotic: it is the powerset of grounded atoms because a state just records *which facts are currently true*.

### Forest preview — the transfer advantage

A tiny navigation domain (the running example from PG3; revisited with a full scoring walk-through in the "PG3's score function" section below) makes this concrete. Objects are three locations $l_1, l_2, l_3$; the initial state contains the facts `at(l1)`, `onTrail(l1, l2)`, `isRock(l2)`, `onTrail(l2, l3)`. A transition like `walk(l1, l2)` *deletes* the fact `at(l1)` and *adds* `at(l2)`.

Because a policy $\pi \in \mathcal{H}$ is written with lifted variables (`?x`, `?y`) rather than grounded objects, the rule

> `IF at(?x) ∧ isRock(?y) ∧ onTrail(?x, ?y) THEN climb(?x, ?y)`

evaluates the *relational structure* of the local state, not the global map. The same rule fires whether $\Pi$ has 3 locations or 300 000. Domain physics (predicates + action schemas) is strictly separated from instance scale ($\mathcal{O}, I, G$) — enabling zero-shot generalization to arbitrarily large state spaces. §5 next checks whether Doors' goal structure exhibits this lift-friendly shape.

## 5. How well Doors fits the classical-planning framework

**Fits cleanly.**
- Typed objects — three types (*room*, *loc*, *key*) with tight cardinalities per instance: $D$ rooms, $M$ locations, $K = D - 1$ keys.
- Unary and binary predicates only; no numerics, no conditional effects, no durative actions.
- Pure STRIPS add/delete: `pick` adds `open(r)` and deletes `key_avail(k)`; `move` adds the new `at(·)` and deletes the old.
- Per-instance static predicates (`in_room`, `key_at`, `unlocks`) that a PDDL compiler would fold into `:constants`. Our Gymnasium env bakes them in; a Stage 3 adapter needs to expose them to guards — see [02_doors_adapter.md](02_doors_adapter.md).

**Fits awkwardly.**
- **Single goal literal.** Doors' goal is the unit atom `at(loc_goal)`. Contrast with Ferry, whose goal is quantified — `forall c:car. at(c, goal_loc)` — or Gripper — `forall b:ball. in_room(b, goal_room)`. Lifted policies that transfer cleanly on Ferry/Gripper do not automatically transfer on Doors because there is no quantifiable object in the goal. This is the single-goal-literal quirk flagged in [overview_project.md Open Q5](../overview_project.md#open-questions) and motivates evaluating the candidate `ForEach` extension in [01_lifted_grammar.md](01_lifted_grammar.md); a sufficiently rich state-guard vocabulary may still make a flat list adequate, which Exp 2 will determine.
- **Implicit dependency chain.** The optimal Doors policy is a *sequence* (pick key 0 → move → pick key 1 → move → reach goal) induced by the static `unlocks` and `key_at` predicates. PG3-style lifted rules express "fire the first applicable rule" but not "this rule before that rule" explicitly — the decision-list order encodes it, and ordering becomes the search burden.

**Implication.** Doors is a legitimate PDDL domain in shape, but evaluating Stage 3 lifted-transfer on Doors is *harder* than on Ferry/Gripper. Any Stage 3 transfer experiment should report Ferry/Gripper first, then Doors flat-first; only invoke `ForEach` if the flat form fails on Doors.

---

## 6. Overview

The original Stage 3 plan was billed as "PG3-style." After re-reading Yang et al. (IJCAI 2022), the honest accounting is: **we adopt PG3's representation, we do not adopt its scoring or its search**. This file pins down the boundary and the reason the phrase "PG3-style search" should be removed from the spec.

## Common misconception

PG3 is often described as "the lifted decision-list approach." That is incorrect. The lifted decision-list **representation** predates PG3 by decades (Rivest 1987; Mooney & Califf 1995; Levine & Humphreys 2003).

**PG3's actual contribution is a new *score function* for generalized policy search.** The setting:

- Given a PDDL domain $\langle\mathcal P, \mathcal A\rangle$ and training problems $\Psi = \{\langle O_i, I_i, G_i\rangle\}$.
- Search over candidate lifted decision-list policies $\pi$ using GBFS.
- Each candidate is scored; the best is returned.

## PG3's score function

For each training problem, run **policy-guided A\*** over the action graph where actions agreeing with $\pi$ have cost 0 and other actions cost 1.

- A perfect $\pi$ ⇒ A* threads through cost-0 edges ⇒ low total cost.
- A poor $\pi$ ⇒ A* deviates often ⇒ high cost.
- Crucially, the score is **dense**, not binary "solved / unsolved" — this is what makes GPS efficient.

```
Score(π) = Σ over training problems: PG3-cost(π, O, I, G)

PG3-cost(π, O, I, G):
  Run A* from I to G over the action graph,
    where action a has cost 0 if a = π(s, G), cost 1 otherwise.
  if A* finds a plan p: return PlanCompare(π, p)
  else: return horizon ℓ
```

### Worked example — PG3 scoring on Forest

To see how policy-guided $A^*$ produces dense, meaningful gradients through the space of candidate policies $\mathcal{H}$, walk through the Forest domain (PG3 Example 1).

**Setup.** The domain has three predicates — `at(x)`, `isRock(x)`, `onTrail(x, y)` — and two actions:

- `walk(x, y)` with precondition `at(x) ∧ onTrail(x, y) ∧ ¬isRock(y)`,
- `climb(x, y)` with precondition `at(x) ∧ onTrail(x, y)` (no anti-rock clause).

Instance: navigate $l_1 \to l_3$ along $l_1 \to l_2 \to l_3$, but $l_2$ has a rock. Initial state contains `at(l1)`, `onTrail(l1, l2)`, `isRock(l2)`, `onTrail(l2, l3)`; goal is `at(l3)`.

**Step 1 — a flawed candidate $\pi_0$.** The search operator proposes

> `π₀(s, g) ↦ walk(?x, ?y)   IF at(?x) ∧ onTrail(?x, ?y)`

Pure execution evaluation (the "did it solve the instance?" baseline) returns **0 successes** — $\pi_0$ gets permanently stuck at $l_2$. That binary signal provides no information about how close $\pi_0$ is to the correct structure.

**Step 2 — policy-guided $A^*$.** Instead of just executing $\pi_0$, PG3 runs $A^*$ from $I$ to $G$, where any action agreeing with $\pi_0$ gets cost $0$:

1. At $l_1$: $A^*$ queries $\pi_0$ → `walk(l1, l2)` → zero-cost transition to a state containing `at(l2)`.
2. At $l_2$: $\pi_0$ still suggests `walk(l2, l3)`, but `walk`'s precondition requires $\neg$ `isRock(y)`, and `isRock(l2)` holds — the suggestion is *invalid*.
3. $A^*$ compensates by exploring standard domain actions at cost $1$; it discovers `climb(l2, l3)` is valid (`climb` has no anti-rock precondition).
4. Goal reached. Final plan $\tau$: `walk(l1, l2)` then `climb(l2, l3)`.

**Step 3 — structural disagreement.** PG3 compares $\tau$ against what $\pi_0$ would have done at each step:

| $t$ | Plan action | $\pi_0$ suggestion | Error |
|---|---|---|---|
| $0$ | `walk(l1, l2)` | `walk(l1, l2)` | $0$ |
| $1$ | `climb(l2, l3)` | `walk(l2, l3)` (invalid) | $1$ |

Score: $\mathrm{PlanCompare}(\pi_0, \tau) = 1$. Low is good — scoring is *dense* even though execution evaluation said "unsolved."

**Step 4 — the GPS update.** Because the score is $1$ (close to the optimal manifold), GBFS expands $\pi_0$'s neighborhood via the "Add Rule" operator, appending

> `IF at(?x) ∧ isRock(?y) ⇒ climb(?x, ?y)`

Evaluating the new $\pi_1$ yields structural disagreement exactly $0$ — a satisficing policy for this training subset.

**Why this matters.** The dense score (here, $1$ — not $0$-or-unsolved) is what makes GBFS tractable over candidate policies. This is exactly the density we *lose* when replacing $A^*$ with rollout reward — revisited in the "Why replacing the scorer matters" section below.

**The hidden requirement:** A* must enumerate per-state successors via the domain's add/delete effects. Without a symbolic forward model, PG3-cost cannot be computed.

## What we have vs. what PG3 assumes

| | PG3 (Yang et al. 2022) | Stage 3 (AlphaZero_PP) |
|---|---|---|
| Domain model | STRIPS/PDDL with add/delete effects | Custom Gymnasium env with `step(s, a)` |
| Per-state successors | Enumerable via domain schema | Require rollout through the env |
| Score function | A* cost with policy-agreeing edges | Environment rollout reward |
| Search strategy | GBFS over policies (GPS) | MCTS over grammar derivations (AlphaZero) |
| Policy class | Lifted decision lists | **Same** |
| Execution semantics | First-applicable, type-respecting existential binding | **Same** |

## Audit: what transfers from PG3?

| PG3 component | Transfers? | Why / why not |
|---|---|---|
| Lifted DL representation (PAR, PRE, GOAL, ACT) | **Yes** | Pure data type. Adopted directly. |
| First-applicable semantics | **Yes** | Pure interpreter logic; works on any $(S, O, G)$ triple. |
| Type-respecting binding | **Yes** | Needs only typed objects from the env's relational view. |
| **PG3 score function** | **No** | Requires A* over predicate-level add/delete effects. We have Gymnasium `step`, not a STRIPS model. |
| **Policy-guided A\*** | **No** | Same reason — no symbolic forward model. |
| **GPS operators** (AddRule, DelRule, AddCond, DelCond, InduceFromPlans) | **No** | Replaced by CFG derivation under AlphaZero MCTS. Search is over derivations, not edits to a policy. |
| Action schemas as productions | Partial | Need a hand-written relational view per Gymnasium env. |

## Connection to our CFG

| Aspect | PG3 | Stage 3 (ours) |
|---|---|---|
| Policy class | Lifted decision lists | **Same** |
| Search method | GPS (hill-climbing over policies) | MCTS + AlphaZero |
| Scoring / oracle | A* policy-guided planning | Neural value network |
| Guard language | Arbitrary conjunctions | Bounded by $K_s$, $K_g$ |
| Binding semantics | Existential, first-applicable | **Same** |
| Tie-breaking | Not specified | Lexicographic (deterministic) |

**Same policy class, different search algorithm.** Our CFG generates exactly PG3's lifted decision lists, bounded by structural hyperparameters $L_{\max}, K_s, K_g, V_{\max}$.

## Why replacing the scorer matters

PG3's GBFS over policies is only tractable because A* cost is cheap per candidate: it evaluates $\pi$ on each training problem in polynomial time (linear in horizon × state size). Our AlphaZero score is *rollout reward* — also cheap, but noisier — and is bootstrapped through self-play rather than queried as a fixed oracle. This has two consequences:

1. **No STRIPS model required.** We can attach to any custom Gymnasium domain via the adapter, including domains where add/delete effects are hard to specify.
2. **Score sparsity replaces score density.** PG3 gets dense feedback from A* deviation count even on unsolved instances. We get sparse terminal reward from rollouts. This is the same signal problem Stage 2 ran into ([stage2/exp4](../stage2/exp4_training_scaling.md)) — the bootstrap problem returns if the random-rollout success probability is too low.

## Recommended phrasing going forward

Stage 3 should be described as:

> "Lifted decision-list synthesis via AlphaZero MCTS over a typed CFG, with score = environment rollout reward."

The phrase "PG3-style search" is wrong and should be removed from the spec wherever it appears. We are borrowing PG3's policy *language*, not its policy *learning algorithm*.

## Takeaways

1. **Representation yes, search no.** We adopt the lifted decision-list data type and first-applicable + existential-binding semantics; we do *not* adopt A* scoring, GBFS, or the GPS operators.
2. **The PG3 score function would require a STRIPS model we don't have.** This is a hard gap, not a temporary one.
3. **Score sparsity is the open risk.** Stage 2 showed that sparse rollout reward becomes a bootstrap problem once solve density drops below a threshold; the same risk applies to Stage 3 at large instances. Any future "Stage 4" should consider reintroducing a dense planning-like signal (e.g. subgoal reward shaping, or a planner subroutine for adapter domains that do have STRIPS).
4. **Doors is PDDL-shaped but the harder transfer benchmark.** Direct AlphaZero already solves Doors up to $D=10$ (§2); the reason to pursue lifted policies is cross-instance transfer, not basic solvability. The single-goal-literal structure (§5) makes Doors a *harder*, not easier, transfer benchmark than Ferry/Gripper. Whether `ForEach` is required is open and tested in Exp 2; any Stage 3 transfer experiment should report Ferry/Gripper first.

## Source slides

Frames 3 (two senses of generalization), 7–10 (PG3 is a score function; PG3 inputs/outputs; PG3 loop in one slide; terminology walkthrough), 23 (connection to PG3: CFG comparison table), 28 (critical evaluation: what transfers from PG3), 34 (known limitations).
