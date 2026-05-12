# Stage 3 — Lifted Typed CFG $G_3$: Design Report

**Source:** [stage3_grammar_experiment.tex](../../presentations/improvementv1/stage3_grammar_experiment.tex) (37 frames, April 2026)

**Status:** Design only — no experimental results yet. This README is the navigation hub for four design documents; experimental result sections under [planned_experiments.md](planned_experiments.md) are scaffolded for later fill-in.

## Where Stages 1–2 left us

Stage 1 ran AlphaZero on the masked surface grammar $G_1$ (right-linear, $\lvert N\rvert = 3^K$, $\lvert L(G_1)\rvert = (2K)!/2^K$). MCTS solved at iteration 1 for every $D \in \{3, 6, 7\}$, but the network plateaued above the entropy floor as $K$ grew: $D_{\mathrm{KL}}^{(15)} = 0.322$ at $D=6$ versus 0.236 at $D=3$. The diagnosis was *coverage* — training tuples per nonterminal $C$ scales as $\Theta(3^K)$ — and the two effective single-parameter levers were patience (long runs) and reduced Dirichlet noise ($\epsilon=0.10$). See [stage1/README1.md](../stage1/README1.md).

Stage 2 stripped all three masks (precedence, uniqueness, completeness) to give the unmasked grammar $G_2$ with $\lvert L_=(G_2)\rvert = (2K)^{2K}$ — a $1.19\!\times\!10^6$ blowup at $D=7$ relative to $G_1$. Two findings stuck. **(i)** Exhaustive enumeration at $D \leq 4$ confirmed the masks filter only junk: $N_{\mathrm{solve}}$ is *identical* in $G_1$ and $G_2$. **(ii)** With tuned $\epsilon = 0, \tau = 0.5$, $G_2$ matches $G_1$ at $D \leq 5$ but **never solves $D \geq 6$** in 50 iterations — $P(\text{random rollout finds a solver}) = 1.8\%$ at $D=6$ falls below the bootstrap threshold. See [stage2/README2.md](../stage2/README2.md).

The shared lesson, restated: any framework that wants to scale beyond $D=5$ *without* domain-specific masks needs structural compression of equivalent strength. Three sources are on offer — a learned prior strong enough to cross the bootstrap, a richer inductive bias, or a *lifted* representation that compresses across objects. Stage 3 takes the third path.

## Goal

> Learn compact symbolic reactive policies for PDDL-style relational domains by combining AlphaZero-style search with a typed lifted policy grammar, and demonstrate that the learned policies transfer from small training instances to larger held-out instances.

This re-targets two new axes simultaneously. **Framework portability**: a small adapter file (4 declarations + observation parser) replaces the Doors-specific `SurfaceCFG` + compiler stack, so Ferry, Gripper, Doors, and beyond all run through one engine. **Policy portability**: a typed program with bound variables can re-execute on a *larger* instance — more keys, more cars, more rooms — without re-search. Concretely $G_3$ **replaces** rather than refines the surface/AST stack of [overview_project.md §2](../overview_project.md); the base AST DSL plays no role at Stage 3.

The mechanism behind both forms of portability is the same: **replace object-indexed ground syntax with typed lifted syntax.** Stage 1's coverage bottleneck $C = \Theta(3^K)$ and Stage 2's $(2K)^{2K}$ blowup are *coverage* problems. Lifting collapses both by sharing structure across objects of the same type — typed variables $k\!:\!\textit{key}, r\!:\!\textit{room}$ in place of $P_0, M_0, P_1, M_1, \ldots$. Stage 3 is therefore not "a more expressive grammar" but the same expressive class refactored for structural reuse.

## Design axes

The next decision in lifted-policy design is *how relational and how reactive* the language should be. Seven axes describe the design space; each has a *weak* and a *strong* end.

| Axis | Weak version | Stronger version | Why it matters |
|---|---|---|---|
| Object representation | Ground tokens ($P_k, M_k$) | Typed variables ($k\!:\!\textit{key}, r\!:\!\textit{room}$) | Removes object-indexed blowup |
| Guards | Atomic predicates | Conjunctions, negation, static predicates, goal guards | Controls useful pruning |
| Binding | No variable sharing | Shared variables across literals | Expresses relational identity |
| Control | Flat sequence | Ordered decision list | Gives reactive policy semantics |
| Iteration | Fixed rule count | `ForEach` / bounded loop / recursion | Needed when policy length otherwise scales with objects |
| Memory | Stateless | Finite-state or progress flags | Needed for domains where current predicates do not encode phase |
| Scoring | Sparse terminal reward | Dense planner-guided or shaped signal | Prevents bootstrap collapse |

**Where Stage 3 sits.** $G_3$ targets the **reactive** form $\pi(S, G) \to a$ — re-applied at every state — not the program-compiler form $\Pi(O, I, G) \to [\,\cdot\,]$ that would emit a macro sequence in one shot (see [01_lifted_grammar.md §"Reactive form vs program-compiler form"](01_lifted_grammar.md)). $G_3$ takes the *strong* version on the first four axes — typed variables (with type-safe binding), conjunctive state-and-goal guards, shared variables across literals, ordered decision list with first-applicable semantics. It takes a *moderate* position on iteration: the base grammar is fixed-length, with a *candidate* `ForEach` extension reserved as a fallback for single-goal-literal domains (gated on Exp 2) and no full recursion. It stays at the *weak* end on memory (stateless; current observation only) and — most consequentially — at the *weak* end on **scoring**: rollout reward, same as Stages 1–2. This last choice is exactly the divergence from PG3 audited in [03_connection_to_pg3.md](03_connection_to_pg3.md): $G_3$ adopts PG3's *representation* but not its *scorer*.

## Design documents

| # | File | Headline |
|---|---|---|
| 1 | [01_lifted_grammar.md](01_lifted_grammar.md) | Typed CFG: rule = PAR / PRE / GOAL / ACT; structural bounds $L_{\max}, K_s, K_g, V_{\max}$; first-applicable decision-list semantics; candidate `ForEach` extension (gated on Exp 2). |
| 2 | [02_doors_adapter.md](02_doors_adapter.md) | 4-declaration adapter (Types $T$, Predicates $P$, Static predicates $P_{\mathrm{static}}$, Action schemas $A$) + observation parser; same pipeline across Ferry / Gripper / Doors. |
| 3 | [03_connection_to_pg3.md](03_connection_to_pg3.md) | Adopt PG3 *representation* (lifted decision list), reject PG3 *scorer* (A\* + heuristic): we use rollout reward + MCTS instead — same as Stages 1–2. |
| 4 | [planned_experiments.md](planned_experiments.md) | Three planned experiments — tiny enumeration, cross-instance generalization, cross-domain software reuse — over a tiered domain rollout (Doors / Gripper / Ferry first). |

## Planned experiments

| # | File | Status | Headline (predicted) |
|---|---|---|---|
| 1 | [tiny exact enumeration](planned_experiments.md) | planned | Enumerate every $G_3$ program at the smallest setting; verify $N_{\mathrm{solve}}$ matches the $G_1/G_2$ exhaustive baseline at $D \leq 4$. |
| 2 | [cross-instance generalization](planned_experiments.md) | planned | Train on $D=3$, evaluate the *same lifted policy* on $D=4..7$ without retraining; expect Ferry / Gripper to transfer cleanly; Doors transfer is the open question (flat-first; `ForEach` only if flat fails). |
| 3 | [cross-domain software reuse](planned_experiments.md) | planned | Same engine + same hyperparameters on Ferry / Gripper / Doors with one new adapter file per domain; acceptance is "no engine code change required". |

Result sections in [planned_experiments.md](planned_experiments.md) are scaffolded with `### Results` headers and will be filled in once runs complete.

## Configuration (intended)

| Component | Value |
|---|---|
| Game | (new) lifted derivation game over $G_3$ |
| Grammar | $G_3$: typed CFG; bounded decision list of $\leq L_{\max}$ rules; per-rule bounds $K_s$ (state-guard literals), $K_g$ (goal-guard literals), $V_{\max}$ (bound variables) |
| Interpreter | Typed-binding interpreter, domain-agnostic — rule applies iff some assignment to its bound variables makes PAR / PRE / GOAL all true |
| Adapter | 4 declarations ($T$, $P$, $P_{\mathrm{static}}$, $A$) + observation parser per domain |
| Network | TBD; likely reuse a `DerivationPolicyValueNet`-style transformer over typed-token sequences |
| MCTS | Inherit Stage 2 tuning as starting point: $\epsilon = 0$, $\tau = 0.5$, 80 sims |
| Scoring | Sparse rollout reward — *not* PG3-style A\* heuristic. See [03_connection_to_pg3.md](03_connection_to_pg3.md) §"Why replacing scorer matters". |

## Open scope and forward path

- **No numerics, no transitive closure, no universal quantification, no recursion** in the chosen $G_3$. These are excluded by design to keep the search space and scoring tractable for the AlphaZero loop.
- **Memory axis is weak.** Domains where the current observation does not encode the policy's phase will need a progress-flag extension.
- **Scoring axis is weak.** The bootstrap collapse seen at Stage 2 $D \geq 6$ may recur on lifted domains. Mitigation paths: curriculum transfer (train at small $D$, evaluate at large $D$ — exp 2 already does this for *evaluation*, not for *training*), learned shaping, or eventually a PG3-style heuristic (which would cross back over the "scorer" line and is flagged as future).
- **Doors quirk (open).** Doors has a single goal literal `at(goal_loc)`, which makes its goal guard degenerate. Whether this requires the `ForEach` extension or whether a sufficiently rich state-guard vocabulary lets a flat decision list transfer is what Exp 2 tests. Ferry / Gripper / Blocksworld-style domains should transfer without `ForEach`.

See [overview_project.md §"Open questions"](../overview_project.md) for the broader list of unresolved items carrying into Stage 3, including the Stage-3-specific item *"Doors vs PG3-style domains"*.

## Reading paths

- **Coming from program synthesis (SyGuS, PBE, CEGIS background)?** Read [appendix/sygus_background.md](../appendix/sygus_background.md) first, then jump straight to [03_connection_to_pg3.md](03_connection_to_pg3.md) for the PG3 audit and recommended phrasing.
- **Coming from Stage 1–2 of this project?** You already know the bootstrap-cliff result. Start with [01_lifted_grammar.md](01_lifted_grammar.md) for the rule anatomy, then [02_doors_adapter.md](02_doors_adapter.md) to see how a single domain ships.
- **Coming from classical / lifted planning?** Skim [03_connection_to_pg3.md](03_connection_to_pg3.md) §"Classical planning formally" for the framing, then [01_lifted_grammar.md](01_lifted_grammar.md). The grammar is recognisable as a bounded-depth PG3-style policy class.

## Quick navigation

- **Design files:** [01_lifted_grammar.md](01_lifted_grammar.md) · [02_doors_adapter.md](02_doors_adapter.md) · [03_connection_to_pg3.md](03_connection_to_pg3.md) · [planned_experiments.md](planned_experiments.md)
- **Source presentation:** [stage3_grammar_experiment.tex](../../presentations/improvementv1/stage3_grammar_experiment.tex)
- **Appendix:** [SyGuS background](../appendix/sygus_background.md) · [AlphaZero + CFG framework](../appendix/alphazero_grammar_framework.md)
- **Project context:** [overview_project.md](../overview_project.md) — §3 places $G_3$ alongside $G_1$ and $G_2$; "Project arc" describes the full Stage 1 → 3 trajectory.
