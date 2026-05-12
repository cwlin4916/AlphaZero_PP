# 00 — Lifted-policy program synthesis with AlphaZero MCTS (orientation)

> **Superseded by [01_draft_lifted_policy_az.md](01_draft_lifted_policy_az.md)** — kept for
> provenance. The rewrite replaces the standalone-`Aux` derivation grammar (§6 below) with a
> proposed occurrence-introduced-variable grammar, adds a literature-positioning section and a
> Doors-PDDL contract, and updates cross-links: the numbered Stage-1/2/3 notes now live under
> [`legacy/`](legacy/).

## §0 Status and reading guide

This is the project's **orientation and theory document**. It fixes the problem, the
two-level architecture, the Gripper-lite MDP, the lifted-policy semantics, and the
synthesis game. Stage 1 ([01_plan.md](legacy/01_plan.md), [01.md](legacy/01.md)) implemented the typed
lifted DSL, the online-unification interpreter, and the `gripper_lite` env; Stage 2
([02_plan.md](legacy/02_plan.md)) implemented the lifted grammar, `LiftedDerivationGame`, the
node encoding, and the leaf evaluator. This document is not an experiment report — it is
the framing the stage plans instantiate.

There are **two decision processes**:

$$\boxed{\;\textbf{Task level: }\ \text{relational MDPs }\mathcal{M}_B\ (\text{Gripper-lite, later Doors}) \;}$$
$$\boxed{\;\textbf{Synthesis level: }\ \text{grammar-derivation game }\mathcal{D}_\Gamma\ \text{searched by MCTS} \;}$$

The **lifted policy** is the object connecting them: MCTS over $\mathcal{D}_\Gamma$
constructs a lifted policy; the task MDP $\mathcal{M}_B$ evaluates it.

How to read: §1 the synthesis objective · §2 the two levels side by side · §3 the task
MDP (Gripper-lite) · §4 what a lifted policy *is* · §5 how the interpreter executes one ·
§6 the derivation game MCTS searches · §7 the hand-policy correctness theorem · §8 the
stage ladder · §9 decisions now fixed · §10 what is not yet claimed.

## §1 Problem statement

Given a family of relational MDPs $\{\mathcal{M}_i\}_{i\in\mathcal{I}}$ and a typed
grammar $\Gamma$ generating lifted decision-list policies $\Pi_\Gamma$, the synthesis
objective is

$$\boxed{\;\pi^\star \in \arg\max_{\pi\in\Pi_\Gamma} J_{\mathrm{train}}(\pi),\qquad
J_{\mathrm{train}}(\pi) = \frac{1}{|\mathcal{I}_{\mathrm{train}}|}\sum_{\mathcal{M}\in\mathcal{I}_{\mathrm{train}}} U_{\mathcal{M}}(\pi)\;}$$

where $U_{\mathcal{M}}(\pi)$ is the leaf score of $\pi$ on $\mathcal{M}$ (§6). The
load-bearing measurement is held-out generalization to larger / structurally different
instances:

$$\boxed{\;J_{\mathrm{out}}(\pi) = \frac{1}{|\mathcal{I}_{\mathrm{out}}|}\sum_{\mathcal{M}\in\mathcal{I}_{\mathrm{out}}} U_{\mathcal{M}}(\pi)\;}$$

**Research question.** Can a *generic* AlphaZero-style MCTS over $\Gamma$ synthesize a
compact lifted policy that fits $\mathcal{I}_{\mathrm{train}}$ and generalizes to
$\mathcal{I}_{\mathrm{out}}$ — replacing PG3's domain-specific GBFS + STRIPS-A* score with
episodic execution reward at complete-program leaves?

## §2 Two levels of decision-making

| Level | state | action | terminal reward |
|---|---|---|---|
| Task MDP $\mathcal{M}_B$ | relational world state $s$ | `move` / `pick` / `drop` (Doors: `MOVE_TO` / `PICK` / `NOOP`) | $-1$ per step, $+100$ on solve |
| Derivation game $\mathcal{D}_\Gamma$ | partial lifted-policy AST $z$ | a grammar production $p$ | $\mathrm{LeafEval}(\mathrm{to\_program}(z_T))$ |

## §3 The task MDP: Gripper-lite

For each ball count $B\ge 1$, Gripper-lite is the deterministic finite-horizon relational
MDP ([gripper_lite/env.py](../../src/alphazeropp/instances/gripper_lite/env.py))

$$\boxed{\;\mathcal{M}_B = \langle\, \mathcal{S}_B,\; \mathcal{A}_B,\; T_B,\; R_B,\; H_B,\; s^0_B,\; G_B \,\rangle\;}$$

**Types and objects.**

$$\boxed{\;\mathcal{T} = \{\textit{ball},\,\textit{room}\},\qquad
\mathcal{O}^B_{\textit{ball}} = \{\textit{ball\_}i : 0\le i < B\},\qquad
\mathcal{O}_{\textit{room}} = \{\textit{room\_a},\,\textit{room\_b}\}\;}$$

$\textit{room\_a}$ is the source room, $\textit{room\_b}$ the target.

**Predicates.** One *implicit* gripper, so the vocabulary uses $\mathrm{carrying}(b)$ and
$\mathrm{handempty}()$ — not the classical $\mathrm{carry}(b,g)$ / $\mathrm{free}(g)$;
there is no $\textit{gripper}$ type.

$$\boxed{\;\mathcal{P} = \{\,\mathrm{at\_robot}(\textit{room}),\;\mathrm{at\_ball}(\textit{ball},\textit{room}),\;\mathrm{carrying}(\textit{ball}),\;\mathrm{handempty}()\,\}\;}$$

**Relational state.** A state $s\in\mathcal{S}_B$ is a finite set of ground atoms — in
code the per-predicate index

$$\boxed{\;\mathrm{RelState} = \mathrm{dict}\bigl[\,\text{predicate name}\,\to\,\mathrm{set}[\,\mathrm{tuple}[\text{object name},\dots]\,]\,\bigr]\;}$$

satisfying the invariants: the robot is in exactly one room,
$\exists!\,r\;\mathrm{at\_robot}(r)\in s$; the hand is empty iff no ball is carried,
$\mathrm{handempty}()\in s \iff \forall b\;\mathrm{carrying}(b)\notin s$; each ball is in
exactly one room *or* carried,
$\forall b:\ \bigl[\exists!\,r\;\mathrm{at\_ball}(b,r)\in s\bigr]\,\oplus\,\bigl[\mathrm{carrying}(b)\in s\bigr]$.
Example ($B=2$): `{"at_robot":{("room_a",)}, "handempty":{()}, "carrying":set(),
"at_ball":{("ball_0","room_a"),("ball_1","room_a")}}`.

**Initial state and goal.**

$$\boxed{\;s^0_B = \{\mathrm{at\_robot}(\textit{room\_a}),\,\mathrm{handempty}()\}\,\cup\,\{\mathrm{at\_ball}(b,\textit{room\_a}):b\in\mathcal{O}^B_{\textit{ball}}\}\;}$$
$$\boxed{\;G_B = \{\mathrm{at\_ball}(b,\textit{room\_b}):b\in\mathcal{O}^B_{\textit{ball}}\},\qquad \mathrm{solved}(s)\iff s|_{\mathrm{at\_ball}} = G_B\ \ (\Leftrightarrow G_B\subseteq s\ \text{under the invariants})\;}$$

The $\cup$ is **set union** over the relational state — recall (above) that a state is a finite
*set* of ground atoms, so $s^0_B$ is built by taking the union of two disjoint atom sets: the
robot/hand facts $\{\mathrm{at\_robot}(\textit{room\_a}),\,\mathrm{handempty}()\}$ and the per-ball
location facts $\{\mathrm{at\_ball}(b,\textit{room\_a}) : b\in\mathcal{O}^B_{\textit{ball}}\}$ (one
atom per ball, all assigning that ball to $\textit{room\_a}$). Colloquially: **at time $0$ the
robot stands in $\textit{room\_a}$ with an empty hand and all $B$ balls sit in $\textit{room\_a}$** —
nothing is carried, nothing is in $\textit{room\_b}$ yet. The goal $G_B$ is the mirror image:
every ball must end up in $\textit{room\_b}$.

**Ground actions.**

$$\boxed{\;\mathcal{A}_B = \{\mathrm{move}(r,r'):r\ne r'\}\,\cup\,\{\mathrm{pick}(b,r)\}\,\cup\,\{\mathrm{drop}(b,r)\},\quad r,r'\in\mathcal{O}_{\textit{room}},\ b\in\mathcal{O}^B_{\textit{ball}}\;}$$

**Legal actions $\mathcal{A}_B(s)$** (the env's `legal_actions()`):

$$\boxed{\begin{aligned}
\mathrm{move}(r,r')\in\mathcal{A}_B(s) &\iff \mathrm{at\_robot}(r)\in s \,\wedge\, r\ne r'\\
\mathrm{pick}(b,r)\in\mathcal{A}_B(s) &\iff \mathrm{at\_robot}(r)\in s \,\wedge\, \mathrm{at\_ball}(b,r)\in s \,\wedge\, \mathrm{handempty}()\in s\\
\mathrm{drop}(b,r)\in\mathcal{A}_B(s) &\iff \mathrm{at\_robot}(r)\in s \,\wedge\, \mathrm{carrying}(b)\in s
\end{aligned}}$$

**Transition $T_B$** (deterministic; defined on legal actions, applied via set add/remove):

$$\boxed{\begin{aligned}
T_B(s,\mathrm{move}(r,r')) &= \bigl(s\setminus\{\mathrm{at\_robot}(r)\}\bigr)\cup\{\mathrm{at\_robot}(r')\}\\
T_B(s,\mathrm{pick}(b,r)) &= \bigl(s\setminus\{\mathrm{at\_ball}(b,r),\,\mathrm{handempty}()\}\bigr)\cup\{\mathrm{carrying}(b)\}\\
T_B(s,\mathrm{drop}(b,r)) &= \bigl(s\setminus\{\mathrm{carrying}(b)\}\bigr)\cup\{\mathrm{at\_ball}(b,r),\,\mathrm{handempty}()\}
\end{aligned}}$$

**Reward and horizon.**

$$\boxed{\;R_B(s,a,s') = -1 + 100\cdot\mathbf{1}[\,\mathrm{solved}(s')\,],\qquad H_B = 4B+4\;}$$

Reward is not load-bearing for Stage 1 — its tests assert on `is_solved()` and action
traces, not return. The hand-policy plan length is $4B-1$ (§7), so $H_B = 4B+4 \ge 4B-1$
is a conservative margin.

**Env contract** (what the interpreter and leaf evaluator consume):
`get_state_atoms() -> RelState`, `get_goal_atoms() -> RelState`,
`get_objects_by_type() -> dict[str, tuple[str,...]]`,
`legal_actions() -> set[GroundAction]`,
`step(GroundAction) -> (RelState, float, bool, dict)`, `is_solved() -> bool`, `reset(...)`.

![Gripper-lite domain, B = 2: robot + balls start in room_a, goal is both balls in room_b](figures/gripper_lite_domain.png)

**Doors** ([doors_pddl_lite.py](../../src/alphazeropp/instances/doors/doors_pddl_lite.py))
is the other intended task domain (types $\{\textit{room},\textit{location},\textit{key}\}$,
sequential key-fetch to a single goal location); wiring it into the lifted interpreter
needs a `get_goal_atoms()` env-contract addition — deferred to a later stage.

## §4 Lifted decision-list policies

A policy is an ordered decision list of typed rules (PG3, Yang et al. IJCAI 2022, Def. 5):

$$\boxed{\;\pi = [\rho_1,\dots,\rho_m]\;}$$

$$\boxed{\;\rho = \langle\, X_\rho,\; \mathrm{body}_\rho,\; \alpha_\rho \,\rangle,\qquad
\mathrm{body}_\rho = (L_1,\dots,L_k),\qquad L_j = \bigl(\mathrm{atom}_j,\ \mathrm{source}_j,\ \mathrm{negated}_j\bigr)\;}$$

where $X_\rho$ is a finite typed variable context; $\alpha_\rho = a(x_1,\dots)$ a lifted
action with $x_i\in X_\rho$ and types matching $\sigma_\mathcal{A}(a)$; each literal $L_j$
carries

$$\boxed{\;\mathrm{source}_j\in\{\mathrm{STATE},\,\mathrm{GOAL}\},\qquad \mathrm{negated}_j\in\{0,1\}\;}$$

A **binding** is a type-respecting map $\theta:X_\rho\to\mathcal{O}$. This is exactly the
`Rule(vars, body, action)` / `Policy(rules)` dataclass pair
([lifted_dsl.py](../../src/alphazeropp/synthesis/lifted_dsl.py)). PG3 *motivates* the
$\langle\mathrm{PAR},\mathrm{PRE},\mathrm{GOAL},\mathrm{ACT}\rangle$ form but the
implemented DSL folds the $(\mathrm{PRE},\mathrm{GOAL})$ split into one ordered
$\mathrm{body}$ tuple discriminated by $\mathrm{source}$.

**The Gripper-lite hand policy** ([policies.py](../../src/alphazeropp/instances/gripper_lite/policies.py)),
$m=4$ rules; $\rho_1,\rho_2$ carry a *positive* goal literal, $\rho_3,\rho_4$ a *negated*
one; every state literal is positive:

$$\boxed{\begin{aligned}
\rho_1:&\ \ \mathrm{carrying}(?b)\,\wedge\,\mathrm{at\_robot}(?r)\,\wedge\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)] \;\Rightarrow\; \mathrm{drop}(?b,?r)\\
\rho_2:&\ \ \mathrm{carrying}(?b)\,\wedge\,\mathrm{at\_robot}(?\textit{from})\,\wedge\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?\textit{to})] \;\Rightarrow\; \mathrm{move}(?\textit{from},?\textit{to})\\
\rho_3:&\ \ \mathrm{at\_ball}(?b,?r)\,\wedge\,\mathrm{at\_robot}(?r)\,\wedge\,\mathrm{handempty}()\,\wedge\,\neg\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)] \;\Rightarrow\; \mathrm{pick}(?b,?r)\\
\rho_4:&\ \ \mathrm{at\_robot}(?\textit{from})\,\wedge\,\mathrm{at\_ball}(?b,?\textit{to})\,\wedge\,\mathrm{handempty}()\,\wedge\,\neg\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?\textit{to})] \;\Rightarrow\; \mathrm{move}(?\textit{from},?\textit{to})
\end{aligned}}$$

## §5 The online-unification interpreter

Closed-world literal satisfaction under a type-correct binding $\theta$ — state literals
read against $S$, goal literals against $G$:

$$\boxed{\begin{aligned}
S\models_\theta\ \ p(\bar x)\ \ &\iff\ p(\theta\bar x)\in S, & G\models_\theta\ \ p(\bar x)\ \ &\iff\ p(\theta\bar x)\in G,\\
S\models_\theta\ \neg p(\bar x)\ &\iff\ p(\theta\bar x)\notin S, & G\models_\theta\ \neg p(\bar x)\ &\iff\ p(\theta\bar x)\notin G.
\end{aligned}}$$

Negation is **safe** (enforced at `Rule.__post_init__`; the grammar also pre-filters):
every variable of a negative literal must already appear in a positive literal or in
$\alpha_\rho$. The satisfying-binding set, requiring legality, and the deterministic
selection:

$$\boxed{\;B_\rho(S,G) = \{\theta : S\models_\theta\mathrm{body}^{S}_\rho \,\wedge\, G\models_\theta\mathrm{body}^{G}_\rho \,\wedge\, \theta(\alpha_\rho)\in\mathcal{A}_B(S)\}\;}$$

$$\boxed{\;\theta^\star_\rho(S,G) = \min_{\preceq} B_\rho(S,G),\qquad \theta\preceq\theta' \iff \bigl(\theta(x)\bigr)_{x\in\mathrm{sort}(X_\rho)} \le_{\mathrm{lex}} \bigl(\theta'(x)\bigr)_{x\in\mathrm{sort}(X_\rho)}\;}$$

(variables sorted by name, then object assignments compared lexicographically by raw
object name — the `find_bindings` sort key `tuple(sorted(theta.items()))`). Rule execution
and the first-applicable policy map:

$$\boxed{\;\mathrm{Exec}(\rho,S,G) = \theta^\star_\rho(S,G)(\alpha_\rho)\;}$$

$$\boxed{\;\mathrm{Interpret} : \mathrm{Policy}\times S\times G\times(\mathcal{T}\to\mathcal{O}^\ast)\times 2^{\mathcal{A}_{\mathrm{ground}}}\;\to\;\mathcal{A}_{\mathrm{ground}}\cup\{\mathrm{None}\}\;}$$
$$\boxed{\;\mathrm{Interpret}(\pi,S,G,\mathcal{O},\mathrm{Legal}) = \begin{cases}\mathrm{Exec}(\rho_{i^\star},S,G), & i^\star = \min\{i : B_{\rho_i}(S,G)\ne\emptyset\}\\[2pt] \mathrm{None}, & \text{no rule applies}\end{cases}\;}$$

The implementation ([lifted_interpreter.py](../../src/alphazeropp/synthesis/lifted_interpreter.py):
`find_bindings`, `interpret`) walks $\mathrm{body}_\rho$ literal-by-literal and never
materializes the full Cartesian product over $\mathcal{O}$ — so it transfers unchanged to
held-out larger instances.

**Renaming caveat (precise).** The matching semantics are name-agnostic *up to the
tie-break order*. For a type-preserving renaming $\sigma$,

$$\boxed{\;\mathrm{Interpret}(\pi,\sigma S,\sigma G,\sigma\mathcal{O},\sigma\mathrm{Legal}) = \sigma\,\mathrm{Interpret}(\pi,S,G,\mathcal{O},\mathrm{Legal})\quad\text{iff the lex order in }\theta^\star\text{ is the }\sigma\text{-transported order}\;}$$

With the implemented *raw object-name* tie-break this holds only for $\sigma$ that
preserve the induced object order (or on states with no contested tie) — it gives
determinism, **not** arbitrary-renaming equivariance. The Stage-1 renaming test uses such
renamings ($\textit{ball\_0}\!\mapsto\!\textit{foo}$, $\textit{room\_a}\!\mapsto\!\textit{left\_room}$,
$B=1$), so it demonstrates that rule structure and binding do not hard-code object names —
not full permutation equivariance. The same caveat is restated and cross-linked in
[01.md §4](legacy/01.md#4-object-renaming-name-agnostic-matching-not-full-permutation-equivariance)
and revisited as hypothesis H4 (grammar-connectedness) in
[02.md](legacy/02.md) §H4 of the Stage-2.5 rewrite, which adds the
`require_goal_var_connected` ablation to prevent the disconnected-aux pathology
seen in Stage 2's best learned policy.

## §6 Synthesis as a derivation game

MCTS searches the grammar-derivation game

$$\boxed{\;\mathcal{D}_\Gamma = \langle\, \mathcal{Z},\; \mathcal{P}_\Gamma,\; T_\Gamma,\; R_\Gamma \,\rangle\;}$$

where $z\in\mathcal{Z}$ is a partial lifted-policy AST (`LiftedDerivationState`:
`completed_rules`, a `PartialRule`, and the `current_hole` $\in$ {policy, action_schema,
aux_var, pre_lit, goal_lit, $\bot$}); $p\in\mathcal{P}_\Gamma(z)$ a grammar production
expanding the current hole; $T_\Gamma(z,p)$ applies it; a terminal $z_T$ ($\mathrm{hole}=\bot$)
yields the complete policy $\mathrm{to\_program}(z_T)\in\mathrm{Policy}$; and

$$\boxed{\;R_\Gamma(z_T) = \mathrm{LeafEval}\bigl(\mathrm{to\_program}(z_T)\bigr)\;}$$

The leaf score rolls the policy out via the §5 interpreter on frozen task instances
($\mathrm{LiftedLeafEvaluator}$):

$$\boxed{\;\mathrm{LeafEval}(\pi) = \mathrm{solve\_rate} + 0.25\cdot\overline{\mathrm{progress}} - 0.01\cdot\overline{\mathrm{steps}} - 0.05\cdot\mathrm{num\_noops}\;}$$
$$\boxed{\;\mathrm{progress}(\pi,\mathcal{M}) = \frac{|\,s_T|_{\mathrm{at\_ball}} \cap G|_{\mathrm{at\_ball}}\,|}{|\,G|_{\mathrm{at\_ball}}\,|}\quad(\text{fraction of balls goal-placed at episode end})\;}$$

cached by `Policy.pretty()`. The MCTS engine ([core/mcts.py](../../src/alphazeropp/core/mcts.py))
and the `Game` protocol are reused unchanged; Stage 2 implements a **new**
`LiftedDerivationGame` ([lifted_derivation.py](../../src/alphazeropp/synthesis/lifted_derivation.py))
rather than refactoring the grounded `DerivationGame` (whose production bounds and
observation encoding presume an integer-indexed bitstring program). The grammar
([lifted_grammar.py](../../src/alphazeropp/synthesis/lifted_grammar.py)) emits Stage-1 DSL
objects directly, so $\mathrm{to\_program}(z_T)$ is immediately executable by
`interpret(...)`; well-formedness is encoded *by construction* — well-typed literals,
canonical literal order (lex key `(pred, args, negated)`), schema-position variable names,
$\le 1$ auxiliary variable, goal negation only when safe, no state negation, bounded caps
($\mathrm{max\_rules}{=}4$, $\mathrm{max\_pre\_literals}{=}3$, $\mathrm{max\_goal\_literals}{=}1$,
$\mathrm{max\_aux\_vars}{=}1$) — so `compute_max_productions` is a closed-form action-space
bound and each derivation node encodes a fixed 7-tuple
$(\mathrm{node\_kind},\mathrm{pred\_id},\mathrm{action\_id},\mathrm{type\_id},\mathrm{var\_local},\mathrm{source},\mathrm{negated})$.

## §7 Proposition: hand-policy plan length

The four-rule hand policy $\pi^{\mathrm{hand}} = [\rho_1,\rho_2,\rho_3,\rho_4]$ (§4) solves
Gripper-lite $\mathcal{M}_B$ in exactly $4B-1$ steps under deterministic dispatch by the §5
interpreter — proof, lemmas, rollout/trace figures, and the inductive case table live in the
companion note [notes_hand_policy_plan_length.md](notes_hand_policy_plan_length.md).

$$\boxed{\;\mathrm{plan\_length}(B) = 3 + 4(B-1) = 4B-1\;}$$

## §8 Stage ladder

| Stage | Question | Status |
|---|---|---|
| 0 ([this doc](00_draft_lifted_policy_az.md)) | What is the question, and what design space does it live in? | orientation |
| 1 ([01_plan.md](legacy/01_plan.md), [01.md](legacy/01.md)) | Does lifted dispatch execute correctly, without MCTS? | **done** — DSL + interpreter + Gripper-lite; 4-rule policy solves $B\in\{1,2,3\}$ |
| 2 ([02_plan.md](legacy/02_plan.md)) | Can grammar-MCTS generate *and* evaluate lifted policies? | **landed** — `LiftedDerivationGame` + grammar + encoding + leaf evaluator; Gripper-lite uniform-MCTS smoke runs A/B |
| 3 | Can a learned policy/value net improve the search? | future |
| 4 | Does the approach scale / generalize beyond Gripper-lite? | future |

(The older [docs/notes/stage1](../stage1)–[stage3](../stage3) are a separate
grounded-grammar lineage on Doors; this Stage-4 ladder uses the new lifted DSL, not those
grammars.)

## §9 Design decisions now fixed

| Question | Decision |
|---|---|
| Interpreter strategy | online unification — `find_bindings` / `interpret` (no full grounding) |
| State representation | `RelState = dict[str, set[tuple[str,...]]]` (per-predicate index) |
| Object identity | string names |
| Rule body | single ordered `body` tuple, each `Literal` tagged `source ∈ {STATE, GOAL}` and `negated` |
| Negation | safe negation checked at `Rule.__post_init__`; grammar pre-filters; goal negation allowed, state negation off |
| Multi-binding tie-break | lex-min over var-name-sorted object assignments |
| Task domain | Gripper-lite implemented; full classical Gripper / Doors-lifted are extensions |
| MCTS integration | new `LiftedDerivationGame` (MCTS core + `Game` protocol reused) |
| Leaf evaluation | `LiftedLeafEvaluator` rolls out via the Stage-1 interpreter; graded score (§6) |
| Search net | Stage 2 uses `UniformPolicyValueNet`; a learned net is Stage 3 |
| Grammar config | `LiftedGrammarConfig(max_rules=4, max_pre_literals=3, max_goal_literals=1, max_aux_vars=1, allow_goal_negation=True, allow_state_negation=False, allow_disjunction=False)` |

## §10 What is not claimed yet

- Stage 1 validates the interpreter and policy semantics with a *hand-written* policy — it
  does not show MCTS can discover that policy.
- Stage 2 is a systems smoke test (uniform-policy MCTS on Gripper-lite) — not learned-AlphaZero superiority.
- Gripper-lite success is not Doors success; Doors needs the `get_goal_atoms()` env-contract change first.
- The renaming test demonstrates name-agnostic matching, not full permutation equivariance (§5 caveat).
- Full classical Gripper (explicit grippers) is a separate extension.
- No baseline, evaluation grid, seed budget, or learning hyperparameters are committed —
  these are deferred to a downstream `0X_plan.md`.

## §11 References

- `[PG3]` Yang, R., Silver, T., Curtis, A., Lozano-Pérez, T., Kaelbling, L. P. (2022).
  *PG3: Policy-Guided Planning for Generalized Policy Generation.* IJCAI 2022, 4686–4692.
  (`papers/Yang-etal_2022_pg3-policy-guided-planning.pdf`.)
- Stage-1/2 source:
  [lifted_dsl.py](../../src/alphazeropp/synthesis/lifted_dsl.py) ·
  [lifted_interpreter.py](../../src/alphazeropp/synthesis/lifted_interpreter.py) ·
  [lifted_grammar.py](../../src/alphazeropp/synthesis/lifted_grammar.py) ·
  [lifted_derivation.py](../../src/alphazeropp/synthesis/lifted_derivation.py) ·
  [lifted_encoding.py](../../src/alphazeropp/synthesis/lifted_encoding.py) ·
  [lifted_leaf_evaluator.py](../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py) ·
  [gripper_lite/env.py](../../src/alphazeropp/instances/gripper_lite/env.py) ·
  [gripper_lite/policies.py](../../src/alphazeropp/instances/gripper_lite/policies.py) ·
  [tests/test_gripper_lite_policy.py](../../tests/test_gripper_lite_policy.py).
- Sibling stage-4 notes: [01_plan.md](legacy/01_plan.md) / [01.md](legacy/01.md) (Stage 1) ·
  [02_plan.md](legacy/02_plan.md) (Stage 2) · [literature_orientation.md](literature_orientation.md) ·
  [proposals_lifted_policy_az.md](proposals_lifted_policy_az.md).
- [notes_hand_policy_plan_length.md](notes_hand_policy_plan_length.md) — full proof of §7 (Lemmas 1–2, Rest-$k$ induction, rollout/trace figures).
