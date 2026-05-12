# 01 — Lifted-policy program synthesis with AlphaZero MCTS (orientation, rev. 1)

## §0 Status and reading guide

This is the project's **orientation and theory document**. It supersedes
[00_draft_lifted_policy_az.md](00_draft_lifted_policy_az.md): same problem, same two-level
architecture, same Gripper-lite MDP, same lifted-policy semantics — but the derivation grammar in §8
is **rewritten**. The old grammar had a *standalone `Aux` phase* — the search committed to a
body-local variable before any literal justified it — which produced the disconnected-variable
pathology Stage 2/2.5 documented ([legacy/02.md](legacy/02.md) §H4 / Table 2). §8 replaces it with an
**occurrence-introduced-variable grammar**: variables enter scope only where they first
*occur* (action arguments; then state literals, which *may* introduce fresh typed body-local
variables; then goal literals, which introduce none). This removes the `Aux` *syntax*, not body-local
variables themselves — hand-policy ρ₂ still needs one. The redesign **landed in Stage 2** ([02_plan.md](02_plan.md) / [02.md](02.md)):
the live code on branch `feature/grammar-redesign` now ships the occurrence-introduced grammar — no
`aux_var` hole, no `add_aux` / `SKIP_AUX` productions, no `max_aux_vars`, no `_KIND_AUX_VAR` encoding
node (see §8.5). This doc is not an experiment report; it is the framing the stage plans instantiate.
A changelog of conceptual changes vs `00_draft` is at the end.

There are **two decision processes**:

$$\boxed{\;\textbf{Task level: }\ \text{relational MDPs }\mathcal{M}_B\ (\text{Gripper-lite; later Doors PDDL}) \;}$$
$$\boxed{\;\textbf{Synthesis level: }\ \text{grammar-derivation game }\mathcal{D}_\Gamma\ \text{searched by MCTS} \;}$$

The **lifted policy** is the object connecting them: MCTS over $\mathcal{D}_\Gamma$ constructs a
lifted policy; the task MDP $\mathcal{M}_B$ evaluates it.

How to read: §1 the synthesis objective · §2 where this sits in the literature · §3 the two levels
side by side · §4 the task domains (Gripper-lite; the Doors PDDL contract) · §5 what a lifted policy
*is* · §6 the Gripper hand policy and its plan length · §7 how the interpreter executes one · §8 the
**redesigned** derivation grammar MCTS searches (why `Aux` is removed; occurrence-introduced
variables; example derivations; invariants; implementation status) · §9 the synthesis game and leaf
evaluator · §10 the experimental ladder · §11 what is *not* claimed · §12 references · changelog.

## §1 Problem statement

Given a family of relational MDPs $\{\mathcal{M}_i\}_{i\in\mathcal{I}}$ and a typed grammar $\Gamma$
generating lifted decision-list policies $\Pi_\Gamma$, the synthesis objective is

$$\boxed{\;\pi^\star \in \arg\max_{\pi\in\Pi_\Gamma} J_{\mathrm{train}}(\pi),\qquad
J_{\mathrm{train}}(\pi) = \frac{1}{|\mathcal{I}_{\mathrm{train}}|}\sum_{\mathcal{M}\in\mathcal{I}_{\mathrm{train}}} U_{\mathcal{M}}(\pi)\;}$$

where $U_{\mathcal{M}}(\pi)$ is the leaf score of $\pi$ on $\mathcal{M}$ (§9). The load-bearing
measurement is held-out generalization to larger / structurally different instances:

$$\boxed{\;J_{\mathrm{out}}(\pi) = \frac{1}{|\mathcal{I}_{\mathrm{out}}|}\sum_{\mathcal{M}\in\mathcal{I}_{\mathrm{out}}} U_{\mathcal{M}}(\pi)\;}$$

**Research question.** Can a *generic* AlphaZero-style MCTS over $\Gamma$ synthesize a compact lifted
policy that fits $\mathcal{I}_{\mathrm{train}}$ and generalizes to $\mathcal{I}_{\mathrm{out}}$ —
replacing PG3's domain-specific GBFS + STRIPS-A* score with episodic execution reward at
complete-program leaves?

## §2 Where this sits in the literature

(Pointers, not a survey — the full survey is [literature_orientation.md](literature_orientation.md).)

- **Generalized planning.** The goal of the field is a single object — a policy, a finite-state
  controller, a program — that solves *every* instance of a parametrized domain, not one plan per
  instance. We work in that frame: a lifted decision-list policy is one such object, and
  $J_{\mathrm{out}}$ measures the generalization that defines success.
- **PDDL / STRIPS domains.** The relational-MDP substrate is classical planning vocabulary: typed
  objects, predicates, lifted action schemas with add/delete effects. Gripper and Doors are standard
  instances. Our envs (§4) speak this vocabulary directly — a state is a set of ground atoms, a
  ground action a schema applied to objects.
- **First-order / lifted decision-list policies.** A policy is an ordered list of typed rules
  "if this lifted condition matches under some binding, fire this lifted action under that binding"
  — Rivest's decision lists (`Rivest_1987_learning-decision-lists.pdf`) lifted to first order, the
  representation relational reinforcement learning and PG3 both use. The condition is read under
  unification against the current state and the goal; first-applicable wins.
- **PG3 as the closest baseline** (`papers/Yang-etal_2022_pg3-policy-guided-planning.pdf`). PG3
  searches *policy space* by greedy best-first search, scoring each candidate policy by how
  suboptimal a STRIPS-A* plan guided by it is. We keep PG3's typed decision-list policy
  representation (its Def. 5) but **swap the search and the score**: MCTS over a grammar-derivation
  game in place of GBFS, and episodic execution reward at complete-program leaves in place of the
  STRIPS-A* suboptimality score (§9).
- **MCTS vs AlphaZero — a distinction we keep sharp.** AlphaZero = MCTS *guided by a learned policy
  / value network*. What is implemented today (§9, §10) is **uniform-prior MCTS**: PUCT with a flat
  policy and a value of zero (`UniformPolicyValueNet`), i.e. essentially PUCT-flavoured random
  search over the derivation game with a leaf rollout. *Uniform MCTS is not learned AlphaZero.* Any
  policy it turns up is a **search artifact**, not a learned policy; the learned-net version is a
  future stage (cf. `Silver-etal_2017_mastering-chess-shogi-alphazero.pdf` for the target shape).

## §3 Two levels of decision-making

| Level | state | action | terminal reward |
|---|---|---|---|
| Task MDP $\mathcal{M}_B$ | relational world state $s$ | `move` / `pick` / `drop` (Doors: `MOVE_TO` / `PICK` / `NOOP`) | $-1$ per step, $+100$ on solve |
| Derivation game $\mathcal{D}_\Gamma$ | partial lifted-policy AST $z$ | a grammar production $p$ | $\mathrm{LeafEval}(\mathrm{to\_program}(z_T))$ |

## §4 Task domains

### §4.1 Gripper-lite

For each ball count $B\ge 1$, Gripper-lite is the deterministic finite-horizon relational MDP
([gripper_lite/env.py](../../../src/alphazeropp/instances/gripper_lite/env.py))

$$\boxed{\;\mathcal{M}_B = \langle\, \mathcal{S}_B,\; \mathcal{A}_B,\; T_B,\; R_B,\; H_B,\; s^0_B,\; G_B \,\rangle\;}$$

**Types and objects.**

$$\boxed{\;\mathcal{T} = \{\textit{ball},\,\textit{room}\},\qquad
\mathcal{O}^B_{\textit{ball}} = \{\textit{ball\_}i : 0\le i < B\},\qquad
\mathcal{O}_{\textit{room}} = \{\textit{room\_a},\,\textit{room\_b}\}\;}$$

$\textit{room\_a}$ is the source room, $\textit{room\_b}$ the target.

**Predicates.** One *implicit* gripper, so the vocabulary uses $\mathrm{carrying}(b)$ and
$\mathrm{handempty}()$ — not the classical $\mathrm{carry}(b,g)$ / $\mathrm{free}(g)$; there is no
$\textit{gripper}$ type.

$$\boxed{\;\mathcal{P} = \{\,\mathrm{at\_robot}(\textit{room}),\;\mathrm{at\_ball}(\textit{ball},\textit{room}),\;\mathrm{carrying}(\textit{ball}),\;\mathrm{handempty}()\,\}\;}$$

**Relational state.** A state $s\in\mathcal{S}_B$ is a finite set of ground atoms — in code the
per-predicate index

$$\boxed{\;\mathrm{RelState} = \mathrm{dict}\bigl[\,\text{predicate name}\,\to\,\mathrm{set}[\,\mathrm{tuple}[\text{object name},\dots]\,]\,\bigr]\;}$$

satisfying the invariants: the robot is in exactly one room, $\exists!\,r\;\mathrm{at\_robot}(r)\in s$;
the hand is empty iff no ball is carried,
$\mathrm{handempty}()\in s \iff \forall b\;\mathrm{carrying}(b)\notin s$; each ball is in exactly one
room *or* carried,
$\forall b:\ \bigl[\exists!\,r\;\mathrm{at\_ball}(b,r)\in s\bigr]\,\oplus\,\bigl[\mathrm{carrying}(b)\in s\bigr]$.
Example ($B=2$): `{"at_robot":{("room_a",)}, "handempty":{()}, "carrying":set(),
"at_ball":{("ball_0","room_a"),("ball_1","room_a")}}`.

**Initial state and goal.**

$$\boxed{\;s^0_B = \{\mathrm{at\_robot}(\textit{room\_a}),\,\mathrm{handempty}()\}\,\cup\,\{\mathrm{at\_ball}(b,\textit{room\_a}):b\in\mathcal{O}^B_{\textit{ball}}\}\;}$$
$$\boxed{\;G_B = \{\mathrm{at\_ball}(b,\textit{room\_b}):b\in\mathcal{O}^B_{\textit{ball}}\},\qquad \mathrm{solved}(s)\iff s|_{\mathrm{at\_ball}} = G_B\;}$$

At time $0$ the robot stands in $\textit{room\_a}$ with an empty hand and all $B$ balls sit in
$\textit{room\_a}$; the goal is the mirror image — every ball ends up in $\textit{room\_b}$.

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

The hand-policy plan length is $4B-1$ (§6), so $H_B = 4B+4 \ge 4B-1$ is a conservative margin.

**Env contract** (what the interpreter and leaf evaluator consume): `get_state_atoms() -> RelState`,
`get_goal_atoms() -> RelState`, `get_objects_by_type() -> dict[str, tuple[str,...]]`,
`legal_actions() -> set[GroundAction]`, `step(GroundAction) -> (RelState, float, bool, dict)`,
`is_solved() -> bool`, `reset(...)`.

![Gripper-lite domain, B = 2: robot + balls start in room_a, goal is both balls in room_b](figures/gripper_lite_domain.png)

### §4.2 The Doors PDDL contract (relational adapter — wired in Stage 1 rev. 1)

Doors is the project's *primary* domain — multi-room navigation with keys, doors, and locking
dependencies — and ships an env, `DoorsPDDLLiteEnv`
([doors/doors_pddl_lite.py](../../../src/alphazeropp/instances/doors/doors_pddl_lite.py)). That env
is a **gymnasium flat-vector** env wired into the AlphaZero *benchmark* harness:

- observation: `spaces.Box(0, 1, shape=(obs_size,), float32)`; action: `spaces.Discrete(action_count)`;
- `reset(seed, options) -> (np.ndarray, dict)`, `step(action: int) -> (np.ndarray, float, bool, bool, dict)`,
  `is_solved(obs: np.ndarray) -> bool`;
- **no** `get_state_atoms` / `get_goal_atoms` / `get_objects_by_type` / relational `legal_actions`.

Stage 1 rev. 1 adds a thin **relational adapter** —
`DoorsPDDLLiteRelationalEnv` ([doors/doors_pddl_lifted.py](../../../src/alphazeropp/instances/doors/doors_pddl_lifted.py),
[01_plan.md](01_plan.md) / [01.md](01.md)) — wrapping `DoorsPDDLLiteEnv` (`.base`; the gym interface
untouched) and exposing the same lifted env contract Gripper-lite exposes
(`get_state_atoms() / get_goal_atoms() / get_objects_by_type() / legal_actions() / step(GroundAction) / reset / is_solved`),
over the Doors vocabulary

$$\boxed{\begin{aligned}
\mathcal{T}_{\mathrm{Doors}} &= \{\textit{room},\,\textit{location},\,\textit{key}\},\\
\mathcal{P}_{\mathrm{Doors}} &= \{\,\mathrm{at\_loc}(\textit{location}),\;\mathrm{unlocked}(\textit{room}),\;\mathrm{key\_avail}(\textit{key}),\\
&\qquad\;\;\mathrm{loc\_in\_room}(\textit{location},\textit{room}),\;\mathrm{key\_at}(\textit{key},\textit{location}),\;\mathrm{key\_unlocks}(\textit{key},\textit{room})\,\},\\
\mathcal{A}_{\mathrm{Doors}} &= \{\,\mathrm{move\_to}(\textit{location}),\;\mathrm{pick}(\textit{key}),\;\mathrm{noop}()\,\}
\end{aligned}}$$

The first three predicates are *fluents* (the obs vector); the last three are *static* relations
(the persistent instance structure `DoorsPDDLLiteEnv` stores as `loc_room` / `key_loc` /
`key_unlocks`), exposed so meaningful lifted Doors rules — "go to a reachable available key, then
pick it" — are expressible. The goal is a single goal location, so `get_goal_atoms()` returns
`{at_loc(<goal>)}` (`goal_predicate_names = ("at_loc",)`). A hand-written 3-rule policy
(`doors_hand_policy()`, [doors/doors_pddl_policies.py](../../../src/alphazeropp/instances/doors/doors_pddl_policies.py))
dispatched by the *same* §7 interpreter solves the two shipped layouts (D2 → 3 steps, D3 → 5 steps;
a chained-unlock layout with $K$ keys takes $2K{+}1$ steps) — a semantic-core demonstration on a
second domain, **not** an MCTS result and **not** a generalization claim (§11).

## §5 Lifted decision-list policies

A policy is an ordered decision list of typed rules (PG3, Yang et al. IJCAI 2022, Def. 5):

$$\boxed{\;\pi = [\rho_1,\dots,\rho_m]\;}$$

$$\boxed{\;\rho = \langle\, X_\rho,\; \mathrm{body}_\rho,\; \alpha_\rho \,\rangle,\qquad
\mathrm{body}_\rho = (L_1,\dots,L_k),\qquad L_j = \bigl(\mathrm{atom}_j,\ \mathrm{source}_j,\ \mathrm{negated}_j\bigr)\;}$$

where $X_\rho$ is a finite typed variable context; $\alpha_\rho = a(x_1,\dots)$ a lifted action with
$x_i\in X_\rho$ and types matching $\sigma_\mathcal{A}(a)$; each literal $L_j$ carries

$$\boxed{\;\mathrm{source}_j\in\{\mathrm{STATE},\,\mathrm{GOAL}\},\qquad \mathrm{negated}_j\in\{0,1\}\;}$$

A **binding** is a type-respecting map $\theta:X_\rho\to\mathcal{O}$. This is exactly the
`Rule(vars, body, action)` / `Policy(rules)` dataclass pair
([lifted_dsl.py](../../../src/alphazeropp/synthesis/lifted_dsl.py)). PG3 *motivates* the
$\langle\mathrm{PAR},\mathrm{PRE},\mathrm{GOAL},\mathrm{ACT}\rangle$ form but the implemented DSL
folds the $(\mathrm{PRE},\mathrm{GOAL})$ split into one ordered $\mathrm{body}$ tuple discriminated
by $\mathrm{source}$.

## §6 The Gripper-lite hand policy and its plan length

The hand-written policy ([gripper_lite/policies.py](../../../src/alphazeropp/instances/gripper_lite/policies.py))
has $m=4$ rules; $\rho_1,\rho_2$ carry a *positive* goal literal, $\rho_3,\rho_4$ a *negated* one;
every state literal is positive:

$$\boxed{\begin{aligned}
\rho_1:&\ \ \mathrm{carrying}(?b)\,\wedge\,\mathrm{at\_robot}(?r)\,\wedge\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)] \;\Rightarrow\; \mathrm{drop}(?b,?r)\\
\rho_2:&\ \ \mathrm{carrying}(?b)\,\wedge\,\mathrm{at\_robot}(?\textit{from})\,\wedge\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?\textit{to})] \;\Rightarrow\; \mathrm{move}(?\textit{from},?\textit{to})\\
\rho_3:&\ \ \mathrm{at\_ball}(?b,?r)\,\wedge\,\mathrm{at\_robot}(?r)\,\wedge\,\mathrm{handempty}()\,\wedge\,\neg\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)] \;\Rightarrow\; \mathrm{pick}(?b,?r)\\
\rho_4:&\ \ \mathrm{at\_robot}(?\textit{from})\,\wedge\,\mathrm{at\_ball}(?b,?\textit{to})\,\wedge\,\mathrm{handempty}()\,\wedge\,\neg\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?\textit{to})] \;\Rightarrow\; \mathrm{move}(?\textit{from},?\textit{to})
\end{aligned}}$$

**Body-local variable.** In $\rho_2$, $?b$ is **not** an argument of $\mathrm{move}$ — it occurs only
in $\mathrm{carrying}(?b)$ and $\mathrm{Goal}[\mathrm{at\_ball}(?b,?\textit{to})]$. It is a
*body-local variable* (the thing the redesign in §8 keeps while removing the `Aux` syntax).

**Plan length.** Under deterministic dispatch by the §7 interpreter, $\pi^{\mathrm{hand}}$ solves
Gripper-lite $\mathcal{M}_B$ in exactly

$$\boxed{\;\mathrm{plan\_length}(B) = 3 + 4(B-1) = 4B-1\;}$$

steps: each ball costs `pick → move → drop` (3 steps) plus a return `move(room\_b, room\_a)`
(1 step) — except the final ball, which needs no return trip. So **$B\ge 2$ needs all four rules**:
$\rho_4$ is what fires the return trip between drops. Proof, lemmas, rollout/trace figures, and the
inductive case table live in [notes_hand_policy_plan_length.md](notes_hand_policy_plan_length.md).

## §7 The online-unification interpreter

Closed-world literal satisfaction under a type-correct binding $\theta$ — state literals read against
$S$, goal literals against $G$:

$$\boxed{\begin{aligned}
S\models_\theta\ \ p(\bar x)\ \ &\iff\ p(\theta\bar x)\in S, & G\models_\theta\ \ p(\bar x)\ \ &\iff\ p(\theta\bar x)\in G,\\
S\models_\theta\ \neg p(\bar x)\ &\iff\ p(\theta\bar x)\notin S, & G\models_\theta\ \neg p(\bar x)\ &\iff\ p(\theta\bar x)\notin G.
\end{aligned}}$$

Negation is **safe** (enforced at `Rule.__post_init__`; the grammar also pre-filters): every
variable of a negative literal must already appear in a positive literal or in $\alpha_\rho$. The
satisfying-binding set, requiring legality, and the deterministic selection:

$$\boxed{\;B_\rho(S,G) = \{\theta : S\models_\theta\mathrm{body}^{S}_\rho \,\wedge\, G\models_\theta\mathrm{body}^{G}_\rho \,\wedge\, \theta(\alpha_\rho)\in\mathcal{A}_B(S)\}\;}$$

$$\boxed{\;\theta^\star_\rho(S,G) = \min_{\preceq} B_\rho(S,G),\qquad \theta\preceq\theta' \iff \bigl(\theta(x)\bigr)_{x\in\mathrm{sort}(X_\rho)} \le_{\mathrm{lex}} \bigl(\theta'(x)\bigr)_{x\in\mathrm{sort}(X_\rho)}\;}$$

(variables sorted by name, then object assignments compared lexicographically by raw object name —
the `find_bindings` sort key `tuple(sorted(theta.items()))`). Rule execution and the first-applicable
policy map:

$$\boxed{\;\mathrm{Exec}(\rho,S,G) = \theta^\star_\rho(S,G)(\alpha_\rho)\;}$$

$$\boxed{\;\mathrm{Interpret}(\pi,S,G,\mathcal{O},\mathrm{Legal}) = \begin{cases}\mathrm{Exec}(\rho_{i^\star},S,G), & i^\star = \min\{i : B_{\rho_i}(S,G)\ne\emptyset\}\\[2pt] \mathrm{None}, & \text{no rule applies}\end{cases}\;}$$

The implementation ([lifted_interpreter.py](../../../src/alphazeropp/synthesis/lifted_interpreter.py):
`find_bindings`, `interpret`) walks $\mathrm{body}_\rho$ literal-by-literal and never materializes
the full Cartesian product over $\mathcal{O}$ — so it transfers unchanged to held-out larger
instances.

**Renaming caveat (precise).** The matching semantics are name-agnostic *up to the tie-break order*.
For a type-preserving renaming $\sigma$,

$$\boxed{\;\mathrm{Interpret}(\pi,\sigma S,\sigma G,\sigma\mathcal{O},\sigma\mathrm{Legal}) = \sigma\,\mathrm{Interpret}(\pi,S,G,\mathcal{O},\mathrm{Legal})\quad\text{iff the lex order in }\theta^\star\text{ is the }\sigma\text{-transported order}\;}$$

With the implemented *raw object-name* tie-break this holds only for $\sigma$ that preserve the
induced object order (or on states with no contested tie) — it gives determinism, **not**
arbitrary-renaming equivariance. The Stage-1 renaming test uses such renamings
($\textit{ball\_0}\!\mapsto\!\textit{foo}$, $\textit{room\_a}\!\mapsto\!\textit{left\_room}$, $B=1$),
so it demonstrates name-agnostic matching — not full permutation equivariance. Same caveat,
cross-linked, in [legacy/01.md §4](legacy/01.md#4-object-renaming-name-agnostic-matching-not-full-permutation-equivariance).

## §8 The redesigned derivation grammar (proposed)

### §8.1 Why the standalone `Aux` phase is removed

In `00_draft`'s grammar, building a rule's variable context was a *separate phase*: after choosing
the action schema the search visited an `aux_var` hole (`current_hole == "aux_var"`) and either
`SKIP_AUX` or `add_aux:{type}` — introducing a body-local variable `?aux_i` *before any literal had
mentioned it*, capped by `max_aux_vars`. The pathology: nothing then forces that variable to occur
positively. The search could place `?aux_0` in `Goal[at_ball(?aux_0, ?r_1)]` and nowhere else, and
the interpreter reads that as an *existential* goal atom ("∃ a ball at a goal location") rather than
the intended "the ball I'm carrying is goal-placed." Stage 2's best uniform-MCTS policy does exactly
this — it leans on a vacuous goal predicate (`Goal[carrying(?b_0)]`) *and* a disconnected goal-bound
variable ([legacy/02.md](legacy/02.md) §H4 / Table 2; in 5 000 sampled permissive-grammar policies,
87.7% use a vacuous goal predicate and 16.2% have a goal-only-bound variable). The Stage-2.5 flag
`require_goal_var_connected` (now a default; see §8.4) filters those literals, but it patches a
*symptom*. The *cause* is decoupling variable introduction from literal occurrence.

### §8.2 Occurrence-introduced variables

The redesign removes the `Aux` nonterminal, the `aux_var` hole, and `max_aux_vars`. **Variables
enter scope only where they first occur:**

1. **Action variables** — introduced by choosing the action schema $a$; one per argument position,
   named by position (e.g. `?r_0, ?r_1` for `move(room, room)`), typed by $\sigma_\mathcal{A}(a)$
   (the current `action_vars_for_schema`).
2. **State literals** — a `pre_lit` production may introduce *fresh typed body-local variables*
   directly in the literal where they first occur (picking `carrying(?b)` with `?b` fresh introduces
   `?b : ball`). This is where body-local variables come from now — there is no separate phase.
3. **Goal literals** — a `goal_lit` production may use *only* variables already introduced by action
   arguments or positive state literals; it introduces none. Consequence: goal-literal-variable
   connectedness is **structural**, not a filter (see §8.4).
4. **Negated goal literals** — safe negation: every variable already covered. Subsumed by (3):
   since goal literals introduce nothing, every goal-literal variable is already positively covered.

Prose change: "auxiliary variables" → **"body-local variables"** (or "existential body variables").
Emphatically, removing the `aux_var` *syntax* does **not** ban body-local variables — rule (2) keeps
them, and rule (2) is exactly what hand-policy $\rho_2$ needs: `?b` is introduced by the state
literal `carrying(?b)`, used again in `Goal[at_ball(?b, ?to)]`, and is never a `move` argument.

Holes under the redesign: $\mathrm{current\_hole}\in\{\textit{policy},\,\textit{action\_schema},\,\textit{pre\_lit},\,\textit{goal\_lit},\,\bot\}$
(no `aux_var`). The partial-rule record drops the `aux_vars` field; body-local variables live
implicitly in the accumulated state literals.

### §8.3 Example derivations for two Gripper rules

**$\rho_3$** — `at_ball(?b,?r) ∧ at_robot(?r) ∧ handempty() ∧ ¬Goal[at_ball(?b,?r)] ⇒ pick(?b,?r)`:

| step | hole | production | scope after |
|---|---|---|---|
| 1 | `policy` | `ADD_RULE` | — |
| 2 | `action_schema` | `pick` | `?b:ball, ?r:room` (action args) |
| 3 | `pre_lit` | `at_ball(?b,?r)` | unchanged (no fresh var) |
| 4 | `pre_lit` | `at_robot(?r)` | unchanged |
| 5 | `pre_lit` | `handempty()` | unchanged |
| 6 | `goal_lit` | `¬at_ball(?b,?r)` | unchanged (`?b,?r` already covered ⇒ safe) |
| 7 | `goal_lit` | `FINISH_RULE` → next rule or `STOP_POLICY` | — |

**$\rho_2$** — `carrying(?b) ∧ at_robot(?from) ∧ Goal[at_ball(?b,?to)] ⇒ move(?from,?to)`:

| step | hole | production | scope after |
|---|---|---|---|
| 1 | `policy` | `ADD_RULE` | — |
| 2 | `action_schema` | `move` | `?from:room, ?to:room` (action args) |
| 3 | `pre_lit` | `carrying(?b)` — **introduces fresh body-local `?b:ball`** | `?from:room, ?to:room, ?b:ball` |
| 4 | `pre_lit` | `at_robot(?from)` | unchanged |
| 5 | `goal_lit` | `at_ball(?b,?to)` (`?b` covered by `carrying`, `?to` is an action arg ⇒ offered) | unchanged |
| 6 | `goal_lit` | `FINISH_RULE` | — |

Under the old grammar, $\rho_2$ needed an explicit `add_aux:ball` at step 3 *before* `carrying(?b)`;
under the redesign that step is gone and `?b` is born in the literal that uses it.

### §8.4 Invariants of the strict grammar

Well-formedness is encoded *by construction* — so `compute_max_productions` is a closed-form
action-space bound and each derivation node encodes a fixed-arity tuple:

- **Well-typed literals** — predicate argument types match $\mathcal{P}$; action arguments match
  $\sigma_\mathcal{A}$.
- **Canonical literal order** — a `pre_lit` / `goal_lit` only offers literals strictly greater than
  the last accepted body literal under the lex key `(source, predicate, args, negated)` (one
  ordering per rule body — kills permutations of the same body).
- **Schema-position action-variable names** — action arguments named by position, deterministically.
- **Body-local variables introduced only by state literals** (§8.2 rule 2); **goal literals
  introduce no variables** (rule 3) — so `require_goal_var_connected` is *vacuous by construction*
  under the redesign rather than an enforced filter.
- **Safe negation** — `allow_state_negation = False` (no state negation); `allow_goal_negation =
  True`, but a negated `goal_lit` is offered only if every variable in it is already covered.
- **Goal-predicate relevance** — `goal_predicate_relevance = True` consults
  `DomainSignature.goal_predicate_names`; for Gripper-lite this whitelist is `("at_ball",)`, so
  `Goal[carrying(...)]` / `Goal[handempty()]` candidates are never offered (a `None` whitelist makes
  the flag inert).
- **Bounded caps** — `max_rules` (default 4), `max_pre_literals` (default 3), `max_goal_literals`
  (default 1). `max_aux_vars` is *removed*; a body-local-variable cap `max_body_local_vars` (default
  2 — the Gripper hand policy needs ≤ 1) bounds how many fresh body-local variables a single rule may
  introduce, keeping `compute_max_productions` closed-form and the encoding fixed-width.
- **Fixed-arity per-node encoding** — `N_FIELDS_PER_NODE = 7`,
  $(\mathrm{node\_kind},\mathrm{predicate\_id},\mathrm{action\_id},\mathrm{type\_id},\mathrm{var\_local\_id},\mathrm{source\_id},\mathrm{negated\_bit})$;
  the redesign drops the `_KIND_AUX_VAR` node kind (no aux-var nodes — body-local vars are implicit
  in the encoded state literals) and the `aux_var` hole id.

The configuration this corresponds to:
`LiftedGrammarConfig(max_rules=4, max_pre_literals=3, max_goal_literals=1, max_body_local_vars=2,
allow_goal_negation=True, allow_state_negation=False, allow_disjunction=False, allow_constants=False,
goal_predicate_relevance=True, require_goal_var_connected=True)` — i.e. the *strict* grammar that
Stage-3-A made the default (`require_goal_var_connected` is now vacuous-by-construction — goal
literals introduce no variables — but kept as a config field). (`legacy_grammar_config()` recovers
the pre-Stage-3-A permissive grammar, with `goal_predicate_relevance=False`, for reproducing
`legacy/02.md`.)

### §8.5 Implementation status — landed in Stage 2

§8.2's grammar is **landed** ([02_plan.md](02_plan.md) / [02.md](02.md)). The live code (branch
`feature/grammar-redesign`): [lifted_grammar.py](../../../src/alphazeropp/synthesis/lifted_grammar.py),
[lifted_derivation.py](../../../src/alphazeropp/synthesis/lifted_derivation.py),
[lifted_encoding.py](../../../src/alphazeropp/synthesis/lifted_encoding.py) — has the
occurrence-introduced grammar: holes `{policy, action_schema, pre_lit, goal_lit, ⊥}` (no `aux_var`);
`pre_lit` productions whose argument positions may take a fresh typed body-local var (`state_literal_candidates`);
`goal_lit` productions over in-scope variables only (`goal_literal_candidates`); `PartialRule.body_local_vars`
(grown lazily in the `pre_lit` "add" branch); `LiftedGrammarConfig.max_body_local_vars` replacing
`max_aux_vars`; the encoding without `_KIND_AUX_VAR`; `compute_max_productions` recomputed against the
maximal-scope construction. `goal_predicate_relevance` / `require_goal_var_connected` remain config
defaults (the latter now a no-op). The Stage-2/2.5/3 *legacy* drivers
(`make_lifted_gripper_canonical.py`, `make_lifted_gripper_landscape.py`,
`run_lifted_gripper_diagnostic_grid.py`) and their committed `data/runA_* / runB_* / landscape_* /
diagnostic_grid/` artifacts are *not* re-pinned — they remain a record of the aux-var grammar; the
new minimal-MCTS run lands under `data/minimal_mcts/`.

## §9 Synthesis as a derivation game; the leaf evaluator

MCTS searches the grammar-derivation game

$$\boxed{\;\mathcal{D}_\Gamma = \langle\, \mathcal{Z},\; \mathcal{P}_\Gamma,\; T_\Gamma,\; R_\Gamma \,\rangle\;}$$

where $z\in\mathcal{Z}$ is a partial lifted-policy AST (`LiftedDerivationState`: `completed_rules`, a
partial rule, and the `current_hole` — under the redesign one of {policy, action_schema, pre_lit,
goal_lit, $\bot$}); $p\in\mathcal{P}_\Gamma(z)$ a grammar production expanding the current hole;
$T_\Gamma(z,p)$ applies it; a terminal $z_T$ ($\mathrm{hole}=\bot$) yields the complete policy
$\mathrm{to\_program}(z_T)\in\mathrm{Policy}$; and

$$\boxed{\;R_\Gamma(z_T) = \mathrm{LeafEval}\bigl(\mathrm{to\_program}(z_T)\bigr)\;}$$

The leaf score rolls the policy out via the §7 interpreter on frozen task instances
($\mathrm{LiftedLeafEvaluator}$,
[lifted_leaf_evaluator.py](../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py)):

$$\boxed{\;\mathrm{LeafEval}(\pi) = \mathrm{solve\_rate} + 0.25\cdot\overline{\mathrm{progress}} - 0.01\cdot\overline{\mathrm{steps}} - 0.05\cdot\mathrm{num\_noops}\;}$$
$$\boxed{\;\mathrm{progress}(\pi,\mathcal{M}) = \frac{|\,s_T|_{\mathrm{at\_ball}} \cap G|_{\mathrm{at\_ball}}\,|}{|\,G|_{\mathrm{at\_ball}}\,|}\quad(\text{fraction of balls goal-placed at episode end});\quad \mathrm{num\_noops} = \#\{\,t : \mathrm{interpret}(\dots)=\mathrm{None}\,\}\;}$$

cached per policy. The MCTS engine ([core/mcts.py](../../../src/alphazeropp/core/mcts.py)) and the
`Game` protocol ([core/game.py](../../../src/alphazeropp/core/game.py)) are reused unchanged; the
synthesis side implements a **new** `LiftedDerivationGame`
([lifted_derivation.py](../../../src/alphazeropp/synthesis/lifted_derivation.py)) rather than
refactoring the grounded `DerivationGame` (whose production bounds and observation encoding presume
an integer-indexed bitstring program). The grammar emits Stage-1 DSL objects directly, so
$\mathrm{to\_program}(z_T)$ is immediately executable by `interpret(...)`. The search prior today is
`UniformPolicyValueNet` — flat policy, zero value — so this is **uniform-prior MCTS**, not learned
AlphaZero (§2, §11); a learned policy/value net is a future stage. The strict grammar (§8.4) — and,
more strongly, the §8.2 redesign — makes the disconnected-variable pathology of §8.1 unreachable, so
MCTS cannot win the leaf score by exploiting it.

## §10 Experimental ladder

| Stage | Question | Status |
|---|---|---|
| 0 ([00_draft](00_draft_lifted_policy_az.md), this doc) | What is the question, and what design space does it live in? | orientation |
| 1 ([01_plan.md](01_plan.md), [01.md](01.md); v0: [legacy/01_plan.md](legacy/01_plan.md), [legacy/01.md](legacy/01.md)) | Does lifted dispatch execute correctly, without MCTS — on *two* relational contracts? | **done** — DSL + interpreter + Gripper-lite (4-rule policy solves $B\in\{1,2,3\}$) **and** the Doors-PDDL relational adapter (`DoorsPDDLLiteRelationalEnv` + the `get_goal_atoms()` env-contract addition; a 3-rule hand policy solves D2/D3). No grammar, no MCTS. |
| 2 ([02_plan.md](02_plan.md), [02.md](02.md); legacy aux-var record: [legacy/02_plan.md](legacy/02_plan.md), [legacy/02_plan2.md](legacy/02_plan2.md), [legacy/02.md](legacy/02.md), [legacy/03_plan.md](legacy/03_plan.md), [legacy/03.md](legacy/03.md)) | **(rev. — occurrence-introduced grammar.)** Land the §8.2 grammar cutover; can *uniform-prior* MCTS over it find reasonable Gripper-lite policies for $B\in\{1,2\}$? (Same-$B$ train/eval; no generalization claim.) | **landed** — the `Aux` phase is removed; `LiftedDerivationGame` + the redesigned grammar + encoding + leaf evaluator + a minimal MCTS run on Gripper $B\in\{1,2\}$ (see [02.md](02.md)). The legacy Stage-2/2.5/3 aux-var work (permissive vs strict grammar; the spurious-solver pathologies; the diagnostic grid) is preserved under `legacy/`. |
| 3 | (folded into Stage 2 — the strict-grammar defaults / structural pathology removal are now part of the §8.2 cutover; the legacy Stage-3-A diagnostic grid is in [legacy/03.md](legacy/03.md).) | superseded |
| 4 | Can a learned policy/value net improve the search? | future — replace `UniformPolicyValueNet`. |
| 5 | Does the approach scale / generalize (larger $B$; Doors PDDL)? | future. |

(The older [docs/notes/stage1](../stage1)–[stage3](../stage3) are a separate grounded-grammar
lineage on Doors; this Stage-4 ladder uses the new lifted DSL, not those grammars.)

## §11 What is not claimed

- Stage 1 validates the interpreter and policy semantics with a *hand-written* policy — it does not
  show MCTS can discover that policy.
- Stage 2's MCTS is **uniform-prior**, not learned AlphaZero; any policy it finds (including the
  Gripper-lite $B{=}1$ solvers reported in [02.md](02.md)) is a *search artifact*, not a "learned
  policy." No learned-AlphaZero superiority is claimed. Stage 2 trains and evaluates on the **same
  $B$** — no held-out generalization result.
- Removing the `aux_var` *syntax* (§8) does **not** remove body-local variables — state literals
  still introduce them (hand-policy $\rho_2$ needs one).
- The renaming test demonstrates name-agnostic matching, not arbitrary permutation equivariance
  (§7 caveat; the implemented tie-break is lexicographic over *raw* object names).
- The Doors relational adapter exists (Stage 1 rev. 1) and its hand policy solves the two shipped
  layouts D2/D3 — but that is a *semantic-core dispatch* demonstration, not generalization to
  arbitrary Doors layouts, and not an MCTS / grammar-search result. No Doors policy was *synthesized*.
- Full classical Gripper (explicit grippers, `carry(b,g)` / `free(g)`) is a separate extension.
- No landscape enumeration, score-variant plots, or learning hyperparameters in Stage 2 beyond what
  [02.md](02.md) reports; the legacy aux-var-grammar landscape / diagnostic-grid study is in
  [legacy/02.md](legacy/02.md) / [legacy/03.md](legacy/03.md).

## §12 References

- `[PG3]` Yang, R., Silver, T., Curtis, A., Lozano-Pérez, T., Kaelbling, L. P. (2022). *PG3:
  Policy-Guided Planning for Generalized Policy Generation.* IJCAI 2022, 4686–4692.
  ([papers/Yang-etal_2022_pg3-policy-guided-planning.pdf](../../../papers/Yang-etal_2022_pg3-policy-guided-planning.pdf)).
- `[AlphaZero]` Silver et al. (2017). *Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm.*
  ([papers/Silver-etal_2017_mastering-chess-shogi-alphazero.pdf](../../../papers/Silver-etal_2017_mastering-chess-shogi-alphazero.pdf)).
- `[Decision lists]` Rivest (1987). *Learning Decision Lists.*
  ([papers/Rivest_1987_learning-decision-lists.pdf](../../../papers/Rivest_1987_learning-decision-lists.pdf)).
- Source — semantic core: [lifted_dsl.py](../../../src/alphazeropp/synthesis/lifted_dsl.py) ·
  [lifted_interpreter.py](../../../src/alphazeropp/synthesis/lifted_interpreter.py) ·
  [gripper_lite/env.py](../../../src/alphazeropp/instances/gripper_lite/env.py) ·
  [gripper_lite/policies.py](../../../src/alphazeropp/instances/gripper_lite/policies.py) ·
  [doors/doors_pddl_lite.py](../../../src/alphazeropp/instances/doors/doors_pddl_lite.py) ·
  [doors/doors_pddl_lifted.py](../../../src/alphazeropp/instances/doors/doors_pddl_lifted.py) ·
  [doors/doors_pddl_policies.py](../../../src/alphazeropp/instances/doors/doors_pddl_policies.py).
- Source — synthesis: [lifted_grammar.py](../../../src/alphazeropp/synthesis/lifted_grammar.py) ·
  [lifted_derivation.py](../../../src/alphazeropp/synthesis/lifted_derivation.py) ·
  [lifted_encoding.py](../../../src/alphazeropp/synthesis/lifted_encoding.py) ·
  [lifted_leaf_evaluator.py](../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py) ·
  [lifted_diagnostics.py](../../../src/alphazeropp/synthesis/lifted_diagnostics.py) ·
  [core/mcts.py](../../../src/alphazeropp/core/mcts.py) · [core/game.py](../../../src/alphazeropp/core/game.py).
- Stage notes: [01_plan.md](01_plan.md) / [01.md](01.md) (Stage 1 rev. 1 — Gripper-lite + Doors PDDL) ·
  [02_plan.md](02_plan.md) / [02.md](02.md) (Stage 2 — the §8.2 grammar cutover + minimal uniform-MCTS run).
- Legacy stage notes (the *aux-var grammar* record): [legacy/01_plan.md](legacy/01_plan.md) / [legacy/01.md](legacy/01.md) (Stage 1, v0 single-domain) ·
  [legacy/02_plan.md](legacy/02_plan.md) / [legacy/02_plan2.md](legacy/02_plan2.md) /
  [legacy/02.md](legacy/02.md) (Stage 2/2.5) · [legacy/03_plan.md](legacy/03_plan.md) /
  [legacy/03.md](legacy/03.md) (Stage 3-A).
- Companion notes: [literature_orientation.md](literature_orientation.md) ·
  [notes_hand_policy_plan_length.md](notes_hand_policy_plan_length.md) (full proof of §6) ·
  [proposals_lifted_policy_az.md](proposals_lifted_policy_az.md) ·
  [notes/derivation_game.md](notes/derivation_game.md) (background reference; rewrite planned —
  [notes/rewrite.md](notes/rewrite.md)) · [notes/pddl_background.md](notes/pddl_background.md).
- Refined implementation plan for this orientation-doc rewrite + the `legacy/` reorg: [rewrite00_plan.md](rewrite00_plan.md).

## Changelog — conceptual changes vs `00_draft_lifted_policy_az.md`

1. **Removed the standalone `Aux` phase** from the grammar description — no `Aux` nonterminal, no
   `aux_var` hole, no `max_aux_vars`. Introduced **occurrence-introduced variables**: action
   arguments (from the schema choice) → state literals *may* introduce fresh typed body-local
   variables → goal literals introduce none (§8.1–§8.3).
2. **Renamed** "auxiliary variables" → "body-local variables" / "existential body variables", and
   stated explicitly that removing the `aux_var` *syntax* does not remove body-local variables —
   hand-policy $\rho_2$ still needs one (§6, §8.2, §11).
3. **Promoted** `goal_predicate_relevance` and `require_goal_var_connected` from Stage-2.5 ablations
   to grammar invariants (they are already `LiftedGrammarConfig` defaults); under the §8.2 redesign
   goal-variable connectedness becomes *structural* rather than an enforced filter (§8.4).
4. **Added §2 — literature positioning**: generalized planning; PDDL/STRIPS domains; first-order /
   lifted decision-list policies; PG3 as the closest baseline (we keep its policy form, swap its
   GBFS + STRIPS-A* score for MCTS + leaf-rollout reward); the MCTS-vs-learned-AlphaZero distinction
   stated sharply (*uniform MCTS is not learned AlphaZero*).
5. **Added §4.2 — the Doors PDDL contract**: a relational adapter (`DoorsPDDLLiteRelationalEnv`)
   over `DoorsPDDLLiteEnv` + the `get_goal_atoms()` env-contract addition. **Implemented in Stage 1
   rev. 1** ([01_plan.md](01_plan.md) / [01.md](01.md)) — the vocabulary box gains three static
   relations (`loc_in_room` / `key_at` / `key_unlocks`); a 3-rule hand policy solves D2/D3.
6. **Restructured into 12 sections** matching the new outline; folded the hand-policy plan-length
   proposition (old §7) into §6.
7. **Sharpened the non-claims** (§11): uniform MCTS ≠ learned AlphaZero; body-local variables
   survive the `Aux` removal; the Doors adapter (Stage 1 rev. 1) is a dispatch demo, not synthesis;
   the renaming caveat restated.
8. **Updated cross-links**: the numbered Stage-1/2/3 notes moved to [`legacy/`](legacy/) and the
   references throughout point there; added a pointer to [rewrite00_plan.md](rewrite00_plan.md).
9. **§8 redesign landed in Stage 2** ([02_plan.md](02_plan.md) / [02.md](02.md)): the `aux_var` hole,
   `add_aux` / `SKIP_AUX` productions, `max_aux_vars`, `PartialRule.aux_vars`, and the `_KIND_AUX_VAR`
   encoding node are gone; `pre_lit` literals may introduce a fresh body-local `?v_i`; goal literals
   introduce none; `LiftedGrammarConfig.max_body_local_vars` replaces `max_aux_vars`. §8.5 is now an
   "implemented" status note. (Doc revised from the original "proposed redesign" framing of `rev. 1`.)
