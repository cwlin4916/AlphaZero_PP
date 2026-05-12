# Appendix (Stage 4) — The derivation game

> Background reference for Stage 4. Cited from [../legacy/01.md](../legacy/01.md), [../legacy/02.md](../legacy/02.md) (the Stage-2/2.5 *aux-var-grammar* landscape results — this note is the expanded form of [Appendix B](../legacy/02.md#appendix-b--the-derivation-game-formally) there), [../01_draft_lifted_policy_az.md](../01_draft_lifted_policy_az.md) §8–§9 and [../02_plan.md](../02_plan.md) (the grammar cutover), and the synthesis modules under [src/alphazeropp/synthesis/](../../../../src/alphazeropp/synthesis/). Companion to [pddl_background.md](pddl_background.md). Style follows [../../appendix/sygus_background.md](../../appendix/sygus_background.md).
>
> **Status — rewritten for the occurrence-introduced-variable grammar (Stage-2 cutover; [../01_draft_lifted_policy_az.md](../01_draft_lifted_policy_az.md) §8, [../02_plan.md](../02_plan.md)).** Supersedes the earlier description, which had a standalone `aux_var` hole and a `?aux_0` "witness variable". In the live grammar there is **no `aux_var` hole**: variables enter scope only where they first *occur* — action arguments from the schema choice, body-local variables `?v_0, ?v_1, …` born inside the precondition literal that first uses them, and goal literals introduce none. The Stage-2 cutover *results* companion (uniform MCTS over the redesigned grammar) is `../02.md` (pending); for the prior aux-var-grammar landscape work see [../legacy/02.md](../legacy/02.md).
>
> **Read it tutorial-first.** §§1–13 need only basic sets, functions, finite-state machines, and a passing acquaintance with context-free grammars. §§14–15 are the dense formal / code-audit layer — safe to skip on a first pass and come back to. The note exists because the early arrow-diagram notation compressed **four genuinely different objects** into one picture, and the result is easy to misread.

## Contents

- [§1 — One-page mental model](#1--one-page-mental-model)
- [§2 — Two kinds of state](#2--two-kinds-of-state)
- [§3 — What the arrow labels mean](#3--what-the-arrow-labels-mean)
- [§4 — The hole-state machine](#4--the-hole-state-machine)
- [§5 — A worked derivation, field by field](#5--a-worked-derivation-field-by-field)
- [§6 — The fields of a partial rule](#6--the-fields-of-a-partial-rule)
- [§7 — Why goal literals are not cheating](#7--why-goal-literals-are-not-cheating)
- [§8 — Body-local variables, introduced by occurrence](#8--body-local-variables-introduced-by-occurrence)
- [§9 — What rules can this language create?](#9--what-rules-can-this-language-create)
- [§10 — Is this a direct cheat?](#10--is-this-a-direct-cheat)
- [§11 — Failure modes of this policy language](#11--failure-modes-of-this-policy-language)
- [§12 — Binding set and execution](#12--binding-set-and-execution)
- [§13 — The grammar as a typed attributed CFG](#13--the-grammar-as-a-typed-attributed-cfg)
- [§14 — The derivation game as an MDP (formal)](#14--the-derivation-game-as-an-mdp-formal)
- [§15 — Code-fidelity audit](#15--code-fidelity-audit)
- [§16 — Figure index, notation bridge, and cross-links](#16--figure-index-notation-bridge-and-cross-links)

## §1 — One-page mental model

The **derivation game is a policy-writing game.** A player (here, MCTS) fills in a half-written program one production at a time until it is a complete *lifted decision-list policy* $\pi$; then $\pi$ is run on Gripper-lite tasks and scored. The whole pipeline:

$$\boxed{\;\underbrace{\Gamma}_{\text{grammar}}\ \longrightarrow\ z_T\ \longrightarrow\ \pi\ \longrightarrow\ \underbrace{\text{rollout in }\mathcal{M}_B}_{\text{interpreter}+\text{env}}\ \longrightarrow\ U_B(\pi).\;}$$

Three things, three jobs, never mixed:

- **The grammar $\Gamma$ writes *syntax*** — it builds well-typed candidate policies and never looks at any world state.
- **The interpreter gives *semantics*** — given a finished policy $\pi$, a world state $x$, and a goal $G$, it decides which action to take.
- **The environment $\mathcal{M}_B$ supplies the task states and goals** — the relational state $x$ and goal $G$ the policy must react to.

There are exactly four objects in play, on three levels (grammar / program / world):

| object | plain English | code field / class | example |
|---|---|---|---|
| $z$ | a **derivation state** — a half-written policy plus a cursor | [`LiftedDerivationState`](../../../../src/alphazeropp/synthesis/lifted_derivation.py) (`completed_rules`, `partial`, `current_hole`) | $((),\ \varnothing,\ \texttt{policy})$ — empty, nothing in progress |
| $q$ | the **partial rule** currently under construction inside $z$ | `z.partial` — a [`PartialRule`](../../../../src/alphazeropp/synthesis/lifted_derivation.py) (`schema`, `action_args`, `body_local_vars`, `state_lits`, `goal_lits`); `None` ⇔ $q=\varnothing$ | schema `move`, two preconditions added so far; the second one, `carrying(?v_0)`, introduced the body-local var `?v_0:ball` |
| $h$ | the **current hole** — which one slot of $z$ is open | `z.current_hole` ∈ {`policy`, `action_schema`, `pre_lit`, `goal_lit`, `None`} | `pre_lit` — "next decision is which precondition to add (or to stop)" |
| $p$ | a **grammar production** — written on an arrow; an *action* of the synthesis MDP | [`LiftedProduction`](../../../../src/alphazeropp/synthesis/lifted_grammar.py) (`hole_kind`, `label`, `payload`) | `schema=move`; `pre:carrying(?v_0)`; `FINISH_RULE` |
| $\rho$ | a **sealed rule** — one finished IF–THEN rule, appended to the policy | [`Rule`](../../../../src/alphazeropp/synthesis/lifted_dsl.py) (`vars`, `body`, `action`) | $\mathrm{at\_robot}(?r_0)\wedge\mathrm{carrying}(?v_0)\wedge\dots\Rightarrow\mathrm{move}(?r_0,?r_1)$ |
| $\pi$ | a **complete policy** — an ordered list of sealed rules | [`Policy`](../../../../src/alphazeropp/synthesis/lifted_dsl.py) (`rules`) | the 4-rule Stage-1 hand policy |
| $\alpha_\rho$ | the **lifted action template** of rule $\rho$ — what the rule outputs, with variables not yet bound | `rule.action` — a [`LiftedAction`](../../../../src/alphazeropp/synthesis/lifted_dsl.py) (`schema`, `args`) | $\mathrm{move}(?r_0,?r_1)$ |
| $x$ | a **task / world state** — a set of ground atoms describing the current Gripper-lite world | the relational state the env exposes ([gripper_lite/env.py](../../../../src/alphazeropp/instances/gripper_lite/env.py)); type `RelState = dict[str, set[tuple]]` | $\{\mathrm{at\_robot}(\textit{room\_a}),\,\mathrm{handempty}(),\,\mathrm{at\_ball}(\textit{ball}_0,\textit{room\_a})\}$ |
| $G$ | a **goal** — the set of ground atoms the task wants made true | another `RelState` ([gripper_lite/env.py](../../../../src/alphazeropp/instances/gripper_lite/env.py)) | $\{\mathrm{at\_ball}(\textit{ball}_0,\textit{room\_b})\}$ |
| $\theta$ | a **runtime binding** — an assignment of rule variables to concrete objects, chosen by the interpreter | `Binding = dict[str, str]` in [lifted_interpreter.py](../../../../src/alphazeropp/synthesis/lifted_interpreter.py) | $\{?r_0\mapsto\textit{room\_a},\ ?r_1\mapsto\textit{room\_b},\ ?v_0\mapsto\textit{ball}_0\}$ |
| $B_\rho(x,G)$ | the **binding set** of rule $\rho$ on $(x,G)$ — every binding under which $\rho$ is applicable right now | output of `find_bindings` in [lifted_interpreter.py](../../../../src/alphazeropp/synthesis/lifted_interpreter.py) | $\{\{?r_0\mapsto\textit{room\_a},\,?r_1\mapsto\textit{room\_b},\,?v_0\mapsto\textit{ball}_0\}\}$ |

The derivation state is written

$$\boxed{\;z = (C,\ q,\ h),\qquad C=\text{completed rules so far},\quad q=\text{partial rule (or }\varnothing),\quad h=\text{current hole}.\;}$$

**The trap.** $x$ is *not* part of $z$. The grammar (levels $z,q,p$) builds *syntax* and never touches a world state. A world state $x$ enters only at the very end: when a derivation finishes, the completed policy $\pi$ is rolled out through the Stage-1 interpreter on Gripper-lite instances and scored by the leaf utility $U_B$ ([../legacy/02.md §Minimal setup](../legacy/02.md#minimal-setup)). So "`state_lits`" in a `PartialRule` does *not* mean "states of the derivation game" — it means *precondition literals*, which will later be matched against a world state $x$ ([§2](#2--two-kinds-of-state), [§6](#6--the-fields-of-a-partial-rule)). They are also where body-local variables are *born* ([§8](#8--body-local-variables-introduced-by-occurrence)).

In one line, the design philosophy:

$$\boxed{\;\underbrace{\Gamma}_{\text{grammar: syntax}}\ \longrightarrow\ z_T\ \longrightarrow\ \pi\ \longrightarrow\ \underbrace{\text{rollout in }\mathcal{M}_B}_{\text{interpreter + env: semantics}}\ \longrightarrow\ U_B(\pi).\;}$$

## §2 — Two kinds of state

The single most common confusion is that "state" means two unrelated things here. Keep them apart.

**Task / world state $x$.** A snapshot of the Gripper-lite world: a set of ground atoms (the robot's room, which balls it carries, where the balls are). For example,

$$x = \{\,\mathrm{at\_robot}(\textit{room\_a}),\ \mathrm{handempty}(),\ \mathrm{at\_ball}(\textit{ball}_0,\textit{room\_a})\,\}.$$

In words: the robot is in room a, holding nothing, with ball 0 also in room a. This is what the *interpreter* reads when it runs a finished policy.

**Derivation-game state $z$.** A snapshot of the *policy being written*:

$$z = (C,\ q,\ h)$$

— the rules finished so far ($C$), the rule currently under construction ($q$, or $\varnothing$ if none), and which one slot is open next ($h$). In words: $z$ describes a half-finished *program*, not a world. This is what *MCTS* searches over.

**The thing to remember.** `state_lits` (a field of a partial rule) has nothing to do with derivation states. It is a list of *precondition literals* — conditions like $\mathrm{carrying}(?v_0)$ — that will eventually be checked against a world state $x$ at runtime. Read `state_lits` as "precondition literals" everywhere in this note; the field is just not renamed in the code. And note: a `pre:ℓ` literal is also where the rule's *body-local variables* come into existence (see [§8](#8--body-local-variables-introduced-by-occurrence)).

## §3 — What the arrow labels mean

A derivation is a path

$$\boxed{\;z_0\ \xrightarrow{\,p_1\,}\ z_1\ \xrightarrow{\,p_2\,}\ z_2\ \xrightarrow{\,p_3\,}\ \cdots\ \xrightarrow{\,p_T\,}\ z_T.\;}$$

The word on each arrow, $p_i$, is a **production** — not a state, not a rule, not a literal. Reading it: $z_i \xrightarrow{p_i} z_{i+1}$ means *"apply grammar production $p_i$ to the one open hole of $z_i$, getting $z_{i+1}$."* The step is deterministic and pure:

$$\boxed{\;z_{i+1} = z_i.\texttt{apply}(p_{i+1})\quad\text{(never mutates }z_i\text{).}\;}$$

In words: each production fills exactly the open hole and produces a fresh state; nothing is overwritten. Here is the whole vocabulary of arrow labels (seven, in the occurrence-introduced grammar — there is no `SKIP_AUX` / `add_aux:T` anymore):

| arrow label | what it does |
|---|---|
| `ADD_RULE` | open a fresh, empty partial rule $q$ (move from the `policy` hole into a rule) |
| `schema=move` / `schema=pick` / `schema=drop` | choose the rule's **action template** — fixes which action the rule will output and introduces its argument variables, named by **schema position** (`move` ⇒ `?r_0:room`, `?r_1:room`; `drop` ⇒ `?b_0:ball`, `?r_1:room`) |
| `pre:ℓ` | add the precondition (state) literal $\ell$ to the rule's body; $\ell$ *may* introduce one or more fresh typed **body-local variables** `?v_i` in argument positions that use them (see [§8](#8--body-local-variables-introduced-by-occurrence)) |
| `STOP_PRE` | stop adding precondition literals; move on to goal literals |
| `goal:ℓ` | add the goal literal $\ell$ (a `Goal[...]` atom, possibly negated) to the body; $\ell$ may *reference* in-scope variables only — it **introduces none** |
| `FINISH_RULE` | **seal** the partial rule $q$ into a completed rule $\rho$, append it to $C$, return to the `policy` hole |
| `STOP_POLICY` | stop writing rules; the policy is now complete (terminal) |

A note on the `pre:ℓ` row. A precondition literal $\ell = P(a_1,\dots,a_k)$ is built one argument slot at a time, left to right: each slot is filled either by an *existing in-scope variable of the right type* or by a *fresh typed body-local variable* `?v_i` — and the latter only while the rule still has room under `max_body_local_vars`. So a literal like `at_ball(?v_0, ?r_1)` (the ball slot fresh, the room slot an existing action argument) is a single `pre:ℓ` production that introduces one variable, while `at_robot(?r_0)` (only an existing argument) introduces none. The naming is deterministic — fresh variables are numbered `?v_{j}` in introduction order across the rule.

Four things that *look* alike but live on different levels, all visible along one derivation:

1. **a production-label string** $p_i$ — e.g. the text `pre:carrying(?v_0)` written on an arrow;
2. **a `Literal`** — e.g. the syntactic atom $\mathrm{carrying}(?v_0)$ sitting inside a rule's body;
3. **a sealed rule** $\rho \in C$ — a whole IF–THEN sentence produced by a `FINISH_RULE` arrow;
4. **a task action** — e.g. `drop` / `pick` / `move`, an action of the *Gripper-lite* MDP, chosen by the interpreter at runtime, **not** an action of the derivation game.

Finally, two kinds of arrow behave differently. A **content** production (`schema=…`, `pre:ℓ` — which may carry a fresh variable, `goal:ℓ`) adds something to $q$. A **cursor** production (`STOP_PRE`, `FINISH_RULE`, `STOP_POLICY`) adds nothing — it just moves the hole $h$ along (and `FINISH_RULE` additionally seals $q$ into $C$).

## §4 — The hole-state machine

At every non-terminal moment there is **exactly one open hole** $h$, and one production is applied per step. The hole runs around a small finite-state machine — this is a *left-to-right policy writer*, not a free-form AST editor with many open frontier nodes:

```
                 ADD_RULE             schema=X            STOP_PRE
   policy  ───────────────▶ action_schema ─────────────▶ pre_lit ──────────▶ goal_lit
     │  ▲                                                  │  ⟲ pre:ℓ          │  ⟲ goal:ℓ
     │  │                                  (≤ L_S literals, each may add a ?v_i)│  (≤ L_G literals, no new vars)
     │  │ FINISH_RULE  (seal q into C, return to the policy hole)               │
     │  └─────────────────────────────────────────────────────────────────────────┘
     │
     │ STOP_POLICY   (#completed rules ≥ 1)
     ▼
     ⊥   (terminal: q = ∅, h = ⊥)
```

The four holes, in plain English:

- **`policy`** — between rules: "start another rule, or stop?" Here $q=\varnothing$.
- **`action_schema`** — just opened a rule: "which action will this rule output?" (this choice also introduces the action-argument variables, named by position).
- **`pre_lit`** — "add another precondition literal, or stop preconditions?" (loops; capped at $L_S$). A precondition literal may introduce one or more fresh typed body-local variables `?v_i` in argument positions that use them (capped per-rule by `max_body_local_vars`).
- **`goal_lit`** — "add another goal literal, or finish the rule?" (loops; capped at $L_G$). A goal literal references in-scope variables only — it never introduces one.

Two invariants worth keeping: $q=\varnothing \iff h\in\{\texttt{policy},\bot\}$ (a partial rule exists exactly while a rule is in progress); and every non-terminal hole offers **at least one** production — a `STOP_*` / `FINISH_RULE` cursor production is always available, so the writer never gets stuck.

![the four hole kinds as a state machine; nodes annotated with legal-production counts; the pre_lit hole is the widest](../figures/02_derivation_state_machine.png)

**Figure D1 — the hole-state machine.** The four holes drawn as a finite-state machine, each node labelled with how many productions it offers in a representative state. The `pre_lit` hole is the widest hole-kind (a state literal may take a fresh typed body-local variable in any argument position), and its analytic worst case, $M = \texttt{compute\_max\_productions}(\mathrm{cfg},\Sigma) = 26$ for the default Gripper-lite config, is what sizes the MCTS action head ([§14](#14--the-derivation-game-as-an-mdp-formal)). Regenerated from the live grammar — see [§16](#16--figure-index-notation-bridge-and-cross-links).

**The full production catalogue** (this is exactly `enumerate_productions`, [lifted_grammar.py:318-371](../../../../src/alphazeropp/synthesis/lifted_grammar.py)):

| hole $h$ | productions (`label`) | `payload` tag | offered when |
|---|---|---|---|
| `policy` | `ADD_RULE` | `("add_rule",)` | $\lvert C\rvert < R$ and the signature has $\ge 1$ action schema |
| `policy` | `STOP_POLICY` | `("stop",)` | $\lvert C\rvert \ge 1$ |
| `action_schema` | `schema={name}` — one per action schema (`schema=move`, `schema=pick`, `schema=drop`) | `("schema", name, arg_types)` | always (introduces the schema-position variables, e.g. `move` ⇒ `?r_0:room`, `?r_1:room`) |
| `pre_lit` | `pre:{ℓ}` — one per legal next precondition literal | `("add", ℓ, new_body_local_vars)` | $\lvert q.\texttt{state\_lits}\rvert < L_S$; only literals strictly greater than the last accepted one under the fixed lex key `(predicate, args, negated)` (so no duplicates, no reorderings); each argument slot is an existing in-scope variable of the right type or a fresh typed body-local variable `?v_i`, with the per-rule total capped at `max_body_local_vars`; state literals are never negated |
| `pre_lit` | `STOP_PRE` | `("stop_pre",)` | always |
| `goal_lit` | `goal:{ℓ}` — one per legal next goal literal | `("add", ℓ, ())` | $\lvert q.\texttt{goal\_lits}\rvert < L_G$; canonical lex order; **every argument is a variable already in scope** (an action argument or a variable of a positive state literal already added) — so a goal literal never introduces a variable, and goal-variable connectedness is *structural*; if `goal_predicate_relevance` is on (default), only predicates in the signature's `goal_predicate_names`; a negated `Goal[…]` is offered only when every variable is positively bound (safe negation) |
| `goal_lit` | `FINISH_RULE` | `("finish",)` | always (seals $q$ into a `Rule`, appends to $C$, hole → `policy`) |
| `⊥` | — (none) | — | terminal |

Two naming conventions ([lifted_grammar.py](../../../../src/alphazeropp/synthesis/lifted_grammar.py) — "Design points" and `action_vars_for_schema` / `body_local_var_name`): action-argument variables are named by **schema position** — `move(?r_0,?r_1)`, `pick(?b_0,?r_1)`, `drop(?b_0,?r_1)` — and body-local variables are `?v_{j}` in **introduction order** (the order their literals were added). The per-rule cap is `max_body_local_vars` (default $2$; the Gripper hand policy needs $\le 1$ — ρ₂'s/ρ₄'s `?v_0`). Because of this, two rules of the same shape pretty-print identically — no separate α-canonicalisation pass is needed at production time.

![legal-production sets at one representative state per hole kind; the pre_lit hole is the widest](../figures/02_derivation_production_sets.png)

**Figure D2 — production sets per hole.** $\mathcal{P}_\Gamma(z) = \texttt{enumerate\_productions}(z,\mathrm{cfg},\Sigma)$ shown at one representative state per hole kind (Gripper-lite signature). The `pre_lit` hole-kind is the widest; its analytic worst case over all states, $M = \texttt{compute\_max\_productions}(\mathrm{cfg},\Sigma) = 26$ for the default config ($33$ with `legacy_grammar_config()`, i.e. both safety flags off), is the upper bound on the MCTS action head ([§14](#14--the-derivation-game-as-an-mdp-formal)). The representative states shown have fewer than $M$ productions; $M$ is computed with the maximal possible variable scope.

## §5 — A worked derivation, field by field

We build the Stage-1 hand-policy rule "move-toward-goal" — chosen because it exercises the new mechanic: `move`'s schema is $\textit{room}\times\textit{room}$, so the carried ball it reasons about is a **body-local variable** `?v_0:ball`, not an action argument:

$$\boxed{\;\rho_2:\ \ \mathrm{at\_robot}(?r_0)\,\wedge\,\mathrm{carrying}(?v_0)\,\wedge\,\mathrm{Goal}[\mathrm{at\_ball}(?v_0,?r_1)]\ \Longrightarrow\ \mathrm{move}(?r_0,?r_1)\;}$$

(α-equivalent to $\rho_2$ as written in [../01_draft_lifted_policy_az.md §6](../01_draft_lifted_policy_az.md#6-the-gripper-lite-hand-policy-and-its-plan-length) — the body-literal order here is the canonical one the grammar emits: $\mathrm{at\_robot}$ before $\mathrm{carrying}$ by lex key, then the goal literal. We keep the hand-policy name $\rho_2$ throughout, even though `Policy.pretty()` numbers it `ρ_1` when it stands alone in the one-rule policy of Figure D3.) The derivation is an 8-step play (no `SKIP_AUX` step — there is no aux phase); each row below is one arrow. Field names are the implementation's: $q=z.\texttt{partial}$, with sub-fields `schema`, `action_args`, `body_local_vars`, `state_lits`, `goal_lits`. (Read `state_lits` as *precondition literals*.)

| step | transition | what the production does | what changes |
|---|---|---|---|
| 0 | — | the start | $z_0 = ((),\ \varnothing,\ \texttt{policy})$ — no rules, no rule in progress |
| 1 | $z_0 \xrightarrow{\texttt{ADD\_RULE}} z_1$ | open a fresh partial rule | $h:\texttt{policy}\to\texttt{action\_schema}$; $q_1 = \texttt{PartialRule.empty()} = (\varnothing_{\text{schema}},\,()_{\text{args}},\,()_{\text{body\_local}},\,()_{\text{pre}},\,()_{\text{goal}})$ |
| 2 | $z_1 \xrightarrow{\texttt{schema=move}} z_2$ | choose the action template | $h\to\texttt{pre\_lit}$; $q_2.\texttt{schema}=\texttt{move}$, $q_2.\texttt{action\_args}=(?r_0{:}\textit{room},\,?r_1{:}\textit{room})$ — this pins $\alpha_{\rho_2}=\mathrm{move}(?r_0,?r_1)$ |
| 3 | $z_2 \xrightarrow{\texttt{pre:at\_robot(?r\_0)}} z_3$ | add a precondition over an action argument | $q_3.\texttt{state\_lits}=(\mathrm{at\_robot}(?r_0))$; no fresh variable; $h$ stays `pre_lit` |
| 4 | $z_3 \xrightarrow{\texttt{pre:carrying(?v\_0)}} z_4$ | add a precondition that **introduces a body-local variable** | $q_4.\texttt{state\_lits}=(\mathrm{at\_robot}(?r_0),\ \mathrm{carrying}(?v_0))$; $q_4.\texttt{body\_local\_vars}=(?v_0{:}\textit{ball})$ — `carrying` needs a `ball` argument and the scope had none, so this slot is a *fresh* `?v_0` |
| 5 | $z_4 \xrightarrow{\texttt{STOP\_PRE}} z_5$ | stop preconditions | $h\to\texttt{goal\_lit}$; $q_5 = q_4$ (cursor only) |
| 6 | $z_5 \xrightarrow{\texttt{goal:Goal[at\_ball(?v\_0,?r\_1)]}} z_6$ | add a goal literal over in-scope vars | $q_6.\texttt{goal\_lits}=(\mathrm{Goal}[\mathrm{at\_ball}(?v_0,?r_1)])$ — both arguments (`?v_0` from a state literal, `?r_1` an action arg) are already in scope; nothing new is introduced |
| 7 | $z_6 \xrightarrow{\texttt{FINISH\_RULE}} z_7$ | seal the rule | $\rho_2$ appended to $C$ — `vars = action_args ∪ body_local_vars = (?r_0, ?r_1, ?v_0)`, body $=$ `state_lits` $+$ `goal_lits`, head $=\mathrm{move}(?r_0,?r_1)$; then $q\to\varnothing$, $h\to\texttt{policy}$ |
| 8 | $z_7 \xrightarrow{\texttt{STOP\_POLICY}} z_8$ | end the policy | $h\to\bot$; $z_8 = ((\rho_2),\ \varnothing,\ \bot)$ is terminal; $\texttt{to\_program}(z_8)=\texttt{Policy}((\rho_2,))$ |

The $q_i$ are just the partial rule as it stands after step $i$: $q_1$ is the empty accumulator, $q_2$ has its schema and action variables, $q_3,q_4$ have growing precondition lists (and $q_4$ also a body-local variable), $q_6$ has the goal literal too — and at step 7 that accumulator becomes the immutable sealed rule $\rho_2$. Note that step 3 had to come before step 4: the grammar only ever offers a *strictly greater* next literal under the lex key, and $\mathrm{at\_robot}$ precedes $\mathrm{carrying}$ — so the body order is forced.

![one derivation as a labelled path through the transition delta, building rho_2 (move-toward-goal) end-to-end](../figures/02_derivation_example.png)

**Figure D3 — the worked derivation, linearly.** The 8-step build of $\rho_2$ drawn as a labelled path through the transition $\delta_\Gamma$, then `STOP_POLICY`. Each state is read off a live `LiftedDerivationState`, so what you see is what the code produces — note `?v_0:ball` appearing in `body_local_vars` exactly when `pre:carrying(?v_0)` fires.

![a field-by-field diff table along the build of rho_2; the pre:carrying(?v_0) row touches two fields](../figures/02_derivation_state_evolution.png)

**Figure D5 — the same derivation as a field-by-field diff.** Each row is a state $z_i$; the highlighted cell is the field the production $\delta_\Gamma$ touched on that step. The `pre:carrying(?v_0)` step is the one that touches *two* fields — `state_lits` and `body_local_vars` — because the literal it adds also introduces a variable. (Figure D4, the static "anatomy" of a state, is in [§6](#6--the-fields-of-a-partial-rule).)

**How long is a derivation?** Each rule costs **4 fixed productions** — `ADD_RULE`, `schema=…`, `STOP_PRE`, `FINISH_RULE` — plus up to $L_S$ `pre:ℓ` and up to $L_G$ `goal:ℓ`. A whole policy of $k$ rules then ends with one `STOP_POLICY`:

$$\ell(\pi) \;=\; 1 + \sum_{i=1}^{k}\bigl(4 + \lvert\texttt{pre}_i\rvert + \lvert\texttt{goal}_i\rvert\bigr),\qquad 1\le k\le R,$$

so $\ell(\pi)\in[\,5,\ R(L_S+L_G+4)+1\,]$ — for Gripper-lite ($L_S=3,L_G=1$) that is $[\,5,\ 8R + 1\,]$. (The $\rho_2$-only example above has $\ell = 1 + (4+2+1) = 8$.)

![per-rule production breakdown (4 fixed + ≤P pre-lits + ≤G goal-lits), the analytic length range, and the empirical length distribution](../figures/02_derivation_production_lengths.png)

**Figure D6 — derivation length.** Per-rule structure ($4$ fixed productions $+\le L_S$ pre-lits $+\le L_G$ goal-lits), the analytic length range $\ell(\pi)\in[5,\ R(L_S+L_G+4)+1]$ vs `max_rules`, and the empirical length distribution over 5000 uniform-random derivations (a uniform prior favours short policies — relevant to [§11](#11--failure-modes-of-this-policy-language)).

## §6 — The fields of a partial rule

A partial rule — and the `Rule` it seals into — is a **half-written IF–THEN sentence**: *IF (these conditions on the world) AND (these conditions on the goal) THEN (this action)*. Formally,

$$\boxed{\;q = (\,\texttt{schema},\ \texttt{action\_args},\ \texttt{body\_local\_vars},\ \texttt{state\_lits},\ \texttt{goal\_lits}\,).\;}$$

In words, field by field (with the $\rho_2$ example from [§5](#5--a-worked-derivation-field-by-field)):

| field | plain English | $\rho_2$ example |
|---|---|---|
| `schema` | which action *form* the rule will output | `move` |
| `action_args` | the variables that appear *in the action* — named by schema position | $(?r_0{:}\textit{room},\ ?r_1{:}\textit{room})$ |
| `body_local_vars` | the body-local (existential) variables used only in conditions, never in the action — each one was *introduced* by the precondition literal that first uses it; grown lazily as `pre:ℓ` literals are added, with no separate phase | $(?v_0{:}\textit{ball})$ — born in $\mathrm{carrying}(?v_0)$ |
| `state_lits` | conditions checked against the **current world state** $x$ — these are precondition literals; positive only in this stage | $(\mathrm{at\_robot}(?r_0),\ \mathrm{carrying}(?v_0))$ |
| `goal_lits` | conditions checked against the **goal** $G$ — written `Goal[…]`, possibly negated; reference in-scope variables only | $(\mathrm{Goal}[\mathrm{at\_ball}(?v_0,?r_1)])$ |

When `FINISH_RULE` fires, the rule is sealed as `Rule(vars = action_args ∪ body_local_vars, body = state_lits + goal_lits, action = LiftedAction(schema, action_args))` — see `LiftedDerivationState.apply` ([lifted_derivation.py:129-136](../../../../src/alphazeropp/synthesis/lifted_derivation.py)). So `state_lits` and `goal_lits` end up in one ordered `body` tuple, distinguished by each literal's `source` tag (`STATE` vs `GOAL`) — [§7](#7--why-goal-literals-are-not-cheating) explains why that split matters.

![six representative LiftedDerivationStates with every field rendered explicitly, including body_local_vars](../figures/02_derivation_state_anatomy.png)

**Figure D4 — state anatomy.** Six representative `LiftedDerivationState`s with *every* field rendered: `completed_rules` / `partial = PartialRule(schema, action_args, body_local_vars, state_lits, goal_lits)` / the single `current_hole`, plus the state's `pretty()` node key (which carries the variable types — see [§4](#4--the-hole-state-machine)). One panel is exactly after `pre:carrying(?v_0)`, where `body_local_vars` has just grown.

## §7 — Why goal literals are not cheating

A natural worry: *the game synthesises a policy, but rules can mention `Goal[at_ball(?b,?r)]` — isn't the goal being smuggled in?* No. The synthesised policy is **goal-conditioned**: at execution time the interpreter is handed *both* the current state $x$ **and** the goal $G$, and a goal literal is just a condition that reads $G$. The policy never invents or hard-codes a fixed goal at derivation time — it is written so that, at runtime, it inspects whichever `at_ball(...)` goal atoms *this* task instance happens to supply. Moreover a goal literal can only *reference* a variable already justified by an action argument or a positive state literal — it cannot *introduce* one — so a `Goal[…]` atom never silently quantifies over a fresh object (see [§8](#8--body-local-variables-introduced-by-occurrence)).

The contrast is sharpest on the drop rule. Without a goal literal:

$$\mathrm{carrying}(?b)\,\wedge\,\mathrm{at\_robot}(?r)\ \Longrightarrow\ \mathrm{drop}(?b,?r)$$

In words: "if I'm carrying a ball, drop it wherever I happen to be." That solves nothing in general — it dumps balls in arbitrary rooms. With a goal literal:

$$\mathrm{carrying}(?b)\,\wedge\,\mathrm{at\_robot}(?r)\,\wedge\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)]\ \Longrightarrow\ \mathrm{drop}(?b,?r)$$

In words: "drop this ball *only* in the room where the goal wants it." Same action, but now it is correct, and it is correct *for every goal*, because the rule reads the goal rather than assuming one.

**Negated goal literals** read the goal the other way. In the pick rule,

$$\neg\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)]$$

means "this ball is *not* already where the goal wants it" — i.e. don't bother re-picking a ball that's already finished. Negation here is **safe**: every variable in a negated goal literal must already appear positively (in `state_lits`) or among the action arguments, so by the time the interpreter evaluates it the variable is bound. The grammar pre-filters at the `goal_lit` hole ([lifted_grammar.py:304-306](../../../../src/alphazeropp/synthesis/lifted_grammar.py)), and [`Rule.__post_init__`](../../../../src/alphazeropp/synthesis/lifted_dsl.py) re-checks it ([lifted_dsl.py:164-180](../../../../src/alphazeropp/synthesis/lifted_dsl.py)). State literals are never negated in this stage (`allow_state_negation=False`).

## §8 — Body-local variables, introduced by occurrence

Sometimes the action itself does not mention every object you need in order to decide whether the action is appropriate. The clean example is "move toward the goal."

`move`'s schema is $\textit{room}\times\textit{room}$, so its schema-position variables are $?r_0,?r_1$ only — **there is no ball variable in a `move` rule by default.** A first try has nothing to say about *which* room to head to:

$$\mathrm{at\_robot}(?r_0)\ \Longrightarrow\ \mathrm{move}(?r_0,?r_1)$$

In words: "go from wherever I am to … some room." $?r_1$ is unconstrained — the rule has no reason to pick the right destination. We need a handle on the carried ball and where it should go. In the occurrence-introduced grammar that handle appears exactly where it is first used: the precondition literal `carrying(·)` needs a `ball` argument and the scope has none, so that slot is filled by a *fresh* body-local variable `?v_0:ball`. The body can then say:

$$\mathrm{at\_robot}(?r_0)\,\wedge\,\mathrm{carrying}(?v_0)\,\wedge\,\mathrm{Goal}[\mathrm{at\_ball}(?v_0,?r_1)]\ \Longrightarrow\ \mathrm{move}(?r_0,?r_1)$$

(this is hand-policy $\rho_2$ up to α-renaming). In words: "if I'm carrying *some* ball and the goal wants that ball in room $?r_1$, then move there." Crucially, **$?v_0$ is a variable, not the object `ball_0`** — it is bound at runtime. With $x=\{\mathrm{at\_robot}(\textit{room\_a}),\,\mathrm{carrying}(\textit{ball}_0)\}$ and $G=\{\mathrm{at\_ball}(\textit{ball}_0,\textit{room\_b})\}$, the interpreter binds

$$?r_0\mapsto\textit{room\_a},\quad ?r_1\mapsto\textit{room\_b},\quad ?v_0\mapsto\textit{ball}_0$$

and the rule fires $\mathrm{move}(\textit{room\_a},\textit{room\_b})$. Notice $?v_0$ *chose the destination* but does **not** appear in the ground action — it is a body-local (existential) variable, present only to make the conditions decidable.

**Where variables come from — the whole story.** A variable enters scope only at its first *occurrence*:

1. **action arguments** — introduced by the `schema=X` production, named by schema position (`?r_0,?r_1` for `move`; `?b_0,?r_1` for `pick`/`drop`);
2. **state literals** — a `pre:ℓ` production *may* introduce a fresh typed body-local variable `?v_i` directly in any argument slot of $\ell$ that uses it, while the rule has room under `max_body_local_vars`;
3. **goal literals** — introduce *nothing*; every argument of a `Goal[…]` atom must already be in scope.

This is why the **disconnected-goal-variable pathology is structurally impossible** in the live grammar. The old aux-var grammar had a standalone `Aux` phase that committed to a variable *before* any literal justified it; a derivation could then place that variable in a goal literal where it occurred nowhere else, getting the unintended "there exists some object" reading (see [../legacy/02.md §Hypotheses (H4)](../legacy/02.md#hypotheses-and-next-experiments) — the `?aux_0` placed in `Goal[at_ball(?aux_0, ?r_1)]` with `?aux_0` bound nowhere). Removing the `Aux` phase removes the *cause*: a goal literal can only reference variables that an action argument or a positive state literal already justified, so goal-variable connectedness holds by construction. The `require_goal_var_connected` flag is consequently a no-op for grammar-produced rules — vacuous-by-construction, kept only for diagnostics on hand-written / legacy policies ([lifted_grammar.py:86-92](../../../../src/alphazeropp/synthesis/lifted_grammar.py)). Cross-link [../01_draft_lifted_policy_az.md §8.1–§8.2](../01_draft_lifted_policy_az.md#8-the-redesigned-derivation-grammar-proposed) and [../legacy/02.md §Hypotheses (H4)](../legacy/02.md#hypotheses-and-next-experiments).

## §9 — What rules can this language create?

Step back from the machinery: what is the **hypothesis class**? A policy $\pi$ is an **ordered list of up to $R$ rules**. Each rule:

- outputs **exactly one action schema** (`move` / `pick` / `drop`);
- carries up to $L_S$ precondition (`state`) literals — positive, well-typed atoms over the rule's variables, where each literal *may* introduce a fresh body-local variable;
- carries up to $L_G$ goal literals — `Goal[…]` atoms over in-scope variables, possibly negated (safely); they introduce no variables;
- has at most $V_b = \texttt{max\_body\_local\_vars}$ body-local (existential) variables in total.

The configuration defaults in [`LiftedGrammarConfig`](../../../../src/alphazeropp/synthesis/lifted_grammar.py) are $R=4$, $L_S=3$, $L_G=1$, $V_b=2$ (the Gripper hand policy needs $V_b\le 1$ — ρ₂'s/ρ₄'s `?v_0`); both Stage-2.5 safety flags (`goal_predicate_relevance`, `require_goal_var_connected`) are **on** by default ([../legacy/03_plan.md](../legacy/03_plan.md) promoted them). `legacy_grammar_config()` recovers the permissive grammar (both off). Some rule families this allows:

- **empty-body rules** — $\top\Rightarrow a(\cdot)$, e.g. $\top\Rightarrow\mathrm{drop}(?b_0,?r_1)$ ("always try to drop");
- **state-reactive rules** — preconditions only, e.g. $\mathrm{at\_ball}(?b_0,?r_1)\wedge\mathrm{handempty}()\Rightarrow\mathrm{pick}(?b_0,?r_1)$;
- **goal-conditioned rules** — a positive `Goal[…]`, e.g. the drop rule of [§7](#7--why-goal-literals-are-not-cheating) ($\rho_1$) — here every variable is an action argument;
- **negated-goal guards** — $\dots\wedge\neg\mathrm{Goal}[\mathrm{at\_ball}(?b_0,?r_1)]\Rightarrow\dots$ ("only when this ball isn't done") — hand-policy $\rho_3$;
- **body-local-variable rules** — $\rho_2$ of [§8](#8--body-local-variables-introduced-by-occurrence), whose `?v_0:ball` is born inside `carrying(?v_0)`.

All four Stage-1 hand-policy rules $\rho_1\dots\rho_4$ are reachable in this grammar up to α-renaming (test `tests/test_lifted_grammar_occurrence_vars.py`; $\rho_2$ and $\rho_4$ use a body-local `?v_0:ball`); the language is **not under-expressive for this task** — see [§10](#10--is-this-a-direct-cheat).

**How big is the program space?** The MCTS action head is sized to the worst-case branching $M(\mathrm{cfg},\Sigma)=\texttt{compute\_max\_productions}=26$ for the default Gripper-lite signature (the `pre_lit` hole dominates — Fig D2; it is $33$ with `legacy_grammar_config()`). As an order-of-magnitude reference, the *old* aux-var grammar's `r1` config ($(R,L_S,L_G,V_{\mathrm{aux}})=(1,2,1,1)$) had $\approx1\,166$ distinct policies — small enough to enumerate exhaustively ([../legacy/02.md](../legacy/02.md) Table 2); that count is not re-enumerated for the occurrence-introduced grammar. One depth fact carries over: **at least $R\ge 3$ is needed for any $\mathcal{M}_B$-solver** — the `r1` ($R=1$) landscape sweep found zero. So the difficulty here is *not* a needle-in-a-galaxy capacity problem; it is a *density / landscape* problem ([§10](#10--is-this-a-direct-cheat), [§11](#11--failure-modes-of-this-policy-language)).

![capacity knobs, program-space size, and one real Rule per rule family — built through the grammar](../figures/02_derivation_hypothesis_class.png)

**Figure D8 — the hypothesis class.** The capacity tuple $\mathrm{cfg}=(R,L_S,L_G,V_b)$ with its `LiftedGrammarConfig` defaults and the disabled switches; the program-space size ($M=26$, the old aux-var `r1` config enumerable at $\approx1\,166$ policies, $R\ge 3$ for any solver); and a gallery of one real `Rule` per family — each built through the grammar, so nothing is hand-transcribed. Regenerated from the live grammar — see [§16](#16--figure-index-notation-bridge-and-cross-links).

| knob | meaning | default | code |
|---|---|---|---|
| $R$ | max rules in the policy | 4 | `max_rules` ([lifted_grammar.py:62](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |
| $L_S$ | max precondition (state) literals per rule | 3 | `max_pre_literals` ([lifted_grammar.py:63](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |
| $L_G$ | max goal literals per rule | 1 | `max_goal_literals` ([lifted_grammar.py:64](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |
| $V_b$ | max body-local (existential) variables per rule | 2 | `max_body_local_vars` ([lifted_grammar.py:65-71](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |
| — | disjunction in a body / negated preconditions / object constants | off | `allow_disjunction` / `allow_state_negation` / `allow_constants` $=$ `False` ([lifted_grammar.py:74-76](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |
| — | goal-predicate whitelist / goal-var-connectedness | on | `goal_predicate_relevance` / `require_goal_var_connected` $=$ `True` ([lifted_grammar.py:80-92](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); the latter is vacuous-by-construction |
| $M(\mathrm{cfg},\Sigma)$ | worst-case branching $=$ MCTS action-head width | 26 | `compute_max_productions` ([lifted_grammar.py:378-413](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |

The **semantics** is an ordered decision list under PG3-style *first-applicable* rules: at each step the interpreter tries the rules top to bottom; the first rule with a non-empty binding set fires, and within that rule the **lex-min** binding is chosen (so, e.g., `ball_0` before `ball_1`). Details in [§12](#12--binding-set-and-execution).

## §10 — Is this a direct cheat?

It is fair to ask whether handing MCTS this grammar is "handing it the answer." Two honest halves:

- **Yes, the grammar is domain-specific.** It is built from a `DomainSignature` — the domain's types, predicate schemas, and action schemas — so it "knows the vocabulary" of Gripper-lite (it will only ever produce `at_robot` / `at_ball` / `carrying` / `handempty` atoms and `move` / `pick` / `drop` actions). That is by design.
- **No, it is not handed the solution policy.** MCTS still has to *choose the production sequence* — which schema, which literals (and where to introduce a body-local variable), in which order, how many rules, where to stop. The grammar only constrains the search to *well-typed, executable lifted policies*: it cannot emit an ill-typed atom, an action with an unbound argument, an unsafe negation, a duplicate / reordered literal, or a goal literal over an unjustified variable. That is the standard posture of syntax-guided synthesis and generalized-policy search — an intentionally structured search space, not a lookup table.

Quantitatively the space is *not* narrow enough to hand over the answer: in the Stage-2/2.5 aux-var-grammar landscape sweep ([../legacy/02.md](../legacy/02.md) Table 2; Fig L1 below) the solver density $\lvert\{\pi:\mathrm{solve}_B(\pi)=1\}\rvert/\lvert\Pi_{\Gamma,\mathrm{cfg}}\rvert$ was $\le 0.1\%$ across every `r3` setting, and uniform-prior MCTS found $B{=}1$ solvers in only one of the committed runs — converging on *other* (spurious) solvers, not the hand policy. The Stage-1 hand-policy *shape* is **expressible by design, but not forced**: the occurrence-introduced grammar expresses $\rho_1\dots\rho_4$ up to α-renaming (test `tests/test_lifted_grammar_occurrence_vars.py`), but *reachable ≠ findable*. So Stage 2's question is "can MCTS search this structured space?" — **not** "did a domain-free learner rediscover the domain model?" That gap is exactly what [../legacy/02.md](../legacy/02.md) Hypotheses **H1** (solver density too low for unguided search), **H3** ($B=1$ identifiability — many spurious solvers hit $\mathrm{solve}_1$), **H4** (the connectedness rule — now subsumed by construction), and **H5** (a *learned* prior is needed) name; Stage 3 attacks H5. The Stage-2 cutover *results* over the redesigned grammar will be reported in `../02.md` (pending).

![B=1 / B=2 solver counts per grammar config; ≤ 0.1% everywhere — from the legacy aux-var landscape sweep](../figures/02_landscape_solver_density.png)

**Figure L1 (from [../legacy/02.md](../legacy/02.md)) — solver density by grammar config.** $B{=}1$ (blue) / $B{=}2$ (red) solver counts per Stage-2.5 safety-flag setting, for the `r1` exhaustive enumeration and the `r3` stratified sample of the *old aux-var grammar*; the annotation `k (p%)` gives the count and the fraction of policies — $\le 0.1\%$ everywhere. The canonical read is in [../legacy/02.md §Empirical result / Fig L1](../legacy/02.md).

## §11 — Failure modes of this policy language

A restricted policy language has characteristic ways of going wrong. Most of these are what the Stage-2/2.5 aux-var-grammar landscape sweep ([../legacy/02.md](../legacy/02.md)) actually observed; the occurrence-introduced grammar removes one class of them outright.

- **Too-general empty-body rules.** $\top\Rightarrow\mathrm{drop}(?b_0,?r_1)$ ([`degenerate_drop_policy`](../../../../src/alphazeropp/instances/gripper_lite/policies.py)): `drop` is never legal from a state where the robot isn't carrying anything, so `interpret` returns `None` on the first step and the rollout stalls — leaf score a small negative ($\approx -0.05$: one noop penalty, zero progress, no solve). The deceptive part is that *doing nothing is good*: in the landscape sweep $\top\Rightarrow\mathrm{drop}$ sat near the $\approx$96th percentile of the uniformly sampled program space, and the near-do-nothing band is what uniform-prior MCTS keeps rediscovering ([../legacy/02.md](../legacy/02.md) Fig L2-right). *That* deceptive sparse-terminal-reward landscape — not under-expressiveness — is the negative finding of [../legacy/02.md](../legacy/02.md).
- **Missing goal guards.** A pick rule *without* $\neg\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)]$ will re-pick balls that are already in the target room, undoing progress and burning steps.
- **Disconnected goal variables and vacuous goal predicates — removed by construction.** A goal literal mentioning a variable bound *nowhere else* in the rule, or a *vacuous* predicate not in the goal vocabulary (e.g. `Goal[carrying(?b_0)]` when `goal_predicate_names = ("at_ball",)`), were common in the old permissive grammar — the landscape sweep found $87.7\%$ of 5 000 sampled policies using a vacuous goal predicate and $16.2\%$ with a disconnected goal variable ([../legacy/02.md](../legacy/02.md) Table 2; Fig L4), and the Stage-2.5 flags drove both to $0\%$. Under the occurrence-introduced grammar both are $0\%$ *by construction* in the default config: a goal literal never introduces a variable (so no disconnected-goal-variable — see [§8](#8--body-local-variables-introduced-by-occurrence)), and `goal_predicate_relevance` is on by default (so a `Goal[…]` atom only ever uses a whitelisted predicate). The detectors `rule_has_vacuous_goal_predicate` / `rule_has_disconnected_goal_var` ([lifted_grammar.py:455-486](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) survive for diagnostics on hand-written / legacy policies.
- **Overfitting to $B=1$.** A policy that solves one-ball Gripper-lite but loops on $B=2$ — the lex-min tie-break and a missing rule can make it shuttle the same ball back and forth. In the landscape sweep, under the combined safety flags the $B{=}1$ solver count rose $4\times$ and exactly one $B{=}1\!\to\!B{=}2$ generaliser appeared among 5 000 sampled policies ([../legacy/02.md](../legacy/02.md) Table 2; Figs L1, L3, L4). *Reachable but rare:* even after grammar restriction, a uniform prior almost never lands on a generalising solver — which is why Stage 3 swaps in a learned prior ([../legacy/02.md](../legacy/02.md) H5).
- **Stop-too-early derivations.** `STOP_POLICY` becomes legal as soon as one rule is sealed, so the search can commit to a short, useless policy (e.g. just the do-nothing rule). Under a uniform prior, shorter derivations are *more* likely (Figure D6), which is part of why the bad region is so reachable.
- **Lexicographic-binding artifacts.** Ties between bindings are broken by `tuple(sorted(theta.items()))`, so `ball_0` is always tried before `ball_1`. Here that is benign, but it is a load-bearing determinism assumption — a policy that "works" might be relying on it.
- **Expressivity limits.** No loops except by re-running the whole policy each environment step; no memory, no counters, no arithmetic; hard caps on the number of rules ($R$), literals per rule ($L_S$, $L_G$), and body-local variables per rule ($V_b$). Tasks needing more than this are simply outside the class.

![vacuous-goal-predicate and disconnected-goal-var fractions per safety-flag setting — from the legacy aux-var landscape sweep](../figures/02_landscape_pathology_fractions.png)

**Figure L4 (from [../legacy/02.md](../legacy/02.md)) — pathology fractions (legacy aux-var grammar).** Fraction of uniformly sampled policies with a vacuous goal predicate / a disconnected goal variable, per Stage-2.5 safety-flag setting; `r3 none` has $87.7\%$ vacuous + $16.2\%$ disconnected, and `both` has $0\%$ / $0\%$. Under the occurrence-introduced grammar the disconnected-goal-variable column is $0\%$ *structurally* (no flag needed), and `goal_predicate_relevance` — on by default — keeps the vacuous-predicate column at $0\%$. For the degenerate-`⊤⇒drop` percentile and the solver/generaliser counts see [../legacy/02.md](../legacy/02.md) Figs L2 / L1 / L3.

## §12 — Binding set and execution

Once a policy $\pi$ is finished, the **interpreter** ([`lifted_interpreter.py`](../../../../src/alphazeropp/synthesis/lifted_interpreter.py)) runs it on a world state $x$ and goal $G$ to pick a task action — unchanged from Stage 1. Here is what `find_bindings(rule, state_atoms, goal_atoms, objects_by_type, legal_actions)` ([lifted_interpreter.py:89-158](../../../../src/alphazeropp/synthesis/lifted_interpreter.py)) actually does, step by step:

1. Split the rule body into positive and negative literals. Start with the trivial binding $\{\}$.
2. **Positive pass.** For each positive literal, try to extend every binding so far by unifying the literal's atom with each matching ground row in $x$ (or in $G$, if the literal's source is `GOAL`). A conflict (a variable already bound to a different object, or a constant that doesn't match) drops that extension. If the candidate set ever empties, return `[]` — the rule does not apply.
3. **Free-variable enumeration.** Any *declared* rule variable still unbound after the positive pass (a variable used only in the action, say) is enumerated over all objects of its declared type, via a Cartesian product over `objects_by_type`. If some type has no objects, drop that candidate.
4. **Negative filter (closed-world).** Discard a candidate $\theta$ if any negative literal's grounded atom *is* present in the relevant relation — i.e. keep $\theta$ only when each $\neg\ell$ genuinely holds. (Safe negation guarantees its variables are bound by this point.)
5. **Legality filter.** Discard $\theta$ unless $\theta(\alpha_\rho)$ — the ground action obtained by substituting $\theta$ into the rule's action template — is in `legal_actions`.
6. **Deterministic sort.** Sort the survivors by `tuple(sorted(theta.items()))`; the first element is the lex-min binding.

In symbols: write a sealed rule as $\rho = (X_\rho,\ \mathrm{body}_\rho,\ \alpha_\rho)$ — its variables $X_\rho$ (action arguments and body-local variables), its body literals (split by `source` into a state side $\mathrm{body}^x_\rho$ and a goal side $\mathrm{body}^G_\rho$), and its lifted action template. A binding is a map $\theta:X_\rho\to\mathcal{O}$ (variables to concrete objects), $\theta(\alpha_\rho)$ is the ground action after substitution, and the **binding set** on world state $x$ with goal $G$ in task $\mathcal{M}_B$ is

$$\boxed{\;B_\rho(x,G) = \bigl\{\,\theta : X_\rho\to\mathcal{O}\ \bigm|\ x\models_\theta\mathrm{body}^x_\rho\ \wedge\ G\models_\theta\mathrm{body}^G_\rho\ \wedge\ \theta(\alpha_\rho)\in\mathcal{A}_B(x)\,\bigr\}.\;}$$

In words: every object assignment that makes $\rho$'s preconditions true in $x$, its goal conditions true in $G$, and its action legal right now. The interpreter takes the lex-min one,

$$\theta^\star_\rho(x,G) = \min\nolimits_{\preceq} B_\rho(x,G),\qquad \theta\preceq\theta'\iff\bigl(\theta(v)\bigr)_{v\in\mathrm{sort}(X_\rho)}\le_{\mathrm{lex}}\bigl(\theta'(v)\bigr)_{v\in\mathrm{sort}(X_\rho)},$$

and the first-applicable policy map fires the first rule with a non-empty binding set: $\mathrm{Interpret}(\pi,x,G) = \theta^\star_{\rho_{i^\star}}(\alpha_{\rho_{i^\star}})$ for $i^\star = \min\{i : B_{\rho_i}(x,G)\ne\varnothing\}$, else `None` (a no-op step) — `interpret`, [lifted_interpreter.py:161-189](../../../../src/alphazeropp/synthesis/lifted_interpreter.py).

**Worked example — the pick rule, $B=2$.** Take

$$\rho_3:\ \ \mathrm{at\_ball}(?b_0,?r_1)\,\wedge\,\mathrm{at\_robot}(?r_1)\,\wedge\,\mathrm{handempty}()\,\wedge\,\neg\mathrm{Goal}[\mathrm{at\_ball}(?b_0,?r_1)]\ \Longrightarrow\ \mathrm{pick}(?b_0,?r_1)$$

on the two-ball start state

$$x_0=\{\,\mathrm{at\_robot}(\textit{room\_a}),\ \mathrm{handempty}(),\ \mathrm{at\_ball}(\textit{ball}_0,\textit{room\_a}),\ \mathrm{at\_ball}(\textit{ball}_1,\textit{room\_a})\,\},\qquad G=\{\,\mathrm{at\_ball}(\textit{ball}_0,\textit{room\_b}),\ \mathrm{at\_ball}(\textit{ball}_1,\textit{room\_b})\,\}.$$

The positive pass over $\mathrm{at\_ball}(?b_0,?r_1)$ produces two candidates, $\theta_0 = \{?b_0\mapsto\textit{ball}_0,\ ?r_1\mapsto\textit{room\_a}\}$ and $\theta_1 = \{?b_0\mapsto\textit{ball}_1,\ ?r_1\mapsto\textit{room\_a}\}$; both survive $\mathrm{at\_robot}(?r_1)$ ($\textit{room\_a}$) and $\mathrm{handempty}()$. The negative literal $\neg\mathrm{Goal}[\mathrm{at\_ball}(?b_0,?r_1)]$ keeps **both** — neither ball is at $\textit{room\_b}$ yet, so $(\textit{ball}_i,\textit{room\_a})\notin G[\mathrm{at\_ball}]$. Both $\mathrm{pick}(\textit{ball}_i,\textit{room\_a})$ are legal. The lex-min sort puts $\theta_0$ first (`ball_0 < ball_1`), so the rule fires

$$\theta_0(\mathrm{pick}(?b_0,?r_1)) = \mathrm{pick}(\textit{ball}_0,\textit{room\_a}).$$

**A falls-through case.** Take the move-toward-goal rule $\rho_2 = \mathrm{at\_robot}(?r_0)\wedge\mathrm{carrying}(?v_0)\wedge\mathrm{Goal}[\mathrm{at\_ball}(?v_0,?r_1)]\Rightarrow\mathrm{move}(?r_0,?r_1)$ on the same $x_0$: the positive pass matches $\mathrm{at\_robot}(?r_0)$, then reaches $\mathrm{carrying}(?v_0)$ — but $x_0$ has no $\mathrm{carrying}(\cdot)$ atom at all, so the candidate set empties and $B_{\rho_2}(x_0,G)=\varnothing$. The interpreter moves on to the next rule. (`?v_0` would have been bound by `carrying`'s positive pass had it succeeded; it never reaches the free-variable phase.)

The implementation walks $\mathrm{body}_\rho$ literal-by-literal and never materialises the full Cartesian product over $\mathcal{O}$, so it transfers unchanged to held-out larger instances. The leaf evaluator's `num_rule_evals` ([lifted_leaf_evaluator.py](../../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py), `_rollout_one`) counts a rule-evaluation proxy (`rule_idx + 1` on a firing step, `len(rules)` on a stall) — it is *not* an instrumentation of `find_bindings` itself.

**Same word, two levels.** $B_\rho$ ranges over *task* variables and objects (level $x$); the legal-production map $\mathcal{P}_\Gamma(z)$ of [§14](#14--the-derivation-game-as-an-mdp-formal) ranges over *productions* (level $p$). Don't conflate them.

## §13 — The grammar as a typed attributed CFG

A plain context-free grammar has nonterminals, terminals, a start symbol, and one-nonterminal-at-a-time productions $A\to\alpha$. **The CFG *skeleton* tells you the order of the choices. The *attributes* tell you which choices are legal at each step.** Without the attributes the skeleton over-generates wildly — it would happily emit ill-typed atoms, duplicate literals, reordered bodies, unsafe negations, and goal literals over unjustified variables. The skeleton, with the production labels of [§4](#4--the-hole-state-machine) as terminals:

$$
\begin{aligned}
\langle\textit{Policy}\rangle &\to \langle\textit{Rule}\rangle\ \langle\textit{Policy}\rangle \;\mid\; \texttt{STOP\_POLICY}\\
\langle\textit{Rule}\rangle &\to \texttt{ADD\_RULE}\ \langle\textit{ActionSchema}\rangle\ \langle\textit{PreList}\rangle\ \texttt{STOP\_PRE}\ \langle\textit{GoalList}\rangle\ \texttt{FINISH\_RULE}\\
\langle\textit{ActionSchema}\rangle &\to \texttt{schema=move} \;\mid\; \texttt{schema=pick} \;\mid\; \texttt{schema=drop}\\
\langle\textit{PreList}\rangle &\to \varepsilon \;\mid\; \texttt{pre:}\ell\ \langle\textit{PreList}\rangle\\
\langle\textit{GoalList}\rangle &\to \varepsilon \;\mid\; \texttt{goal:}\ell\ \langle\textit{GoalList}\rangle
\end{aligned}
$$

A literal $\ell$ is itself built argument-by-argument: $\ell = P(a_1,\dots,a_k)$ with each $a_i$ either an *existing in-scope variable of type $\tau_i$* or — for a `pre:ℓ` literal only, while the rule has room — a *fresh typed variable* $\texttt{NEW\_VAR}(\tau_i)$:

$$\textit{Arg}_\tau \;\to\; v\ (v\in\textit{scope},\ \mathrm{type}(v)=\tau) \;\mid\; \texttt{NEW\_VAR}(\tau)\quad(\text{the }\texttt{NEW\_VAR}\text{ alternative is absent inside a }\langle\textit{GoalList}\rangle).$$

The legal literals $\ell$ at a hole, and which `NEW_VAR` alternatives are open, depend on **attributes already chosen**:

- the **action schema** picked for this rule;
- the **variable context** — which action arguments and body-local variables `?v_i` are in scope, and how many fresh slots remain under $V_b$; an attribute rule extends the context with any `NEW_VAR`s a `pre:ℓ` literal uses;
- the **last literal** accepted in this list, under the canonical lex key `_lit_key = (predicate, args, negated)` (so the next must be strictly greater — no duplicates, no reorderings);
- the **remaining caps** $L_S,L_G$ (and $R$ at the policy hole);
- the **positive-variable set** — for safe negation, a negated `Goal[ℓ]` is offered only when every variable of $\ell$ is already positively bound; for the goal-predicate whitelist, $\ell$'s predicate must be in `goal_predicate_names`.

So the honest object is a **typed attributed context-free grammar** — equivalently a *one-hole typed derivation grammar*:

$$\boxed{\;\Gamma = (N,\ \Sigma_{\mathrm{prod}},\ P,\ S;\ \mathrm{Attr}),\qquad \mathrm{Attr} = (\text{chosen schema},\ \text{scope vars \& remaining }V_b,\ \text{last literal},\ \text{remaining caps},\ \text{positive-var set}),\;}$$

with $N$ = the hole kinds $\cup\ \langle\textit{Rule}\rangle$, $\Sigma_{\mathrm{prod}}$ = the production labels, $S=\langle\textit{Policy}\rangle$. The honest indexed nonterminals look like $\langle\textit{PreList}(j,\ X,\ b,\ \ell_{\mathrm{last}})\rangle$ ($j$ = literals so far, $X$ = scope, $b$ = body-local vars used so far, $\ell_{\mathrm{last}}$ = last literal) — and $\mathrm{Attr}$ is precisely what `enumerate_productions` reads off the state $z$ together with $(\mathrm{cfg},\Sigma)$.

The payoff: **a derivation tree of $\Gamma$ and an MCTS play are the same object viewed two ways** — the tree view shows *syntax*, the play view shows *decisions over time*. The terminal leaves of the tree, read left to right, are the production sequence $p_1\dots p_T$ of [§3](#3--what-the-arrow-labels-mean); the same tree, read through `to_program`, is a `Policy`.

![one full derivation of a 3-rule policy drawn as a grammar parse tree; no ⟨Aux⟩ node](../figures/02_derivation_tree.png)

**Figure D7 — a derivation as a parse tree.** One full derivation of a 3-rule policy — the Stage-1 hand-policy prefix $\rho_1$ drop-at-goal / $\rho_2$ move-toward-goal (with body-local `?v_0`) / $\rho_3$ pick — drawn as a grammar parse tree: $\langle\textit{Policy}\rangle$ at the root, expanding through $\langle\textit{Rule}\rangle\to\texttt{ADD\_RULE}\ \langle\textit{ActionSchema}\rangle\ \langle\textit{PreList}\rangle\ \texttt{STOP\_PRE}\ \langle\textit{GoalList}\rangle\ \texttt{FINISH\_RULE}$ (no $\langle\textit{Aux}\rangle$) down to the terminal production labels. The leaves, top to bottom, are the production sequence $p_1\dots p_T$ — the same play as Figure D3, but as a tree; colour = hole kind; the right column shows the sealed rule $\rho_i$ each $\langle\textit{Rule}\rangle$ subtree yields.

## §14 — The derivation game as an MDP (formal)

> Dense layer — optional for a first pass. Everything below is the precise restatement of [§§1–13](#1--one-page-mental-model). The package's [`Game`](../../../../src/alphazeropp/core/game.py) protocol is single-player and MCTS-driven (`reset` / `step` / `get_action_mask` / `hashable_obs` / `stash_state` / `clone`); [`LiftedDerivationGame`](../../../../src/alphazeropp/synthesis/lifted_derivation.py) instantiates it over the grammar so the unchanged [`MCTS`](../../../../src/alphazeropp/core/mcts.py) engine and `UniformPolicyValueNet` drive it.

Fix a grammar config $\mathrm{cfg}$ ([`LiftedGrammarConfig`](../../../../src/alphazeropp/synthesis/lifted_grammar.py): `max_rules` $R$, `max_pre_literals` $L_S$, `max_goal_literals` $L_G$, `max_body_local_vars` $V_b$, plus the two on-by-default safety toggles `goal_predicate_relevance`, `require_goal_var_connected`; defaults $R{=}4,L_S{=}3,L_G{=}1,V_b{=}2$) and a domain signature $\Sigma$ ([`DomainSignature`](../../../../src/alphazeropp/synthesis/lifted_grammar.py): `types`, `predicates`, `action_schemas`, optional `goal_predicate_names`; `gripper_lite_signature()` sets `types=("ball","room")`, `goal_predicate_names=("at_ball",)`). The derivation game is

$$\boxed{\;\mathcal{D}_\Gamma \;=\; \bigl(\,\mathcal{Z},\ \mathcal{P}_\Gamma,\ \delta_\Gamma,\ R_\Gamma,\ z_0\,\bigr).\;}$$

**State space.** $\mathcal{Z} = \{\,z=(C,q,h)\,\}$ with hole kind $h(z)\in\{\texttt{policy},\texttt{action\_schema},\texttt{pre\_lit},\texttt{goal\_lit},\bot\}$ and the invariant $q=\varnothing \iff h\in\{\texttt{policy},\bot\}$. Initial state and terminal set:

$$z_0 = \bigl((),\ \varnothing,\ \texttt{policy}\bigr) = \texttt{LiftedDerivationState.initial()},\qquad \mathcal{T}_\Gamma = \{\,z : h(z)=\bot\,\} = \{\,(C,\varnothing,\bot)\,\}.$$

**Actions / legal set.** $\mathcal{P}_\Gamma(z) = \texttt{enumerate\_productions}(z,\ \mathrm{cfg},\ \Sigma)$, with $\mathcal{P}_\Gamma(z)=\varnothing \iff h(z)=\bot$ (every non-terminal state offers at least one production — the `STOP_*` / `FINISH_RULE` cursor production is always there). The per-hole catalogue is the table in [§4](#4--the-hole-state-machine).

**Transition.** $\delta_\Gamma(z,p) = z.\texttt{apply}(p)$ — the partial map "apply the chosen production to the open hole", defined for $p\in\mathcal{P}_\Gamma(z)$, pure (`dataclasses.replace`, never mutates $z$).

**Reward.** Sparse — zero everywhere except the leaf, where it equals the training-objective score of the synthesised policy (see $U_B$, $J_{\mathrm{train}}$ in [../legacy/02.md §Minimal setup](../legacy/02.md#minimal-setup)):

$$R_\Gamma(z,p) \;=\; \begin{cases} J_{\mathrm{train}}\bigl(\texttt{to\_program}(\delta_\Gamma(z,p))\bigr), & \delta_\Gamma(z,p)\in\mathcal{T}_\Gamma,\\[2pt] 0, & \text{otherwise,} \end{cases}\qquad \texttt{to\_program}(z) = \texttt{Policy}(z.\texttt{completed\_rules}).$$

Concretely the leaf score (per [`LiftedLeafEvaluator`](../../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py)) is $\texttt{solve\_rate} + 0.25\cdot\texttt{avg\_progress} - 0.01\cdot\texttt{avg\_steps} - 0.05\cdot\texttt{num\_noops}$ over the train instances (`num_noops` = number of steps where `interpret` returned `None`).

**MCTS over $\mathcal{D}_\Gamma$.** PUCT selection on the search tree of $\mathcal{D}_\Gamma$:

$$\mathrm{UCB}(z,p) = \hat{Q}_{\mathrm{norm}}(z,p) + c\,\pi_0(p\mid z)\,\frac{\sqrt{N(z)}}{1+N(z,p)},\qquad c=1.5,\quad \pi_0\ \text{uniform}$$

([core/mcts.py](../../../../src/alphazeropp/core/mcts.py)). The action head is sized to the worst-case branching $M(\mathrm{cfg},\Sigma) = \max_{z}\lvert\mathcal{P}_\Gamma(z)\rvert = \texttt{compute\_max\_productions}(\mathrm{cfg},\Sigma)$ — for the default Gripper-lite signature, $M = 26$ (the `pre_lit` hole, taken with the maximal possible variable scope so the fresh-variable alternative is open in every slot, dominates; with `legacy_grammar_config()` — both safety flags off — it is $33$). The encoded observation length is $L(\mathrm{cfg}) = N_{\text{fields}}\bigl(R\,(1+L_S+L_G)+1\bigr)$ with $N_{\text{fields}}=7$ (one node per finished literal + one per rule header + one trailing hole-marker node; `nodes_per_rule = 1 + max_pre_literals + max_goal_literals` — [lifted_encoding.py:37-42](../../../../src/alphazeropp/synthesis/lifted_encoding.py)); that is $112$ at $R{=}3$ and $147$ at $R{=}4$ — uniform MCTS ignores it; it becomes load-bearing only when Stage 3 swaps in a learned net. Under uniform-prior MCTS, *all* search structure therefore comes from the backed-up $R_\Gamma$ values — which is why the deceptive landscape of [§11](#11--failure-modes-of-this-policy-language) (the negative finding of [../legacy/02.md](../legacy/02.md)) defeats it.

## §15 — Code-fidelity audit

> Optional for first-pass readers. Every formal symbol above ties to a live site (post-Stage-2-cutover source).

| Claim / symbol | Formal statement | Impl site | Status |
|---|---|---|---|
| Derivation state | $z = (C,q,h)$ — completed rules, partial rule, current hole; hole set $\{\texttt{policy},\texttt{action\_schema},\texttt{pre\_lit},\texttt{goal\_lit},\texttt{None}\}$ — **no `aux_var`** | `LiftedDerivationState` ([lifted_derivation.py:75-83](../../../../src/alphazeropp/synthesis/lifted_derivation.py)); `partial` is non-`None` iff $h\in\{\texttt{action\_schema},\texttt{pre\_lit},\texttt{goal\_lit}\}$ | ✓ exact |
| Partial rule | $q = (\texttt{schema},\texttt{action\_args},\texttt{body\_local\_vars},\texttt{state\_lits},\texttt{goal\_lits})$; `body_local_vars` grown lazily in the `pre_lit` "add" branch | `PartialRule` ([lifted_derivation.py:46-72](../../../../src/alphazeropp/synthesis/lifted_derivation.py)); `all_vars()` = action args ∪ body-local vars (deduped); `positive_var_names()` = action-arg names ∪ vars of `state_lits` | ✓ exact |
| Initial / terminal | $z_0 = ((),\varnothing,\texttt{policy})$; $\mathcal{T}_\Gamma=\{h=\bot\}$ | `LiftedDerivationState.initial()` ([:85-87](../../../../src/alphazeropp/synthesis/lifted_derivation.py)); `is_terminal()` ⇔ `current_hole is None` ([:141-142](../../../../src/alphazeropp/synthesis/lifted_derivation.py)) | ✓ exact |
| Production | $p = (\texttt{hole\_kind},\texttt{label},\texttt{payload})$; `pre_lit` "add" payload is `("add", lit, new_body_local_vars)`; `goal_lit` "add" payload is `("add", lit, ())` | `LiftedProduction` ([lifted_grammar.py:173-182](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) | ✓ exact |
| Legal set | $\mathcal{P}_\Gamma(z) = \texttt{enumerate\_productions}(z,\mathrm{cfg},\Sigma)$; the `pre_lit` enumerator's fresh-var path is `state_literal_candidates`; the `goal_lit` enumerator's "all args already in scope" is `goal_literal_candidates` | [lifted_grammar.py:318-371](../../../../src/alphazeropp/synthesis/lifted_grammar.py); `state_literal_candidates` ([:221-270](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); `goal_literal_candidates` ([:273-313](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); `body_local_var_name` ([:195-200](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) | ✓ exact |
| Transition | $\delta_\Gamma(z,p) = z.\texttt{apply}(p)$, pure (`dataclasses.replace`, never mutates `self`); no `aux_var` branch | [lifted_derivation.py:93-138](../../../../src/alphazeropp/synthesis/lifted_derivation.py) | ✓ exact |
| Sparse reward | nonzero only at $h=\bot$, $= J_{\mathrm{train}}(\texttt{to\_program}(z))$ | `LiftedDerivationGame.step` ([lifted_derivation.py:210-233](../../../../src/alphazeropp/synthesis/lifted_derivation.py)); `to_program() = Policy(completed_rules)` ([:144-145](../../../../src/alphazeropp/synthesis/lifted_derivation.py)) | ✓ exact |
| Leaf objective $J_{\mathrm{train}}$ / $U_B$ | $\texttt{solve\_rate}+0.25\,\texttt{avg\_progress}-0.01\,\texttt{avg\_steps}-0.05\,\texttt{num\_noops}$, over the train instances | [`LiftedLeafEvaluator`](../../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py); $U_B$ / $J_{\mathrm{train}}$ pinned in [../legacy/02.md §Minimal setup](../legacy/02.md#minimal-setup) | ✓ exact (↗ ../legacy/02.md for the symbol names) |
| Action-head width | $M(\mathrm{cfg},\Sigma) = \max_z\lvert\mathcal{P}_\Gamma(z)\rvert$; default cfg $=26$, `legacy_grammar_config()` $=33$; the `pre_lit` hole at the maximal-scope construction dominates | `compute_max_productions` ([lifted_grammar.py:378-413](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) | ✓ exact |
| Observation length | $L(\mathrm{cfg}) = 7\bigl(R(1+L_S+L_G)+1\bigr)$; `nodes_per_rule = 1 + max_pre_literals + max_goal_literals`; no aux-var node | `encode_max_len` ([lifted_encoding.py:37-42](../../../../src/alphazeropp/synthesis/lifted_encoding.py)); hole-id table `_HOLE_ID = {None:0,policy:1,action_schema:2,pre_lit:3,goal_lit:4}` ([:34](../../../../src/alphazeropp/synthesis/lifted_encoding.py)) | ✓ exact ($112$ at $R{=}3$, $147$ at $R{=}4$) |
| Node key | `pretty()` carries `vars=[?r_0:room, …, ?v_0:ball]` (type-disambiguation — a body-local `?v_0` alone is type-ambiguous) | [lifted_derivation.py:147-162](../../../../src/alphazeropp/synthesis/lifted_derivation.py); `LiftedDerivationGame.hashable_obs` | ✓ exact |
| α-canonical form | action-position names `?{prefix}_{i}`, body-local extras `?v_{j}` in first-occurrence order; `canonical_rule_form` collapses renaming + body reorder | [lifted_grammar.py:493-524](../../../../src/alphazeropp/synthesis/lifted_grammar.py) | ✓ exact |
| Safe negation | negated `Goal[…]` offered only when every var is positively bound or an action arg; re-checked at `Rule.__post_init__` | `goal_literal_candidates` ([lifted_grammar.py:304-306](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); [lifted_dsl.py:164-180](../../../../src/alphazeropp/synthesis/lifted_dsl.py) | ✓ exact |
| `require_goal_var_connected` | vacuous-by-construction — goal literals introduce no variables, so every goal-literal variable is connected; the flag is a no-op; `goal_vars_locally_connected` kept for diagnostics on hand-written / legacy policies | `LiftedGrammarConfig` ([lifted_grammar.py:86-92](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); `goal_vars_locally_connected` ([:436-452](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) | ✓ exact (no-op by design) |
| Pathology detectors (§11) | vacuous goal predicate (not in `goal_predicate_names`); disconnected goal var (sole binding site is a goal lit) — diagnostics on arbitrary policies; both $0\%$ for grammar output in the default config | `rule_has_vacuous_goal_predicate` ([lifted_grammar.py:455-464](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); `rule_has_disconnected_goal_var` ([:467-486](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) | ✓ exact (87.7% / 16.2% legacy fractions ↗ ../legacy/02.md Fig L4) |
| Capacity tuple defaults (§9) | $\mathrm{cfg}=(R,L_S,L_G,V_b)=(4,3,1,2)$; `allow_disjunction=allow_state_negation=allow_constants=False`; `goal_predicate_relevance=require_goal_var_connected=True`; `strict_grammar_config()` = these defaults, `legacy_grammar_config()` = both safety flags off | `LiftedGrammarConfig` ([lifted_grammar.py:60-92](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); `strict_grammar_config` ([:107-113](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); `legacy_grammar_config` ([:95-104](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) | ✓ exact |
| MCTS unchanged | PUCT, $c=1.5$, $\pi_0$ uniform; `lifted_*` modules import nothing grounded | [core/mcts.py](../../../../src/alphazeropp/core/mcts.py); `LiftedDerivationGame` ([lifted_derivation.py:169-287](../../../../src/alphazeropp/synthesis/lifted_derivation.py)) | ✓ exact |
| Binding set $B_\rho(x,G)$ / $\theta^\star_\rho$ | satisfying type-correct bindings + legality; lex-min by var-name-sorted object names | `find_bindings`, lex-min sort, `interpret` ([lifted_interpreter.py:89-158, 161-189](../../../../src/alphazeropp/synthesis/lifted_interpreter.py)) | ✓ exact (unchanged from Stage 1) |
| `num_rule_evals` proxy | `rule_idx + 1` on a firing step, `len(rules)` on a stall; not an instrumentation of `find_bindings` | [lifted_leaf_evaluator.py](../../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py) (`_rollout_one`); see [../legacy/02.md Appendix A](../legacy/02.md#appendix-a--code-fidelity-audit) | ✓ exact |

## §16 — Figure index, notation bridge, and cross-links

**Figure index.** Eight D-figures live inline next to the section they illustrate, and are regenerated from the real grammar / derivation modules (no hand-transcription) by `python scripts/plotting/plot_derivation_game_diagrams.py`:

- **D1 — hole-state machine** ([../figures/02_derivation_state_machine.png](../figures/02_derivation_state_machine.png)) — the four holes as an FSM, nodes labelled with production counts; `pre_lit` is the widest, $M=26$. *([§4](#4--the-hole-state-machine).)*
- **D2 — production sets per hole** ([../figures/02_derivation_production_sets.png](../figures/02_derivation_production_sets.png)) — $\mathcal{P}_\Gamma(z)$ at one representative state per hole kind; the `pre_lit` hole-kind is the widest, $M=\texttt{compute\_max\_productions}=26$. *([§4](#4--the-hole-state-machine).)*
- **D3 — worked derivation, linear** ([../figures/02_derivation_example.png](../figures/02_derivation_example.png)) — the 8-step build of $\rho_2$ (move-toward-goal) as a labelled path through $\delta_\Gamma$; `?v_0:ball` appears at `pre:carrying(?v_0)`. *([§5](#5--a-worked-derivation-field-by-field).)*
- **D4 — state anatomy** ([../figures/02_derivation_state_anatomy.png](../figures/02_derivation_state_anatomy.png)) — six `LiftedDerivationState`s with every field rendered, including `body_local_vars`. *([§6](#6--the-fields-of-a-partial-rule).)*
- **D5 — state evolution (diff)** ([../figures/02_derivation_state_evolution.png](../figures/02_derivation_state_evolution.png)) — the $\rho_2$ build as a field-by-field diff table; the `pre:carrying(?v_0)` row touches two fields. *([§5](#5--a-worked-derivation-field-by-field).)*
- **D6 — derivation length** ([../figures/02_derivation_production_lengths.png](../figures/02_derivation_production_lengths.png)) — per-rule structure (4 fixed + ≤$L_S$ pre-lits + ≤$L_G$ goal-lits), analytic range $[5, R(L_S+L_G+4)+1]$, empirical length distribution. *([§5](#5--a-worked-derivation-field-by-field).)*
- **D7 — derivation as a parse tree** ([../figures/02_derivation_tree.png](../figures/02_derivation_tree.png)) — one 3-rule policy drawn as a grammar parse tree (no $\langle\textit{Aux}\rangle$); leaves = the production sequence. *([§13](#13--the-grammar-as-a-typed-attributed-cfg).)*
- **D8 — the hypothesis class** ([../figures/02_derivation_hypothesis_class.png](../figures/02_derivation_hypothesis_class.png)) — the capacity tuple $(R,L_S,L_G,V_b)$ with its defaults + disabled switches, the program-space size ($M=26$; the old aux-var `r1` config enumerable at $\approx1\,166$ policies; $R\ge 3$ for any solver), and one real `Rule` per rule family. *([§9](#9--what-rules-can-this-language-create).)*

Three figures *owned by* [../legacy/02.md](../legacy/02.md) (the Stage-2/2.5 aux-var-grammar landscape work) are reproduced inline here for the cheat / failure-mode discussion — their canonical captions and reads live there: **Fig L1** `02_landscape_solver_density.png` ([§10](#10--is-this-a-direct-cheat)) and **Fig L4** `02_landscape_pathology_fractions.png` ([§11](#11--failure-modes-of-this-policy-language)).

**Notation bridge.** [../legacy/02.md Appendix B](../legacy/02.md#appendix-b--the-derivation-game-formally) writes the derivation state $s = (\rho_{1:k},\ \texttt{partial},\ \texttt{hole}(s))$; this note writes it $z = (C,\ q,\ h)$ — same object, with $z$ chosen to avoid the visual collision with the task/world state (which [../01_draft_lifted_policy_az.md](../01_draft_lifted_policy_az.md) writes $s$/$S$ and this note writes $x$). The maps line up: $\mathcal{P}(s)\equiv\mathcal{P}_\Gamma(z)$, $\delta\equiv\delta_\Gamma$. The old `aux_vars` field is now `body_local_vars` (and the old `?aux_j` are `?v_j`) — these are **body-local / existential variables**, born inside the precondition literal that uses them, not a separate phase. The implementation field is `state_lits`; throughout this note read it as "precondition literals" — it is *not* renamed in code, and it has nothing to do with derivation states. The lifted action template of rule $\rho$ is written $\alpha_\rho$ here (the `rule.action: LiftedAction`).

**Cross-links.**

- [../01_draft_lifted_policy_az.md](../01_draft_lifted_policy_az.md) — orientation (rev. 1): [§4](../01_draft_lifted_policy_az.md#4-task-domains) (the task MDP, Gripper-lite), [§5](../01_draft_lifted_policy_az.md#5-lifted-decision-list-policies) (lifted decision-list policies), [§6](../01_draft_lifted_policy_az.md#6-the-gripper-lite-hand-policy-and-its-plan-length) (the hand policy $\rho_1..\rho_4$), [§7](../01_draft_lifted_policy_az.md#7-the-online-unification-interpreter) (the interpreter, $B_\rho$, $\theta^\star_\rho$), **[§8](../01_draft_lifted_policy_az.md#8-the-redesigned-derivation-grammar-proposed) (the redesigned derivation grammar — why the `Aux` phase is removed; occurrence-introduced variables)**, [§9](../01_draft_lifted_policy_az.md#9-synthesis-as-a-derivation-game-the-leaf-evaluator) (synthesis as a derivation game).
- [../02_plan.md](../02_plan.md) — Stage 2: the grammar cutover plan (the occurrence-introduced grammar; module/class changes; the minimal uniform-MCTS run). The Stage-2 *results* companion `../02.md` is pending.
- [../legacy/02.md](../legacy/02.md) — the Stage-2/2.5 *aux-var-grammar* landscape results: [Appendix B](../legacy/02.md#appendix-b--the-derivation-game-formally) (the compact version of this note, pre-cutover), [Appendix A](../legacy/02.md#appendix-a--code-fidelity-audit) (code-fidelity audit), [§Minimal setup](../legacy/02.md#minimal-setup) ($U_B$, $J_{\mathrm{train}}$, the task family $\mathcal{M}_B$, the grammar config), [§Hypotheses](../legacy/02.md#hypotheses-and-next-experiments) (H1–H5; H4, the connectedness rule).
- [../legacy/01.md](../legacy/01.md) — Stage 1: the lifted DSL, the Stage-1 interpreter, the Gripper-lite env, the hand policy. [../legacy/00_draft_lifted_policy_az.md](../legacy/00_draft_lifted_policy_az.md) — the original orientation draft.
- [pddl_background.md](pddl_background.md) — PDDL and its mapping to this repo's lifted DSL; [../../appendix/sygus_background.md](../../appendix/sygus_background.md) — syntax-guided synthesis grammars in general.
- Source: [lifted_grammar.py](../../../../src/alphazeropp/synthesis/lifted_grammar.py), [lifted_derivation.py](../../../../src/alphazeropp/synthesis/lifted_derivation.py), [lifted_dsl.py](../../../../src/alphazeropp/synthesis/lifted_dsl.py), [lifted_interpreter.py](../../../../src/alphazeropp/synthesis/lifted_interpreter.py), [lifted_encoding.py](../../../../src/alphazeropp/synthesis/lifted_encoding.py), [lifted_leaf_evaluator.py](../../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py); [gripper_lite/env.py](../../../../src/alphazeropp/instances/gripper_lite/env.py), [gripper_lite/policies.py](../../../../src/alphazeropp/instances/gripper_lite/policies.py); [core/mcts.py](../../../../src/alphazeropp/core/mcts.py), [core/game.py](../../../../src/alphazeropp/core/game.py). Tests: [tests/test_lifted_grammar_occurrence_vars.py](../../../../tests/test_lifted_grammar_occurrence_vars.py).
