# Appendix (Stage 4) — The derivation game

> Background reference for Stage 4. Cited from [../01.md](../01.md), [../02.md](../02.md) (this note is the expanded form of [Appendix B](../02.md#appendix-b--the-derivation-game-formally) there), [../00_draft_lifted_policy_az.md](../00_draft_lifted_policy_az.md) §6, and the synthesis modules under [src/alphazeropp/synthesis/](../../../../src/alphazeropp/synthesis/). Companion to [pddl_background.md](pddl_background.md). Style follows [../../appendix/sygus_background.md](../../appendix/sygus_background.md).
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
- [§8 — Auxiliary variables as witnesses](#8--auxiliary-variables-as-witnesses)
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
| $q$ | the **partial rule** currently under construction inside $z$ | `z.partial` — a [`PartialRule`](../../../../src/alphazeropp/synthesis/lifted_derivation.py) (`schema`, `action_args`, `aux_vars`, `state_lits`, `goal_lits`); `None` ⇔ $q=\varnothing$ | schema `drop`, two preconditions added so far |
| $h$ | the **current hole** — which one slot of $z$ is open | `z.current_hole` ∈ {`policy`, `action_schema`, `aux_var`, `pre_lit`, `goal_lit`, `None`} | `pre_lit` — "next decision is which precondition to add" |
| $p$ | a **grammar production** — written on an arrow; an *action* of the synthesis MDP | [`LiftedProduction`](../../../../src/alphazeropp/synthesis/lifted_grammar.py) (`hole_kind`, `label`, `payload`) | `schema=drop`; `pre:carrying(?b_0)`; `FINISH_RULE` |
| $\rho$ | a **sealed rule** — one finished IF–THEN rule, appended to the policy | [`Rule`](../../../../src/alphazeropp/synthesis/lifted_dsl.py) (`vars`, `body`, `action`) | $\mathrm{carrying}(?b_0)\wedge\dots\Rightarrow\mathrm{drop}(?b_0,?r_1)$ |
| $\pi$ | a **complete policy** — an ordered list of sealed rules | [`Policy`](../../../../src/alphazeropp/synthesis/lifted_dsl.py) (`rules`) | the 4-rule Stage-1 hand policy |
| $\alpha_\rho$ | the **lifted action template** of rule $\rho$ — what the rule outputs, with variables not yet bound | `rule.action` — a [`LiftedAction`](../../../../src/alphazeropp/synthesis/lifted_dsl.py) (`schema`, `args`) | $\mathrm{drop}(?b_0,?r_1)$ |
| $x$ | a **task / world state** — a set of ground atoms describing the current Gripper-lite world | the relational state the env exposes ([gripper_lite/env.py](../../../../src/alphazeropp/instances/gripper_lite/env.py)); type `RelState = dict[str, set[tuple]]` | $\{\mathrm{at\_robot}(\textit{room\_a}),\,\mathrm{handempty}(),\,\mathrm{at\_ball}(\textit{ball}_0,\textit{room\_a})\}$ |
| $G$ | a **goal** — the set of ground atoms the task wants made true | another `RelState` ([gripper_lite/env.py](../../../../src/alphazeropp/instances/gripper_lite/env.py)) | $\{\mathrm{at\_ball}(\textit{ball}_0,\textit{room\_b})\}$ |
| $\theta$ | a **runtime binding** — an assignment of rule variables to concrete objects, chosen by the interpreter | `Binding = dict[str, str]` in [lifted_interpreter.py](../../../../src/alphazeropp/synthesis/lifted_interpreter.py) | $\{?b_0\mapsto\textit{ball}_0,\ ?r_1\mapsto\textit{room\_b}\}$ |
| $B_\rho(x,G)$ | the **binding set** of rule $\rho$ on $(x,G)$ — every binding under which $\rho$ is applicable right now | output of `find_bindings` in [lifted_interpreter.py](../../../../src/alphazeropp/synthesis/lifted_interpreter.py) | $\{\{?b_0\mapsto\textit{ball}_0,\,?r_1\mapsto\textit{room\_b}\}\}$ |

The derivation state is written

$$\boxed{\;z = (C,\ q,\ h),\qquad C=\text{completed rules so far},\quad q=\text{partial rule (or }\varnothing),\quad h=\text{current hole}.\;}$$

**The trap.** $x$ is *not* part of $z$. The grammar (levels $z,q,p$) builds *syntax* and never touches a world state. A world state $x$ enters only at the very end: when a derivation finishes, the completed policy $\pi$ is rolled out through the Stage-1 interpreter on Gripper-lite instances and scored by the leaf utility $U_B$ ([../02.md §Minimal setup](../02.md#minimal-setup)). So "`state_lits`" in a `PartialRule` does *not* mean "states of the derivation game" — it means *precondition literals*, which will later be matched against a world state $x$ ([§2](#2--two-kinds-of-state), [§6](#6--the-fields-of-a-partial-rule)).

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

**The thing to remember.** `state_lits` (a field of a partial rule) has nothing to do with derivation states. It is a list of *precondition literals* — conditions like $\mathrm{carrying}(?b_0)$ — that will eventually be checked against a world state $x$ at runtime. Read `state_lits` as "precondition literals" everywhere in this note; the field is just not renamed in the code.

## §3 — What the arrow labels mean

A derivation is a path

$$\boxed{\;z_0\ \xrightarrow{\,p_1\,}\ z_1\ \xrightarrow{\,p_2\,}\ z_2\ \xrightarrow{\,p_3\,}\ \cdots\ \xrightarrow{\,p_T\,}\ z_T.\;}$$

The word on each arrow, $p_i$, is a **production** — not a state, not a rule, not a literal. Reading it: $z_i \xrightarrow{p_i} z_{i+1}$ means *"apply grammar production $p_i$ to the one open hole of $z_i$, getting $z_{i+1}$."* The step is deterministic and pure:

$$\boxed{\;z_{i+1} = z_i.\texttt{apply}(p_{i+1})\quad\text{(never mutates }z_i\text{).}\;}$$

In words: each production fills exactly the open hole and produces a fresh state; nothing is overwritten. Here is the whole vocabulary of arrow labels:

| arrow label | what it does |
|---|---|
| `ADD_RULE` | open a fresh, empty partial rule $q$ (move from the `policy` hole into a rule) |
| `schema=move` / `schema=pick` / `schema=drop` | choose the rule's **action template** — fixes which action the rule will output and introduces its argument variables (`drop` ⇒ `?b_0:ball`, `?r_1:room`) |
| `SKIP_AUX` | do *not* add an extra variable |
| `add_aux:ball` / `add_aux:room` | introduce one **witness variable** `?aux_0` of that type (used only in conditions, never in the action) |
| `pre:ℓ` | add the precondition (state) literal $\ell$ to the rule's body |
| `STOP_PRE` | stop adding precondition literals; move on to goal literals |
| `goal:ℓ` | add the goal literal $\ell$ (a `Goal[...]` atom, possibly negated) to the body |
| `FINISH_RULE` | **seal** the partial rule $q$ into a completed rule $\rho$, append it to $C$, return to the `policy` hole |
| `STOP_POLICY` | stop writing rules; the policy is now complete (terminal) |

Four things that *look* alike but live on different levels, all visible along one derivation:

1. **a production-label string** $p_i$ — e.g. the text `pre:carrying(?b_0)` written on an arrow;
2. **a `Literal`** — e.g. the syntactic atom $\mathrm{carrying}(?b_0)$ sitting inside a rule's body;
3. **a sealed rule** $\rho \in C$ — a whole IF–THEN sentence produced by a `FINISH_RULE` arrow;
4. **a task action** — e.g. `drop` / `pick` / `move`, an action of the *Gripper-lite* MDP, chosen by the interpreter at runtime, **not** an action of the derivation game.

Finally, two kinds of arrow behave differently. A **content** production (`schema=…`, `add_aux:T`, `pre:ℓ`, `goal:ℓ`) adds something to $q$. A **cursor** production (`SKIP_AUX`, `STOP_PRE`, `FINISH_RULE`, `STOP_POLICY`) adds nothing — it just moves the hole $h$ along (and `FINISH_RULE` additionally seals $q$ into $C$).

## §4 — The hole-state machine

At every non-terminal moment there is **exactly one open hole** $h$, and one production is applied per step. The hole runs around a small finite-state machine — this is a *left-to-right policy writer*, not a free-form AST editor with many open frontier nodes:

```
                 ADD_RULE             schema=X            SKIP_AUX | add_aux:T          STOP_PRE
   policy  ───────────────▶ action_schema ─────────▶ aux_var ──────────────────▶ pre_lit ──────────▶ goal_lit
     │  ▲                                                                          │  ⟲ pre:ℓ          │  ⟲ goal:ℓ
     │  │                                                                          (≤ L_S literals)    (≤ L_G literals)
     │  │ FINISH_RULE  (seal q into C, return to the policy hole)                                       │
     │  └───────────────────────────────────────────────────────────────────────────────────────────────┘
     │
     │ STOP_POLICY   (#completed rules ≥ 1)
     ▼
     ⊥   (terminal: q = ∅, h = ⊥)
```

The five holes, in plain English:

- **`policy`** — between rules: "start another rule, or stop?" Here $q=\varnothing$.
- **`action_schema`** — just opened a rule: "which action will this rule output?"
- **`aux_var`** — picked an action: "do I need an extra witness variable, and of what type?"
- **`pre_lit`** — "add another precondition literal, or stop preconditions?" (loops; capped at $L_S$).
- **`goal_lit`** — "add another goal literal, or finish the rule?" (loops; capped at $L_G$).

Two invariants worth keeping: $q=\varnothing \iff h\in\{\texttt{policy},\bot\}$ (a partial rule exists exactly while a rule is in progress); and every non-terminal hole offers **at least one** production — a `STOP_*` / `FINISH_RULE` cursor production is always available, so the writer never gets stuck.

![the five hole kinds as a state machine; nodes annotated with legal-production counts](../figures/02_derivation_state_machine.png)

**Figure D1 — the hole-state machine.** The five holes drawn as a finite-state machine, each node labelled with how many productions it offers in a representative state. The `goal_lit` hole is the widest (it offers a positive *and* a negated `Goal[ℓ]` per legal atom), which is what sizes the MCTS action head ([§14](#14--the-derivation-game-as-an-mdp-formal)). Regenerated from the live grammar — see [§16](#16--figure-index-notation-bridge-and-cross-links).

**The full production catalogue** (this is exactly `enumerate_productions`, [lifted_grammar.py:201-266](../../../../src/alphazeropp/synthesis/lifted_grammar.py)):

| hole $h$ | productions (`label`) | `payload` tag | offered when |
|---|---|---|---|
| `policy` | `ADD_RULE` | `("add_rule",)` | $\lvert C\rvert < R$ and the signature has $\ge 1$ action schema |
| `policy` | `STOP_POLICY` | `("stop",)` | $\lvert C\rvert \ge 1$ |
| `action_schema` | `schema={name}` — one per action schema (`schema=move`, `schema=pick`, `schema=drop`) | `("schema", name, arg_types)` | always (introduces the schema-position variables, e.g. `drop` ⇒ `?b_0:ball`, `?r_1:room`) |
| `aux_var` | `SKIP_AUX` | `("skip_aux",)` | always |
| `aux_var` | `add_aux:{type}` — one per type (`add_aux:ball`, `add_aux:room`) | `("add_aux", type)` | $V_{\mathrm{aux}}\ge 1$ and $\lvert q.\texttt{aux\_vars}\rvert < V_{\mathrm{aux}}$ (with $V_{\mathrm{aux}}=1$ this is offered only on the first visit; introduces `?aux_0`) |
| `pre_lit` | `pre:{ℓ}` — one per legal next precondition literal | `("add", ℓ)` | $\lvert q.\texttt{state\_lits}\rvert < L_S$; only literals strictly greater than the last accepted one under a fixed lex key (so no duplicates, no reorderings); state literals are never negated |
| `pre_lit` | `STOP_PRE` | `("stop_pre",)` | always |
| `goal_lit` | `goal:{ℓ}` — one per legal next goal literal | `("add", ℓ)` | $\lvert q.\texttt{goal\_lits}\rvert < L_G$; canonical lex order; a negated `Goal[…]` is offered only when every variable is positively bound (safe negation); if `goal_predicate_relevance` is on, only predicates in the signature's `goal_predicate_names`; if `require_goal_var_connected` is on, every variable of the literal must occur in the action arguments or in some positive state literal already added |
| `goal_lit` | `FINISH_RULE` | `("finish",)` | always (seals $q$ into a `Rule`, appends to $C$, hole → `policy`) |
| `⊥` | — (none) | — | terminal |

Two naming conventions ([lifted_grammar.py §"Design points"](../../../../src/alphazeropp/synthesis/lifted_grammar.py)): action-argument variables are named by **schema position** — `move(?r_0,?r_1)`, `pick(?b_0,?r_1)`, `drop(?b_0,?r_1)` — and any extra variable is the **auxiliary** `?aux_{j}` (Stage 2 caps $V_{\mathrm{aux}}=1$, so at most `?aux_0`). Because of this, two rules of the same shape pretty-print identically — no separate α-canonicalisation pass is needed at production time.

![legal-production sets at one representative state per hole kind](../figures/02_derivation_production_sets.png)

**Figure D2 — production sets per hole.** $\mathcal{P}_\Gamma(z) = \texttt{enumerate\_productions}(z,\mathrm{cfg},\Sigma)$ shown at one representative state per hole kind (Gripper-lite signature). The `goal_lit` hole is the widest; its size, $M = \texttt{compute\_max\_productions} = 13$ for the default config, is the upper bound on the MCTS action head ([§14](#14--the-derivation-game-as-an-mdp-formal)).

## §5 — A worked derivation, field by field

We build the Stage-1 hand-policy rule "drop-at-goal":

$$\boxed{\;\rho_1:\ \ \mathrm{at\_robot}(?r_1)\,\wedge\,\mathrm{carrying}(?b_0)\,\wedge\,\mathrm{Goal}[\mathrm{at\_ball}(?b_0,?r_1)]\ \Longrightarrow\ \mathrm{drop}(?b_0,?r_1)\;}$$

(α-equivalent to $\rho_1$ as written in [../00_draft_lifted_policy_az.md §4](../00_draft_lifted_policy_az.md#4-lifted-decision-list-policies) — the body-literal order here is the canonical one the grammar emits: $\mathrm{at\_robot}$ before $\mathrm{carrying}$ by lex key, then the goal literal.) The derivation is a 9-step play; each row below is one arrow. Field names are the implementation's: $q=z.\texttt{partial}$, with sub-fields `schema`, `action_args`, `aux_vars`, `state_lits`, `goal_lits`. (Read `state_lits` as *precondition literals*.)

| step | transition | what the production does | what changes |
|---|---|---|---|
| 0 | — | the start | $z_0 = ((),\ \varnothing,\ \texttt{policy})$ — no rules, no rule in progress |
| 1 | $z_0 \xrightarrow{\texttt{ADD\_RULE}} z_1$ | open a fresh partial rule | $h:\texttt{policy}\to\texttt{action\_schema}$; $q_1 = \texttt{PartialRule.empty()} = (\varnothing_{\text{schema}},\,()_{\text{args}},\,()_{\text{aux}},\,()_{\text{pre}},\,()_{\text{goal}})$ |
| 2 | $z_1 \xrightarrow{\texttt{schema=drop}} z_2$ | choose the action template | $h\to\texttt{aux\_var}$; $q_2.\texttt{schema}=\texttt{drop}$, $q_2.\texttt{action\_args}=(?b_0{:}\textit{ball},\,?r_1{:}\textit{room})$ — this pins $\alpha_{\rho_1}=\mathrm{drop}(?b_0,?r_1)$ |
| 3 | $z_2 \xrightarrow{\texttt{SKIP\_AUX}} z_3$ | no witness variable | $h\to\texttt{pre\_lit}$; $q_3 = q_2$ (cursor only — nothing added) |
| 4 | $z_3 \xrightarrow{\texttt{pre:at\_robot(?r\_1)}} z_4$ | add a precondition literal | $q_4.\texttt{state\_lits}=(\mathrm{at\_robot}(?r_1))$; $h$ stays `pre_lit` |
| 5 | $z_4 \xrightarrow{\texttt{pre:carrying(?b\_0)}} z_5$ | add a precondition literal | $q_5.\texttt{state\_lits}=(\mathrm{at\_robot}(?r_1),\ \mathrm{carrying}(?b_0))$ |
| 6 | $z_5 \xrightarrow{\texttt{STOP\_PRE}} z_6$ | stop preconditions | $h\to\texttt{goal\_lit}$; $q_6 = q_5$ (cursor only) |
| 7 | $z_6 \xrightarrow{\texttt{goal:Goal[at\_ball(?b\_0,?r\_1)]}} z_7$ | add a goal literal | $q_7.\texttt{goal\_lits}=(\mathrm{Goal}[\mathrm{at\_ball}(?b_0,?r_1)])$ |
| 8 | $z_7 \xrightarrow{\texttt{FINISH\_RULE}} z_8$ | seal the rule | $\rho_1$ appended to $C$ — body $=$ `state_lits` $+$ `goal_lits`, head $=\mathrm{drop}(?b_0,?r_1)$; then $q\to\varnothing$, $h\to\texttt{policy}$ |
| 9 | $z_8 \xrightarrow{\texttt{STOP\_POLICY}} z_9$ | end the policy | $h\to\bot$; $z_9 = ((\rho_1),\ \varnothing,\ \bot)$ is terminal; $\texttt{to\_program}(z_9)=\texttt{Policy}((\rho_1,))$ |

The $q_i$ are just the partial rule as it stands after step $i$: $q_1$ is the empty accumulator, $q_2$ has its schema and action variables, $q_4,q_5$ have growing precondition lists, $q_7$ has the goal literal too — and at step 8 that accumulator becomes the immutable sealed rule $\rho_1$.

![one derivation as a labelled path through the transition delta, building rho_1 end-to-end](../figures/02_derivation_example.png)

**Figure D3 — the worked derivation, linearly.** The same 9-step play drawn as a labelled path through the transition $\delta_\Gamma$: $\rho_1$ built in 8 productions, then `STOP_POLICY`. Each state is read off a live `LiftedDerivationState`, so what you see is what the code produces.

![a field-by-field diff table along the build of rho_1](../figures/02_derivation_state_evolution.png)

**Figure D5 — the same derivation as a field-by-field diff.** Each row is a state $z_i$; the highlighted cell is the single field that the production $\delta_\Gamma$ touched on that step. (Figure D4, the static "anatomy" of a state, is in [§6](#6--the-fields-of-a-partial-rule).)

**How long is a derivation?** Each rule costs $5$ fixed productions — `ADD_RULE`, `schema=…`, the aux decision (`SKIP_AUX` or `add_aux:T`), `STOP_PRE`, `FINISH_RULE` — plus up to $L_S$ `pre:ℓ` and up to $L_G$ `goal:ℓ`. A whole policy of $k$ rules then ends with one `STOP_POLICY`:

$$\ell(\pi) \;=\; 1 + \sum_{i=1}^{k}\bigl(5 + \lvert\texttt{pre}_i\rvert + \lvert\texttt{goal}_i\rvert\bigr),\qquad 1\le k\le R,$$

so $\ell(\pi)\in[\,6,\ R(L_S+L_G+5)+1\,]$ — for Gripper-lite ($L_S=3,L_G=1$) that is $[\,6,\ R\cdot 9 + 1\,]$.

![per-rule production breakdown, the analytic length range, and the empirical length distribution](../figures/02_derivation_production_lengths.png)

**Figure D6 — derivation length.** Per-rule structure ($5$ fixed productions $+\le L_S$ pre-lits $+\le L_G$ goal-lits), the analytic length range vs `max_rules`, and the empirical length distribution over 5000 uniform-random derivations (a uniform prior favours short policies — relevant to [§11](#11--failure-modes-of-this-policy-language)).

## §6 — The fields of a partial rule

A partial rule — and the `Rule` it seals into — is a **half-written IF–THEN sentence**: *IF (these conditions on the world) AND (these conditions on the goal) THEN (this action)*. Formally,

$$\boxed{\;q = (\,\texttt{schema},\ \texttt{action\_args},\ \texttt{aux\_vars},\ \texttt{state\_lits},\ \texttt{goal\_lits}\,).\;}$$

In words, field by field (with the $\rho_1$ example from [§5](#5--a-worked-derivation-field-by-field)):

| field | plain English | $\rho_1$ example |
|---|---|---|
| `schema` | which action *form* the rule will output | `drop` |
| `action_args` | the variables that appear *in the action* — named by schema position | $(?b_0{:}\textit{ball},\ ?r_1{:}\textit{room})$ |
| `aux_vars` | extra **witness** variables used only in conditions, never in the action | $()$ — none (drop already mentions every object it needs) |
| `state_lits` | conditions checked against the **current world state** $x$ — these are precondition literals; positive only in Stage 2 | $(\mathrm{at\_robot}(?r_1),\ \mathrm{carrying}(?b_0))$ |
| `goal_lits` | conditions checked against the **goal** $G$ — written `Goal[…]`, possibly negated | $(\mathrm{Goal}[\mathrm{at\_ball}(?b_0,?r_1)])$ |

When `FINISH_RULE` fires, the rule is sealed as `Rule(vars = action_args ∪ aux_vars, body = state_lits + goal_lits, action = LiftedAction(schema, action_args))`. So `state_lits` and `goal_lits` end up in one ordered `body` tuple, distinguished by each literal's `source` tag (`STATE` vs `GOAL`) — [§7](#7--why-goal-literals-are-not-cheating) explains why that split matters.

![six representative LiftedDerivationStates with every field rendered explicitly](../figures/02_derivation_state_anatomy.png)

**Figure D4 — state anatomy.** Six representative `LiftedDerivationState`s with *every* field rendered: `completed_rules` / `partial = PartialRule(schema, action_args, aux_vars, state_lits, goal_lits)` / the single `current_hole`, plus the state's `pretty()` node key (which carries the variable types — see [§4](#4--the-hole-state-machine)).

## §7 — Why goal literals are not cheating

A natural worry: *the game synthesises a policy, but rules can mention `Goal[at_ball(?b,?r)]` — isn't the goal being smuggled in?* No. The synthesised policy is **goal-conditioned**: at execution time the interpreter is handed *both* the current state $x$ **and** the goal $G$, and a goal literal is just a condition that reads $G$. The policy never invents or hard-codes a fixed goal at derivation time — it is written so that, at runtime, it inspects whichever `at_ball(...)` goal atoms *this* task instance happens to supply.

The contrast is sharpest on the drop rule. Without a goal literal:

$$\mathrm{carrying}(?b)\,\wedge\,\mathrm{at\_robot}(?r)\ \Longrightarrow\ \mathrm{drop}(?b,?r)$$

In words: "if I'm carrying a ball, drop it wherever I happen to be." That solves nothing in general — it dumps balls in arbitrary rooms. With a goal literal:

$$\mathrm{carrying}(?b)\,\wedge\,\mathrm{at\_robot}(?r)\,\wedge\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)]\ \Longrightarrow\ \mathrm{drop}(?b,?r)$$

In words: "drop this ball *only* in the room where the goal wants it." Same action, but now it is correct, and it is correct *for every goal*, because the rule reads the goal rather than assuming one.

**Negated goal literals** read the goal the other way. In the pick rule,

$$\neg\,\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)]$$

means "this ball is *not* already where the goal wants it" — i.e. don't bother re-picking a ball that's already finished. Negation here is **safe**: every variable in a negated goal literal must already appear positively (in `state_lits`) or among the action arguments, so by the time the interpreter evaluates it the variable is bound. The grammar pre-filters at the `goal_lit` hole, and [`Rule.__post_init__`](../../../../src/alphazeropp/synthesis/lifted_dsl.py) re-checks it ([lifted_dsl.py:166-180](../../../../src/alphazeropp/synthesis/lifted_dsl.py)). State literals are never negated in this stage (`allow_state_negation=False`).

## §8 — Auxiliary variables as witnesses

Sometimes the action itself does not mention every object you need in order to decide whether the action is appropriate. The clean example is "move toward the goal."

`move`'s schema is $\textit{room}\times\textit{room}$, so its schema-position variables are $?r_0,?r_1$ only — **there is no `?b_0` in a `move` rule by default.** A first try has nothing to say about *which* room to head to:

$$\mathrm{at\_robot}(?r_0)\ \Longrightarrow\ \mathrm{move}(?r_0,?r_1)$$

In words: "go from wherever I am to … some room." $?r_1$ is unconstrained — the rule has no reason to pick the right destination. We need a handle on the carried ball and where it should go. That is what `add_aux:ball` introduces — a variable $?aux_0:\textit{ball}$ — letting the body say:

$$\mathrm{at\_robot}(?r_0)\,\wedge\,\mathrm{carrying}(?aux_0)\,\wedge\,\mathrm{Goal}[\mathrm{at\_ball}(?aux_0,?r_1)]\ \Longrightarrow\ \mathrm{move}(?r_0,?r_1)$$

(this is hand-policy $\rho_2$ up to renaming). In words: "if I'm carrying *some* ball and the goal wants that ball in room $?r_1$, then move there." Crucially, **$?aux_0$ is a variable, not the object `ball_0`** — it is bound at runtime. With $x=\{\mathrm{at\_robot}(\textit{room\_a}),\,\mathrm{carrying}(\textit{ball}_0)\}$ and $G=\{\mathrm{at\_ball}(\textit{ball}_0,\textit{room\_b})\}$, the interpreter binds

$$?aux_0\mapsto\textit{ball}_0,\quad ?r_0\mapsto\textit{room\_a},\quad ?r_1\mapsto\textit{room\_b}$$

and the rule fires $\mathrm{move}(\textit{room\_a},\textit{room\_b})$. Notice $?aux_0$ *chose the destination* but does **not** appear in the ground action — it is a *witness* variable, present only to make the conditions decidable.

The discipline "action variables come from schema positions; every other body variable is introduced as an auxiliary" buys: explicit typing of `?aux_0` (ball vs room — and the type must be in the `pretty()` node key, since two such states have different production sets), a bounded search space (at most one aux), easy canonical naming, and local safety checks. It is also why `require_goal_var_connected` (Stage 2.5) is the **local** rule and not a transitive closure from action arguments: a literal transitive reading would reject $\rho_2$ — there $?aux_0$ is bound only via $\mathrm{carrying}(?aux_0)$, a *positive state literal* that co-occurs with no action argument — and Stage 2.5 requires $\rho_2$ to stay expressible (see [../02.md §Hypotheses (H4)](../02.md#hypotheses-and-next-experiments) and [../02.md Appendix E](../02.md#appendix-e--refinement-deltas-vs-the-plans)).

## §9 — What rules can this language create?

Step back from the machinery: what is the **hypothesis class**? A policy $\pi$ is an **ordered list of up to $R$ rules**. Each rule:

- outputs **exactly one action schema** (`move` / `pick` / `drop`);
- carries up to $L_S$ precondition (`state`) literals — positive, well-typed atoms over the rule's variables;
- carries up to $L_G$ goal literals — `Goal[…]` atoms, possibly negated (safely);
- may introduce up to $V_{\mathrm{aux}}$ auxiliary variables.

The configuration defaults in [`LiftedGrammarConfig`](../../../../src/alphazeropp/synthesis/lifted_grammar.py) are $R=4$, $L_S=3$, $L_G=1$, $V_{\mathrm{aux}}=1$ (the Stage-2 / 2.5 *experiments* in [../02.md](../02.md) run $R=3$ — Run B uses $R=4$; everything else as default). Some rule families this allows:

- **empty-body rules** — $\top\Rightarrow a(\cdot)$, e.g. $\top\Rightarrow\mathrm{drop}(?b_0,?r_1)$ ("always try to drop");
- **state-reactive rules** — preconditions only, e.g. $\mathrm{at\_ball}(?b_0,?r_1)\wedge\mathrm{handempty}()\Rightarrow\mathrm{pick}(?b_0,?r_1)$;
- **goal-conditioned rules** — a positive `Goal[…]`, e.g. the drop rule of [§7](#7--why-goal-literals-are-not-cheating) ($\rho_1$);
- **negated-goal guards** — $\dots\wedge\neg\mathrm{Goal}[\mathrm{at\_ball}(?b_0,?r_1)]\Rightarrow\dots$ ("only when this ball isn't done") — hand-policy $\rho_3$;
- **auxiliary-witness rules** — $\rho_2$ of [§8](#8--auxiliary-variables-as-witnesses).

All four Stage-1 hand-policy rules $\rho_1\dots\rho_4$ are reachable in this grammar up to α-renaming (tests `test_grammar_can_express_hand_policy_rules` / `test_current_grammar_still_expresses_hand_policy_rules` / `test_connectedness_allows_hand_policy_move_toward_goal`); the language is **not under-expressive for this task** — see [§10](#10--is-this-a-direct-cheat).

**How big is the program space?** Two cardinality knobs settle it. The MCTS action head is sized to the worst-case branching $M(\mathrm{cfg},\Sigma)=\texttt{compute\_max\_productions}=13$ for the default Gripper-lite signature (the `goal_lit` hole dominates — Fig D2; it drops to $7$ with `goal_predicate_relevance` on). And the policy count $|\Pi_{\Gamma,\mathrm{cfg}}|$ is small: at the `r1` config $(R,L_S,L_G,V_{\mathrm{aux}})=(1,2,1,1)$ it is $1\,166$ distinct policies — small enough to enumerate exhaustively (that is the `r1` row of [../02.md](../02.md) Table 2); the `r3` config ($R=3$) is no longer exhaustively enumerable, so [../02.md](../02.md) stratified-samples $5\,000$ policies from it. One depth fact matters up front: **at least $R\ge 3$ is needed for any $\mathcal{M}_B$-solver** — the `r1` ($R=1$) row finds zero. So the difficulty here is *not* a needle-in-a-galaxy capacity problem; it is a *density / landscape* problem ([§10](#10--is-this-a-direct-cheat), [§11](#11--failure-modes-of-this-policy-language)).

![capacity knobs, program-space size, and one real Rule per rule family](../figures/02_derivation_hypothesis_class.png)

**Figure D8 — the hypothesis class.** The capacity tuple $\mathrm{cfg}=(R,L_S,L_G,V_{\mathrm{aux}})$ with its `LiftedGrammarConfig` defaults and the disabled switches; the program-space size ($M=13$, the `r1` config fully enumerable at $1\,166$ policies, `r3` stratified-sampled, $R\ge 3$ for any solver); and a gallery of one real `Rule` per family — each built through the grammar, so nothing is hand-transcribed. Regenerated from the live grammar — see [§16](#16--figure-index-notation-bridge-and-cross-links).

| knob | meaning | default | experiments | code |
|---|---|---|---|---|
| $R$ | max rules in the policy | 4 | 3 (Run B: 4) | `max_rules` ([lifted_grammar.py:51](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |
| $L_S$ | max precondition (state) literals per rule | 3 | 3 | `max_pre_literals` ([lifted_grammar.py:52](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |
| $L_G$ | max goal literals per rule | 1 | 1 | `max_goal_literals` ([lifted_grammar.py:53](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |
| $V_{\mathrm{aux}}$ | max auxiliary witness vars per rule | 1 | 1 | `max_aux_vars` ([lifted_grammar.py:54](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |
| — | disjunction in a body / negated preconditions / object constants | off | off | `allow_disjunction` / `allow_state_negation` / `allow_constants` $=$ `False` ([lifted_grammar.py:56-58](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |
| $M(\mathrm{cfg},\Sigma)$ | worst-case branching $=$ MCTS action-head width | 13 | 13 (7 if `goal_predicate_relevance`) | `compute_max_productions` ([lifted_grammar.py:273](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) |

The **semantics** is an ordered decision list under PG3-style *first-applicable* rules: at each step the interpreter tries the rules top to bottom; the first rule with a non-empty binding set fires, and within that rule the **lex-min** binding is chosen (so, e.g., `ball_0` before `ball_1`). Details in [§12](#12--binding-set-and-execution).

What the configuration *excludes*: no disjunction in bodies (`allow_disjunction=False`), no object constants in rules (`allow_constants=False`), no negated precondition literals (`allow_state_negation=False`).

## §10 — Is this a direct cheat?

It is fair to ask whether handing MCTS this grammar is "handing it the answer." Two honest halves:

- **Yes, the grammar is domain-specific.** It is built from a `DomainSignature` — the domain's types, predicate schemas, and action schemas — so it "knows the vocabulary" of Gripper-lite (it will only ever produce `at_robot` / `at_ball` / `carrying` / `handempty` atoms and `move` / `pick` / `drop` actions). That is by design.
- **No, it is not handed the solution policy.** MCTS still has to *choose the production sequence* — which schema, which literals, in which order, how many rules, where to stop. The grammar only constrains the search to *well-typed, executable lifted policies*: it cannot emit an ill-typed atom, an action with an unbound argument, an unsafe negation, or a duplicate / reordered literal. That is the standard posture of syntax-guided synthesis and generalized-policy search — an intentionally structured search space, not a lookup table.

And quantitatively the space is *not* narrow enough to hand over the answer: the solver density $\lvert\{\pi:\mathrm{solve}_B(\pi)=1\}\rvert/\lvert\Pi_{\Gamma,\mathrm{cfg}}\rvert$ is $\le 0.1\%$ across every `r3` setting ([../02.md](../02.md) Table 2; Fig L1 below), and uniform-prior MCTS finds $B{=}1$ solvers in only **one** of the five committed runs (seed 0 / 128 sims — 8 solvers; both other seeds at 128 sims, and seed 0 at 512 sims, find *none* — [../02.md](../02.md) Fig L5). The Stage-1 hand-policy *shape* is **expressible by design, but not forced** — the grammar expresses $\rho_1\dots\rho_4$ up to α-renaming (tests `test_grammar_can_express_hand_policy_rules` / `test_current_grammar_still_expresses_hand_policy_rules` / `test_connectedness_allows_hand_policy_move_toward_goal`) — but *reachable ≠ findable*: [../02.md](../02.md) reports uniform-prior MCTS converging to *other* (spurious) solvers, not the hand policy. So Stage 2's question is "can MCTS search this structured space?" — **not** "did a domain-free learner rediscover the domain model?" That gap is exactly what [../02.md](../02.md) Hypotheses **H1** (solver density too low for unguided search), **H3** ($B=1$ identifiability — many spurious solvers hit $\mathrm{solve}_1$), **H4** (the connectedness rule), and **H5** (a *learned* prior is needed) name; Stage 3 attacks H5.

![B=1 / B=2 solver counts per grammar config; ≤ 0.1% everywhere](../figures/02_landscape_solver_density.png)

**Figure L1 (from [../02.md](../02.md)) — solver density by grammar config.** $B{=}1$ (blue) / $B{=}2$ (red) solver counts per Stage-2.5 safety-flag setting, for the `r1` exhaustive enumeration and the `r3` stratified sample; the annotation `k (p%)` gives the count and the fraction of policies — $\le 0.1\%$ everywhere, and only the `connected` / `both` settings ever produce a $B{=}2$ solver. The canonical read is in [../02.md §Empirical result / Fig L1](../02.md).

## §11 — Failure modes of this policy language

A restricted policy language has characteristic ways of going wrong. Most of these are what [../02.md](../02.md) actually observed.

- **Too-general empty-body rules.** $\top\Rightarrow\mathrm{drop}(?b_0,?r_1)$ ([`degenerate_drop_policy`](../../../../src/alphazeropp/instances/gripper_lite/policies.py)): `drop` is never legal from a state where the robot isn't carrying anything, so `interpret` returns `None` on the first step and the rollout stalls — leaf score a small negative ($\approx -0.1$: noop penalty, zero progress, no solve). The deceptive part is that *doing nothing is good*: $\top\Rightarrow\mathrm{drop}$ sits at the **≈96th percentile** of the uniformly sampled program space (rank 187/5 000 under both Stage-2.5 safety flags; 60/5 000 — ≈99th percentile — under `none`), and the near-do-nothing band is what uniform-prior MCTS actually keeps rediscovering ([../02.md](../02.md) Fig 3; Fig L2-right). *That* deceptive sparse-terminal-reward landscape — not under-expressiveness — is the negative finding of [../02.md](../02.md).
- **Missing goal guards.** A pick rule *without* $\neg\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)]$ will re-pick balls that are already in the target room, undoing progress and burning steps.
- **Disconnected goal variables and vacuous goal predicates.** A goal literal can mention a variable bound *nowhere else* in the rule — e.g. `Goal[at_ball(?aux_0, ?r_1)]` where `?aux_0` appears in no precondition and no action argument — or a *vacuous* predicate not even in the goal vocabulary, e.g. `Goal[carrying(?b_0)]` (the signature's `goal_predicate_names` is `("at_ball",)`). Both are common: in 5 000 uniformly sampled default-grammar `r3` policies, **87.7 %** (4 384) use a vacuous goal predicate and **16.2 %** (811) have a disconnected goal variable ([../02.md](../02.md) Table 2; Fig L4). The Run A **seed 0 / 128 sims** "best learned policy" is itself one of these spurious solvers — its $B{=}1$ solve leans on a vacuous `Goal[carrying(?b_0)]` *and* a disconnected aux in `Goal[at_ball(?aux_0, ?r_1)]` ([../02.md](../02.md) Executive summary / Appendix A). Stage 2.5's two default-off flags fix it by construction: `goal_predicate_relevance` blocks the vacuous-predicate version (→ 0 %), and `require_goal_var_connected` — the **local** rule "every goal-literal variable occurs in the action arguments or in some positive state literal" ([`goal_vars_locally_connected`](../../../../src/alphazeropp/synthesis/lifted_grammar.py), which keeps $\rho_2$, bound via `carrying(?aux_0)` — see [§6](#6--the-fields-of-a-partial-rule)) — blocks the disconnected-variable version (→ 0 %).
- **Overfitting to $B=1$.** A policy that solves one-ball Gripper-lite but loops on $B=2$ — the lex-min tie-break and a missing rule can make it shuttle the same ball back and forth. Under the combined Stage-2.5 safety flags the $B{=}1$ solver count rises **4× (1 → 4)** and exactly **one** $B{=}1\!\to\!B{=}2$ generaliser appears among 5 000 sampled policies (while the vacuous-goal and disconnected-aux pathologies above go to 0 %) — see [../02.md](../02.md) Table 2; Figs L1, L3, L4. *Reachable but rare:* even after grammar restriction, a uniform prior almost never lands on a generalising solver — which is why Stage 3 swaps in a learned prior ([../02.md](../02.md) H5).
- **Stop-too-early derivations.** `STOP_POLICY` becomes legal as soon as one rule is sealed, so the search can commit to a short, useless policy (e.g. just the do-nothing rule). Under a uniform prior, shorter derivations are *more* likely (Figure D6), which is part of why the bad region is so reachable.
- **Lexicographic-binding artifacts.** Ties between bindings are broken by `tuple(sorted(theta.items()))`, so `ball_0` is always tried before `ball_1`. Here that is benign, but it is a load-bearing determinism assumption — a policy that "works" might be relying on it.
- **Expressivity limits.** No loops except by re-running the whole policy each environment step; no memory, no counters, no arithmetic; hard caps on the number of rules ($R$), literals per rule ($L_S$, $L_G$), and auxiliary variables ($V_{\mathrm{aux}}$). Tasks needing more than this are simply outside the class.

![vacuous-goal-predicate and disconnected-goal-var fractions per safety-flag setting](../figures/02_landscape_pathology_fractions.png)

**Figure L4 (from [../02.md](../02.md)) — pathology fractions.** Fraction of uniformly sampled policies with a vacuous goal predicate / a disconnected goal variable, per Stage-2.5 safety-flag setting; `r3 none` has 87.7 % vacuous + 16.2 % disconnected, and `both` has 0 % / 0 % — not an estimate, *by construction* (the grammar cannot emit those once the flags are on). For the degenerate-`⊤⇒drop` percentile and the solver/generaliser counts see [../02.md](../02.md) Figs L2 / L1 / L3; for the rest of the quantitative story — solver density, $B{=}1\!\to\!B{=}2$ generalisation rate, score-shape sensitivity, the spurious "best learned" policy — see [../02.md](../02.md) (Empirical result, Interpretation, Appendix A, Hypotheses H1–H5).

## §12 — Binding set and execution

Once a policy $\pi$ is finished, the **interpreter** ([`lifted_interpreter.py`](../../../../src/alphazeropp/synthesis/lifted_interpreter.py)) runs it on a world state $x$ and goal $G$ to pick a task action. Here is what `find_bindings(rule, state_atoms, goal_atoms, objects_by_type, legal_actions)` actually does, step by step:

1. Split the rule body into positive and negative literals. Start with the trivial binding $\{\}$.
2. **Positive pass.** For each positive literal, try to extend every binding so far by unifying the literal's atom with each matching ground row in $x$ (or in $G$, if the literal's source is `GOAL`). A conflict (a variable already bound to a different object, or a constant that doesn't match) drops that extension. If the candidate set ever empties, return `[]` — the rule does not apply.
3. **Free-variable enumeration.** Any *declared* rule variable still unbound after the positive pass (a variable used only in the action, say) is enumerated over all objects of its declared type, via a Cartesian product over `objects_by_type`. If some type has no objects, drop that candidate.
4. **Negative filter (closed-world).** Discard a candidate $\theta$ if any negative literal's grounded atom *is* present in the relevant relation — i.e. keep $\theta$ only when each $\neg\ell$ genuinely holds.
5. **Legality filter.** Discard $\theta$ unless $\theta(\alpha_\rho)$ — the ground action obtained by substituting $\theta$ into the rule's action template — is in `legal_actions`.
6. **Deterministic sort.** Sort the survivors by `tuple(sorted(theta.items()))`; the first element is the lex-min binding.

In symbols: write a sealed rule as $\rho = (X_\rho,\ \mathrm{body}_\rho,\ \alpha_\rho)$ — its variables $X_\rho$, its body literals (split by `source` into a state side $\mathrm{body}^x_\rho$ and a goal side $\mathrm{body}^G_\rho$), and its lifted action template. A binding is a map $\theta:X_\rho\to\mathcal{O}$ (variables to concrete objects), $\theta(\alpha_\rho)$ is the ground action after substitution, and the **binding set** on world state $x$ with goal $G$ in task $\mathcal{M}_B$ is

$$\boxed{\;B_\rho(x,G) = \bigl\{\,\theta : X_\rho\to\mathcal{O}\ \bigm|\ x\models_\theta\mathrm{body}^x_\rho\ \wedge\ G\models_\theta\mathrm{body}^G_\rho\ \wedge\ \theta(\alpha_\rho)\in\mathcal{A}_B(x)\,\bigr\}.\;}$$

In words: every object assignment that makes $\rho$'s preconditions true in $x$, its goal conditions true in $G$, and its action legal right now. The interpreter takes the lex-min one,

$$\theta^\star_\rho(x,G) = \min\nolimits_{\preceq} B_\rho(x,G),\qquad \theta\preceq\theta'\iff\bigl(\theta(v)\bigr)_{v\in\mathrm{sort}(X_\rho)}\le_{\mathrm{lex}}\bigl(\theta'(v)\bigr)_{v\in\mathrm{sort}(X_\rho)},$$

and the first-applicable policy map fires the first rule with a non-empty binding set: $\mathrm{Interpret}(\pi,x,G) = \theta^\star_{\rho_{i^\star}}(\alpha_{\rho_{i^\star}})$ for $i^\star = \min\{i : B_{\rho_i}(x,G)\ne\varnothing\}$, else `None` (a no-op step).

**Worked example — the pick rule, $B=2$.** Take

$$\rho_3:\ \ \mathrm{at\_ball}(?b_0,?r_1)\,\wedge\,\mathrm{at\_robot}(?r_1)\,\wedge\,\mathrm{handempty}()\,\wedge\,\neg\mathrm{Goal}[\mathrm{at\_ball}(?b_0,?r_1)]\ \Longrightarrow\ \mathrm{pick}(?b_0,?r_1)$$

on the two-ball start state

$$x_0=\{\,\mathrm{at\_robot}(\textit{room\_a}),\ \mathrm{handempty}(),\ \mathrm{at\_ball}(\textit{ball}_0,\textit{room\_a}),\ \mathrm{at\_ball}(\textit{ball}_1,\textit{room\_a})\,\},\qquad G=\{\,\mathrm{at\_ball}(\textit{ball}_0,\textit{room\_b}),\ \mathrm{at\_ball}(\textit{ball}_1,\textit{room\_b})\,\}.$$

The positive pass over $\mathrm{at\_ball}(?b_0,?r_1)$ produces two candidates, $\theta_0 = \{?b_0\mapsto\textit{ball}_0,\ ?r_1\mapsto\textit{room\_a}\}$ and $\theta_1 = \{?b_0\mapsto\textit{ball}_1,\ ?r_1\mapsto\textit{room\_a}\}$; both survive $\mathrm{at\_robot}(?r_1)$ ($\textit{room\_a}$) and $\mathrm{handempty}()$. The negative literal $\neg\mathrm{Goal}[\mathrm{at\_ball}(?b_0,?r_1)]$ keeps **both** — neither ball is at $\textit{room\_b}$ yet, so $(\textit{ball}_i,\textit{room\_a})\notin G[\mathrm{at\_ball}]$. Both $\mathrm{pick}(\textit{ball}_i,\textit{room\_a})$ are legal. The lex-min sort puts $\theta_0$ first (`ball_0 < ball_1`), so the rule fires

$$\theta_0(\mathrm{pick}(?b_0,?r_1)) = \mathrm{pick}(\textit{ball}_0,\textit{room\_a}).$$

**A falls-through case.** Take the drop rule $\rho_1 = \mathrm{at\_robot}(?r_1)\wedge\mathrm{carrying}(?b_0)\wedge\mathrm{Goal}[\mathrm{at\_ball}(?b_0,?r_1)]\Rightarrow\mathrm{drop}(?b_0,?r_1)$ on the same $x_0$: the positive pass matches $\mathrm{at\_robot}(?r_1)$, then reaches $\mathrm{carrying}(?b_0)$ — but $x_0$ has no $\mathrm{carrying}(\cdot)$ atom at all, so the candidate set empties and $B_{\rho_1}(x_0,G)=\varnothing$. The interpreter moves on to the next rule.

The implementation walks $\mathrm{body}_\rho$ literal-by-literal and never materialises the full Cartesian product over $\mathcal{O}$, so it transfers unchanged to held-out larger instances. The leaf evaluator's `num_rule_evals` ([../02.md Appendix A](../02.md#appendix-a--code-fidelity-audit)) counts a rule-evaluation proxy (`rule_idx + 1` on a firing step, `len(rules)` on a stall) — it is *not* an instrumentation of `find_bindings` itself.

**Same word, two levels.** $B_\rho$ ranges over *task* variables and objects (level $x$); the legal-production map $\mathcal{P}_\Gamma(z)$ of [§14](#14--the-derivation-game-as-an-mdp-formal) ranges over *productions* (level $p$). Don't conflate them.

## §13 — The grammar as a typed attributed CFG

A plain context-free grammar has nonterminals, terminals, a start symbol, and one-nonterminal-at-a-time productions $A\to\alpha$. **The CFG *skeleton* tells you the order of the choices. The *attributes* tell you which choices are legal at each step.** Without the attributes the skeleton over-generates wildly — it would happily emit ill-typed atoms, duplicate literals, reordered bodies, and unsafe negations. The skeleton, with the production labels of [§4](#4--the-hole-state-machine) as terminals:

$$
\begin{aligned}
\langle\textit{Policy}\rangle &\to \langle\textit{Rule}\rangle\ \langle\textit{Policy}\rangle \;\mid\; \texttt{STOP\_POLICY}\\
\langle\textit{Rule}\rangle &\to \texttt{ADD\_RULE}\ \langle\textit{ActionSchema}\rangle\ \langle\textit{Aux}\rangle\ \langle\textit{PreList}\rangle\ \texttt{STOP\_PRE}\ \langle\textit{GoalList}\rangle\ \texttt{FINISH\_RULE}\\
\langle\textit{ActionSchema}\rangle &\to \texttt{schema=move} \;\mid\; \texttt{schema=pick} \;\mid\; \texttt{schema=drop}\\
\langle\textit{Aux}\rangle &\to \texttt{SKIP\_AUX} \;\mid\; \texttt{add\_aux:ball} \;\mid\; \texttt{add\_aux:room}\\
\langle\textit{PreList}\rangle &\to \varepsilon \;\mid\; \texttt{pre:}\ell\ \langle\textit{PreList}\rangle\\
\langle\textit{GoalList}\rangle &\to \varepsilon \;\mid\; \texttt{goal:}\ell\ \langle\textit{GoalList}\rangle
\end{aligned}
$$

The legal literals $\ell$ at a hole depend on **attributes already chosen**:

- the **action schema** picked for this rule;
- the **variable context** — which of `?b_0,?r_1,?aux_0` are in scope;
- the **last literal** accepted in this list, under the canonical lex key `_lit_key` (so the next must be strictly greater — no duplicates, no reorderings);
- the **remaining caps** $L_S,L_G$ (and $R$ at the policy hole, $V_{\mathrm{aux}}$ at the aux hole);
- the **positive-variable set** — for safe negation, a negated `Goal[ℓ]` is offered only when every variable of $\ell$ is already positively bound;
- if `require_goal_var_connected` is on, the **local goal-connectivity** check (every goal-literal variable occurs in the action arguments or in some positive state literal).

So the honest object is a **typed attributed context-free grammar** — equivalently a *one-hole typed derivation grammar*:

$$\boxed{\;\Gamma = (N,\ \Sigma_{\mathrm{prod}},\ P,\ S;\ \mathrm{Attr}),\qquad \mathrm{Attr} = (\text{chosen schema},\ \text{scope variables},\ \text{last literal},\ \text{remaining caps},\ \text{positive-var set}),\;}$$

with $N$ = the hole kinds $\cup\ \langle\textit{Rule}\rangle$, $\Sigma_{\mathrm{prod}}$ = the production labels, $S=\langle\textit{Policy}\rangle$. The honest indexed nonterminals look like $\langle\textit{PreList}(j,\ X,\ \ell_{\mathrm{last}})\rangle$ ($j$ = literals so far, $X$ = scope, $\ell_{\mathrm{last}}$ = last literal) — and $\mathrm{Attr}$ is precisely what `enumerate_productions` reads off the state $z$ together with $(\mathrm{cfg},\Sigma)$.

The payoff: **a derivation tree of $\Gamma$ and an MCTS play are the same object viewed two ways** — the tree view shows *syntax*, the play view shows *decisions over time*. The terminal leaves of the tree, read left to right, are the production sequence $p_1\dots p_T$ of [§3](#3--what-the-arrow-labels-mean); the same tree, read through `to_program`, is a `Policy`.

![one full derivation of a 3-rule policy drawn as a grammar parse tree](../figures/02_derivation_tree.png)

**Figure D7 — a derivation as a parse tree.** One full derivation of a 3-rule policy — the Stage-1 hand-policy prefix $\rho_1$ drop-at-goal / $\rho_2$ move-toward-goal / $\rho_3$ pick — drawn as a grammar parse tree: $\langle\textit{Policy}\rangle$ at the root, expanding through $\langle\textit{Rule}\rangle\to\texttt{ADD\_RULE}\ \langle\textit{ActionSchema}\rangle\ \langle\textit{Aux}\rangle\ \langle\textit{PreList}\rangle\ \texttt{STOP\_PRE}\ \langle\textit{GoalList}\rangle\ \texttt{FINISH\_RULE}$ down to the terminal production labels. The leaves, top to bottom, are the production sequence $p_1\dots p_T$ — the same play as Figure D3, but as a tree; colour = hole kind; the right column shows the sealed rule $\rho_i$ each $\langle\textit{Rule}\rangle$ subtree yields.

## §14 — The derivation game as an MDP (formal)

> Dense layer — optional for a first pass. Everything below is the precise restatement of [§§1–13](#1--one-page-mental-model). The package's [`Game`](../../../../src/alphazeropp/core/game.py) protocol is single-player and MCTS-driven (`reset` / `step` / `get_action_mask` / `hashable_obs` / `stash_state` / `clone`); [`LiftedDerivationGame`](../../../../src/alphazeropp/synthesis/lifted_derivation.py) instantiates it over the grammar so the unchanged [`MCTS`](../../../../src/alphazeropp/core/mcts.py) engine and `UniformPolicyValueNet` drive it.

Fix a grammar config $\mathrm{cfg}$ ([`LiftedGrammarConfig`](../../../../src/alphazeropp/synthesis/lifted_grammar.py): `max_rules` $R$, `max_pre_literals` $L_S$, `max_goal_literals` $L_G$, `max_aux_vars` $V_{\mathrm{aux}}$, plus the two default-off Stage-2.5 toggles `goal_predicate_relevance`, `require_goal_var_connected`; defaults $R{=}4,L_S{=}3,L_G{=}1,V_{\mathrm{aux}}{=}1$, the Stage-2/2.5 experiments using $R{=}3$) and a domain signature $\Sigma$ ([`DomainSignature`](../../../../src/alphazeropp/synthesis/lifted_grammar.py): `types`, `predicates`, `action_schemas`, optional `goal_predicate_names`; `gripper_lite_signature()` sets `types=("ball","room")`, `goal_predicate_names=("at_ball",)`). The derivation game is

$$\boxed{\;\mathcal{D}_\Gamma \;=\; \bigl(\,\mathcal{Z},\ \mathcal{P}_\Gamma,\ \delta_\Gamma,\ R_\Gamma,\ z_0\,\bigr).\;}$$

**State space.** $\mathcal{Z} = \{\,z=(C,q,h)\,\}$ with hole kind $h(z)\in\{\texttt{policy},\texttt{action\_schema},\texttt{aux\_var},\texttt{pre\_lit},\texttt{goal\_lit},\bot\}$ and the invariant $q=\varnothing \iff h\in\{\texttt{policy},\bot\}$. Initial state and terminal set:

$$z_0 = \bigl((),\ \varnothing,\ \texttt{policy}\bigr) = \texttt{LiftedDerivationState.initial()},\qquad \mathcal{T}_\Gamma = \{\,z : h(z)=\bot\,\} = \{\,(C,\varnothing,\bot)\,\}.$$

**Actions / legal set.** $\mathcal{P}_\Gamma(z) = \texttt{enumerate\_productions}(z,\ \mathrm{cfg},\ \Sigma)$, with $\mathcal{P}_\Gamma(z)=\varnothing \iff h(z)=\bot$ (every non-terminal state offers at least one production — the `STOP_*` / `FINISH_RULE` cursor production is always there). The per-hole catalogue is the table in [§4](#4--the-hole-state-machine).

**Transition.** $\delta_\Gamma(z,p) = z.\texttt{apply}(p)$ — the partial map "apply the chosen production to the open hole", defined for $p\in\mathcal{P}_\Gamma(z)$, pure (`dataclasses.replace`, never mutates $z$).

**Reward.** Sparse — zero everywhere except the leaf, where it equals the training-objective score of the synthesised policy (see $U_B$, $J_{\mathrm{train}}$ in [../02.md §Minimal setup](../02.md#minimal-setup)):

$$R_\Gamma(z,p) \;=\; \begin{cases} J_{\mathrm{train}}\bigl(\texttt{to\_program}(\delta_\Gamma(z,p))\bigr), & \delta_\Gamma(z,p)\in\mathcal{T}_\Gamma,\\[2pt] 0, & \text{otherwise,} \end{cases}\qquad \texttt{to\_program}(z) = \texttt{Policy}(z.\texttt{completed\_rules}).$$

Concretely the leaf score (per [`LiftedLeafEvaluator`](../../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py)) is $\texttt{solve\_rate} + 0.25\cdot\texttt{avg\_progress} - 0.01\cdot\texttt{avg\_steps} - 0.05\cdot\texttt{num\_noops}$ over the train instances (`num_noops` = number of steps where `interpret` returned `None`).

**MCTS over $\mathcal{D}_\Gamma$.** PUCT selection on the search tree of $\mathcal{D}_\Gamma$:

$$\mathrm{UCB}(z,p) = \hat{Q}_{\mathrm{norm}}(z,p) + c\,\pi_0(p\mid z)\,\frac{\sqrt{N(z)}}{1+N(z,p)},\qquad c=1.5,\quad \pi_0\ \text{uniform}$$

([mcts.py:400-447](../../../../src/alphazeropp/core/mcts.py)). The action head is sized to the worst-case branching $M(\mathrm{cfg},\Sigma) = \max_{z}\lvert\mathcal{P}_\Gamma(z)\rvert = \texttt{compute\_max\_productions}(\mathrm{cfg},\Sigma)$ — for the default Gripper-lite signature, $M = 13$ (the `goal_lit` hole, taken with every variable positively bound so safe-negation never filters, dominates; with `goal_predicate_relevance` on it drops to $7$). The encoded observation length is $L(\mathrm{cfg}) = 7\bigl(R(1+L_S+L_G+V_{\mathrm{aux}})+1\bigr)$ ([lifted_encoding.py](../../../../src/alphazeropp/synthesis/lifted_encoding.py); $=133$ at $R{=}3$, $175$ at $R{=}4$) — uniform MCTS ignores it; it becomes load-bearing only when Stage 3 swaps in a learned net. Under Stage 2 / 2.5, *all* search structure therefore comes from the backed-up $R_\Gamma$ values — which is exactly why the deceptive landscape of [§11](#11--failure-modes-of-this-policy-language) (the negative finding of [../02.md](../02.md)) defeats it.

## §15 — Code-fidelity audit

> Optional for first-pass readers. Every formal symbol above ties to a live site.

| Claim / symbol | Formal statement | Impl site | Status |
|---|---|---|---|
| Derivation state | $z = (C,q,h)$ — completed rules, partial rule, current hole | `LiftedDerivationState` ([lifted_derivation.py:71-83](../../../../src/alphazeropp/synthesis/lifted_derivation.py)); `partial` is non-`None` iff $h\in\{\texttt{action\_schema},\texttt{aux\_var},\texttt{pre\_lit},\texttt{goal\_lit}\}$ | ✓ exact |
| Partial rule | $q = (\texttt{schema},\texttt{action\_args},\texttt{aux\_vars},\texttt{state\_lits},\texttt{goal\_lits})$ | `PartialRule` ([lifted_derivation.py:42-69](../../../../src/alphazeropp/synthesis/lifted_derivation.py)); `positive_var_names()` = action-arg names ∪ vars of `state_lits` | ✓ exact |
| Initial / terminal | $z_0 = ((),\varnothing,\texttt{policy})$; $\mathcal{T}_\Gamma=\{h=\bot\}$ | `LiftedDerivationState.initial()`; `is_terminal()` ⇔ `current_hole is None` ([lifted_derivation.py:81-83,142-143](../../../../src/alphazeropp/synthesis/lifted_derivation.py)) | ✓ exact |
| Production | $p = (\texttt{hole\_kind},\texttt{label},\texttt{payload})$, payload self-contained | `LiftedProduction` ([lifted_grammar.py:113-120](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) | ✓ exact |
| Legal set | $\mathcal{P}_\Gamma(z) = \texttt{enumerate\_productions}(z,\mathrm{cfg},\Sigma)$; per-hole catalogue of [§4](#4--the-hole-state-machine) | [lifted_grammar.py:201-266](../../../../src/alphazeropp/synthesis/lifted_grammar.py) | ✓ exact |
| Transition | $\delta_\Gamma(z,p) = z.\texttt{apply}(p)$, pure (`dataclasses.replace`, never mutates `self`) | [lifted_derivation.py:89-139](../../../../src/alphazeropp/synthesis/lifted_derivation.py) | ✓ exact |
| Sparse reward | nonzero only at $h=\bot$, $= J_{\mathrm{train}}(\texttt{to\_program}(z))$ | `LiftedDerivationGame.step` ([lifted_derivation.py:211-234](../../../../src/alphazeropp/synthesis/lifted_derivation.py)); `to_program() = Policy(completed_rules)` ([:145-146](../../../../src/alphazeropp/synthesis/lifted_derivation.py)) | ✓ exact |
| Leaf objective $J_{\mathrm{train}}$ / $U_B$ | $\texttt{solve\_rate}+0.25\,\texttt{avg\_progress}-0.01\,\texttt{avg\_steps}-0.05\,\texttt{num\_noops}$, over the train instances | [`LiftedLeafEvaluator`](../../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py); $U_B$ / $J_{\mathrm{train}}$ pinned in [../02.md §Minimal setup](../02.md#minimal-setup) | ✓ exact (↗ ../02.md for the symbol names) |
| Action-head width | $M(\mathrm{cfg},\Sigma) = \max_z\lvert\mathcal{P}_\Gamma(z)\rvert$; default cfg $=13$ | `compute_max_productions` ([lifted_grammar.py:273-306](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) | ✓ exact (verified across all 4 Stage-2.5 flag combinations in `tests/test_lifted_grammar.py`) |
| Observation length | $L(\mathrm{cfg}) = 7\bigl(R(1+L_S+L_G+V_{\mathrm{aux}})+1\bigr)$ | `encode_max_len` ([lifted_encoding.py](../../../../src/alphazeropp/synthesis/lifted_encoding.py)) | ✓ exact ($133$ at $R{=}3$, $175$ at $R{=}4$) |
| Node key | `pretty()` carries `vars=[?r_0:room,…]` (type-disambiguation) | [lifted_derivation.py:148-163](../../../../src/alphazeropp/synthesis/lifted_derivation.py); `LiftedDerivationGame.hashable_obs` | ✓ exact |
| α-canonical form | schema-position names `?{prefix}_{i}`, extras `?aux_{j}`; `canonical_rule_form` collapses renaming + body reorder | [lifted_grammar.py:388-419](../../../../src/alphazeropp/synthesis/lifted_grammar.py) | ✓ exact |
| Safe negation | negated `Goal[…]` offered only when every var is positively bound or an action arg; re-checked at `Rule.__post_init__` | `literal_candidates` ([lifted_grammar.py:151-196](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); [lifted_dsl.py:166-180](../../../../src/alphazeropp/synthesis/lifted_dsl.py) | ✓ exact |
| `require_goal_var_connected` (local) | every goal-lit var occurs in action args ∪ positive state lits | `goal_vars_locally_connected` ([lifted_grammar.py:327-349](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); local-rule rationale in [../02.md Appendix E](../02.md#appendix-e--refinement-deltas-vs-the-plans) | ✓ exact |
| Stage-2.5 pathology detectors (§11) | vacuous goal predicate (not in `goal_predicate_names`); disconnected goal var (sole binding site is a goal lit) | `rule_has_vacuous_goal_predicate` ([lifted_grammar.py:352-361](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); `rule_has_disconnected_goal_var` ([lifted_grammar.py:364-381](../../../../src/alphazeropp/synthesis/lifted_grammar.py)) | ✓ exact (87.7% / 16.2% fractions ↗ ../02.md Fig L4) |
| Capacity tuple defaults (§9) | $\mathrm{cfg}=(R,L_S,L_G,V_{\mathrm{aux}})=(4,3,1,1)$; `allow_disjunction=allow_state_negation=allow_constants=False`; $M=13$ | `LiftedGrammarConfig` ([lifted_grammar.py:51-58](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); `compute_max_productions` ([:273-306](../../../../src/alphazeropp/synthesis/lifted_grammar.py)); $\lvert\Pi_{\Gamma,r1}\rvert=1\,166$ ↗ `data/landscape_r1.json` | ✓ exact |
| MCTS unchanged | PUCT, $c=1.5$, $\pi_0$ uniform; `lifted_*` modules import nothing grounded | [mcts.py:400-447](../../../../src/alphazeropp/core/mcts.py); `LiftedDerivationGame` ([lifted_derivation.py:170-287](../../../../src/alphazeropp/synthesis/lifted_derivation.py)) | ✓ exact |
| Binding set $B_\rho(x,G)$ / $\theta^\star_\rho$ | satisfying type-correct bindings + legality; lex-min by var-name-sorted object names | `find_bindings`, lex-min sort, `interpret` ([lifted_interpreter.py:89,157,161](../../../../src/alphazeropp/synthesis/lifted_interpreter.py)) | ✓ exact |
| `num_rule_evals` proxy | `rule_idx + 1` on a firing step, `len(rules)` on a stall; not an instrumentation of `find_bindings` | [lifted_leaf_evaluator.py](../../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py) (`_rollout_one`); see [../02.md Appendix A](../02.md#appendix-a--code-fidelity-audit) | ✓ exact |

## §16 — Figure index, notation bridge, and cross-links

**Figure index.** Eight D-figures live inline next to the section they illustrate, and are regenerated from the real grammar / derivation modules (no hand-transcription) by `python scripts/plotting/plot_derivation_game_diagrams.py`:

- **D1 — hole-state machine** ([../figures/02_derivation_state_machine.png](../figures/02_derivation_state_machine.png)) — the five holes as an FSM, nodes labelled with production counts. *([§4](#4--the-hole-state-machine).)*
- **D2 — production sets per hole** ([../figures/02_derivation_production_sets.png](../figures/02_derivation_production_sets.png)) — $\mathcal{P}_\Gamma(z)$ at one representative state per hole; `goal_lit` is widest, $M=13$. *([§4](#4--the-hole-state-machine).)*
- **D3 — worked derivation, linear** ([../figures/02_derivation_example.png](../figures/02_derivation_example.png)) — the 9-step build of $\rho_1$ as a labelled path through $\delta_\Gamma$. *([§5](#5--a-worked-derivation-field-by-field).)*
- **D4 — state anatomy** ([../figures/02_derivation_state_anatomy.png](../figures/02_derivation_state_anatomy.png)) — six `LiftedDerivationState`s with every field rendered. *([§6](#6--the-fields-of-a-partial-rule).)*
- **D5 — state evolution (diff)** ([../figures/02_derivation_state_evolution.png](../figures/02_derivation_state_evolution.png)) — the $\rho_1$ build as a field-by-field diff table. *([§5](#5--a-worked-derivation-field-by-field).)*
- **D6 — derivation length** ([../figures/02_derivation_production_lengths.png](../figures/02_derivation_production_lengths.png)) — per-rule structure, analytic range, empirical length distribution. *([§5](#5--a-worked-derivation-field-by-field).)*
- **D7 — derivation as a parse tree** ([../figures/02_derivation_tree.png](../figures/02_derivation_tree.png)) — one 3-rule policy drawn as a grammar parse tree; leaves = the production sequence. *([§13](#13--the-grammar-as-a-typed-attributed-cfg).)*
- **D8 — the hypothesis class** ([../figures/02_derivation_hypothesis_class.png](../figures/02_derivation_hypothesis_class.png)) — the capacity tuple $(R,L_S,L_G,V_{\mathrm{aux}})$ with its defaults + disabled switches, the program-space size ($M=13$, the `r1` config enumerable at $1\,166$ policies, `r3` stratified-sampled, $R\ge 3$ for any solver), and one real `Rule` per rule family. *([§9](#9--what-rules-can-this-language-create).)*

Three figures *owned by* [../02.md](../02.md) are reproduced inline here for the cheat / failure-mode discussion — their canonical captions and reads live there: **Fig L1** `02_landscape_solver_density.png` ([§10](#10--is-this-a-direct-cheat)), **Fig L4** `02_landscape_pathology_fractions.png` ([§11](#11--failure-modes-of-this-policy-language)), and **Fig L2** `02_landscape_score_variants.png` (cross-linked from [§11](#11--failure-modes-of-this-policy-language) for the degenerate-`⊤⇒drop` percentile).

**Notation bridge.** [../02.md Appendix B](../02.md#appendix-b--the-derivation-game-formally) writes the derivation state $s = (\rho_{1:k},\ \texttt{partial},\ \texttt{hole}(s))$; this note writes it $z = (C,\ q,\ h)$ — same object, with $z$ chosen to avoid the visual collision with the task/world state (which [../00_draft_lifted_policy_az.md](../00_draft_lifted_policy_az.md) writes $s$/$S$ and this note writes $x$). The maps line up: $\mathcal{P}(s)\equiv\mathcal{P}_\Gamma(z)$, $\delta\equiv\delta_\Gamma$, $M(\mathrm{cfg},\Sigma)$ is the same number. The implementation field is `state_lits`; throughout this note read it as "precondition literals" — it is *not* renamed in code, and it has nothing to do with derivation states. The lifted action template of rule $\rho$ is written $\alpha_\rho$ here (the `rule.action: LiftedAction`).

**Cross-links.**

- [../02.md](../02.md) — Stage-2/2.5 results companion: [Appendix B](../02.md#appendix-b--the-derivation-game-formally) (the compact version of this note), [Appendix A](../02.md#appendix-a--code-fidelity-audit) (code-fidelity audit), [§Minimal setup](../02.md#minimal-setup) ($U_B$, $J_{\mathrm{train}}$, $J_{\mathrm{out}}$, the task family $\mathcal{M}_B$, the grammar config), [§Hypotheses](../02.md#hypotheses-and-next-experiments) (H4, the connectedness rule), [Appendix E](../02.md#appendix-e--refinement-deltas-vs-the-plans) (the `require_goal_var_connected` local-rule deviation).
- [../00_draft_lifted_policy_az.md](../00_draft_lifted_policy_az.md) — orientation: [§3](../00_draft_lifted_policy_az.md#3-the-task-mdp-gripper-lite) (the task MDP), [§4](../00_draft_lifted_policy_az.md#4-lifted-decision-list-policies) (lifted decision-list policies + the hand policy $\rho_1..\rho_4$), [§5](../00_draft_lifted_policy_az.md#5-the-online-unification-interpreter) (the interpreter, $B_\rho$, $\theta^\star_\rho$), [§6](../00_draft_lifted_policy_az.md#6-synthesis-as-a-derivation-game) (synthesis as a derivation game).
- [../01.md](../01.md) — Stage 1: the lifted DSL, the Stage-1 interpreter, the Gripper-lite env, the hand policy.
- [pddl_background.md](pddl_background.md) — PDDL and its mapping to this repo's lifted DSL; [../../appendix/sygus_background.md](../../appendix/sygus_background.md) — syntax-guided synthesis grammars in general.
- Source: [lifted_grammar.py](../../../../src/alphazeropp/synthesis/lifted_grammar.py), [lifted_derivation.py](../../../../src/alphazeropp/synthesis/lifted_derivation.py), [lifted_dsl.py](../../../../src/alphazeropp/synthesis/lifted_dsl.py), [lifted_interpreter.py](../../../../src/alphazeropp/synthesis/lifted_interpreter.py), [lifted_encoding.py](../../../../src/alphazeropp/synthesis/lifted_encoding.py), [lifted_leaf_evaluator.py](../../../../src/alphazeropp/synthesis/lifted_leaf_evaluator.py); [gripper_lite/env.py](../../../../src/alphazeropp/instances/gripper_lite/env.py), [gripper_lite/policies.py](../../../../src/alphazeropp/instances/gripper_lite/policies.py); [core/mcts.py](../../../../src/alphazeropp/core/mcts.py), [core/game.py](../../../../src/alphazeropp/core/game.py).
