# Appendix (Stage 4) — PDDL Background

> Background reference for Stage 4. Cited from [../01.md](../01.md), [../02.md](../02.md), and the lifted DSL/grammar modules under [src/alphazeropp/synthesis/](../../../../src/alphazeropp/synthesis/). Style follows [../../appendix/sygus_background.md](../../appendix/sygus_background.md).

This note is self-contained: it defines PDDL on its own terms (philosophy, history, syntax), works through a small example, and then maps every PDDL concept onto this repository's lifted DSL so future stage-4 work can cite "PDDL" precisely.

## Contents

- [§1 Introduction — what PDDL is](#1-introduction--what-pddl-is)
- [§2 Philosophy — why PDDL looks the way it does](#2-philosophy--why-pddl-looks-the-way-it-does)
- [§3 History — a one-page timeline](#3-history--a-one-page-timeline)
- [§4 The PDDL 1.2 / typed-STRIPS grammar](#4-the-pddl-12--typed-strips-grammar)
- [§5 Versions & extensions — survey](#5-versions--extensions--survey)
- [§6 Worked example — Gripper](#6-worked-example--gripper)
- [§7 Connection to this repo's lifted DSL](#7-connection-to-this-repos-lifted-dsl)
- [§8 References](#8-references)

## §1 Introduction — what PDDL is

> **PDDL = Planning Domain Definition Language**
> ❀ standard encoding language for "classical" planning tasks
>
> Components of a PDDL planning task:
> - **Objects** — Things in the world that interest us.
> - **Predicates** — Properties of objects that we are interested in; can be true or false.
> - **Initial state** — The state of the world that we start in.
> - **Goal specification** — Things that we want to be true.
> - **Actions / Operators** — Ways of changing the state of the world.

PDDL was introduced in 1998 by Drew McDermott and the AIPS-98 planning-competition committee as the official input format for the **International Planning Competition (IPC)**. Before PDDL, every classical planner shipped its own bespoke domain file format; entries to the competition had to be hand-ported. PDDL fixed a common Lisp-like surface syntax so that a domain authored by one group could be solved, unchanged, by any conformant planner — and so that benchmark results from different teams were directly comparable.

Two ideas anchor the language. First, a **planning task is split into two files**: a *domain* file naming the physics (types, predicates, action schemas) and a *problem* file naming the situation (objects, initial state, goal). Second, every constituent is **lifted** — predicates and actions are universally quantified over typed variables, not enumerated over ground tuples. The planner is responsible for grounding when it needs to; the model is small.

## §2 Philosophy — why PDDL looks the way it does

### §2.1 Separation of domain and problem

A *domain file* declares `:types`, `:predicates`, and one or more `:action` schemas — the reusable physics of a world. A *problem file* declares `:objects`, `:init`, and `:goal` — the disposable situation. The same Gripper domain solves a problem with two balls and a problem with twenty; the same BlocksWorld domain solves any tower configuration. This separation is what makes IPC benchmarks meaningful: a planner is graded on its ability to generalise across problems sharing a fixed domain.

In practice, the domain file is checked into the benchmark repository once. Problem files are generated programmatically by an instance generator that knows the domain's predicates but produces a fresh `:init` / `:goal` pair per seed.

### §2.2 Lifted, declarative, action-centric

PDDL is **lifted**: an action schema like `(move ?from ?to)` stands for the whole family of ground instances `(move room-a room-b), (move room-a room-c), …`. Grounding is left to the planner. This is structural — the language *cannot* express a ground action without using a schema with all its parameters bound.

PDDL is **declarative**: a domain says *what* is true (`(at-robby ?r)`) and *what changes* (the `:effect` of `move` adds `(at-robby ?to)` and deletes `(at-robby ?from)`), but it never says *how* to search. There are no loops, no control flow, no heuristics inside the domain file. The planner — a separate executable — is a black box that consumes the domain + problem and emits a plan.

PDDL is **action-centric**: state changes happen only via action schemas. There are no spontaneous events (until PDDL+) and no derived state (until PDDL 2.2's `:derived-predicates`).

### §2.3 STRIPS lineage

PDDL inherits its action model from STRIPS (Fikes & Nilsson, 1971). Each action has:

- a **precondition**: a conjunction of literals over the action's parameters that must hold in the current state;
- an **effect**: a conjunction of literals to assert (add list) and literals to negate (delete list).

Every fluent obeys the **closed-world assumption** — atoms not asserted in `:init` are false; atoms not on an action's add list or delete list are unchanged (the **STRIPS frame assumption**). There are no implicit frame axioms beyond this default-persistence rule.

The philosophical commitment of vanilla PDDL is therefore: worlds are **finite**, **deterministic**, **fully observable**, with **instantaneous** actions and **discrete** state. Every PDDL extension since 1998 has been an attempt to relax one of those constraints while keeping the rest of the language intact.

## §3 History — a one-page timeline

| Year | Version | Key additions | Reference |
|---|---|---|---|
| 1971 | (pre-PDDL) STRIPS | Add/delete lists, conjunctive goals, closed-world | Fikes & Nilsson, AIJ 1971 |
| 1998 | PDDL 1.2 | Standardized STRIPS + ADL; types; IPC-1 input format | McDermott et al., 1998 |
| 2002 | PDDL 2.1 | Numeric fluents, durative actions, plan metrics; IPC-3 | Fox & Long, JAIR 2003 |
| 2004 | PDDL 2.2 | Timed initial literals, derived predicates; IPC-4 | Edelkamp & Hoffmann, 2004 |
| 2005 | PDDL 3.0 | Preferences, state-trajectory constraints; IPC-5 | Gerevini & Long, 2005 |
| 2008 | PDDL 3.1 | Object fluents (functions returning objects); IPC-6 | Helmert et al., 2008 |
| (parallel) | PDDL+ | Continuous processes & events for mixed discrete-continuous | Fox & Long, JAIR 2006 |

PDDL 1.2 was the AIPS-98 competition language and is what most textbooks mean by "PDDL". Each subsequent revision was driven by an IPC track that demanded expressivity vanilla PDDL could not provide: temporal planning (2.1), exogenous events at fixed times (2.2), soft goals and quality metrics (3.0), and richer object-valued state (3.1). PDDL+, developed in parallel by Fox & Long, addresses hybrid discrete-continuous domains and is used outside the IPC mainline.

Modern research papers on generalized planning, learning-to-plan, and neural planners almost always restrict themselves to the **typed STRIPS subset** of PDDL 1.2 with negative preconditions enabled. That is also the slice this project's lifted DSL targets (see §7).

## §4 The PDDL 1.2 / typed-STRIPS grammar

A *domain* file has the form:

$$
\begin{aligned}
 \texttt{domain} &::= \texttt{(define (domain}\;\mathit{name}\texttt{)}\;\;\mathit{req}^?\;\mathit{types}^?\;\mathit{preds}\;\mathit{action}^+\texttt{)} \\
 \mathit{req}   &::= \texttt{(:requirements}\;\texttt{:strips}\;\texttt{:typing}\;\texttt{:negative-preconditions}\;\ldots\texttt{)} \\
 \mathit{types} &::= \texttt{(:types}\;\mathit{name}^+\;[\texttt{- }\mathit{name}]\texttt{)} \\
 \mathit{preds} &::= \texttt{(:predicates}\;\mathit{atom}^+\texttt{)} \\
 \mathit{atom}  &::= \texttt{(}\;\mathit{name}\;(\texttt{?}\mathit{var}\;[\texttt{- }\mathit{type}])^*\texttt{)} \\
 \mathit{action}&::= \texttt{(:action}\;\mathit{name} \\
                &\phantom{::=\;}\;\;\texttt{:parameters (}(\texttt{?}\mathit{var}\;\texttt{- }\mathit{type})^*\texttt{)} \\
                &\phantom{::=\;}\;\;\texttt{:precondition }\mathit{gd} \\
                &\phantom{::=\;}\;\;\texttt{:effect }\mathit{eff}\texttt{)} \\
 \mathit{gd}    &::= \mathit{atom}\;\mid\;\texttt{(not }\mathit{atom}\texttt{)}\;\mid\;\texttt{(and }\mathit{gd}^+\texttt{)} \\
 \mathit{eff}   &::= \mathit{atom}\;\mid\;\texttt{(not }\mathit{atom}\texttt{)}\;\mid\;\texttt{(and }\mathit{eff}^+\texttt{)}
\end{aligned}
$$

A *problem* file has the form:

$$
\begin{aligned}
 \texttt{problem} &::= \texttt{(define (problem}\;\mathit{name}\texttt{)} \\
                 &\phantom{::=\;}\;\;\texttt{(:domain }\mathit{name}\texttt{)} \\
                 &\phantom{::=\;}\;\;\texttt{(:objects }(\mathit{name}\;\texttt{- }\mathit{type})^+\texttt{)} \\
                 &\phantom{::=\;}\;\;\texttt{(:init }\mathit{ground\text{-}atom}^+\texttt{)} \\
                 &\phantom{::=\;}\;\;\texttt{(:goal }\mathit{gd}\texttt{)}\texttt{)}
\end{aligned}
$$

In the precondition and goal `gd`, every atom is either a positive predicate application or its negation; conjunctions are the only connective in the STRIPS subset. The effect `eff` reads exactly the same way but is interpreted as an add/delete list: positive atoms are added to the state, negated atoms are removed.

The `:requirements` block declares which language features the domain uses. The minimum useful set for typed STRIPS is `(:requirements :strips :typing :negative-preconditions)`. Other flags include `:equality` (the `=` predicate is built-in), `:disjunctive-preconditions` (allow `or` in preconditions), `:existential-preconditions`, `:universal-preconditions`, `:quantified-preconditions` (both), and `:adl` (the union of the previous four plus conditional effects). The grammar above does not cover numeric fluents, durative actions, or PDDL+ syntax — these are surveyed in §5 but not formalised here.

## §5 Versions & extensions — survey

**Numeric fluents (PDDL 2.1).** State can include numeric quantities updated by `assign`, `increase`, `decrease` effects: e.g. `(fuel ?vehicle)` with `(decrease (fuel ?v) 5)` per move. Used for resource-bounded planning. Most classical-planning benchmarks ignore numeric fluents because forward search blows up when state is unbounded.

**Durative actions (PDDL 2.1).** Actions have a duration and overlap on a timeline; preconditions are tagged "at start / over all / at end" and effects "at start / at end". Used for temporal planning (IPC temporal track). Ignored by classical-planning work because the planner must reason about concurrency, not just sequence.

**Timed initial literals (PDDL 2.2).** Facts that become true (or false) at fixed absolute times, independent of any action. Models exogenous events. Largely confined to temporal planning.

**Derived predicates (PDDL 2.2).** Predicates defined by a rule body, computed from other predicates rather than asserted by an action. Equivalent to Datalog views over state. Useful for expressing transitive closure; ignored when the domain author can manually maintain the closure with action effects.

**Preferences (PDDL 3.0).** Soft constraints whose violation costs plan quality but does not invalidate the plan. Encodes "if you can, do X". Used in net-benefit planning; ignored when only hard goals matter.

**State-trajectory constraints (PDDL 3.0).** Temporal-logic constraints on the entire plan trajectory (`(always …)`, `(sometime …)`, `(within k …)`). Mostly used in IPC-5 trajectory-constrained tracks.

**Object fluents (PDDL 3.1).** Functions that return objects rather than numbers: e.g. `(location-of ?ball)` returns a `room`. Replaces some relational predicates with functional state. Adopted by some learning-to-plan benchmarks; many tools still parse only the predicate form.

**PDDL+ (Fox & Long, 2006).** Adds **processes** (continuous changes active while preconditions hold) and **events** (instantaneous, triggered, exogenous transitions) on top of PDDL 2.1's durative actions. Targets hybrid discrete-continuous systems (e.g. tanks filling, planes climbing). Used by a small specialised community; not part of the mainline IPC classical track.

**Bottom line for readers of this repo.** When a paper says "PDDL" without qualification it almost always means *typed STRIPS PDDL 1.2 with negative preconditions* — the slice formalised in §4 above. That is the slice the project's lifted DSL emulates (§7).

## §6 Worked example — Gripper

The Gripper domain (Koehler & Hoffmann, IPC-1) is the canonical "small but nontrivial" PDDL benchmark and the one the project's `gripper_lite` env mirrors. The original IPC files look like this.

`gripper.pddl` (the **domain**):

```pddl
(define (domain gripper)
  (:requirements :strips :typing)
  (:types room ball gripper)
  (:predicates
    (at-robby ?r - room)
    (at ?b - ball ?r - room)
    (free ?g - gripper)
    (carry ?b - ball ?g - gripper))

  (:action move
    :parameters (?from - room ?to - room)
    :precondition (at-robby ?from)
    :effect (and (at-robby ?to)
                 (not (at-robby ?from))))

  (:action pick
    :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and (at ?b ?r) (at-robby ?r) (free ?g))
    :effect (and (carry ?b ?g)
                 (not (at ?b ?r))
                 (not (free ?g))))

  (:action drop
    :parameters (?b - ball ?r - room ?g - gripper)
    :precondition (and (carry ?b ?g) (at-robby ?r))
    :effect (and (at ?b ?r)
                 (free ?g)
                 (not (carry ?b ?g)))))
```

`gripper-2balls.pddl` (a **problem**):

```pddl
(define (problem gripper-2balls)
  (:domain gripper)
  (:objects
    rooma roomb - room
    ball1 ball2 - ball
    left - gripper)
  (:init
    (at-robby rooma)
    (free left)
    (at ball1 rooma)
    (at ball2 rooma))
  (:goal (and (at ball1 roomb) (at ball2 roomb))))
```

Walkthrough mapping back to §1's component list:

- **Objects** ↔ `:objects`. Here: `rooma`, `roomb` (rooms); `ball1`, `ball2` (balls); `left` (gripper).
- **Predicates** ↔ `:predicates` in the domain. Here: `at-robby`, `at`, `free`, `carry`.
- **Initial state** ↔ `:init` in the problem. The robot starts in `rooma`, the gripper is `free`, both balls are in `rooma`.
- **Goal specification** ↔ `:goal` in the problem. Both balls end up in `roomb`.
- **Actions/Operators** ↔ `:action` blocks in the domain. Here: `move`, `pick`, `drop`, each lifted over typed parameters.

A plan is a sequence of *grounded* actions, e.g. `pick(ball1, rooma, left); move(rooma, roomb); drop(ball1, roomb, left); …`. The planner is responsible for finding this sequence; the domain and problem do not encode any search strategy.

## §7 Connection to this repo's lifted DSL

This project does **not** parse `.pddl` files. It encodes the same conceptual model — typed, lifted, STRIPS-flavored — directly in Python, and synthesises **policies** (mappings from state to action) rather than per-instance **plans** (sequences of actions). The mapping is one-to-one for the typed-STRIPS subset:

| PDDL | This repo (lifted DSL) | File |
|---|---|---|
| `:types` | type strings on `Var` (e.g. `Var("?b", "ball")`) | [lifted_dsl.py](../../../../src/alphazeropp/synthesis/lifted_dsl.py) |
| `:predicates` | `Literal` over typed variables | [lifted_dsl.py](../../../../src/alphazeropp/synthesis/lifted_dsl.py) |
| `:action` schema | `Rule.schema` + `action_args` in `PartialRule` | [lifted_derivation.py](../../../../src/alphazeropp/synthesis/lifted_derivation.py) |
| `:precondition` (state part) | `state_lits` (rule body, state side) | [lifted_dsl.py](../../../../src/alphazeropp/synthesis/lifted_dsl.py) |
| Goal literals in body | `goal_lits` (rule body, goal side; `Goal[...]` syntax) | [lifted_dsl.py](../../../../src/alphazeropp/synthesis/lifted_dsl.py) |
| `:init` of a problem | `env.get_state_atoms()` | [gripper_lite/env.py](../../../../src/alphazeropp/instances/gripper_lite/env.py) |
| `:goal` of a problem | `env.get_goal_atoms()` | [gripper_lite/env.py](../../../../src/alphazeropp/instances/gripper_lite/env.py) |
| Plan (sequence of grounded actions) | `interpret(policy, state, goal, objs, legal) → GroundAction`, applied step-by-step | [lifted_interpreter.py](../../../../src/alphazeropp/synthesis/lifted_interpreter.py) |

Two structural differences from textbook PDDL are worth flagging.

**Plans vs policies.** PDDL planners search for a *plan* — a finite sequence of grounded actions solving one problem. This repo synthesises a *lifted policy* — an ordered list of rules with first-applicable semantics that, when dispatched at every step, solves every problem in a family (Gripper-lite for any $B$). The synthesis search space is therefore over decision-list programs, not over action sequences.

**Goals inside rule bodies.** A PDDL action's precondition is a conjunction of literals over the *current state*. A lifted-DSL rule's body has two sides: `state_lits` (over the current state) **and** `goal_lits` (over the goal), with `goal_lits` written as `Goal[...]` in source. This is what lets a single policy generalise across problems with different goals: a rule can fire only when the state matches *and* the goal demands a particular configuration. PDDL achieves the same effect by re-running the planner per problem; lifted policies bake the goal dependency into the rule.

**The Doors instance.** This repo's Doors domain has a "PDDL-lite" wrapper ([doors_pddl_lite.py](../../../../src/alphazeropp/instances/doors/doors_pddl_lite.py)) that exposes a predicate-and-goal interface mimicking the §1 component list (typed objects, predicates, init, goal, action schemas) — without a parser and without a `.pddl` file on disk. Gripper-lite reuses the same convention. The lifted DSL of [src/alphazeropp/synthesis/](../../../../src/alphazeropp/synthesis/) consumes either env identically.

## §8 References

- Fikes, R. E., & Nilsson, N. J. (1971). *STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving*. Artificial Intelligence, 2(3–4), 189–208.
- McDermott, D., Ghallab, M., Howe, A., Knoblock, C., Ram, A., Veloso, M., Weld, D., & Wilkins, D. (1998). *PDDL — The Planning Domain Definition Language*. Technical Report CVC TR-98-003, Yale Center for Computational Vision and Control. (AIPS-98 competition spec.)
- Fox, M., & Long, D. (2003). *PDDL2.1: An Extension to PDDL for Expressing Temporal Planning Domains*. JAIR, 20, 61–124.
- Edelkamp, S., & Hoffmann, J. (2004). *PDDL2.2: The Language for the Classical Part of the 4th International Planning Competition*. Technical Report 195, Institut für Informatik, Freiburg.
- Gerevini, A., & Long, D. (2005). *Plan Constraints and Preferences in PDDL3*. Technical Report, University of Brescia.
- Fox, M., & Long, D. (2006). *Modelling Mixed Discrete-Continuous Domains for Planning*. JAIR, 27, 235–297. (PDDL+.)
- Helmert, M., Do, M., & Refanidis, I. (2008). *Deterministic Track Description of IPC-6* (PDDL 3.1 object fluents).
- IPC archive of domains and reference planners: <https://www.icaps-conference.org/competitions/>.
