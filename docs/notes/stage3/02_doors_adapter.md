# Stage 3 / Design — The Domain Adapter Interface

## Overview

For each PDDL-style domain $\mathcal D$, a **domain adapter** exposes a small relational vocabulary plus an observation parser. Everything else — CFG generation, the relational interpreter, MCTS, the training loop — is generic and unchanged across domains.

**The adapter is the only domain-specific contract.** Same codebase + different adapter file = different domain.

## Interface

| Component | Symbol | Example (Ferry) |
|---|---|---|
| Types | $\mathcal T$ | $\{\textit{car}, \textit{location}\}$ |
| Predicate signatures | $\mathcal P$ | $\texttt{at}/2, \texttt{on}/1, \texttt{empty\_ferry}/0$ |
| Static predicates | $\mathcal P_{\mathrm{static}}$ | Instance-fixed relations (e.g. $\texttt{key\_unlocks}/2$ in Doors) |
| Action schemas | $\mathcal A$ | $\texttt{board}(c\!:\!\textit{car}, \ell\!:\!\textit{loc})$, $\texttt{sail}(\ell)$, $\texttt{debark}(c, \ell)$ |
| Observation parser | — | $\texttt{obs} \mapsto (S, O, G)$ |

where $S$ = true state literals (dynamic), $O$ = typed objects, $G$ = goal literals. Static predicates $\mathcal P_{\mathrm{static}}$ are instance-fixed and available to guards without appearing in $S$.

## Notation

The lifted grammar uses four notational pieces that the rest of the design files refer to without re-defining.

**Type set.** $T$ is a finite set of object types. For Doors, $T = \{\textit{room}, \textit{key}, \textit{loc}\}$.

**Per-type object sets.** Each instance partitions $O$ into per-type subsets $O_\tau$ for $\tau \in T$. For Doors at $D = 3$:
$$
O_{\textit{room}} = \{r_0, r_1, r_2\}, \qquad O_{\textit{key}} = \{k_0, k_1\}.
$$

**Ground atom vs lifted literal.**
- A *ground atom* names concrete objects: $\textit{key\_for}(k_0, r_1)$.
- A *lifted literal* uses typed variables: $\textit{key\_for}(k\!:\!\textit{key},\, r\!:\!\textit{room})$. The annotation $k\!:\!\textit{key}$ means "any object of type $\textit{key}$" — not "the key named $k$".

**Typed binding.** A binding (substitution) is a map $\theta : V \to O$ from a rule's variables $V$ to objects, satisfying the type-respecting condition
$$
\theta(x_i) \in O_{\textit{type}(x_i)} \quad \text{for every } x_i \in V.
$$
Applying $\theta$ to a literal grounds it: $\textit{key\_for}(k, r)\,\theta = \textit{key\_for}(\theta(k), \theta(r))$.

**Why typing compresses.** At $D = 3$, the lifted literal $\textit{key\_for}(k\!:\!\textit{key},\, r\!:\!\textit{room})$ has $\lvert O_{\textit{key}}\rvert \times \lvert O_{\textit{room}}\rvert = 2 \times 3 = 6$ candidate bindings, not $\lvert O\rvert^2 = 5^2 = 25$ untyped pairs. This is the structural compression that lifting buys at every literal.

[01_lifted_grammar.md](01_lifted_grammar.md) refers to $\theta$ in execution semantics and the four structural bounds $L_{\max}, K_s, K_g, V_{\max}$ build on top of this notation.

## Role of each component

**Types $\mathcal T$.** Object categories (e.g. *car*, *location*). Determines which variables bind to which objects — a *car* variable never binds to a location. The foundation of **type-safe binding**.

**Predicate signatures $\mathcal P$.** Relations over typed objects describing the world state. Arity matters: $\texttt{at}/2$ takes two arguments, $\texttt{empty\_ferry}/0$ takes none. These become the **atoms in rule guards**.

**Static predicates $\mathcal P_{\mathrm{static}}$.** Instance-fixed relations that never change during execution (e.g. $\texttt{key\_unlocks}(k, r)$ in Doors). Available to guards but **not part of the dynamic observation** $S$. Stored as compile-time constants.

**Action schemas $\mathcal A$.** Parameterized actions the agent can take. Arguments must be bound from guard variables (**action vars ⊆ guard vars**, enforced at grammar-generation time).

**Observation parser.** Converts the Gymnasium environment's raw observation (typically a flat `float32` vector) into the relational triple $(S, O, G)$. This is the **bridge** between the environment and the lifted grammar. Every adapter provides a `parse_obs` method.

## Concrete example — Ferry adapter

```
Types:      car, location
Predicates: at(c: car, ℓ: loc)
            at_ferry(ℓ: loc)
            on(c: car)
            empty_ferry()          # 0-ary
Actions:    board(c: car, ℓ: loc)
            sail(ℓ: loc)
            debark(c: car, ℓ: loc)
```

A 3-rule lifted policy for Ferry:

```
ρ₁: if at_ferry(ℓ) ∧ at(c, ℓ) ∧ goal:at(c, d) ∧ ¬at(c, d)   then board(c, ℓ)
ρ₂: if on(c) ∧ at_ferry(ℓ) ∧ goal:at(c, d) ∧ ¬at_ferry(d)   then sail(d)
ρ₃: if on(c) ∧ at_ferry(d) ∧ goal:at(c, d)                  then debark(c, d)
```

The same three rules handle any number of cars. PG3 reports **perfect solved fraction** on Ferry with lifted decision-list policies of this form.

## Concrete example — Doors adapter

```
Types:      key, room, loc
Predicates: at_agent(ℓ: loc)                                (dynamic)
            locked(r: room)                                 (dynamic)
            key_available(k: key)                           (dynamic)
            key_for(k: key, r: room)                        (static)
            loc_of(k: key) → loc                            (static relation)
Actions:    pick(k: key)
            move_to(ℓ: loc)
```

A lifted Doors policy (any $K$):

```
ρ₁: if key_available(k) ∧ key_for(k, r) ∧ locked(r)
      then pick(k)
ρ₂: if locked(r) ∧ key_for(k, r) ∧ ¬key_available(k)
      then move_to(loc_of(k))
default: move_to_goal
```

No index appears. If a flat reactive decision list transfers on Doors (Exp 2), the same two rules suffice for any $K$; if it does not, `ForEach` is the planned fallback (see [01_lifted_grammar.md](01_lifted_grammar.md)).

## What already exists in the repo

Doors has a partial lifted layer; the Stage 3 work is to **generalize these in place**, not build a parallel stack.

| File | Lines | What it does |
|---|---:|---|
| `instances/doors/dsl/lifted_dsl.py` | 267 | Typed lifted AST: `KeyFor`, `LocOf`, `Pickable`, `NeedKey`, `ForEachLockedRoom`, `LiftedPolicy` |
| `instances/doors/dsl/lifted_compiler.py` | 132 | Compiles a `LiftedPolicy` to a flat surface program by iterating over lockable rooms |
| `instances/doors/dsl/relational_runtime.py` | 259 | `DoorsRelationalRuntime`: typed objects, dynamic literals |
| `instances/doors/dsl/reactive_typed_grammar.py` | 166 | Enumeration / counting / evaluation of typed reactive policies via a `ReactiveBranchCatalog` |

Planned refactor:
- Extract `lifted_dsl.py` into a generic core + Doors-specific subclass.
- Abstract `relational_runtime.py` into a `DomainAdapter` protocol that the existing Doors runtime *implements*.
- Treat `ForEachLockedRoom` as a *prototype to evaluate* against the flat reactive decision list; promote to a general `ForEach(τ, rules)` only if Exp 2 requires it.

## Codebase impact map

| Module / layer | Status | Rationale |
|---|---|---|
| `derivation_game.py` | Reused | Grammar derivation as MCTS game — logic unchanged |
| MCTS + AlphaZero loop | Reused | Search and self-play are grammar-agnostic |
| Checkpoint / evaluation | Reused | Save/load and metric logging unchanged |
| `derivation.py` | Modified | `DerivationState` extended for typed variable pools |
| `ast_nodes.py` | Modified | New node types for predicates, guards, lifted actions |
| **Lifted grammar generator** | **New** | Enumerates productions from $\mathcal T, \mathcal P, \mathcal A$ |
| **Relational interpreter** | **New** | Evaluates lifted rules against $(S, O, G)$ observations |
| **Variable binding engine** | **New** | Type-safe existential binding with lex tie-break |
| **Domain adapters** | **New** | Doors, Gripper, Ferry (≈100–200 lines each) |
| `grammargame.py` (bitstring) | Superseded | Flat bitstring grammar — Stage 1/2 only |
| `unmasked_surface_cfg.py` | Superseded | Domain-specific surface CFG — replaced by generic schema |

## Takeaways

1. The adapter interface is small and strictly typed: 4 declarations + 1 parser function.
2. Static predicates are a first-class concept — crucial for Doors (`key_for(k, r)`) and likely for other domains with instance-specific geometry.
3. Custom Gymnasium environments are assumed throughout; no PDDLGym dependency. Each adapter provides `parse_obs` to convert flat observations into relational form.
4. The lifted policy is written over variables and predicates, not specific objects — guaranteeing cross-instance portability when the domain has per-object goal literals.

## Source slides

Frames 17–18 (domain adapter interface, unpacking), frames 30 (Keys and Doors: Ground to Lifted), frame 38 (Ferry concrete policy), frame 43 (what changes in the codebase), frame 44 (Doors already has a lifted layer).
