# Appendix — Syntax-Guided Synthesis (SyGuS) Background

**Source:** [sygus_discussion.tex](../../presentations/sygus_discussion.tex) (~12 frames).

This note summarizes the SyGuS formalism, its canonical solver (CEGIS), its scaling variant (divide-and-conquer), PBE as a restricted SyGuS, and a property-by-property comparison with the Doors problem. It is the conceptual backdrop for Stages 1–3.

## 1. SyGuS: the problem

**Goal:** find an expression $e$ satisfying *both* a semantic and a syntactic constraint.

| | Semantic constraint — *what* $e$ must do | Syntactic constraint — *how* $e$ may be built |
|---|---|---|
| Form | Logical specification $\Phi$ (SMT formula) | Context-free grammar $G$ |
| Role | Correctness requirement on all inputs | Restricts the form of candidates |
| Relation to $e$ | $e \models \Phi$ | $S \to_G^* e$ |

A **SyGuS instance** is a pair $\langle \Phi, G \rangle$. A **solution** is an expression $e \in \llbracket G\rrbracket$ such that $e \models \Phi$.

### Formal definition

- $\Phi$ is an SMT formula over the function to synthesize and universally quantified inputs. Substituting $e$ for the unknown function makes $\Phi$ valid.
- $G = \langle \mathcal N, S, \mathcal R\rangle$ with nonterminals $\mathcal N$, start symbol $S$, and production rules $\mathcal R$.

Reference: Alur et al., FMCAD 2013 (original formulation); Padhi et al., *SyGuS Language Standard v2.1*, 2021.

### Worked example — synthesizing $\max(x, y)$

Grammar $G$ (linear integer arithmetic):
$$
\begin{aligned}
 S &::= T \;\mid\; \texttt{if}\;C\;\texttt{then}\;T\;\texttt{else}\;T \\
 T &::= 0 \;\mid\; 1 \;\mid\; x \;\mid\; y \;\mid\; T + T \\
 C &::= T \leq T \;\mid\; C \wedge C \;\mid\; \neg\, C
\end{aligned}
$$

Specification $\Phi$:
$$
\forall x, y:\; f(x,y) \geq x \;\wedge\; f(x,y) \geq y \;\wedge\; (f(x,y) = x \vee f(x,y) = y).
$$

Solution: $f(x,y) \equiv \texttt{if}\;x \leq y\;\texttt{then}\;y\;\texttt{else}\;x$. Derivable from $G$ ✓, satisfies $\Phi$ on all inputs ✓.

In SyGuS v2.1 concrete syntax, the problem is declared with a `synth-fun` command that pairs the function signature with its grammar.

### Why both constraints

1. **Spec alone is under-determined.** Many programs satisfy $\Phi$; the grammar focuses the solver.
2. **Grammar alone is too large.** $\llbracket G\rrbracket$ is infinite; the spec prunes semantically.
3. **Together:** grammar bounds syntax, spec bounds semantics. Solvers enumerate from $G$ in size order and verify against $\Phi$ via an SMT oracle.

## 2. Solving SyGuS: enumerative synthesis + CEGIS

### Basic enumerative algorithm (Alur, Radhakrishna, Udupa — TACAS 2017, Algorithm 1)

```
pts ← ∅
loop:
  for e ∈ Enumerate(G, pts):        # size order, observational-equivalence pruned
    if e ⊭ Φ↓pts: continue
    cexpt ← Verify(e, Φ)
    if cexpt = ⊥: return e           # verified correct on all inputs
    pts ← pts ∪ {cexpt}
```

Key ideas:
- **Maintain concrete input points $\mathrm{pts}$.** Initially empty; grown by counterexamples.
- **Enumerate** expressions in size order, skipping any $e$ that is *observationally equivalent* on $\mathrm{pts}$ to one already seen. Only one representative per equivalence class is kept.

### CEGIS — Counter-Example Guided Inductive Synthesis

Two-phase loop:

1. **Check** a candidate $e$ against $\mathrm{pts}$ (cheap concrete evaluation). If it passes all known points, call the SMT verifier.
2. **Verify** via the SMT solver (expensive). If the verifier returns $\bot$, $e$ is correct on *all* inputs — done. Otherwise, add the counterexample to $\mathrm{pts}$ and continue.

```
Learner (enumerator) ──── candidate e ────▶ Verifier (SMT)
Learner ◀──── counterexample or ⊥ ─────── Verifier
```

Most candidates are rejected cheaply by concrete evaluation; the expensive SMT call is rare.

### Enumerated vs. candidate expressions

- **Enumerated expressions:** all derivable from $G$ up to the current size, after observational-equivalence pruning on $\mathrm{pts}$.
- **Candidate expression:** the first enumerated expression that satisfies $\Phi$ on *all* current $\mathrm{pts}$. Only the candidate is sent to the SMT verifier.

### Combinatorial cost

Even with pruning, basic enumeration is **exponential in solution size**. For $\max(x,y)$ (size 6), enumeration blows up over thousands of smaller expressions before reaching the solution.

## 3. Divide and conquer (TACAS 2017)

**Observation.** The $\max$ solution has size 6, but its components are small: terms $x, y$ (size 1), predicate $x \leq y$ (size 3). Largest component = 3, not 6. Enumeration cost is exponential in component size — so this saves a huge amount.

**Algorithm.**
1. **Enumerate terms** from the $T$ nonterminal until they *cover* all $\mathrm{pts}$ (for every pt, some term gives the correct output).
2. **Enumerate predicates** from the $C$ nonterminal.
3. **Combine** terms and predicates into a conditional expression by learning a decision tree: predicates as branch conditions, terms as leaves.

**Why this works.** The basic enumerator must build the full `if C then T else T` as a single size-6 expression. D&C enumerates only up to size 3 (largest component) and assembles the pieces.

Execution trace on $\max(x,y)$ reaches the canonical solution in 4 CEGIS rounds using ~15 enumerated expressions total (4 terms + 11 predicates), vs. hundreds for the basic enumerator.

## 4. PBE — Programming by Examples

**What if we don't have a logical spec?** In many domains, the easiest specification is concrete input/output examples.

PBE restricts SyGuS specifications to conjunctions of I/O equalities:
$$
\Phi \;\equiv\; \bigwedge_{i=1}^m f(\vec c_i) = d_i \qquad \text{where } \vec c_i, d_i \text{ are constants}.
$$

| | General SyGuS | PBE |
|---|---|---|
| Specification | Arbitrary SMT formula $\Phi$ | Conjunction of I/O equalities |
| Verifier | SMT solver needed | Evaluate on examples |
| Quantifiers | $\forall$ over inputs | None |

Each I/O pair constrains $f$ **independently** on one point. In the SyGuS v2.1 standard, the logic is declared as $\texttt{PBE\_}X$ for a base theory $X$ (e.g. $\texttt{PBE\_SLIA}$ for Strings + Linear Integer Arithmetic). No quantifiers, no logical connectives beyond $\wedge$ and $=$.

### Example — string concatenation

I/O examples (the entire specification):

| `fname` | `lname` | $f(\texttt{fname}, \texttt{lname})$ |
|---|---|---|
| `"Nancy"` | `"FreeHafer"` | `"Nancy FreeHafer"` |
| `"Andrew"` | `"Cencici"` | `"Andrew Cencici"` |
| `"Jan"` | `"Kotas"` | `"Jan Kotas"` |
| `"Mariya"` | `"Sergienko"` | `"Mariya Sergienko"` |

Grammar (SyGuS v2.1 `synth-fun`):
$$
\begin{aligned}
 y\_\mathrm{str} &::= \texttt{" "} \mid \texttt{fname} \mid \texttt{lname} \mid \texttt{str.++}\;y\_\mathrm{str}\;y\_\mathrm{str} \mid \ldots \\
 y\_\mathrm{int} &::= 0 \mid 1 \mid 2 \mid \texttt{str.len}\;y\_\mathrm{str} \mid \ldots
\end{aligned}
$$

Solution: $f = \texttt{str.++}(\texttt{fname},\; \texttt{str.++}(\texttt{" "},\; \texttt{lname}))$.

**Key property.** Each I/O pair constrains $f$ pointwise and exactly — no approximation, no reward shaping.

## 5. Doors vs. SyGuS — where it fits, where it breaks

### The synthesis conjecture, reformulated

| | Standard SyGuS | Doors (current formulation) |
|---|---|---|
| Formulation | $\exists f_1, \ldots, f_n.\; \forall v_1, \ldots, v_m.\; \alpha \Rightarrow \varphi$ | $\max_\pi \mathbb E\!\left[\sum_t \gamma^t r(s_t, \pi(s_t))\right]$ |
| Constraint on policy | Grammars $G_i$ | Grammar $G$ ✓ |
| Correctness | **Hard** — logical formula on *all* inputs | **Scalar reward** — compressed, approximate |
| Feedback | Counterexample $\vec v^\ast$ | Rollout return $R \in \mathbb R$ |

### Property-by-property comparison

| Property | SyGuS | Doors (current) |
|---|---|---|
| Syntactic constraint (grammar) | ✓ | ✓ |
| Semantic constraint (logical formula) | ✓ | ✗ (scalar reward) |
| Specification type | Logical formula / PBE equalities | Reward function |
| Pointwise separability | ✓ | ✗ (transition system) |
| Exact counterexamples (CEGIS) | ✓ | ✗ (reward desert) |

Three of four SyGuS properties fail in the current Doors formulation. But Doors has a crucial advantage: **it is finite and deterministic**.

### The modeling mismatch

Doors is a deterministic, finite-state transition system $M = (\mathcal S, \mathcal A, T, r, H)$ with $T: \mathcal S \times \mathcal A \to \mathcal S$ deterministic and $\lvert\mathcal S\rvert < \infty$. A policy $\pi: \mathcal S \to \mathcal A$ generated by grammar $G$ induces a trajectory $s_0, \pi(s_0), s_1, \pi(s_1), \ldots, s_H$. Correctness means reaching the goal — a **reachability** property, not a pointwise I/O relation.

Two breakdowns:

- **CEGIS breaks.** A low-reward rollout compresses many mistakes (wrong key, wrong room, wrong order) into one scalar. No structural feedback.
- **D&C is inapplicable.** Sequential dependence ($s_{t+1} = T(s_t, a_t)$) means guards and actions at step $t$ cannot be verified independently of earlier steps.

**Conclusion.** The SyGuS formulation does not naturally fit Doors. The key SyGuS mechanisms (CEGIS, divide-and-conquer, pointwise verification) all break down for sequential transition systems. This is why the current codebase substitutes MCTS + neural value bootstrapping for the SMT+enumeration pair.

## Why this matters for Stages 1–3

- **Stage 1** diagnoses why AlphaZero doesn't converge on Doors even when MCTS finds solutions. The problem is the scalar-reward feedback — exactly the property CEGIS needs and Doors lacks.
- **Stage 2** shows the bootstrap problem when the grammar is unmasked: without a sharp prior, sparse terminal reward is insufficient to guide search. Compare to SyGuS's dense counterexamples.
- **Stage 3** reintroduces structural bias via *typing* (a SyGuS-style restriction on the grammar), but still uses rollout reward rather than logical specifications. See [../stage3/03_connection_to_pg3.md](../stage3/03_connection_to_pg3.md) for why we cannot adopt PG3's planning-based scorer.

## References

- R. Alur et al., *Syntax-Guided Synthesis*, FMCAD 2013.
- R. Alur, A. Radhakrishna, A. Udupa, *Scaling Enumerative Program Synthesis via Divide and Conquer*, TACAS 2017.
- S. Padhi et al., *The SyGuS Language Standard Version 2.1*, arXiv:2312.06001, 2023.
- S. A. Seshia, *Syntax-Guided Synthesis — EECS 219C Tutorial*, adapted from Alur FMCAD'13.
