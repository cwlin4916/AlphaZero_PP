# Project Status Overview

**Date:** 2026-04-02
**Branch:** `feature/grammar-redesign`
**Authors:** Gabriel Konar-Steenberg, Peter Graf (Alliance for Energy Innovation / US DOE NREL)

---

## 1. Project Overview

AlphaZero_PP is a **single-player reimplementation of AlphaZero** designed for program synthesis research. The core idea is to cast grammar-guided program derivation as a single-player game and solve it with MCTS + neural networks.

**Primary application domain:** The **Doors** environment, a multi-room navigation task with key-door dependencies, serves as a testbed for infrastructure planning. The system synthesizes reactive policies (programs) that map observations to actions, rather than learning a monolithic neural policy.

**Key insight driving the work:** Standard AlphaZero assumes intermediate states carry predictive value and rewards are differentiated. Program synthesis violates both assumptions — partial ASTs have no predictive value, and 99% of completed programs score identically. This tension is the central research challenge.

---

## 2. Architecture

### Core Modules (`src/alphazeropp/`)

| Module | Purpose |
|--------|---------|
| `core/` | MCTS, Agent, Game abstraction, PolicyValueNet, Config |
| `training/` | Self-play training loop, evaluator, gated acceptance |
| `synthesis/` | AST nodes, budget grammar, derivation game, interpreter, leaf evaluator |
| `instances/` | Game implementations: Doors, Bitstring, CartPole |
| `benchmark/` | Algorithm comparison harness (AlphaZero, PPO, DQN, Q-learning, oracle, random) |
| `utils/` | Checkpointing, multiprocessing, derivation utilities, statistics |

### Game Domains

1. **Doors** (primary) — Multi-room planning with keys, doors, and a goal. Configurable difficulty via D (number of rooms). Optimal policy length = 2(D-1)+1 steps.
2. **Bitstring** — Maximize ones in a bit vector. Used for synthesis experiments with scan grammar.
3. **CartPole** — Standard Gymnasium control task. Baseline for direct AlphaZero.

### DSL Layer Stack (Doors)

Five derivation approaches, ordered from low-level to high-level abstraction:

| Layer | Approach | Search Space | Key Property |
|-------|----------|-------------|--------------|
| **Surface** | Rule-ordering over domain macros (PickRule, MoveRule, GoalRule) | 2K+1 rules, ~15 actions | Eliminates dead ends; 50%+ solve at D=3 |
| **Stage** | Guard/action with hierarchical derivation + relational queries | Variable | Structured decomposition |
| **Lifted** | D-independent policy compilation via relational runtime | Constant (independent of D) | Generalizes across problem sizes |
| **Reactive Sketch** | Tick-based behavior tree interpreter | Fixed skeleton | Reactive execution model |
| **Reactive Typed** | Full typed grammar with AlphaZero integration | Low branching, semantic rules | Best structural match to domain |

---

## 3. Development Timeline

### Phase 1: Foundation (Aug 2024 – Feb 2025)
- Initial AlphaZero implementation with line-extending game
- Basic game-gym interface, MCTS, training loop
- **Zoning Game** — grammar-based infrastructure planning prototype with CFG rules and distance constraints

### Phase 2: Multi-Game & Infrastructure (Jul 2025 – Jan 2026)
- Added **CartPole** (Jul 2025) and **Bitstring** (Jul–Aug 2025) domains
- Multiprocessing parallelization for game collection
- Determinism enforcement, GPU utilization improvements
- MCTS enhancements: configurable backup strategies (mean, max, top-k, softmax)

### Phase 3: Unified Framework Refactor (Jan – Feb 2026)
- Modular `src/` package structure with `instances/` folder
- Generic synthesis logic extraction from game-specific code
- Interactive config editor and CLI improvements
- Multi-seed training capability

### Phase 4: Doors & Grammar Redesign (Feb – Mar 2026)
- **Doors environment** implementation with oracle baseline, state enumeration, PDDL-Lite integration
- **Direct play** experiments (D=2 through D=14)
- **5 DSL layers** implemented and compared
- **Diagnostic investigations** identifying why AlphaZero fails at synthesis
- **Small-scale verification** confirming AlphaZero correctness and establishing compute thresholds
- Comprehensive test coverage (28+ test files)

---

## 4. Report & Experiment Catalog

### Spec Reports (`spec/`)

| Date | Report | Type | Key Finding |
|------|--------|------|-------------|
| 2026-03-20 | `report_doors_synthesis_full_context` | Reference | Self-contained diagnostic context: D=2 solved reliably, D=3 unreliable (~50% with macros, 0% flat grammar) |
| 2026-03-21 | `report-surface-derivation-explained` | Specification | Surface derivation reduces search space from 200-300 productions to 2K+1 rules; eliminates dead ends |
| 2026-03-22 | `report-reactive-typed-grammar` | Specification | Behavior tree representation shifts search from syntactic AST expansion to semantic rule ordering |
| 2026-03-26 | `report-alphazero-synthesis-diagnostic` | Diagnostic | AlphaZero is correctly implemented but **every domain assumption is violated** for synthesis: no intermediate value signal, flat reward landscape, Q-values remain uniform |
| 2026-03-26 | `report-doors-direct-D14-failure-diagnosis` | Root Cause | D=14 failure caused by **data starvation** (20 games/iter insufficient for 40K-param network), not code bugs |
| 2026-03-27 | `plan-doors-direct-small-scale-verification` | Experimental Plan | 13 systematic runs designed across D=2-6 to reproduce failures at small scale |
| 2026-03-27 | `report-doors-direct-small-scale-verification` | Results | All 13 runs behaved as predicted. Three regimes: comfortable (sims/K >= 1.5), fragile (~1.25), slow bootstrap (~1.0) |

### Learning Notes (`spec/learning/`)

| Date | Report | Topic |
|------|--------|-------|
| 2026-03-06 | `multiprocessing_and_pickle` | Worker pool dynamics, pickle serialization, cache inflation bug (factor = 1 + num_tasks), fix via `__getstate__` |

### Design Documents (`docs/`)

| Date | Document | Purpose |
|------|----------|---------|
| 2026-03-09 | `ablation_semantics` | Verified semantic map of MCTS edge cases: 10 confirmed, 3 broken assumptions (n_sims=0, pickle, temperature) |
| 2026-03-09 | `doors_env_audit` | BFS-verified environment correctness for D=2,3,5,10,20,50 |
| — | `benchmark_protocol` | Standardized evaluation: solve = 95% success AND 95% oracle return for 3 consecutive checkpoints |
| — | `doors_benchmark_interface` | Action space cleanup: removed padding for fair comparison |
| — | `doors_relational_semantics` | Typed compile-time query interface for lifted DSL compiler |

### Algorithm Specifications (`docs/specs/`)

| Date | Document | Purpose |
|------|----------|---------|
| 2026-03-09 | `alphazero_ablation_plan` | Design for isolating network vs MCTS contribution (frozen, uniform-policy, zero-value wrappers) |
| 2026-03-09 | `refined-plan-ablation-semantics` | Ablation infrastructure validated: 24 test cases across 11 classes, all pass |
| — | `direct_play_algorithm_comparison` | Catalog of 5 algorithms (Q-learning, DQN, PPO, AlphaZero, SAC) with scaling expectations |
| — | `grammar_search_algorithm_comparison` | Catalog of 8 synthesis algorithms; diagnoses 5 bottlenecks in AlphaZero for synthesis |

### Experimental Results (`presentations/`, `experiments/`)

| Date | Experiment | Key Result |
|------|-----------|------------|
| 2026-03-26 | D=8/D=10 Surface vs Grammar comparison | Surface: 100% solve at both D=8 and D=10. Grammar: 0% at both. Surface is 10-20x faster per iteration. |
| 2026-03-22 | Reactive BT D=2 proof of concept | 100% solve, optimal reward (+1.07), 3 steps, with only 5 MCTS sims and 2 training iterations |

---

## 5. Key Findings

### AlphaZero Direct Play: Verified Correct
- D=2 through D=6 solved reliably with appropriate hyperparameters
- **Critical thresholds:** sims/K >= 1.56 (sharp phase transition), games >= 10 for D=4
- D=14 failure was data starvation (20 games insufficient), not a bug
- Catastrophic forgetting occurs under low-data regime (spike at iter 9, collapse at iter 10)

### AlphaZero for Synthesis: Structurally Mismatched
Five assumption violations identified:
1. **No intermediate value signal** — partial ASTs carry no predictive information
2. **Sparse reward** — terminal-only, in a space of ~10^30 programs (D=3, budget=34)
3. **Flat reward landscape** — 99% of completed programs score identically (~-0.075)
4. **Q-values remain uniform** — UCB becomes pure exploration; exploitation never kicks in
5. **Neural network learns nothing useful** — MCTS noise is the sole solver

### Surface Derivation >> Grammar Derivation
- At D=8 and D=10: surface achieves 100% solve rate, grammar achieves 0%
- Surface value loss increases (learning happening); grammar value loss stays ~0 (no reward diversity)
- Surface is 10-20x faster per training iteration

### Reactive Typed Grammar: Promising Direction
- Proof of concept at D=2: solves with minimal compute (5 sims, 3 games, 2 iters)
- Shifts search from syntactic AST expansion to semantic rule ordering
- Fixed skeleton exploits domain structure; search focuses on meaningful choices

---

## 6. Open Questions

1. **Does the neural network contribute to synthesis?** The ablation plan (2026-03-09) was designed but results are not yet documented. The synthesis diagnostic (2026-03-26) suggests MCTS noise alone drives solving.

2. **Can reactive typed grammar scale beyond D=2?** The proof of concept works, but larger instances remain untested.

3. **What is the right search algorithm for synthesis?** The grammar search comparison spec catalogs 8 alternatives, but systematic comparison experiments have not been run.

4. **Can lifted DSL achieve D-independent generalization?** The relational runtime is specified, but generalization across problem sizes is unverified.

5. **Benchmark suite completion.** The benchmark protocol and algorithm comparison specs exist, but the full standardized comparison (AlphaZero vs PPO vs DQN vs Q-learning vs SAC on Doors direct play) has not been executed at scale.
