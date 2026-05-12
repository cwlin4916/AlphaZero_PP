# Research proposals: lifted-policy program synthesis with AlphaZero MCTS

## Proposal 1 — Planner-free AlphaZero search for lifted decision-list policies

### 1. Question

Can a generic AlphaZero-style MCTS over a typed lifted-policy grammar synthesize PG3-style decision-list policies that generalize across object counts using only episodic execution reward at complete-program leaves?

### 2. Methodology sketch

Use small relational planning curricula drawn from doors, gripper, and a small set of standard PDDL generalized-planning domains with typed predicates, action schemas, and held-out larger instances. Implement a lifted decision-list grammar with typed variables, state literals, goal literals, actions, flat negation, and controlled auxiliary-variable introduction. The search method is a grammar derivation game: MCTS actions expand the leftmost hole, terminal programs are executed by an online unification interpreter, and a learned policy/value model is trained from completed derivations and rewards. Evaluation compares synthesized policy success on training instances, held-out same-size instances, and held-out larger instances; synthesis cost; policy size; and dispatch cost. The comparison points are PG3-style generalized policy search, unguided grammar MCTS, and simple hand-specified lifted-policy templates where available.

### 3. Literature justification

This builds directly on the generalized-planning framing: generalized policies are the target object in [Khardon1999], [MartinGeffner2004], [SrivastavaImmermanZilberstein2011], and [YangEtAl2022PG3]. It keeps PG3’s lifted decision-list representation but replaces PG3’s planner-guided scoring and policy-edit operators with grammar-level derivation search [YangEtAl2022PG3]. The search mechanism comes from MCTS/AlphaZero [KocsisSzepesvari2006], [SilverEtAl2018AlphaZero] and from syntax-guided/neural-guided synthesis [AlurEtAl2013SyGuS], [ParsertPolgreen2024].

### 4. Why realistic

The draft already has the hard outer loop: grammar-agnostic `DerivationGame`, terminal leaf evaluation, caching, and episodic execution metrics. The new work is bounded to a typed grammar, an online unification interpreter, and a small number of relational domains. A 3–4 month prototype can answer feasibility on doors and gripper; a 6 month paper-scale study can include PG3-style baselines and additional domains. The bottleneck is not MCTS infrastructure but reward sparsity and production fan-out: many complete policies will score zero unless curricula and grammar caps keep partial search informative.

### 5. Why top-journal-worthy

The contribution is a planner-free, grammar-generic generalized-policy synthesis method: lifted relational programs are discovered by AlphaZero search rather than by planner-specific policy edits or LLM prompting. The closest comparable paper is [YangEtAl2022PG3] at IJCAI 2022. The delta is precise: PG3 evaluates candidate lifted policies by policy-guided planning over a STRIPS model, while this proposal evaluates them by direct execution reward and learns a reusable search policy over grammar derivations. The closest synthesis-side comparable is [ParsertPolgreen2024] at AAAI 2024, but that work targets SyGuS logical specifications, not relational generalized-planning policies with cross-instance object-count extrapolation.

## Proposal 2 — Expressiveness and search-cost calculus for lifted decision-list grammars

### 1. Question

What is the smallest typed lifted decision-list grammar that can express the generalized policies needed by common relational planning domains, and what search branching cost is paid for each added language feature?

### 2. Methodology sketch

Define a family of grammar fragments: action-parameter-only rules; auxiliary-variable rules; positive-only versus flat-negated literals; separate `GOAL` fields versus `Goal(...)` wrappers; and bounded numbers of rules and literals. For each fragment, give formal semantics under first-applicable unification and canonicalize α-equivalent variable names. Analyze expressiveness by proving inclusions, constructing minimal policies for representative domain schemas, and proving non-expressibility for fragments that cannot represent needed relational chains or goal tests. Analyze search cost by deriving production fan-out bounds as functions of predicate arity, type counts, variable scope, and literal budget. Evaluation is by coverage of known policy templates and by analytic branching/canonicalization results, not by a hyperparameter grid.

### 3. Literature justification

The representation starts with first-order decision lists [MooneyCaliff1995] and their use in learning action strategies [Khardon1999]. The generalized-policy target follows [MartinGeffner2004], [FrancesBonetGeffner2021], and [YangEtAl2022PG3]. The grammar perspective follows syntax-guided synthesis [AlurEtAl2013SyGuS], but the object of study is a lifted relational policy language rather than an arithmetic or string expression language.

### 4. Why realistic

This is mostly a symbolic and combinatorial project. Doors and gripper are small enough to make the key distinctions crisp: doors stresses typed relational chains and negation, while gripper stresses goal-conditioned object quantification. The work is achievable in 6–9 months because the fragments are finite, the semantics are simple, and the proofs can be staged around a handful of typed schemas. The bottleneck is proving clean non-expressibility results without overfitting them to one handpicked domain encoding.

### 5. Why top-journal-worthy

The contribution is a search-space theory for lifted generalized-policy synthesis: what features buy expressiveness, what they cost MCTS, and how to canonicalize the resulting grammar. The closest comparable paper is [YangEtAl2022PG3], which instantiates lifted decision lists and reports empirical search but does not provide a grammar-minimality or branching-factor theory. [FrancesBonetGeffner2021] gives a feature-based formulation for learning general policies, but not a calculus for CFG-based synthesis of typed decision lists. No close comparable exists for the specific problem of designing a lifted decision-list grammar for AlphaZero-style program synthesis.

## Proposal 3 — Cost-aware lifted-policy execution as part of synthesis

### 1. Question

Can lifted-policy synthesis optimize policies whose generalization advantage survives the runtime cost of unification and pattern matching as the number of objects grows?

### 2. Methodology sketch

Build three interpreters for the same synthesized lifted policies: ground-and-reuse, online unification, and compile-once indexed matching inspired by production-rule systems. Use doors, gripper, and additional typed relational domains to separate policy success from execution cost. The leaf evaluator records task reward plus secondary operational metrics: literals examined, binding attempts, predicate-index lookups, compiled matcher memory, and per-step dispatch time. The synthesis objective can be reported as reward-first with Pareto analysis over cost, or as a cost-regularized score when comparing policies that solve the same tasks. Evaluation asks whether MCTS discovers policies that remain compact and fast under larger object sets, not merely policies that solve small training instances.

### 3. Literature justification

This builds on the relational-MDP framing, where avoiding full grounding is central [BoutilierReiterPrice2001], [GuestrinEtAl2003], [SannerBoutilier2009]. It also uses the production-system insight that many-pattern/many-object matching has specialized algorithms [Forgy1982]. PG3 gives the lifted policy representation [YangEtAl2022PG3], but it does not make interpreter cost a search objective. The synthesis framing supplies the mechanism for feeding execution cost back into grammar-level search [AlurEtAl2013SyGuS], [ParsertPolgreen2024].

### 4. Why realistic

The three interpreters are incremental implementations over the same rule semantics, and the metrics are directly observable during execution. A 4–6 month scope is enough to implement online unification and ground-and-reuse, then add a compiled matcher if profiling shows interpretation is the bottleneck. The bottleneck is fair comparison: a compiled matcher can spend time before execution, while online unification spends time per step, so the study must report synthesis-time, compile-time, memory, and episode-time costs separately.

### 5. Why top-journal-worthy

The contribution is an operational semantics and cost model for synthesized lifted generalized policies, connecting compact symbolic representation to actual scalable execution. The closest comparable paper is [SannerBoutilier2009] in *Artificial Intelligence*, which derives domain-independent policies for first-order MDPs without grounding at intermediate steps. The delta is that this proposal studies executable decision-list policies synthesized by grammar search and asks whether the search can prefer policies that are both correct and cheap to dispatch. PG3’s IJCAI 2022 result is also close, but it treats policy matching cost as incidental rather than as a measured and optimizable property.
