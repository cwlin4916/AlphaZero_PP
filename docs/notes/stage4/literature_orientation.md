# Literature orientation: lifted-policy program synthesis with AlphaZero MCTS

## §1 Canonical framings (with bibliography)

The observation in the draft sits at the intersection of three traditions. None of the three, on its own, exactly studies “AlphaZero-style MCTS over a typed grammar of lifted decision-list policies, with episodic execution reward as the leaf evaluation.” The closest existing work either searches lifted policies with planner-guided scores, learns relational policies/value functions in object-centric MDPs, or uses grammar/tree-search methods for program synthesis under logical or I/O specifications rather than generalized-planning execution reward.

### Framing 1 — Generalized planning as generalized policy synthesis

**Name.** Generalized planning / generalized policy search / learned general policies.

**Canonical question.** Given a planning domain and a few small instances, synthesize a policy or program that solves unseen instances from the same domain, often with more objects than appeared during training.

**Why it fits §A.** This is the most direct parent framing. PG3’s lifted decision-list policy is a modern representative of this line: a candidate policy is evaluated over training problems and judged by whether it generalizes to held-out instances. Your draft’s difference is the search mechanism and score: replace PG3’s domain-specific policy-edit GBFS plus planner-guided score with a grammar derivation game and leaf execution reward.

**Core bibliography.**

- [Khardon1999] Roni Khardon. “Learning Action Strategies for Planning Domains.” *Artificial Intelligence*, 113(1–2):125–148, 1999.
- [MartinGeffner2004] Mario Martín and Héctor Geffner. “Learning Generalized Policies from Planning Examples Using Concept Languages.” *Applied Intelligence*, 20(1):9–19, 2004.
- [BonetPalaciosGeffner2009] Blai Bonet, Héctor Palacios, and Hector Geffner. “Automatic Derivation of Memoryless Policies and Finite-State Controllers Using Classical Planners.” *Proceedings of the 19th International Conference on Automated Planning and Scheduling (ICAPS)*, 34–41, 2009.
- [SrivastavaImmermanZilberstein2011] Siddharth Srivastava, Neil Immerman, and Shlomo Zilberstein. “A New Representation and Associated Algorithms for Generalized Planning.” *Artificial Intelligence*, 175(2):615–647, 2011.
- [FrancesBonetGeffner2021] Guillem Francès, Blai Bonet, and Hector Geffner. “Learning General Planning Policies from Small Examples Without Supervision.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(13):11801–11808, 2021.
- [YangEtAl2022PG3] Ryan Yang, Tom Silver, Aidan Curtis, Tomas Lozano-Perez, and Leslie Pack Kaelbling. “PG3: Policy-Guided Planning for Generalized Policy Generation.” *Proceedings of the 31st International Joint Conference on Artificial Intelligence (IJCAI)*, 4686–4692, 2022.

**Current-state markers, not all foundational.**

- [SilverEtAl2024LLM] Tom Silver, Soham Dan, Kavitha Srinivas, Joshua B. Tenenbaum, Leslie Pack Kaelbling, and Michael Katz. “Generalized Planning in PDDL Domains with Pretrained Large Language Models.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(18):20256–20264, 2024.
- [HofmannGeffner2024] Till Hofmann and Hector Geffner. “Learning Generalized Policies for Fully Observable Non-Deterministic Planning Domains.” *Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI)*, 6733–6742, 2024.
- [BonetGeffner2025] Blai Bonet and Hector Geffner. “Learning General Policies from Examples.” *Proceedings of the 22nd International Conference on Principles of Knowledge Representation and Reasoning (KR)*, 740–750, 2025.
- [ChenEtAl2026GoalRegression] Dillon Z. Chen, Till Hofmann, Toryn Q. Klassen, and Sheila A. McIlraith. “Satisficing and Optimal Generalised Planning via Goal Regression.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 2026.

**Gap relative to §A.** This framing has studied lifted policies, learned features, plan generalization, and planner-guided scoring. It has not made the policy language itself a generic CFG derivation game with AlphaZero-style MCTS and learned policy/value guidance, nor has it made purely episodic execution reward the central replacement for planner-guided policy scoring.

### Framing 2 — Relational reinforcement learning and first-order/relational MDPs

**Name.** Relational reinforcement learning (RRL), relational MDPs, first-order MDPs, object-centric policy/value learning.

**Canonical question.** How can an agent learn policies or value functions in domains whose states and actions are described by objects, predicates, and relations, so that learned structure transfers across object identities and instance sizes?

**Why it fits §A.** A lifted decision-list policy is a relational policy representation. Episodic execution reward, object-count extrapolation, and unification-based dispatch are all natural in RRL/RMDP terms. This framing supplies the vocabulary for “lifted vs grounded,” “policy language bias,” “object generalization,” and “the cost of relational inference.”

**Core bibliography.**

- [MooneyCaliff1995] Raymond J. Mooney and Mary Elaine Califf. “Induction of First-Order Decision Lists: Results on Learning the Past Tense of English Verbs.” *Journal of Artificial Intelligence Research*, 3:1–24, 1995.
- [DzeroskiDeRaedtDriessens2001] Sašo Džeroski, Luc De Raedt, and Kurt Driessens. “Relational Reinforcement Learning.” *Machine Learning*, 43(1–2):7–52, 2001.
- [BoutilierReiterPrice2001] Craig Boutilier, Ray Reiter, and Bryce Price. “Symbolic Dynamic Programming for First-Order MDPs.” *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI)*, 690–697, 2001.
- [GuestrinEtAl2003] Carlos Guestrin, Daphne Koller, Chris Gearhart, and Neal Kanodia. “Generalizing Plans to New Environments in Relational MDPs.” *Proceedings of the 18th International Joint Conference on Artificial Intelligence (IJCAI)*, 1003–1010, 2003.
- [FernYoonGivan2006] Alan Fern, SungWook Yoon, and Robert Givan. “Approximate Policy Iteration with a Policy Language Bias: Solving Relational Markov Decision Processes.” *Journal of Artificial Intelligence Research*, 25:75–118, 2006.
- [SannerBoutilier2009] Scott Sanner and Craig Boutilier. “Practical Solution Techniques for First-Order MDPs.” *Artificial Intelligence*, 173(5–6):748–788, 2009.
- [RivlinHazanKarpas2020] Or Rivlin, Tamir Hazan, and Erez Karpas. “Generalized Planning with Deep Reinforcement Learning.” *Proceedings of the ICAPS Workshop on Bridging the Gap Between AI Planning and Reinforcement Learning*, 2020.

**Supporting runtime reference.**

- [Forgy1982] Charles L. Forgy. “Rete: A Fast Algorithm for the Many Pattern/Many Object Pattern Match Problem.” *Artificial Intelligence*, 19(1):17–37, 1982.

**Gap relative to §A.** RRL/RMDP work studies relational generalization and lifted policy/value representation, but usually not as explicit synthesis of human-readable PG3-style decision lists by grammar derivation. Conversely, PG3 has lifted decision lists but does not treat the score as a pure execution reward nor the search as learned tree search over a CFG.

### Framing 3 — Syntax-guided and neural-guided program synthesis as tree search over derivations

**Name.** Syntax-guided synthesis, neural-guided synthesis, MCTS/RL-guided program synthesis, AlphaZero-style policy/value-guided tree search.

**Canonical question.** Given a grammar of candidate programs and a semantic objective, how can search over partial derivations be guided by learned policies, learned values, rollout rewards, or counterexamples?

**Why it fits §A.** Your `DerivationGame` is exactly the synthesis-as-game view: each action expands a hole in a partial AST; the terminal reward is obtained only after the program is complete and evaluated. The lifted-policy case is a domain-specific instantiation where the target grammar emits typed relational decision-list programs rather than arithmetic expressions, string programs, or low-level code.

**Core bibliography.**

- [KocsisSzepesvari2006] Levente Kocsis and Csaba Szepesvári. “Bandit Based Monte-Carlo Planning.” *Machine Learning: ECML 2006*, Lecture Notes in Computer Science 4212, 282–293, 2006.
- [SilverEtAl2017AlphaGoZero] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. “Mastering the Game of Go without Human Knowledge.” *Nature*, 550(7676):354–359, 2017.
- [SilverEtAl2018AlphaZero] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. “A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play.” *Science*, 362(6419):1140–1144, 2018.
- [AlurEtAl2013SyGuS] Rajeev Alur, Rastislav Bodík, Garvit Juniwal, Milo M. K. Martin, Mukund Raghothaman, Sanjit A. Seshia, Rishabh Singh, Armando Solar-Lezama, Emina Torlak, and Abhishek Udupa. “Syntax-Guided Synthesis.” *Proceedings of the IEEE International Conference on Formal Methods in Computer-Aided Design (FMCAD)*, 1–17, 2013.
- [KalyanEtAl2018NGDS] Ashwin Kalyan, Abhishek Mohta, Oleksandr Polozov, Dhruv Batra, Prateek Jain, and Sumit Gulwani. “Neural-Guided Deductive Search for Real-Time Program Synthesis from Examples.” *6th International Conference on Learning Representations (ICLR)*, 2018.
- [SimmonsEdlerMiltnerSeung2018] Riley Simmons-Edler, Anders Miltner, and Sebastian Seung. “Program Synthesis Through Reinforcement Learning Guided Tree Search.” *arXiv:1806.02932*, 2018.
- [EllisEtAl2019WEA] Kevin Ellis, Maxwell Nye, Yewen Pu, Felix Sosa, Joshua B. Tenenbaum, and Armando Solar-Lezama. “Write, Execute, Assess: Program Synthesis with a REPL.” *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.
- [EllisEtAl2023DreamCoder] Kevin Ellis, Catherine Wong, Maxwell I. Nye, Mathias Sablé-Meyer, Luc Cary, Lucas Morales, Luke B. Hewitt, Armando Solar-Lezama, and Joshua B. Tenenbaum. “DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning.” *Philosophical Transactions of the Royal Society A*, 381(2251):20220050, 2023.
- [ParsertPolgreen2024] Julian Parsert and Elizabeth Polgreen. “Reinforcement Learning and Data-Generation for Syntax-Guided Synthesis.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(10):10670–10678, 2024.

**Gap relative to §A.** SyGuS and neural-guided synthesis have grammars and learned search, but their objectives are typically logical satisfaction or I/O consistency. Generalized planning has relational execution reward and cross-instance policy generalization, but not usually AlphaZero-style learned derivation search over PG3-style policy grammars. The draft’s proposed loop occupies precisely this missing cross-product.

## §2 Unexplored directions

### Direction 1 — Planner-free generalized policy search

PG3 showed that lifted decision-list policies can be searched effectively when a planner-guided score supplies dense information. The visible but underexplored direction is to remove the planner from scoring entirely and ask how far generic grammar MCTS can go with only episodic execution reward. This is not just a baseline swap: it changes the search problem from “policy edits evaluated by a planner” to “derivation choices evaluated by completed executable programs.” The open conceptual issue is how much of PG3’s success comes from the lifted representation versus the planner-backed score.

### Direction 2 — Expressiveness/fan-out calculus for lifted policy grammars

The literature contains many policy languages—first-order decision lists, feature-based rules, finite-state controllers, Python programs—but little systematic accounting of which small grammar features are load-bearing for which generalized-planning structures. Auxiliary variables, flat negation, explicit goal literals, and universal-goal encodings each expand expressiveness and MCTS branching factor. A useful missing object is a calculus that says, for a typed PDDL schema class, what grammar fragment is sufficient, what is impossible, and what branching-factor penalty each feature imposes.

### Direction 3 — Unification cost as a first-class synthesis objective

Lifted policies are compact syntactically, but their runtime interpreter can pay for unification, binding search, and pattern matching. Existing generalized-policy work usually reports policy success and synthesis cost; relational-MDP work often focuses on avoiding full grounding; production-rule systems optimize matching. The unexplored synthesis direction is to treat dispatch cost—number of literal checks, binding attempts, memory use for compiled matchers—as an objective that shapes the discovered lifted policy, not just an implementation detail measured after the fact.

### Direction 4 — Search-to-certificate loops for reward-synthesized lifted policies

Program-synthesis tree search can discover candidates by reward, while generalized planning values formal claims over unbounded instance families. A visible gap is a loop in which MCTS proposes a lifted decision list, then a symbolic checker either certifies it for a domain family or returns a counterexample instance/state that refines the curriculum. This separates discovery from proof and would make planner-free reward search compatible with the correctness ambitions of generalized planning.

### Direction 5 — Goal conditioning and quantifier-sensitive rule design

The draft’s doors/gripper contrast exposes a gap in how goal information enters lifted policies. In sequential single-target tasks, goal literals can be nearly vacuous; in universally quantified goal tasks, they are the only way to identify unsatisfied objects. The literature has examples of goal-conditioned relational policies, but there is no standard design rule for whether goal atoms should be separate rule fields, ordinary state literals under a `Goal` wrapper, or compiled into derived predicates such as “misplaced object.” This direction is about the language boundary between `PRE`, `GOAL`, and derived goal-progress predicates.
