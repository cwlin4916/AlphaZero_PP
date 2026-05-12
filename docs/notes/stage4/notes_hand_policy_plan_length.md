# Hand-policy plan length for Gripper-lite (Proposition + proof)

This note proves that the four-rule hand policy $\pi^{\mathrm{hand}}$ solves
Gripper-lite $\mathcal{M}_B$ in exactly $4B-1$ steps. It is the proof companion to
§7 of the Stage-4 orientation doc
[00_draft_lifted_policy_az.md](00_draft_lifted_policy_az.md); the orientation doc
fixes the task MDP, the rule definitions, and the interpreter — this note assumes
those and develops the per-rule binding conditions, the pick order, and the
inductive plan-length argument.

## §0 Prerequisites

Read first, in order:

- [orientation §3](00_draft_lifted_policy_az.md) — the task MDP $\mathcal{M}_B$:
  predicates, relational state, initial state $s^0_B$, goal $G_B$, ground actions
  $\mathcal{A}_B$, legality, transitions $T_B$, horizon $H_B = 4B+4$.
- [orientation §4](00_draft_lifted_policy_az.md) — the four-rule hand policy
  $\pi^{\mathrm{hand}} = [\rho_1,\rho_2,\rho_3,\rho_4]$ (recalled in §1 below for
  convenience).
- [orientation §5](00_draft_lifted_policy_az.md) — the online-unification
  interpreter: closed-world satisfaction, the binding set $B_\rho(S,G)$, the
  lex-min tie-break $\theta^\star_\rho(S,G)$, and the first-applicable policy map
  $\mathrm{Interpret}$.

Throughout this note, "orientation §k" means §k of `00_draft_lifted_policy_az.md`;
unqualified §k refers to a section of *this* note.

## §1 Proposition

> **Proposition.** For every $B\ge 1$, under the canonical $s^0_B$ and $G_B$, the four-rule
> hand policy $\pi^{\mathrm{hand}} = [\rho_1,\rho_2,\rho_3,\rho_4]$ (orientation §4) solves
> $\mathcal{M}_B$ in exactly $4B-1$ steps; the rollout is deterministic — at every reachable
> state at most one rule has a non-empty binding set (exactly one until the goal is reached),
> and its lex-min binding delivers the misplaced balls in index order
> $\textit{ball\_0},\textit{ball\_1},\dots$

$$\boxed{\;\mathrm{plan\_length}(B) = 3 + 4(B-1) = 4B-1\;}$$

The proof has three pieces: a per-rule **binding-condition lemma** (§2), a **pick-order
lemma** (§3), and an **induction** over the number of balls delivered (§4), each step of which
is a four-row case table — also rendered panel by panel in
[figures/gripper_lite_proof_trace_B2.png](figures/gripper_lite_proof_trace_B2.png) /
[figures/gripper_lite_proof_trace_B3.png](figures/gripper_lite_proof_trace_B3.png).
Throughout, $S$ is the current relational state (assumed only to satisfy the orientation §3
invariants), $r_S$ the unique room with $\mathrm{at\_robot}(r_S)\in S$, $G_B$ the orientation §3
goal, and $b$ is **in $\textit{room\_a}$** in $S$ iff $\mathrm{at\_ball}(b,\textit{room\_a})\in S$.
Recall the rules (orientation §4):

$$\boxed{\begin{aligned}
\rho_1:&\ \mathrm{carrying}(?b)\wedge\mathrm{at\_robot}(?r)\wedge\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)] &&\Rightarrow\ \mathrm{drop}(?b,?r)\\
\rho_2:&\ \mathrm{carrying}(?b)\wedge\mathrm{at\_robot}(?\textit{from})\wedge\mathrm{Goal}[\mathrm{at\_ball}(?b,?\textit{to})] &&\Rightarrow\ \mathrm{move}(?\textit{from},?\textit{to})\\
\rho_3:&\ \mathrm{at\_ball}(?b,?r)\wedge\mathrm{at\_robot}(?r)\wedge\mathrm{handempty}()\wedge\neg\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)] &&\Rightarrow\ \mathrm{pick}(?b,?r)\\
\rho_4:&\ \mathrm{at\_robot}(?\textit{from})\wedge\mathrm{at\_ball}(?b,?\textit{to})\wedge\mathrm{handempty}()\wedge\neg\mathrm{Goal}[\mathrm{at\_ball}(?b,?\textit{to})] &&\Rightarrow\ \mathrm{move}(?\textit{from},?\textit{to})
\end{aligned}}$$

## §2 Lemma 1 — per-rule binding conditions

**Lemma 1 (per-rule binding conditions).** For any $S$ satisfying the orientation §3 invariants,
with $B_\rho$ and $\mathcal{A}_B(\cdot)$ as in orientation §5/§3 and $G_B$ assigning every ball
to $\textit{room\_b}$:

$$\boxed{\begin{aligned}
B_{\rho_1}(S,G_B)\ne\emptyset &\iff (\exists b)\,\mathrm{carrying}(b)\in S \ \wedge\ r_S=\textit{room\_b}\\
B_{\rho_2}(S,G_B)\ne\emptyset &\iff (\exists b)\,\mathrm{carrying}(b)\in S \ \wedge\ r_S=\textit{room\_a}\\
B_{\rho_3}(S,G_B)\ne\emptyset &\iff \mathrm{handempty}()\in S \ \wedge\ r_S=\textit{room\_a} \ \wedge\ (\exists b)\,\text{$b$ in $\textit{room\_a}$}\\
B_{\rho_4}(S,G_B)\ne\emptyset &\iff \mathrm{handempty}()\in S \ \wedge\ r_S=\textit{room\_b} \ \wedge\ (\exists b)\,\text{$b$ in $\textit{room\_a}$}
\end{aligned}}$$

and the fired action is $\mathrm{drop}(b,\textit{room\_b})$ for $\rho_1$ (the unique carried ball),
$\mathrm{move}(\textit{room\_a},\textit{room\_b})$ for $\rho_2$, $\mathrm{pick}(b^\star,\textit{room\_a})$ for $\rho_3$,
$\mathrm{move}(\textit{room\_b},\textit{room\_a})$ for $\rho_4$ — with $b^\star$ the lex-min ball in
$\textit{room\_a}$ (§3 below).

*Proof.* `find_bindings` ([lifted_interpreter.py:89–158](../../src/alphazeropp/synthesis/lifted_interpreter.py))
runs the positive literals of $\mathrm{body}_\rho$ left-to-right against $S$ (goal literals
against $G_B$), then drops bindings failing a negated literal under the closed-world reading,
then drops bindings whose ground action $\notin\mathcal{A}_B(S)$.

- $\rho_1$: $\mathrm{carrying}(?b)$ matches iff a ball is carried (binds $?b$); $\mathrm{at\_robot}(?r)$
  binds $?r=r_S$; $\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)]$ needs $(b,r_S)\in G_B$, i.e. $r_S=\textit{room\_b}$;
  then $\mathrm{drop}(b,\textit{room\_b})\in\mathcal{A}_B(S)$ (robot co-located, ball carried). At most one
  ball is carried (orientation §3 invariant), so the binding is unique.
- $\rho_2$: first two literals as for $\rho_1$; $\mathrm{Goal}[\mathrm{at\_ball}(?b,?\textit{to})]$ binds
  $?\textit{to}=\textit{room\_b}$ (the only goal room for the carried ball); $\mathrm{move}(r_S,\textit{room\_b})$
  is legal iff $r_S\ne\textit{room\_b}$, i.e. $r_S=\textit{room\_a}$ in the two-room domain.
- $\rho_3$: $\mathrm{at\_ball}(?b,?r)$ ranges over ball–room pairs in $S$; $\mathrm{at\_robot}(?r)$ then
  forces $?r=r_S$, so $?b$ ranges over balls in $r_S$; $\mathrm{handempty}()$ requires the empty hand;
  $\neg\mathrm{Goal}[\mathrm{at\_ball}(?b,?r)]$ requires $(?b,r_S)\notin G_B$. If $r_S=\textit{room\_b}$ every
  ball there has $(\cdot,\textit{room\_b})\in G_B$, so the negated literal kills all candidates —
  $\rho_3$ is dead in $\textit{room\_b}$. If $r_S=\textit{room\_a}$ then $(\cdot,\textit{room\_a})\notin G_B$, so
  the condition reduces to "$\exists$ ball in $\textit{room\_a}$" and $\mathrm{pick}(b,\textit{room\_a})$ is legal.
- $\rho_4$: $\mathrm{at\_robot}(?\textit{from})$ binds $?\textit{from}=r_S$; $\mathrm{at\_ball}(?b,?\textit{to})$
  ranges over ball–room pairs in $S$; $\mathrm{handempty}()$; $\neg\mathrm{Goal}[\mathrm{at\_ball}(?b,?\textit{to})]$
  requires $(?b,?\textit{to})\notin G_B$, i.e. $?\textit{to}=\textit{room\_a}$; $\mathrm{move}(r_S,\textit{room\_a})$
  is legal iff $r_S\ne\textit{room\_a}$, i.e. $r_S=\textit{room\_b}$. So: hand empty, robot in $\textit{room\_b}$,
  some ball in $\textit{room\_a}$. $\square$

## §3 Lemma 2 — lex-min pick order

**Lemma 2 (lex-min pick order).** Whenever $\rho_3$ (resp. $\rho_4$) fires, $\theta^\star$
binds $?b$ to the lexicographically least object name among balls in $\textit{room\_a}$. Under the
canonical naming $\textit{ball\_0}<\textit{ball\_1}<\dots$ this is the least-indexed misplaced ball,
and since a picked-and-dropped ball sits in $\textit{room\_b}$ — excluded from every later $\rho_3$/$\rho_4$
binding set by the $\neg\mathrm{Goal}$ literal and never returned to $\textit{room\_a}$ by $\pi^{\mathrm{hand}}$
— the deliveries occur in index order $\textit{ball\_0},\dots,\textit{ball\_}{B-1}$.

*Proof.* From §2's proof the $\rho_3$ binding set is
$\{\{?b\mapsto b,\,?r\mapsto\textit{room\_a}\}:\text{$b$ in }\textit{room\_a}\}$ (and the $\rho_4$ set
has $?\textit{from}\mapsto\textit{room\_b},?\textit{to}\mapsto\textit{room\_a}$ fixed and $?b$ ranging the
same way). `find_bindings` returns these sorted by $\theta\mapsto\mathrm{tuple}(\mathrm{sorted}(\theta.\mathrm{items()}))$
and $\mathrm{interpret}$ takes element $0$ ([lifted_interpreter.py:157,183](../../src/alphazeropp/synthesis/lifted_interpreter.py));
the variable names compare $\textit{?b}<\textit{?from}<\textit{?r}<\textit{?to}$, so the tuples order
first on the $?b$ value and the minimum is the lex-least ball. Index order then follows because
$\mathrm{drop}(b,\textit{room\_b})$ adds $\mathrm{at\_ball}(b,\textit{room\_b})$, after which $b$ is goal-placed
and the $\neg\mathrm{Goal}$ literal excludes it. $\square$

## §4 Rest configurations and the induction

**Rest configurations.** For $0\le k\le B$ define

$$\boxed{\;\mathrm{Rest}_k \;=\; \{\mathrm{at\_robot}(\sigma_k),\,\mathrm{handempty}()\}\,\cup\,\{\mathrm{at\_ball}(\textit{ball\_}j,\textit{room\_b}):0\le j<k\}\,\cup\,\{\mathrm{at\_ball}(\textit{ball\_}j,\textit{room\_a}):k\le j<B\},\quad \sigma_0=\textit{room\_a},\ \ \sigma_{k\ge1}=\textit{room\_b}\;}$$

so $\mathrm{Rest}_0 = s^0_B$ and $\mathrm{Rest}_B\supseteq G_B$, hence $\mathrm{solved}(\mathrm{Rest}_B)$
(orientation §3).

**Claim (one delivery cycle).** For $0\le k<B$, started from $\mathrm{Rest}_k$, $\pi^{\mathrm{hand}}$
emits the actions below and reaches $\mathrm{Rest}_{k+1}$ — $3$ steps for $k=0$, $4$ steps for $k\ge1$:

- $k=0$: $\ \rho_3\!:\mathrm{pick}(\textit{ball\_0},\textit{room\_a})\ \to\ \rho_2\!:\mathrm{move}(\textit{room\_a},\textit{room\_b})\ \to\ \rho_1\!:\mathrm{drop}(\textit{ball\_0},\textit{room\_b})$
- $k\ge1$: $\ \rho_4\!:\mathrm{move}(\textit{room\_b},\textit{room\_a})\ \to\ \rho_3\!:\mathrm{pick}(\textit{ball\_}k,\textit{room\_a})\ \to\ \rho_2\!:\mathrm{move}(\textit{room\_a},\textit{room\_b})\ \to\ \rho_1\!:\mathrm{drop}(\textit{ball\_}k,\textit{room\_b})$

*Proof.* Read each row off Lemma 1 (which $B_{\rho_i}$ are non-empty), take the first applicable
rule ($\mathrm{Interpret}$, orientation §5), apply $T_B$ (orientation §3). Write $b_k=\textit{ball\_}k$.
The table is the $k\ge1$ cycle — for $k=0$ delete the first row (the robot already starts in
$\textit{room\_a}$, and with no ball in the *other* room $B_{\rho_4}=\emptyset$ there). The parenthetical
after each $\emptyset$ names the failing Lemma-1 condition: *¬carry* = no ball carried, *hand full* =
$\mathrm{handempty}()\notin S$, *$r{\ne}a$* / *$r{\ne}b$* = robot not in $\textit{room\_a}$ / $\textit{room\_b}$.

| sub-state ($S$) | $B_{\rho_1}$ | $B_{\rho_2}$ | $B_{\rho_3}$ | $B_{\rho_4}$ | fires → action | $T_B$ → next |
|---|:--:|:--:|:--:|:--:|---|---|
| $\mathrm{Rest}_k$: robot $\textit{room\_b}$, hand empty; $b_k\!..\!b_{B-1}$ in $\textit{room\_a}$ | $\emptyset$ (¬carry) | $\emptyset$ (¬carry) | $\emptyset$ ($r{\ne}a$) | **✓** ($b_k\!..$ in $\textit{room\_a}$) | $\rho_4$ → $\mathrm{move}(\textit{room\_b},\textit{room\_a})$ | robot → $\textit{room\_a}$ |
| robot $\textit{room\_a}$, hand empty; $b_k\!..\!b_{B-1}$ in $\textit{room\_a}$ | $\emptyset$ (¬carry) | $\emptyset$ (¬carry) | **✓** (lex-min $b_k$) | $\emptyset$ ($r{\ne}b$) | $\rho_3$ → $\mathrm{pick}(b_k,\textit{room\_a})$ | $b_k$: $\textit{room\_a}$ → carried; hand full |
| carrying $b_k$, robot $\textit{room\_a}$; $b_{k+1}\!..\!b_{B-1}$ in $\textit{room\_a}$ | $\emptyset$ ($r{\ne}b$) | **✓** ($?\textit{to}{=}\textit{room\_b}$) | $\emptyset$ (hand full) | $\emptyset$ (hand full) | $\rho_2$ → $\mathrm{move}(\textit{room\_a},\textit{room\_b})$ | robot → $\textit{room\_b}$ |
| carrying $b_k$, robot $\textit{room\_b}$; $b_{k+1}\!..\!b_{B-1}$ in $\textit{room\_a}$ | **✓** ($(b_k,\textit{room\_b}){\in}G_B$) | $\emptyset$ ($r{\ne}a$) | $\emptyset$ (hand full) | $\emptyset$ (hand full) | $\rho_1$ → $\mathrm{drop}(b_k,\textit{room\_b})$ | $b_k$: carried → $\textit{room\_b}$; hand empty |
| $=\mathrm{Rest}_{k+1}$: robot $\textit{room\_b}$, hand empty; $b_{k+1}\!..\!b_{B-1}$ in $\textit{room\_a}$ | | | | | | |

The last row matches the definition of $\mathrm{Rest}_{k+1}$ (when $k+1=B$ the "balls in $\textit{room\_a}$"
set is empty and the configuration is $\mathrm{Rest}_B$). $\square$

**Proof of the Proposition.** Induct on $k$: the rollout visits $\mathrm{Rest}_0,\dots,\mathrm{Rest}_B$.
Base $\mathrm{Rest}_0=s^0_B$. Step: from $\mathrm{Rest}_k$ ($k<B$) the Claim reaches $\mathrm{Rest}_{k+1}$
deterministically in $3$ steps ($k=0$) or $4$ ($k\ge1$), with exactly one non-empty $B_{\rho_i}$ at
every sub-state (the "fires" column). At $\mathrm{Rest}_B$, $\mathrm{solved}$ holds and by Lemma 1 all
$B_{\rho_i}(\mathrm{Rest}_B,G_B)=\emptyset$ (no ball carried; $\rho_3,\rho_4$ need a ball in $\textit{room\_a}$,
none left), so $\mathrm{Interpret}$ returns $\mathrm{None}$ — but the episode already terminates on
$\mathrm{solved}$. Total steps $= 3 + \sum_{k=1}^{B-1}4 = 3 + 4(B-1) = 4B-1$. $\blacksquare$

## §5 Corollary — rule-firing trace, horizon, tests

**Corollary (rule-firing trace, horizon, tests).** The fired-rule sequence is
$(\rho_3,\rho_2,\rho_1)\,(\rho_4,\rho_3,\rho_2,\rho_1)^{\times(B-1)}$ — so $\rho_3,\rho_2,\rho_1$
for $B=1$; $\rho_3,\rho_2,\rho_1,\rho_4,\rho_3,\rho_2,\rho_1$ for $B=2$;
$\rho_3,\rho_2,\rho_1,(\rho_4,\rho_3,\rho_2,\rho_1)^{\times2}$ for $B=3$. Since $H_B = 4B+4 > 4B-1$
(orientation §3), the episode never truncates. The action-schema sequences, the $4B-1$ step counts,
and the $\textit{ball\_0},\textit{ball\_1},\dots$ pick order for $B\in\{1,2,3\}$ are asserted in
[tests/test_gripper_lite_policy.py](../../tests/test_gripper_lite_policy.py); the per-step
$\rho_1$–$\rho_4$ binding-set status — the "fires" columns above, recomputed from `find_bindings`
so the figure cannot drift — is rendered by
[plot_gripper_lite_rollout.py](../../scripts/plotting/plot_gripper_lite_rollout.py) into the two
proof-trace figures in §6.

## §6 Figures

![Gripper-lite B = 2 rollout: pick → move → drop → move → pick → move → drop](figures/gripper_lite_rollout_B2.png)

![Gripper-lite B = 2 proof trace: each panel shows the state and the ρ1–ρ4 binding-set status — ✓ fires with lex-min binding, ✗ empty with the first failing literal](figures/gripper_lite_proof_trace_B2.png)

![Gripper-lite B = 3 proof trace: the (ρ4,ρ3,ρ2,ρ1) delivery cycle repeats once per remaining ball](figures/gripper_lite_proof_trace_B3.png)
