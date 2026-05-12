#!/usr/bin/env python3
"""Formal-formulation diagrams for the lifted derivation game.

Companion to ``docs/notes/stage4/notes/derivation_game.md`` (figures D1–D8) —
the **occurrence-introduced-variable** grammar (no ``aux_var`` hole; body-local
variables ``?v_0, ?v_1, …`` are born inside the precondition literal that first
uses them; goal literals introduce none). Produces, into
``docs/notes/stage4/figures/``:

- ``02_derivation_state_machine.png`` — the four hole kinds as a finite-state
  machine over which a derivation runs (``policy -> action_schema -> pre_lit ->
  goal_lit``, back to ``policy`` on ``FINISH_RULE``, to ``TERMINAL`` on
  ``STOP_POLICY``); each node annotated with its legal-production count. The
  ``pre_lit`` hole is the widest (state literals may introduce a fresh body-local
  variable in any argument position), and ``compute_max_productions(cfg, sig)``
  sizes the MCTS action head.
- ``02_derivation_example.png`` — one concrete derivation (the Stage-1
  hand-policy rule ρ₂, move-toward-goal — chosen because its ``?v_0:ball`` is a
  body-local variable born inside ``carrying(?v_0)``) drawn as a labelled path
  through the transition delta, ending at a terminal complete policy. The state
  at each step is read from a live ``LiftedDerivationState``; the edge labels are
  the actual ``LiftedProduction.label``s.
- ``02_derivation_production_sets.png`` — the legal-production set
  ``A(s) = enumerate_productions(s, cfg, sig)`` at one representative state per
  hole kind, for the Gripper-lite signature; the ``pre_lit`` hole-kind is the
  widest and ``M = compute_max_productions(cfg, sig)`` is the action-space size.
- ``02_derivation_state_anatomy.png`` — six representative ``LiftedDerivationState``
  values with *every* field rendered explicitly (``completed_rules`` / ``partial``
  = ``PartialRule(schema, action_args, body_local_vars, state_lits, goal_lits)`` /
  the single ``current_hole``), plus the state's ``pretty()`` key.
- ``02_derivation_state_evolution.png`` — a field-by-field *diff* table along one
  derivation (the build of hand-policy rule ρ₂): each row a state ``s_i``, each
  highlighted cell the one field the applied production changed (the
  ``pre:carrying(?v_0)`` step both extends ``state_lits`` *and* grows
  ``body_local_vars``).
- ``02_derivation_production_lengths.png`` — how long a derivation is: the per-rule
  production breakdown (4 fixed + ≤P pre-lits + ≤G goal-lits), the analytic
  complete-policy length range ``ℓ(π) ∈ [5, R·(P+G+4)+1]`` vs ``max_rules``, and
  the empirical length distribution over 5000 uniform-random derivations.
- ``02_derivation_tree.png`` — one full derivation of a 3-rule policy (the
  Stage-1 hand-policy prefix ρ₁ drop-at-goal / ρ₂ move-toward-goal / ρ₃ pick)
  drawn as a grammar *parse tree*: the start symbol ``⟨Policy⟩`` at the root,
  expanding through ``⟨Rule⟩ → ADD_RULE ⟨ActionSchema⟩ ⟨PreList⟩ STOP_PRE
  ⟨GoalList⟩ FINISH_RULE`` down to the terminal production labels (leaves,
  left-to-right = ``p₁ … p_T``, colour-coded by hole kind).  Complements
  ``02_derivation_example.png`` (the *linear* path through δ).
- ``02_derivation_hypothesis_class.png`` — what the grammar can *say*: the
  capacity tuple ``cfg = (R, L_S, L_G, V_b)`` (``V_b = max_body_local_vars``)
  with its ``LiftedGrammarConfig`` defaults and the disabled switches; the
  program-space size (action-head width ``M = compute_max_productions``; the old
  aux-var-grammar ``r1`` config enumerable at ≈1,166 distinct policies — see
  ``legacy/02.md``, not re-enumerated here; ``R ≥ 3`` needed for any solver); and
  a gallery of one real ``Rule`` per rule family (empty-body / state-reactive /
  goal-conditioned / negated-goal guard / body-local-variable), each built
  through the grammar — used by ``derivation_game.md`` §9 (Figure D8).

All of these are derived from the real grammar/derivation modules — nothing is
hand-transcribed. Run from repo root::

    python scripts/plotting/plot_derivation_game_diagrams.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alphazeropp.synthesis.lifted_derivation import LiftedDerivationState  # noqa: E402
from alphazeropp.synthesis.lifted_dsl import Var  # noqa: E402
from alphazeropp.synthesis.lifted_grammar import (  # noqa: E402
    LiftedGrammarConfig,
    compute_max_productions,
    enumerate_productions,
    gripper_lite_signature,
)

OUT = REPO_ROOT / "docs" / "notes" / "stage4" / "figures"
DPI = 160

# hole-kind -> (face colour, edge colour) — shared across the figures
HOLE_FACE = {
    "policy": "#e8f0fe",
    "action_schema": "#fde8e6",
    "pre_lit": "#e6f4ea",
    "goal_lit": "#f3e8fd",
    None: "#eceff1",            # terminal
}
HOLE_EDGE = {
    "policy": "#1a73e8",
    "action_schema": "#d93025",
    "pre_lit": "#188038",
    "goal_lit": "#8430ce",
    None: "#5f6368",
}
HOLE_ORDER = ["policy", "action_schema", "pre_lit", "goal_lit"]
_SUB = "₀₁₂₃₄₅₆₇₈₉"  # subscript digits for state indices


# ---------------------------------------------------------------------------
# shared: walk a real derivation
# ---------------------------------------------------------------------------

def _grammar():
    return LiftedGrammarConfig(max_rules=3), gripper_lite_signature()


def _pick(prods, *, label=None, pred=None, negated=None, argnames=None):
    """Select one production from ``prods`` by exact label, or by add-literal
    predicate — optionally pinned to a negation flag and/or an argument-name
    tuple (to disambiguate e.g. ``Goal[at_ball(?v_0, ?r_0)]`` from
    ``Goal[at_ball(?v_0, ?r_1)]`` when both are legal)."""
    for p in prods:
        if label is not None and p.label == label:
            return p
        if pred is not None and p.payload and p.payload[0] == "add":
            lit = p.payload[1]
            if lit.atom.pred != pred:
                continue
            if negated is not None and lit.negated != negated:
                continue
            if argnames is not None:
                names = tuple(a.name if isinstance(a, Var) else a for a in lit.atom.args)
                if names != tuple(argnames):
                    continue
            return p
    raise KeyError(f"no production matching label={label!r} pred={pred!r} "
                   f"negated={negated!r} argnames={argnames!r}; "
                   f"have {[p.label for p in prods]}")


def _state_after(cfg, sig, selectors):
    """Apply a sequence of selectors (each a dict of kwargs for ``_pick``) to the
    initial state; return the list of (state_before, production) pairs and the
    final state."""
    st = LiftedDerivationState.initial()
    steps = []
    for sel in selectors:
        prod = _pick(enumerate_productions(st, cfg, sig), **sel)
        steps.append((st, prod))
        st = st.apply(prod)
    return steps, st


# the Stage-1 hand-policy rule ρ₁ (drop-at-goal), as a selector sequence — its
# variables are both action arguments (drop's schema is ball × room), so it
# introduces no body-local variable. Used as the "goal-conditioned" gallery
# exemplar (Fig D8).
_RHO1 = [
    dict(label="ADD_RULE"),
    dict(label="schema=drop"),
    dict(pred="at_robot"),                                       # at_robot(?r_1)
    dict(pred="carrying"),                                       # carrying(?b_0)
    dict(label="STOP_PRE"),
    dict(pred="at_ball", negated=False),                         # Goal[at_ball(?b_0, ?r_1)]
    dict(label="FINISH_RULE"),
    dict(label="STOP_POLICY"),
]

# the Stage-1 hand-policy rule ρ₂ (move-toward-goal), as a selector sequence —
# move's schema is room × room, so the carried ball is a *body-local* variable
# ``?v_0:ball``, born inside ``carrying(?v_0)`` and reused in the goal literal.
# This is the running example for the worked-derivation figures (D3, D5).
_RHO2 = [
    dict(label="ADD_RULE"),
    dict(label="schema=move"),
    dict(pred="at_robot"),                                       # at_robot(?r_0)
    dict(pred="carrying"),                                       # carrying(?v_0)  ← introduces ?v_0:ball
    dict(label="STOP_PRE"),
    dict(pred="at_ball", negated=False, argnames=("?v_0", "?r_1")),  # Goal[at_ball(?v_0, ?r_1)]
    dict(label="FINISH_RULE"),
    dict(label="STOP_POLICY"),
]

# one representative state per hole kind, as a selector prefix
_REP_PREFIX = {
    "policy": [dict(label="ADD_RULE"), dict(label="schema=move"),
               dict(label="STOP_PRE"), dict(label="FINISH_RULE")],     # at policy, 1 rule done
    "action_schema": [dict(label="ADD_RULE")],
    "pre_lit": [dict(label="ADD_RULE"), dict(label="schema=pick")],
    "goal_lit": [dict(label="ADD_RULE"), dict(label="schema=move"),
                 dict(pred="at_robot"), dict(pred="carrying"), dict(label="STOP_PRE")],
}


def _rep_states(cfg, sig):
    out = {}
    for hole, prefix in _REP_PREFIX.items():
        _, st = _state_after(cfg, sig, prefix)
        assert st.current_hole == hole, (hole, st.current_hole)
        out[hole] = (st, enumerate_productions(st, cfg, sig))
    return out


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, *, face, edge, fontsize=9, fontweight="normal",
         family="sans-serif", lw=1.6, pad=0.25):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad={pad}", fc=face, ec=edge, lw=lw,
                                zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight,
            family=family, zorder=3)


def _arrow(ax, p0, p1, *, label=None, rad=0.0, color="#444", lw=1.6, fs=8,
           lab_dx=0.0, lab_dy=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=15, color=color,
                                 lw=lw, zorder=1, shrinkA=2, shrinkB=2, linestyle=ls))
    if label:
        mx, my = (p0[0] + p1[0]) / 2 + lab_dx, (p0[1] + p1[1]) / 2 + lab_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs, family="monospace",
                color="#202124", zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#cfcfcf", lw=0.4,
                          alpha=0.95))


def _partial_str(st):
    p = st.partial
    if p is None:
        n = len(st.completed_rules)
        if st.current_hole is None:
            return "complete policy"
        return f"{n} rule{'s' if n != 1 else ''} committed; no rule in progress" if n else \
            "no rule in progress"
    body = " ∧ ".join(l.pretty() for l in (p.state_lits + p.goal_lits)) or "⊤"
    act = f"{p.schema}({', '.join(v.name for v in p.action_args)})" if p.schema else "<action?>"
    blv = (f"   body_local=[{', '.join(f'{v.name}:{v.type_name}' for v in p.body_local_vars)}]"
           if p.body_local_vars else "")
    return f"{body}  ⇒  {act}{blv}"


# ---------------------------------------------------------------------------
# Figure D1 — the hole-kind state machine (state space S)
# ---------------------------------------------------------------------------

def fig_state_machine() -> Path:
    cfg, sig = _grammar()
    M = compute_max_productions(cfg, sig)
    reps = _rep_states(cfg, sig)
    count_label = {
        "policy": "|A(s)| = 1 or 2",
        "action_schema": f"|A(s)| = {len(reps['action_schema'][1])}",
        "pre_lit": f"|A(s)| ≤ M = {M}",
        "goal_lit": f"|A(s)| ≤ {len(reps['goal_lit'][1])}",
    }

    fig, ax = plt.subplots(figsize=(16.0, 5.8))
    y0 = 2.6
    xs = {"policy": 1.6, "action_schema": 6.1, "pre_lit": 10.6, "goal_lit": 15.1}
    pos = {h: (xs[h], y0) for h in HOLE_ORDER}
    pos[None] = (1.6, 0.5)
    bw, bh = 2.5, 0.95
    for h in HOLE_ORDER:
        x, y = pos[h]
        _box(ax, x, y, bw, bh, f"{h}\n{count_label[h]}", face=HOLE_FACE[h], edge=HOLE_EDGE[h],
             fontsize=11, fontweight="bold", family="monospace")
    _box(ax, *pos[None], bw, bh, "TERMINAL\ncomplete policy", face=HOLE_FACE[None],
         edge=HOLE_EDGE[None], fontsize=11, fontweight="bold", family="monospace")

    def Lx(h):
        return pos[h][0] - bw / 2

    def Rx(h):
        return pos[h][0] + bw / 2

    # main chain
    _arrow(ax, (Rx("policy"), y0), (Lx("action_schema"), y0), label="ADD_RULE",
           color=HOLE_EDGE["policy"], fs=8.5)
    _arrow(ax, (Rx("action_schema"), y0), (Lx("pre_lit"), y0), label="schema=X",
           color=HOLE_EDGE["action_schema"], fs=8.5)
    _arrow(ax, (Rx("pre_lit"), y0), (Lx("goal_lit"), y0), label="STOP_PRE",
           color=HOLE_EDGE["pre_lit"], fs=8.5)
    # self-loops (add a literal)
    _arrow(ax, (pos["pre_lit"][0] - 0.5, y0 + bh / 2), (pos["pre_lit"][0] + 0.5, y0 + bh / 2),
           rad=-1.7, color=HOLE_EDGE["pre_lit"], lw=1.3)
    ax.text(pos["pre_lit"][0], y0 + bh / 2 + 0.95, "pre:lit\n(may introduce ?v_i)", ha="center",
            va="center", fontsize=7.6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#cfcfcf", lw=0.4))
    _arrow(ax, (pos["goal_lit"][0] - 0.5, y0 + bh / 2), (pos["goal_lit"][0] + 0.5, y0 + bh / 2),
           rad=-1.7, color=HOLE_EDGE["goal_lit"], lw=1.3)
    ax.text(pos["goal_lit"][0], y0 + bh / 2 + 0.95, "goal:lit  (±)\n(no new vars)", ha="center",
            va="center", fontsize=7.6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#cfcfcf", lw=0.4))
    # FINISH_RULE: goal_lit -> policy, big arc over the top
    _arrow(ax, (pos["goal_lit"][0], y0 + bh / 2 + 0.05), (pos["policy"][0], y0 + bh / 2 + 0.05),
           rad=0.30, color=HOLE_EDGE["goal_lit"], lw=1.9)
    ax.text((pos["goal_lit"][0] + pos["policy"][0]) / 2, y0 + 2.55,
            "FINISH_RULE  (commit the rule, return to the policy hole)", ha="center", va="center",
            fontsize=9, family="monospace", color=HOLE_EDGE["goal_lit"],
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=HOLE_EDGE["goal_lit"], lw=0.8))
    # STOP_POLICY: policy -> TERMINAL
    _arrow(ax, (pos["policy"][0], y0 - bh / 2), (pos[None][0], pos[None][1] + bh / 2),
           color=HOLE_EDGE["policy"], lw=1.7)
    ax.text(pos["policy"][0] + 0.18, (y0 - bh / 2 + pos[None][1] + bh / 2) / 2,
            "STOP_POLICY  (#rules ≥ 1)", ha="left", va="center", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#cfcfcf", lw=0.4))
    # widest-hole marker on pre_lit
    ax.text(pos["pre_lit"][0], y0 - bh / 2 - 0.32, "← widest hole-kind", ha="center", va="center",
            fontsize=8.0, color=HOLE_EDGE["pre_lit"], fontweight="bold")
    # initial state
    _arrow(ax, (Lx("policy") - 1.0, y0), (Lx("policy"), y0), color="#202124", lw=1.7)
    ax.text(Lx("policy") - 1.1, y0, "s₀", ha="right", va="center", fontsize=12, fontweight="bold")

    ax.set_title("Figure D1 — the lifted derivation game: state space as a hole-kind state machine "
                 "(occurrence-introduced-variable grammar — no aux_var hole)\n"
                 "a state is  s = (completed rules, partial rule under construction, current hole);  "
                 "the per-hole legal-production counts |A(s)| size the MCTS action head (M = "
                 f"compute_max_productions(cfg, sig) = {M}, attained at the pre_lit hole)",
                 fontsize=11.0, fontweight="bold")
    ax.set_xlim(-1.4, 17.2)
    ax.set_ylim(-0.8, 6.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = OUT / "02_derivation_state_machine.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure D3 — one derivation as a labelled path through delta
# ---------------------------------------------------------------------------

def fig_example_derivation() -> Path:
    cfg, sig = _grammar()
    steps, final = _state_after(cfg, sig, _RHO2)
    assert final.is_terminal()
    rows = [(st.current_hole, _partial_str(st), prod.label) for st, prod in steps]
    n = len(rows)

    fig, ax = plt.subplots(figsize=(13.5, 1.0 * (n + 2) + 1.2))
    bw, bh = 4.0, 0.6
    x_node, x_state = 2.4, 5.0
    dy = 1.0
    y_top = (n + 1) * dy

    for i, (hole, partial, plabel) in enumerate(rows):
        y = y_top - i * dy
        _box(ax, x_node, y, bw, bh, f"hole = {hole}", face=HOLE_FACE[hole],
             edge=HOLE_EDGE[hole], fontsize=10.5, fontweight="bold", family="monospace")
        ax.text(x_state, y, partial, ha="left", va="center", fontsize=9.0, family="monospace",
                color="#202124")
        y_next = y_top - (i + 1) * dy
        _arrow(ax, (x_node, y - bh / 2), (x_node, y_next + bh / 2), color="#5f6368", lw=1.7)
        is_struct = not plabel.startswith(("pre:", "goal:"))
        ax.text(x_node + 0.32, (y - bh / 2 + y_next + bh / 2) / 2, plabel, ha="left", va="center",
                fontsize=8.8, family="monospace",
                color=("#1a73e8" if is_struct else "#188038"),
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#cfcfcf", lw=0.4))
    # terminal node + the completed policy
    y = y_top - n * dy
    _box(ax, x_node, y, bw, bh, "hole = ⊥   (terminal)", face=HOLE_FACE[None],
         edge=HOLE_EDGE[None], fontsize=10.5, fontweight="bold", family="monospace")
    ax.text(x_state, y + 0.16, "complete policy  π = to_program(s):", ha="left", va="center",
            fontsize=9.0, family="monospace", color="#202124")
    ax.text(x_state, y - 0.42, final.to_program().pretty(), ha="left", va="center", fontsize=9.6,
            family="monospace", color="#8430ce", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f7f2fc", ec="#8430ce", lw=1.0))

    ax.set_title("Figure D3 — one derivation as a path through the transition δ\n"
                 "actions are grammar productions; the Stage-1 hand-policy rule ρ₂ (move-toward-goal) is "
                 "built in 8 productions, then STOP_POLICY — note ?v_0:ball is born inside pre:carrying(?v_0) "
                 "(a body-local variable, used in conditions only); state read from a live LiftedDerivationState",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlim(0.0, 13.5)
    ax.set_ylim(y - 1.1, y_top + bh + 0.5)
    ax.axis("off")
    fig.tight_layout()
    out = OUT / "02_derivation_example.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure D3 — legal-production sets A(s) per hole kind
# ---------------------------------------------------------------------------

def _fmt_prod(label: str) -> str:
    if label.startswith(("pre:", "goal:")):
        kind, body = label.split(":", 1)
        return f"{kind}:  {body}"
    return label


def fig_production_sets() -> Path:
    cfg, sig = _grammar()
    reps = _rep_states(cfg, sig)
    M = compute_max_productions(cfg, sig)

    fig, axes = plt.subplots(1, 4, figsize=(20.0, 7.4))
    for ax, hole in zip(axes, HOLE_ORDER):
        st, prods = reps[hole]
        widest = (hole == "pre_lit")
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.895), 0.96, 0.09, boxstyle="round,pad=0.01",
                                    transform=ax.transAxes, fc=HOLE_FACE[hole],
                                    ec=HOLE_EDGE[hole], lw=2.4 if widest else 1.5))
        ax.text(0.5, 0.94, f"hole = {hole}", transform=ax.transAxes, ha="center", va="center",
                fontsize=12.5, fontweight="bold", family="monospace")
        ax.text(0.5, 0.872, f"|A(s)| = {len(prods)}" + ("   ← widest hole" if widest else ""),
                transform=ax.transAxes, ha="center", va="center", fontsize=10.5,
                color=HOLE_EDGE[hole], fontweight="bold")
        ax.text(0.04, 0.835, "at s :  " + _partial_str(st), transform=ax.transAxes, ha="left",
                va="top", fontsize=7.6, family="monospace", color="#3c4043")
        lines = [f"• {_fmt_prod(p.label)}" for p in prods]
        ax.text(0.05, 0.745, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
                fontsize=8.6, family="monospace", color="#202124")

    fig.suptitle("Figure D2 — the action space: legal-production sets  A(s) = enumerate_productions(s, cfg, sig)  "
                 "at one representative state per hole kind (Gripper-lite)\n"
                 f"the pre_lit hole-kind is the widest — a state literal may take a fresh body-local variable in any "
                 f"argument position; the MCTS action head is sized to  M = compute_max_productions(cfg, sig) = {M}  "
                 f"(the analytic max over all states, computed with the maximal possible variable scope; the rep "
                 f"states shown have fewer)",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT / "02_derivation_production_sets.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure D8 — the hypothesis class: capacity knobs, program size, rule families
# ---------------------------------------------------------------------------

# one real Rule per rule family — built through the grammar (selectors), so the
# variable names come out canonical (action args ?b_0 / ?r_1 by schema position;
# body-local vars ?v_0 by introduction order). ``_RHO1`` (drop-at-goal, action
# args only) is the goal-conditioned exemplar; ``_RHO2`` (move-toward-goal) is
# the body-local-variable exemplar.
_FAMILY_GALLERY = [
    ("empty-body",
     [dict(label="ADD_RULE"), dict(label="schema=drop"),
      dict(label="STOP_PRE"), dict(label="FINISH_RULE"), dict(label="STOP_POLICY")],
     "no body — fires from (almost) any state; the ⊤ ⇒ drop do-nothing attractor lives here (§11)"),
    ("state-reactive",
     [dict(label="ADD_RULE"), dict(label="schema=pick"),
      dict(pred="at_ball"), dict(pred="handempty"), dict(label="STOP_PRE"),
      dict(label="FINISH_RULE"), dict(label="STOP_POLICY")],
     "preconditions only — reacts to the world state, never reads the goal"),
    ("goal-conditioned   (= hand-policy ρ₁ drop-at-goal)",
     _RHO1,
     "a positive Goal[…] literal — acts only where this task's goal wants it (§7); all vars are action args"),
    ("negated-goal guard   (= hand-policy ρ₃ pick)",
     [dict(label="ADD_RULE"), dict(label="schema=pick"),
      dict(pred="at_ball"), dict(pred="at_robot"), dict(pred="handempty"),
      dict(label="STOP_PRE"), dict(pred="at_ball", negated=True),
      dict(label="FINISH_RULE"), dict(label="STOP_POLICY")],
     "¬Goal[…] (safe — every variable already positively bound) — skip what is already done"),
    ("body-local-variable   (= hand-policy ρ₂ move-toward-goal)",
     _RHO2,
     "?v_0:ball is born in carrying(?v_0) (a body-local / existential var, used in conditions only, never "
     "in the action) and reused in Goal[at_ball(?v_0, ?r_1)] — picks the move target (§8)"),
]


def _r1_policy_count() -> int:
    """|Π_Γ,cfg| at the *old aux-var grammar's* r1 config — read from the
    committed landscape data (``unique_policies_evaluated`` under the ``none``
    setting), not transcribed. Not re-enumerated for the occurrence-introduced
    grammar; shown as an order-of-magnitude reference (see legacy/02.md)."""
    p = REPO_ROOT / "docs" / "notes" / "stage4" / "data" / "landscape_r1.json"
    try:
        return int(json.loads(p.read_text())["none"]["unique_policies_evaluated"])
    except Exception:
        return 1166


def fig_hypothesis_class() -> Path:
    cfg_default = LiftedGrammarConfig()
    cfg_exp = LiftedGrammarConfig(max_rules=3)
    sig = gripper_lite_signature()
    from alphazeropp.synthesis.lifted_grammar import legacy_grammar_config  # noqa: E402
    M = compute_max_productions(cfg_default, sig)
    M_legacy = compute_max_productions(legacy_grammar_config(), sig)
    n_r1 = _r1_policy_count()

    fig, axd = plt.subplot_mosaic(
        [["caps", "space"], ["gallery", "gallery"]],
        figsize=(19.5, 11.2), height_ratios=[1.0, 1.85],
    )
    for ax in axd.values():
        ax.axis("off")

    # --- panel A: the capacity tuple ---------------------------------------
    ax = axd["caps"]
    ax.add_patch(FancyBboxPatch((0.02, 0.875), 0.96, 0.105, boxstyle="round,pad=0.012",
                                transform=ax.transAxes, fc=HOLE_FACE["policy"],
                                ec=HOLE_EDGE["policy"], lw=2.0))
    ax.text(0.5, 0.925, "the capacity tuple   cfg = (R, L_S, L_G, V_b)",
            transform=ax.transAxes, ha="center", va="center", fontsize=13.5,
            fontweight="bold", family="monospace")
    caps_lines = [
        f"R      = max_rules            = {cfg_default.max_rules}     rules in a policy        (experiments: 3 — Run B: 4)",
        f"L_S    = max_pre_literals      = {cfg_default.max_pre_literals}     precondition (state) literals / rule",
        f"L_G    = max_goal_literals     = {cfg_default.max_goal_literals}     goal literals / rule",
        f"V_b    = max_body_local_vars   = {cfg_default.max_body_local_vars}     body-local (existential) vars / rule",
        "         (born inside the precondition literal that first uses them — no Aux phase)",
        "",
        "switched OFF this stage:",
        f"  allow_disjunction    = {cfg_default.allow_disjunction}     (no ∨ inside a rule body)",
        f"  allow_state_negation = {cfg_default.allow_state_negation}     (preconditions are positive only)",
        f"  allow_constants      = {cfg_default.allow_constants}     (no object constants in rules)",
        f"  allow_goal_negation  = {cfg_default.allow_goal_negation}      (¬Goal[…] allowed — when safe)",
        "",
        "ON by default (Stage 3-A): goal_predicate_relevance, require_goal_var_connected",
        "  (the latter is vacuous-by-construction here — goal lits introduce no vars)",
        "",
        "→ L(Γ_cfg) = ordered first-applicable decision lists of ≤ R such rules",
    ]
    ax.text(0.035, 0.80, "\n".join(caps_lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=9.6, family="monospace", color="#202124")

    # --- panel B: how big is the program space -----------------------------
    ax = axd["space"]
    ax.add_patch(FancyBboxPatch((0.02, 0.875), 0.96, 0.105, boxstyle="round,pad=0.012",
                                transform=ax.transAxes, fc=HOLE_FACE["pre_lit"],
                                ec=HOLE_EDGE["pre_lit"], lw=2.0))
    ax.text(0.5, 0.925, "how big is the program space?", transform=ax.transAxes,
            ha="center", va="center", fontsize=13.5, fontweight="bold", family="monospace")
    space_lines = [
        f"action-head width   M(cfg, Σ) = compute_max_productions = {M}",
        f"   (the pre_lit hole dominates — a state literal may take a fresh",
        f"    body-local var in any position; {M_legacy} with legacy_grammar_config(),",
        f"    i.e. both safety flags off)",
        "",
        f"|Π_Γ,cfg|: order-of-magnitude reference from the OLD aux-var",
        f"   grammar's r1 config — ≈{n_r1:,} distinct policies (legacy/02.md",
        f"   Table 2); not re-enumerated for the occurrence-introduced grammar",
        "",
        f"depth: R ≥ 3 is required for ANY M_B-solver — the r1 (R=1)",
        f"   landscape sweep finds zero (legacy/02.md Table 2)",
        "",
        "the bottleneck is NOT capacity — it is solver density (≤ 0.1 %",
        "in the legacy aux-var landscape) and a deceptive sparse-reward",
        "landscape (§10–§11; legacy/02.md Figs L1, L2, L4)",
    ]
    ax.text(0.035, 0.80, "\n".join(space_lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=10.2, family="monospace", color="#202124")

    # --- panel C: rule-family gallery --------------------------------------
    ax = axd["gallery"]
    ax.add_patch(FancyBboxPatch((0.01, 0.935), 0.98, 0.055, boxstyle="round,pad=0.008",
                                transform=ax.transAxes, fc=HOLE_FACE["goal_lit"],
                                ec=HOLE_EDGE["goal_lit"], lw=2.0))
    ax.text(0.5, 0.9625,
            "rule families this grammar can build   (one real Rule per family, built through the grammar — nothing hand-transcribed)",
            transform=ax.transAxes, ha="center", va="center", fontsize=12.5,
            fontweight="bold", family="monospace")
    y_top, dy = 0.882, 0.158
    for name, sels, gloss in _FAMILY_GALLERY:
        _, st = _state_after(cfg_exp, sig, sels)
        rule = st.completed_rules[0]
        ax.add_patch(FancyBboxPatch((0.02, y_top - dy + 0.030), 0.96, dy - 0.026,
                                    boxstyle="round,pad=0.006", transform=ax.transAxes,
                                    fc="#fbfbfd", ec="#cfcfcf", lw=1.0))
        ax.text(0.035, y_top, name, transform=ax.transAxes, ha="left", va="top",
                fontsize=11.5, fontweight="bold", color=HOLE_EDGE["goal_lit"], family="monospace")
        ax.text(0.035, y_top - 0.050, "    " + rule.pretty(), transform=ax.transAxes,
                ha="left", va="top", fontsize=12.0, family="monospace", color="#202124")
        ax.text(0.035, y_top - 0.098, "  ↳ " + gloss, transform=ax.transAxes, ha="left",
                va="top", fontsize=9.6, family="sans-serif", color="#5f6368", style="italic")
        y_top -= dy

    fig.suptitle(
        "Figure D8 — the hypothesis class  L(Γ_cfg):  an ordered list of ≤ R rules, each  "
        "(≤ L_S preconditions ∧ ≤ L_G goal literals, with ≤ V_b body-local vars)  ⇒  one action schema\n"
        f"the four caps + the disabled switches fix what is expressible; the space is small "
        f"(M = {M} productions wide; ≈{n_r1:,} policies at the old aux-var r1 config) — the difficulty is density / landscape, not capacity",
        fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    out = OUT / "02_derivation_hypothesis_class.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure D4 — what a state looks like: six representative states, every field
# ---------------------------------------------------------------------------

def _fmt_vars(vs) -> str:
    return "(" + ", ".join(f"{v.name}:{v.type_name}" for v in vs) + ")" if vs else "()"


def _fmt_lits(ls) -> str:
    return "(" + ", ".join(l.pretty() for l in ls) + ")" if ls else "()"


def _state_card_lines(st) -> list[str]:
    """Monospace lines describing *every* field of a ``LiftedDerivationState``."""
    cr = st.completed_rules
    n = len(cr)
    cr_repr = ("(" + ", ".join(f"ρ_{i + 1}" for i in range(n)) + (",)" if n == 1 else ")")) if n else "()"
    lines = [
        "LiftedDerivationState(",
        f"  completed_rules = {cr_repr}   # {n} sealed rule" + ("" if n == 1 else "s"),
    ]
    p = st.partial
    if p is None:
        lines.append("  partial         = None")
    else:
        lines.append("  partial         = PartialRule(")
        lines.append(f"      schema          = {p.schema!r}" if p.schema is not None else "      schema          = None")
        lines.append(f"      action_args     = {_fmt_vars(p.action_args)}")
        lines.append(f"      body_local_vars = {_fmt_vars(p.body_local_vars)}")
        lines.append(f"      state_lits      = {_fmt_lits(p.state_lits)}")
        lines.append(f"      goal_lits       = {_fmt_lits(p.goal_lits)}  )")
    ch = st.current_hole
    lines.append(f"  current_hole    = {ch!r}" + ("   (= ⊥ — terminal)" if ch is None else ""))
    lines.append(")")
    for i, r in enumerate(cr):
        lines.append(f"  where ρ_{i + 1} = {r.pretty()}")
    lines.append("")
    lines.append(f"hashable_obs / pretty() = {st.pretty()}")
    if st.is_terminal():
        lines.append(f"to_program() = Policy: {st.to_program().pretty()}")
    return lines


_ANATOMY = [
    ("s₀ — the initial state\n(no rules, no rule in progress)",
     []),
    ("after  ADD_RULE → schema=move\n(action chosen; action_args = (?r_0, ?r_1) — no body-local vars yet)",
     [dict(label="ADD_RULE"), dict(label="schema=move")]),
    ("after  … → pre:at_robot(?r_0)\n(one state precondition over an action argument)",
     [dict(label="ADD_RULE"), dict(label="schema=move"), dict(pred="at_robot")]),
    ("after  … → pre:carrying(?v_0)\n(this literal *introduces* ?v_0:ball — body_local_vars grows)",
     [dict(label="ADD_RULE"), dict(label="schema=move"), dict(pred="at_robot"),
      dict(pred="carrying")]),
    ("after  … → STOP_PRE → goal:Goal[at_ball(?v_0, ?r_1)]\n(one goal literal — references only in-scope vars; introduces none)",
     [dict(label="ADD_RULE"), dict(label="schema=move"), dict(pred="at_robot"),
      dict(pred="carrying"), dict(label="STOP_PRE"),
      dict(pred="at_ball", negated=False, argnames=("?v_0", "?r_1"))]),
    ("terminal — after  … → FINISH_RULE → STOP_POLICY\n(rule ρ₂ sealed; policy complete)",
     [dict(label="ADD_RULE"), dict(label="schema=move"), dict(pred="at_robot"),
      dict(pred="carrying"), dict(label="STOP_PRE"),
      dict(pred="at_ball", negated=False, argnames=("?v_0", "?r_1")),
      dict(label="FINISH_RULE"), dict(label="STOP_POLICY")]),
]


def fig_state_anatomy() -> Path:
    cfg, sig = _grammar()
    fig, axes = plt.subplots(2, 3, figsize=(22.0, 10.6))
    for ax, (title, selectors) in zip(axes.flat, _ANATOMY):
        _, st = _state_after(cfg, sig, selectors)
        hole = st.current_hole
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.015, 0.855), 0.97, 0.13, boxstyle="round,pad=0.012",
                                    transform=ax.transAxes, fc=HOLE_FACE[hole],
                                    ec=HOLE_EDGE[hole], lw=2.0, zorder=1))
        ax.text(0.5, 0.93, title, transform=ax.transAxes, ha="center", va="center",
                fontsize=9.4, fontweight="bold", family="monospace", color="#202124", zorder=2)
        ax.text(0.5, 0.832, f"current_hole = {hole!r}", transform=ax.transAxes, ha="center",
                va="center", fontsize=9.4, color=HOLE_EDGE[hole], fontweight="bold",
                family="monospace", zorder=2)
        body = "\n".join(_state_card_lines(st))
        ax.text(0.03, 0.80, body, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.1, family="monospace", color="#202124")
    fig.suptitle(
        "Figure D4 — what a derivation state is, explicitly: six representative LiftedDerivationStates, "
        "every field shown\n"
        "s = (completed_rules, partial, current_hole);  partial is a PartialRule"
        "(schema, action_args, body_local_vars, state_lits, goal_lits) while a rule is being built, else None;  "
        "current_hole ∈ {policy, action_schema, pre_lit, goal_lit, ⊥}  —  exactly one open hole;  body_local_vars "
        "grows when a pre:ℓ literal introduces a fresh ?v_i  (values read live off the state object; pretty() is the MCTS node key)",
        fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT / "02_derivation_state_anatomy.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure D5 — how the state evolves: a field-by-field diff along one derivation
# ---------------------------------------------------------------------------

def _walk(cfg, sig, selectors):
    """Return ``[(state, production_that_produced_it), ...]`` from the initial
    state (production ``None``) through ``selectors`` applied in order."""
    st = LiftedDerivationState.initial()
    out = [(st, None)]
    for sel in selectors:
        prod = _pick(enumerate_productions(st, cfg, sig), **sel)
        st = st.apply(prod)
        out.append((st, prod))
    return out


def _evo_cells(st) -> list[str]:
    """The 7 field cells for the diff table; partial sub-fields collapse to '·'
    when ``partial`` is None (no rule in progress)."""
    p = st.partial
    n = len(st.completed_rules)
    cr = ("(" + ",".join(f"ρ_{i + 1}" for i in range(n)) + (",)" if n == 1 else ")")) if n else "()"
    if p is None:
        sub = ["·", "·", "·", "·", "·"]
    else:
        sub = [
            (repr(p.schema) if p.schema is not None else "None"),
            (", ".join(v.name for v in p.action_args) if p.action_args else "()"),
            (", ".join(v.name for v in p.body_local_vars) if p.body_local_vars else "()"),
            (" ∧ ".join(l.pretty() for l in p.state_lits) if p.state_lits else "()"),
            (" ∧ ".join(l.pretty() for l in p.goal_lits) if p.goal_lits else "()"),
        ]
    hole = st.current_hole if st.current_hole is not None else "⊥"
    return [cr] + sub + [hole]


def fig_state_evolution() -> Path:
    cfg, sig = _grammar()
    trace = _walk(cfg, sig, _RHO2)                       # 9 states, 8 productions
    rho2 = trace[-1][0].completed_rules[0].pretty()

    field_titles = ["completed_rules", "partial.schema", "partial.action_args",
                    "partial.body_local_vars", "partial.state_lits", "partial.goal_lits", "current_hole"]
    headers = ["state", "production applied"] + field_titles
    widths = [0.042, 0.150, 0.066, 0.058, 0.096, 0.118, 0.215, 0.122, 0.133]   # sums to 1.0
    starts, acc = [], 0.0
    for w in widths:
        starts.append(acc)
        acc += w

    n = len(trace)
    fig, ax = plt.subplots(figsize=(19.5, 0.62 * (n + 2) + 1.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.7, n + 1.4)
    ax.axis("off")

    def row_y(i):                                        # i = 0 is the header
        return (n + 0.4) - i

    # header
    ax.add_patch(Rectangle((0, row_y(0) - 0.45), 1.0, 0.9, fc="#eceff1", ec="#bdc1c6", lw=0.8, zorder=0))
    for h, s in zip(headers, starts):
        ax.text(s + 0.004, row_y(0), h, ha="left", va="center", fontsize=7.8, fontweight="bold",
                family="monospace", color="#202124")
    ax.axhline(row_y(0) + 0.45, color="#bdc1c6", lw=0.8)
    ax.axhline(row_y(0) - 0.45, color="#bdc1c6", lw=0.8)

    prev = None
    for i, (st, prod) in enumerate(trace):
        y = row_y(i + 1)
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, y - 0.45), 1.0, 0.9, fc="#fafafa", ec="none", zorder=0))
        cells = _evo_cells(st)
        if prev is not None:
            for k, (a, b) in enumerate(zip(prev, cells)):
                if a != b:
                    ax.add_patch(Rectangle((starts[k + 2], y - 0.43), widths[k + 2], 0.86,
                                           fc="#fff3c4", ec="none", zorder=0))
        prev = cells
        row_vals = [f"s{_SUB[i]}", ("—  (initial)" if prod is None else _fmt_prod(prod.label))] + cells
        for k, (val, s) in enumerate(zip(row_vals, starts)):
            if val == "·":
                color, fw = "#9aa0a6", "normal"
            elif k == 8 and val == "⊥":
                color, fw = "#8430ce", "bold"
            elif k == 0:
                color, fw = "#202124", "bold"
            else:
                color, fw = "#202124", "normal"
            ax.text(s + 0.004, y, val, ha="left", va="center", fontsize=7.3, family="monospace",
                    color=color, fontweight=fw)
        ax.axhline(y - 0.45, color="#ececec", lw=0.5, zorder=0)

    ax.text(0.0, -0.45, f"ρ₂  =  {rho2}", ha="left", va="center", fontsize=8.4, family="monospace",
            color="#8430ce", fontweight="bold")
    ax.text(0.0, -1.05, "highlighted cell = the one field the production changed;   "
            "“·” = partial is None (no rule in progress);   ⊥ = terminal (current_hole is None);   "
            "pre:carrying(?v_0) changes BOTH partial.state_lits and partial.body_local_vars",
            ha="left", va="center", fontsize=7.8, color="#5f6368")
    ax.set_title("Figure D5 — how a derivation state evolves: a field-by-field diff along the build of "
                 "hand-policy rule ρ₂ (move-toward-goal)\n"
                 "8 productions (then STOP_POLICY); each row is a state s_i, each highlighted cell the field "
                 "the production δ changed — note ?v_0:ball is born inside pre:carrying(?v_0)",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    out = OUT / "02_derivation_state_evolution.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure D6 — production-sequence lengths
# ---------------------------------------------------------------------------

_PER_RULE_SEGMENTS = [   # (label, colour, group)  — bottom-to-top build order of one rule
    ("ADD_RULE",                          "#1a73e8", "fixed"),
    ("schema = X",                        "#d93025", "fixed"),
    ("pre : lit 1   (may introduce ?v_i)", "#34a853", "pre"),
    ("pre : lit 2",                       "#62bd77", "pre"),
    ("pre : lit 3",                       "#90d3a0", "pre"),
    ("STOP_PRE",                          "#188038", "fixed"),
    ("goal : lit 1",                      "#8430ce", "goal"),
    ("FINISH_RULE",                       "#6a1b9a", "fixed"),
]


def _random_lengths(cfg, sig, n=5000, seed=0):
    """Lengths (production counts) of ``n`` uniform-random complete derivations."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        st = LiftedDerivationState.initial()
        length = 0
        while not st.is_terminal():
            prods = enumerate_productions(st, cfg, sig)
            if not prods:                                # unreachable (every non-terminal has ≥1)
                break
            st = st.apply(rng.choice(prods))
            length += 1
        out.append(length)
    return out


def fig_production_lengths() -> Path:
    cfg, sig = _grammar()
    P, G = cfg.max_pre_literals, cfg.max_goal_literals          # 3, 1
    FIXED = 4                                                   # ADD_RULE, schema, STOP_PRE, FINISH_RULE
    per_rule_min, per_rule_max = FIXED, FIXED + P + G           # 4, 8

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.6))

    # --- (a) one rule, stacked by build order ---
    ax = axes[0]
    for i, (name, color, grp) in enumerate(_PER_RULE_SEGMENTS):
        ax.bar(0, 1.0, bottom=i, width=0.42, color=color, edgecolor="white", lw=0.8,
               hatch=("" if grp == "fixed" else "//"))
        ax.text(0.27, i + 0.5, name, ha="left", va="center", fontsize=8.4, family="monospace")
    ax.plot([-0.30, -0.30], [2, 5], color="#188038", lw=2.4)
    ax.text(-0.36, 3.5, "≤ P = 3\npre-literals", ha="right", va="center", fontsize=8.2, color="#188038")
    ax.plot([-0.30, -0.30], [6, 7], color="#8430ce", lw=2.4)
    ax.text(-0.36, 6.5, "≤ G = 1\ngoal-literal", ha="right", va="center", fontsize=8.2, color="#8430ce")
    ax.text(0.3, -0.6, "solid = fixed       hatched = variable  (0 … P  /  0 … G)", ha="center",
            va="center", fontsize=8.0, color="#5f6368")
    ax.set_xlim(-1.25, 2.2)
    ax.set_ylim(-1.0, 8.6)
    ax.set_xticks([])
    ax.set_yticks(range(0, 9))
    ax.set_ylabel("productions, in build order")
    ax.set_title(f"(a) one rule  =  {FIXED} fixed  +  ≤P pre-lits  +  ≤G goal-lits\n"
                 f"→  {per_rule_min} … {per_rule_max} productions per rule  (Gripper-lite: P={P}, G={G})  "
                 f"— no aux phase",
                 fontsize=9.0)

    # --- (b) complete-policy production count vs the grammar cap R ---
    ax = axes[1]
    Rs = list(range(1, 5))
    lmax = [R * (P + G + FIXED) + 1 for R in Rs]                # [9, 17, 25, 33]
    lmin = [FIXED + 1] * len(Rs)                                # 5 (one empty rule + STOP_POLICY), any R
    ax.fill_between(Rs, lmin, lmax, color="#e8f0fe", alpha=0.95, label="reachable  ℓ(π)")
    ax.plot(Rs, lmax, "-o", color="#1a73e8", lw=1.9, label="max = R·(P+G+4)+1")
    ax.plot(Rs, lmin, "--", color="#5f6368", lw=1.5, label="min = 5  (one empty rule + STOP_POLICY)")
    for R, m in zip(Rs, lmax):
        ax.annotate(str(m), (R, m), textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8.6, color="#1a73e8")
    ax.scatter([3], [25], s=80, facecolor="none", edgecolor="#d93025", lw=1.9, zorder=5)
    ax.annotate("experiments' grammar (R=3)", (3, 25), textcoords="offset points", xytext=(-8, -20),
                ha="center", fontsize=8.0, color="#d93025")
    ax.scatter([4], [33], s=80, facecolor="none", edgecolor="#d93025", lw=1.9, zorder=5)
    ax.annotate("default / Run B (R=4)", (4, 33), textcoords="offset points", xytext=(-40, -16),
                ha="center", fontsize=8.0, color="#d93025")
    ax.axhline(8, color="#8430ce", lw=1.3, ls=":")
    ax.text(1.05, 9.1, "ρ₂-only example (Fig. D5):  ℓ = 8", fontsize=7.9, color="#8430ce")
    ax.set_xticks(Rs)
    ax.set_xlim(0.7, 4.5)
    ax.set_xlabel("grammar cap   R = max_rules")
    ax.set_ylabel("ℓ(π)  =  #productions to build a complete policy")
    ax.set_ylim(0, 38)
    ax.set_title("(b) complete-policy production count\nℓ(π) = 1 + Σᵢ ( 4 + |preᵢ| + |goalᵢ| ),   1 ≤ k ≤ R",
                 fontsize=9.5)
    ax.legend(fontsize=7.7, loc="upper left")
    ax.grid(alpha=0.25)

    # --- (c) empirical length distribution under a uniform prior, R=3 ---
    ax = axes[2]
    lengths = _random_lengths(cfg, sig, n=5000, seed=0)
    lo, hi = min(lengths), max(lengths)
    mean = sum(lengths) / len(lengths)
    ax.hist(lengths, bins=range(5, 28), color="#1a73e8", alpha=0.85, edgecolor="white", lw=0.6)
    ax.axvline(mean, color="#d93025", lw=2.1, label=f"mean = {mean:.1f}")
    ax.axvline(25, color="#5f6368", lw=1.3, ls="--", label="analytic max = 25  (R=3)")
    ax.axvline(5, color="#5f6368", lw=1.3, ls=":", label="analytic min = 5")
    ax.set_xlabel("ℓ(π)  =  #productions   (R=3, uniform-random derivation)")
    ax.set_ylabel("count   (of 5000 random complete policies)")
    ax.set_title("(c) the uniform prior favours short policies\n5000 uniform-random derivations, "
                 f"max_rules=3   (observed range [{lo}, {hi}])", fontsize=9.5)
    ax.legend(fontsize=7.9)

    fig.suptitle("Figure D6 — how long is a derivation?  per-rule structure (a), the analytic length range "
                 "ℓ(π) ∈ [5, R·(P+G+4)+1] (b), and the empirical distribution under a uniform prior (c)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "02_derivation_production_lengths.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure D7 — one full derivation of a 3-rule policy as a grammar parse tree
# ---------------------------------------------------------------------------

# The three Stage-1 hand-policy rules, each as a selector segment (no trailing
# STOP_POLICY). ρ₁ — drop a carried ball when it is already at its goal room;
# ρ₂ — move toward the room a carried ball should go to (its ?v_0:ball is a
# body-local variable, born inside carrying(?v_0)); ρ₃ — pick up a misplaced ball
# you are standing on with a free hand.
_RHO1_SEG = [
    dict(label="ADD_RULE"),
    dict(label="schema=drop"),
    dict(pred="at_robot"),                                       # at_robot(?r_1)
    dict(pred="carrying"),                                       # carrying(?b_0)
    dict(label="STOP_PRE"),
    dict(pred="at_ball", negated=False),                         # Goal[at_ball(?b_0, ?r_1)]
    dict(label="FINISH_RULE"),
]
_RHO2_SEG = [
    dict(label="ADD_RULE"),
    dict(label="schema=move"),
    dict(pred="at_robot"),                                       # at_robot(?r_0)
    dict(pred="carrying"),                                       # carrying(?v_0)  ← introduces ?v_0:ball
    dict(label="STOP_PRE"),
    dict(pred="at_ball", negated=False, argnames=("?v_0", "?r_1")),  # Goal[at_ball(?v_0, ?r_1)]
    dict(label="FINISH_RULE"),
]
_RHO3_SEG = [
    dict(label="ADD_RULE"),
    dict(label="schema=pick"),
    dict(pred="at_ball", negated=False),                         # at_ball(?b_0, ?r_1)
    dict(pred="at_robot"),                                       # at_robot(?r_1)
    dict(pred="handempty"),                                      # handempty()
    dict(label="STOP_PRE"),
    dict(pred="at_ball", negated=True),                          # ¬Goal[at_ball(?b_0, ?r_1)]
    dict(label="FINISH_RULE"),
]
_POLICY_DERIV = _RHO1_SEG + _RHO2_SEG + _RHO3_SEG + [dict(label="STOP_POLICY")]


class _PNode:
    """A node of the rendered parse tree. ``kind`` is a hole-kind key (for the
    leaf colour) or ``"nt"`` for a nonterminal. ``sub`` is an optional small
    annotation drawn under the node (used to show ρ_i's pretty form)."""
    __slots__ = ("label", "kind", "children", "sub", "x", "y", "depth")

    def __init__(self, label, kind, sub=None):
        self.label = label
        self.kind = kind
        self.sub = sub
        self.children: list["_PNode"] = []
        self.x = 0.0
        self.y = 0.0
        self.depth = 0


def _build_policy_tree(steps, final) -> _PNode:
    """Fold the production sequence ``steps`` (= ``[(state_before, prod), ...]``)
    into a parse tree mirroring the grammar skeleton
    ``⟨Policy⟩ → ⟨Rule⟩* STOP_POLICY``,
    ``⟨Rule⟩ → ADD_RULE ⟨ActionSchema⟩ ⟨PreList⟩ STOP_PRE ⟨GoalList⟩ FINISH_RULE``."""
    root = _PNode("⟨Policy⟩", "nt")
    rules = final.completed_rules
    cur_rule: _PNode | None = None
    seg: _PNode | None = None          # current ⟨PreList⟩ / ⟨GoalList⟩ accumulator
    rule_idx = 0
    for _st, prod in steps:
        hk, tag = prod.hole_kind, prod.payload[0]
        if hk == "policy":
            if tag == "add_rule":
                rule_idx += 1
                cur_rule = _PNode("⟨Rule⟩", "nt")
                root.children.append(cur_rule)
                cur_rule.children.append(_PNode("ADD_RULE", "policy"))
                seg = None
            else:                       # stop -> STOP_POLICY
                root.children.append(_PNode("STOP_POLICY", "policy"))
        elif hk == "action_schema":
            nt = _PNode("⟨ActionSchema⟩", "nt")
            cur_rule.children.append(nt)
            nt.children.append(_PNode(prod.label, "action_schema"))   # schema=move|pick|drop
            seg = None
        elif hk == "pre_lit":
            if tag == "add":
                if seg is None or seg.label != "⟨PreList⟩":
                    seg = _PNode("⟨PreList⟩", "nt")
                    cur_rule.children.append(seg)
                seg.children.append(_PNode(prod.label, "pre_lit"))    # pre:<literal>
            else:                       # stop_pre
                if seg is None or seg.label != "⟨PreList⟩":           # zero pre-lits -> ε
                    seg = _PNode("⟨PreList⟩", "nt")
                    cur_rule.children.append(seg)
                    seg.children.append(_PNode("ε", "nt"))
                cur_rule.children.append(_PNode("STOP_PRE", "pre_lit"))
                seg = None
        elif hk == "goal_lit":
            if tag == "add":
                if seg is None or seg.label != "⟨GoalList⟩":
                    seg = _PNode("⟨GoalList⟩", "nt")
                    cur_rule.children.append(seg)
                seg.children.append(_PNode(prod.label, "goal_lit"))   # goal:<literal>
            else:                       # finish -> FINISH_RULE
                if seg is None or seg.label != "⟨GoalList⟩":          # zero goal-lits -> ε
                    seg = _PNode("⟨GoalList⟩", "nt")
                    cur_rule.children.append(seg)
                    seg.children.append(_PNode("ε", "nt"))
                cur_rule.children.append(_PNode("FINISH_RULE", "goal_lit"))
                cur_rule.label = f"⟨Rule⟩ = ρ_{rule_idx}"
                cur_rule.sub = rules[rule_idx - 1].pretty()
                seg = None
    return root


def _layout_tree(root: _PNode) -> tuple[int, float]:
    """Left→right layout: x = tree depth, y = post-order leaf order (parents
    centred on their children). Returns ``(max_depth, n_leaves)``."""
    max_depth = [0]
    counter = [0.0]

    def walk(node: _PNode, depth: int) -> None:
        node.depth = depth
        node.x = float(depth)
        max_depth[0] = max(max_depth[0], depth)
        if not node.children:
            node.y = counter[0]
            counter[0] += 1.0
            return
        for c in node.children:
            walk(c, depth + 1)
        node.y = sum(c.y for c in node.children) / len(node.children)

    walk(root, 0)
    return max_depth[0], counter[0]


def _iter_nodes(node: _PNode):
    yield node
    for c in node.children:
        yield from _iter_nodes(c)


def fig_derivation_tree() -> Path:
    cfg, sig = _grammar()
    steps, final = _state_after(cfg, sig, _POLICY_DERIV)
    assert final.is_terminal(), "policy derivation did not terminate"
    assert len(final.completed_rules) == 3, final.completed_rules
    root = _build_policy_tree(steps, final)
    max_depth, n_leaves = _layout_tree(root)

    col_w = 4.7                                              # x-spacing per depth level
    row_h = 0.52                                             # y-spacing per leaf
    leaf_x = max_depth * col_w                               # x of the deepest column
    summary_x = leaf_x + 4.6                                 # right-hand "sealed rule" column
    fig_w = max(22.0, summary_x + 14.0)
    fig_h = max(7.0, row_h * (n_leaves + 3.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    def X(node):
        return node.x * col_w

    # edges first (under the nodes)
    for node in _iter_nodes(root):
        for c in node.children:
            ax.plot([X(node), X(c)], [node.y, c.y], color="#bdc1c6", lw=1.2,
                    solid_capstyle="round", zorder=1)

    # nodes
    for node in _iter_nodes(root):
        x, y = X(node), node.y
        if node.children or node.kind == "nt":              # nonterminal -> rounded box
            is_rule = node.label.startswith("⟨Rule⟩")
            face = "#fef6e0" if is_rule else ("#ffffff" if node.label == "ε" else "#f1f3f4")
            _box(ax, x, y, max(1.7, 0.165 * len(node.label) + 0.55), 0.52, node.label,
                 face=face, edge=("#b06000" if is_rule else "#5f6368"), fontsize=9.4,
                 fontweight=("bold" if is_rule else "normal"), family="serif", lw=1.3, pad=0.12)
        else:                                               # leaf -> coloured marker + label
            ax.scatter([x], [y], s=130, marker="s", facecolor=HOLE_FACE[node.kind],
                       edgecolor=HOLE_EDGE[node.kind], linewidths=1.5, zorder=3)
            ax.text(x + 0.18, y, node.label, ha="left", va="center", fontsize=8.6,
                    family="monospace", color="#202124", zorder=3)

    # right-hand summary: the sealed form of each ⟨Rule⟩ subtree
    for rn in root.children:
        if rn.sub is None:
            continue
        ax.plot([leaf_x + 2.4, summary_x - 0.18], [rn.y, rn.y], color="#dadce0", lw=0.9,
                ls=(0, (1, 2)), zorder=0)
        name = rn.label.split("=")[-1].strip()              # "⟨Rule⟩ = ρ_1" -> "ρ_1"
        ax.text(summary_x, rn.y, f"{name}:  {rn.sub}", ha="left", va="center", fontsize=8.0,
                family="monospace", color="#8430ce",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f7f2fc", ec="#8430ce", lw=0.9))

    # start-symbol arrow into ⟨Policy⟩
    ax.annotate("", xy=(X(root) - 0.05, root.y), xytext=(X(root) - col_w * 0.55, root.y),
                arrowprops=dict(arrowstyle="-|>", color="#202124", lw=1.7))
    ax.text(X(root) - col_w * 0.58, root.y, "start", ha="right", va="center", fontsize=10.5,
            fontweight="bold")

    # legend (hole-kind colours + nonterminal)
    handles = [Patch(facecolor=HOLE_FACE[h], edgecolor=HOLE_EDGE[h], label=f"{h}-hole production")
               for h in HOLE_ORDER]
    handles += [
        Patch(facecolor="#fef6e0", edgecolor="#b06000", label="nonterminal  ⟨Rule⟩"),
        Patch(facecolor="#f1f3f4", edgecolor="#5f6368", label="nonterminal  ⟨ActionSchema⟩ / ⟨PreList⟩ / ⟨GoalList⟩"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.4, framealpha=0.96)

    ax.set_xlim(-col_w * 0.85, fig_w - 0.5)
    ax.set_ylim(-2.0, n_leaves + 0.6)
    ax.invert_yaxis()                                       # first leaf at the top
    ax.set_title(
        "Figure D7 — one full derivation of a 3-rule policy, as a grammar parse tree (no ⟨Aux⟩ — "
        "variables are introduced by literal occurrence)\n"
        "⟨Policy⟩ → ⟨Rule⟩* STOP_POLICY ;   ⟨Rule⟩ → ADD_RULE ⟨ActionSchema⟩ ⟨PreList⟩ "
        "STOP_PRE ⟨GoalList⟩ FINISH_RULE.   Leaves top-to-bottom = the production sequence "
        "p₁…p_T; colour = hole kind; right column = the sealed rule ρᵢ each ⟨Rule⟩ subtree yields "
        "(ρ₁ drop-at-goal · ρ₂ move-toward-goal, with body-local ?v_0 · ρ₃ pick — all read live from the grammar).",
        fontsize=10.0, fontweight="bold")
    fig.tight_layout()
    out = OUT / "02_derivation_tree.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = (
        fig_state_machine, fig_example_derivation, fig_production_sets,
        fig_hypothesis_class,
        fig_state_anatomy, fig_state_evolution, fig_production_lengths,
        fig_derivation_tree,
    )
    for fn in figures:
        print(f"wrote {fn()}")


if __name__ == "__main__":
    main()
