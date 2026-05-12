#!/usr/bin/env python3
"""Stage-2 (lifted grammar × MCTS) figures for ``docs/notes/stage4/02.md``.

Produces, into ``docs/notes/stage4/figures/``:

- ``02_mcts_search_tree.png`` — *how MCTS drives the lifted derivation*: the
  reconstructed search tree (``MCTS.nodes``) rooted at a mid-derivation state,
  drawn at three simulation budgets (16 / 64 / 128); node area ∝ visit count,
  fill ∝ backed-up Q of the incoming edge, edge width ∝ visit share.
- ``02_uniform_prior_exploration.png`` — *how the uniform prior explores (and
  stalls)*: a 2×3 diagnostic panel built from the canonical-run JSONL
  (best-so-far curve, score histogram, score-by-size, score CDF Run A vs B,
  grammar branching by depth, more-search-≠-better-solver bars).
- ``02_learned_policies_runs.png`` — *what the learned policies are*: rule-by-rule
  "policy cards" for the best B=1 solver (Run A) and the degenerate do-nothing
  attractor (Run B).
- ``02_learned_policies_hand.png`` — the Stage-1 hand policy as a reference
  policy card (the grammar can express this up to α-renaming).
- ``02_learned_policy_rollout.png`` — the best learned B=1 policy run through
  the Gripper-lite environment: it solves B=1 in 3 steps and loops to the
  horizon on B=2 (re-picks an already-placed ball).
- ``02_graded_reward_runA.png`` — the graded leaf-reward landscape over every
  policy Run A evaluated (regenerated; has a generator now).

Reads the committed canonical artifacts under ``docs/notes/stage4/data/``
(``{stem}_scores.csv``, ``{stem}_best.jsonl``, ``summary.json``) — regenerate
with ``python scripts/run/make_lifted_gripper_canonical.py``. Figures 1 and 5
also run a fresh, deterministic uniform-MCTS pass (seed 0).

Run from repo root:

    python scripts/plotting/plot_lifted_gripper_mcts.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_PLOT_DIR = REPO_ROOT / "scripts" / "plotting"
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

from alphazeropp.core.mcts import MCTS  # noqa: E402
from alphazeropp.instances.gripper_lite.env import GripperLiteEnv  # noqa: E402
from alphazeropp.instances.gripper_lite.policies import hand_policy  # noqa: E402
from alphazeropp.synthesis.derivation_game import UniformPolicyValueNet  # noqa: E402
from alphazeropp.synthesis.lifted_derivation import (  # noqa: E402
    LiftedDerivationGame,
    LiftedDerivationState,
)
from alphazeropp.synthesis.lifted_grammar import (  # noqa: E402
    LiftedGrammarConfig,
    gripper_lite_signature,
)
from alphazeropp.synthesis.lifted_interpreter import interpret  # noqa: E402
from alphazeropp.synthesis.lifted_leaf_evaluator import LiftedLeafEvaluator  # noqa: E402
from plot_gripper_lite_rollout import _draw_frame  # noqa: E402

OUT = REPO_ROOT / "docs" / "notes" / "stage4" / "figures"
DATA = REPO_ROOT / "docs" / "notes" / "stage4" / "data"
DPI = 160

RUN_A = "runA_seed0_sims128"
RUN_B = "runB_seed0_sims512"

RUN_A_COLOR = "#1f77b4"
RUN_B_COLOR = "#d62728"
HAND_COLOR = "#2ca02c"


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

_SCORE_COLS = ("score", "train_solve_rate", "eval_out_solve_rate", "num_rules", "num_literals")


def _load_scores(stem: str) -> dict[str, np.ndarray]:
    """Per-policy metrics (discovery order) from ``docs/notes/stage4/data/{stem}_scores.csv``."""
    cols: dict[str, list] = {c: [] for c in _SCORE_COLS}
    with (DATA / f"{stem}_scores.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            for c in _SCORE_COLS:
                cols[c].append(float(row[c]))
    return {c: np.asarray(v) for c, v in cols.items()}


def _load_summary() -> dict:
    return json.loads((DATA / "summary.json").read_text())


def _grammar_setup(max_rules: int = 3):
    cfg = LiftedGrammarConfig(max_rules=max_rules)
    sig = gripper_lite_signature()
    train = [GripperLiteEnv(n_balls=1, seed=0)]
    eval_out = [GripperLiteEnv(n_balls=2, seed=0)]
    evaluator = LiftedLeafEvaluator(train, train, eval_out)
    net = UniformPolicyValueNet(LiftedDerivationGame(cfg, sig, evaluator)._max_productions)
    return cfg, sig, evaluator, net


# ---------------------------------------------------------------------------
# Figure 1 — MCTS search-tree snapshots
# ---------------------------------------------------------------------------

# Deterministic prefix that lands the derivation at the first rule's goal-literal
# hole — wide-branching (≈9 productions) and close enough to terminals that the
# subtree contains complete one-rule policies (with their leaf scores).
_PREFIX_LABELS = ["ADD_RULE", "schema=pick", "SKIP_AUX", "STOP_PRE"]
_SNAPSHOT_DESC = "partial rule  ⊤ ⇒ pick(?b_0, ?r_1)  at the goal-literal hole"
_TREE_BUDGETS = (16, 64, 128)
_TREE_TOPK = 4
_TREE_MAXDEPTH = 6


def _label_to_action(game: LiftedDerivationGame, label: str) -> int:
    for i, prod in enumerate(game._current_productions):
        if prod.label == label:
            return i
    raise KeyError(f"production {label!r} not legal here: have "
                   f"{[p.label for p in game._current_productions]}")


def _capture_tree(cfg, sig, evaluator, net, sims: int):
    game = LiftedDerivationGame(cfg, sig, evaluator)
    game.reset_wrapper()
    for lab in _PREFIX_LABELS:
        game.step_wrapper(_label_to_action(game, lab))
    mcts = MCTS(game, net, n_simulations=sims, temperature=1.0, c_exploration=1.5)
    mcts.perform_simulations(None)  # restores game to the snapshot state on exit
    return mcts, game


def _abbrev_label(label: str) -> str:
    fixed = {"ADD_RULE": "+rule", "STOP_POLICY": "STOP", "SKIP_AUX": "−aux",
             "STOP_PRE": "end-pre", "FINISH_RULE": "⇒FIN"}
    if label in fixed:
        return fixed[label]
    if label.startswith("schema="):
        return "=" + label[len("schema="):]
    if label.startswith("add_aux:"):
        return "+aux:" + label[len("add_aux:"):][0]
    if label.startswith(("pre:", "goal:")):
        kind = "+s" if label.startswith("pre:") else "+g"
        body = label.split(":", 1)[1]
        body = (body.replace("¬Goal[", "¬").replace("Goal[", "").replace("]", "")
                .replace("?b_", "b").replace("?r_", "r").replace("?aux_", "a")
                .replace(" ", ""))
        return f"{kind} {body}"
    if label.startswith("…"):
        return label
    return label


class _TNode:
    __slots__ = ("total_n", "direct_reward", "is_terminal", "is_stub", "depth",
                 "children", "y")

    def __init__(self, total_n, direct_reward, is_terminal, depth, is_stub=False):
        self.total_n = total_n
        self.direct_reward = direct_reward
        self.is_terminal = is_terminal
        self.is_stub = is_stub
        self.depth = depth
        self.children: list[tuple[str, int, float, "_TNode"]] = []
        self.y = 0.0


def _build_tree(mcts: MCTS, game: LiftedDerivationGame, depth: int = 0) -> _TNode:
    node = mcts.nodes.get(game.hashable_obs)
    if node is None:
        is_term = bool(game.terminated or game.truncated)
        return _TNode(0, (game.reward if is_term else None), is_term, depth)
    t = _TNode(node.total_N, node.direct_reward, node.is_terminal_state, depth)
    if node.is_terminal_state or depth >= _TREE_MAXDEPTH or not node.action_N:
        return t
    edges = []
    for akey, n in node.action_N.items():
        a = int(akey[0]) if isinstance(akey, (tuple, list, np.ndarray)) else int(akey)
        q = float(node.action_Q.get(akey, 0.0))
        child_game = game.clone()
        child_game.step_wrapper(a)
        edges.append((child_game.info["production"], n, q, child_game))
    edges.sort(key=lambda e: e[1], reverse=True)
    for (label, n, q, child_game) in edges[:_TREE_TOPK]:
        t.children.append((label, n, q, _build_tree(mcts, child_game, depth + 1)))
    pruned = edges[_TREE_TOPK:]
    if pruned:
        stub = _TNode(sum(e[1] for e in pruned), None, False, depth + 1, is_stub=True)
        t.children.append((f"…(+{len(pruned)} more)", sum(e[1] for e in pruned), 0.0, stub))
    return t


def _layout(t: _TNode, y0: float = 0.0) -> float:
    if not t.children:
        t.y = y0 + 0.5
        return 1.0
    cy = y0
    centers = []
    for (_lab, _n, _q, ct) in t.children:
        h = _layout(ct, cy)
        centers.append(ct.y)
        cy += h
    t.y = sum(centers) / len(centers)
    return cy - y0


def _iter_tree(t: _TNode):
    yield t
    for (_lab, _n, _q, ct) in t.children:
        yield from _iter_tree(ct)


def _draw_tree(ax, root: _TNode, cmap, qmin: float, qmax: float):
    norm = Normalize(qmin, qmax) if qmax > qmin else Normalize(qmin - 1e-3, qmax + 1e-3)

    def msize(n):                      # marker area (pts^2)
        return 24.0 + 95.0 * np.sqrt(max(n, 0))

    def rec(node: _TNode, n_in: int, q_in):
        x, y = node.depth, node.y
        for (label, n, q, ct) in node.children:
            if ct.is_stub:
                ax.plot([x, ct.depth], [y, ct.y], ":", color="#bdbdbd", lw=0.8, zorder=1)
            else:
                share = (n / node.total_n) if node.total_n else 0.0
                ax.plot([x, ct.depth], [y, ct.y], "-", color="#9aa0a6",
                        lw=0.5 + 4.5 * share, solid_capstyle="round", zorder=1)
            ax.text((x + ct.depth) / 2, (y + ct.y) / 2, _abbrev_label(label),
                    fontsize=6.2, ha="center", va="center", fontfamily="monospace",
                    color="#1a1a1a", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#cfcfcf",
                              lw=0.3, alpha=0.92))
            rec(ct, n, q)
        if node.is_stub:
            ax.scatter([x], [y], s=30, marker="x", color="#bdbdbd", lw=1.0, zorder=5)
            return
        face = cmap(norm(q_in)) if q_in is not None else "#b8c4d0"
        if node.is_terminal:
            ax.scatter([x], [y], s=msize(n_in), marker="s", color=face,
                       edgecolor="black", lw=0.5, zorder=5)
            if node.direct_reward is not None:
                ax.annotate(f"{node.direct_reward:+.2f}", (x, y),
                            textcoords="offset points", xytext=(7, 0), fontsize=5.6,
                            ha="left", va="center", fontfamily="monospace", zorder=6)
        else:
            ax.scatter([x], [y], s=msize(n_in), marker="o", color=face,
                       edgecolor="black", lw=0.5, zorder=5)

    rec(root, root.total_n, None)


def fig_search_tree() -> Path:
    cfg, sig, evaluator, net = _grammar_setup(max_rules=3)
    trees = []
    all_q = []
    for sims in _TREE_BUDGETS:
        mcts, game = _capture_tree(cfg, sig, evaluator, net, sims)
        root = _build_tree(mcts, game)
        _layout(root)
        trees.append((sims, root))
        for nd in _iter_tree(root):
            for (_lab, _n, q, _ct) in nd.children:
                if not _ct.is_stub:
                    all_q.append(q)
    qmin, qmax = (min(all_q), max(all_q)) if all_q else (0.0, 1.0)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.4))
    for ax, (sims, root) in zip(axes, trees):
        _draw_tree(ax, root, cmap, qmin, qmax)
        nodes = [n for n in _iter_tree(root) if not n.is_stub]
        n_leaves = sum(1 for n in nodes if n.is_terminal)
        ax.set_title(f"{sims} simulations\n{len(nodes)} tree nodes · {n_leaves} terminal leaves",
                     fontsize=10)
        ax.axis("off")
        max_y = max(n.y for n in _iter_tree(root))
        max_d = max(n.depth for n in _iter_tree(root))
        ax.set_xlim(-0.4, max_d + 0.9)
        ax.set_ylim(max_y + 0.6, -0.6)   # first leaf on top

    sm = ScalarMappable(norm=Normalize(qmin, qmax), cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axes), fraction=0.022, pad=0.02)
    cb.set_label("MCTS backed-up Q(s,a) of the incoming edge", fontsize=9)

    fig.suptitle("Stage 2 — how MCTS drives the lifted derivation: search tree rooted at the "
                 f"{_SNAPSHOT_DESC}", fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.015,
             "node area ∝ visit count · edge width ∝ visit share · ○ = open hole, ▢ = complete "
             "policy (number = leaf score) · top-4 children kept per node, ✕ + dotted edge = "
             "remaining (pruned) siblings\n"
             "edge labels:  +rule = ADD_RULE · STOP = STOP_POLICY · =X = pick schema X · "
             "−aux/+aux:b = aux-var hole · +s/+g = add state/goal literal · end-pre = STOP_PRE · "
             "⇒FIN = FINISH_RULE · b0/r1/a0 = ?b_0/?r_1/?aux_0",
             ha="center", va="bottom", fontsize=6.8, color="#333333")
    fig.subplots_adjust(left=0.01, right=0.93, top=0.86, bottom=0.10, wspace=0.04)
    out = OUT / "02_mcts_search_tree.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — uniform-prior exploration / diagnostics
# ---------------------------------------------------------------------------

def _branching_by_depth(max_rules: int = 3, n_walks: int = 500, seed: int = 0) -> dict[int, list[int]]:
    cfg = LiftedGrammarConfig(max_rules=max_rules)
    sig = gripper_lite_signature()
    rng = np.random.default_rng(seed)
    out: dict[int, list[int]] = {}
    for _ in range(n_walks):
        st = LiftedDerivationState.initial()
        depth = 0
        while not st.is_terminal():
            prods = st.legal_productions(cfg, sig)
            out.setdefault(depth, []).append(len(prods))
            if not prods:
                break
            st = st.apply(prods[int(rng.integers(len(prods)))])
            depth += 1
    return out


def fig_uniform_prior(scores_a: dict, scores_b: dict, summary: dict) -> Path:
    sA, sB = scores_a["score"], scores_b["score"]
    sumA, sumB, sumA512 = summary["runA"], summary["runB"], summary["runA_512sims"]
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.8))

    # (1) best-so-far curve
    ax = axes[0, 0]
    for s, c, lab in [(sA, RUN_A_COLOR, "Run A — B=1 train, ≤3 rules, 128 sims"),
                      (sB, RUN_B_COLOR, "Run B — B=2 train, ≤4 rules, 512 sims")]:
        ax.step(np.arange(1, len(s) + 1), np.maximum.accumulate(s), where="post",
                color=c, lw=1.7, label=lab)
    solv_i = sumA["first_solver_index"]
    if solv_i is not None:
        ax.scatter([solv_i + 1], [sA[solv_i]], color=RUN_A_COLOR, s=48, edgecolor="black",
                   zorder=6, label=f"first B=1 solver — policy #{solv_i + 1:,}")
    ax.set_xscale("log")
    ax.set_xlabel("distinct policies evaluated (discovery order)")
    ax.set_ylabel("best leaf score so far")
    ax.set_title("(1) Best-so-far leaf score — the 'learning' curve\n"
                 f"(no learned prior: flat at −0.05 for ~{(solv_i or 0) + 1:,} policies, then one jump)",
                 fontsize=10)
    ax.legend(fontsize=6.6, loc="upper left")
    ax.grid(alpha=0.3)

    # (2) Run A score histogram
    ax = axes[0, 1]
    ax.hist(sA, bins=70, color=RUN_A_COLOR, alpha=0.85)
    ax.set_yscale("log")
    n_dn = int(np.sum(np.isclose(sA, -0.05)))
    smax = float(sA.max())
    n_solv = int(np.sum(sA > 1.0))
    ax.axvline(-0.05, color="#444444", ls="--", lw=1.2)
    ax.axvline(smax, color=HAND_COLOR, ls="--", lw=1.2)
    ax.annotate(f"do-nothing (−0.05): {n_dn:,} policies", (-0.05, 0.6),
                xycoords=("data", "axes fraction"), xytext=(40, 0),
                textcoords="offset points", fontsize=7.2, va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate(f"B=1 solvers (≈{smax:.2f}): {n_solv}", (smax, 0.3),
                xycoords=("data", "axes fraction"), xytext=(-110, 0),
                textcoords="offset points", fontsize=7.2, va="center", color=HAND_COLOR,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=HAND_COLOR))
    ax.set_xlabel("leaf score")
    ax.set_ylabel("# policies (log scale)")
    ax.set_title(f"(2) Run A leaf-score distribution\n"
                 f"(graded, not 0/1: {len(sA):,} policies span [{sA.min():.2f}, {smax:.2f}])",
                 fontsize=10)

    # (3) score by #rules
    ax = axes[0, 2]
    nr = scores_a["num_rules"]
    ks = sorted(set(int(k) for k in np.unique(nr)))
    parts = ax.violinplot([sA[nr == k] for k in ks], positions=ks, widths=0.7,
                          showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(RUN_A_COLOR)
        pc.set_alpha(0.5)
    ax.set_xticks(ks)
    ax.set_xlabel("# rules in the policy")
    ax.set_ylabel("leaf score")
    ax.set_title("(3) Run A leaf score by policy size\n(B=1 solvers occur only at 3 rules)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # (4) score CDF, Run A vs Run B
    ax = axes[1, 0]
    for s, c, lab in [(np.sort(sA), RUN_A_COLOR, f"Run A — {len(sA):,} policies"),
                      (np.sort(sB), RUN_B_COLOR, f"Run B — {len(sB):,} policies")]:
        ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=c, lw=1.8, label=lab)
    ax.set_xlabel("leaf score")
    ax.set_ylabel("fraction of evaluated policies ≤ score")
    ax.set_title(f"(4) Leaf-score CDF — the deceptive landscape\n"
                 f"(Run B collapses into [{sB.min():.2f}, {sB.max():.2f}]: no policy ever drops a "
                 f"ball in room_b)", fontsize=10)
    ax.legend(fontsize=7.2, loc="lower right")
    ax.grid(alpha=0.3)

    # (5) grammar branching by derivation depth
    ax = axes[1, 1]
    dc = _branching_by_depth(max_rules=3)
    depths = sorted(dc)
    means = np.array([np.mean(dc[d]) for d in depths])
    ax.bar(depths, means, color="#8c8c8c")
    ax.set_ylim(0, means.max() * 1.22)  # headroom so the annotation clears the top spine
    mxd = int(depths[int(np.argmax(means))])
    ax.annotate(f"goal-literal hole peak\n≈ {means.max():.1f} legal productions\n"
                f"(≤ compute_max_productions = 13)", (mxd, means.max()),
                textcoords="offset points", xytext=(16, -16), fontsize=6.8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("derivation depth (# productions applied)")
    ax.set_ylabel("mean # legal productions")
    ax.set_title("(5) Grammar branching by derivation depth\n(why the policy space is large; "
                 "≤3-rule grammar, 500 random derivations)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # (6) more search != better solver
    ax = axes[1, 2]
    sims_x = [sumA["mcts_sims"], sumA512["mcts_sims"]]
    uniq = [sumA["n_unique"], sumA512["n_unique"]]
    solvers = [sumA["n_solving"], sumA512["n_solving"]]
    x = np.arange(2)
    ax.bar(x, uniq, width=0.5, color="#5b8db8")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} sims" for s in sims_x])
    ax.set_ylabel("# distinct policies evaluated")
    ax.set_ylim(0, max(uniq) * 1.25)
    for xi, u, sv in zip(x, uniq, solvers):
        ax.annotate(f"{u:,} policies\n{sv} B=1 solver(s)", (xi, u),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_title("(6) More search ≠ better solver (Run A, ≤3 rules)\n"
                 "(4× the simulations → ~4× the policies, 0 new solvers)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Stage 2 — uniform-prior MCTS over lifted policies on Gripper-lite: "
                 "exploration & diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = OUT / "02_uniform_prior_exploration.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3 — learned-policy cards (split: runs vs hand reference)
# ---------------------------------------------------------------------------

def _wrap_rule_line(line: str, width: int = 88) -> list[str]:
    if len(line) <= width:
        return [line]
    head, _, rest = line.partition(": ")
    pieces = rest.split(" ∧ ")
    out, cur = [], head + ": "
    for i, p in enumerate(pieces):
        sep = "" if i == 0 else " ∧ "
        if len(cur) + len(sep) + len(p) > width and cur.strip() not in ("", head + ":"):
            out.append(cur)
            cur = " " * (len(head) + 2) + p
        else:
            cur = cur + sep + p
    out.append(cur)
    return out


def _draw_card(ax, accent: str, header: str, sub: str, wrapped: list[str]) -> None:
    """``wrapped`` is the already line-wrapped list of rule lines. The ax height
    is sized (by the caller's gridspec) to ``len(wrapped) + 2`` text rows, so a
    fixed per-row step gives uniform line spacing across all cards."""
    n_rows = len(wrapped) + 2
    step = 1.0 / n_rows
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.012, 0.04), 0.976, 0.92,
                                boxstyle="round,pad=0.0,rounding_size=0.02",
                                transform=ax.transAxes, fc="#fbfbfb", ec=accent, lw=2.0,
                                zorder=1))
    ax.text(0.03, 1 - 0.5 * step, header, fontsize=11.5, fontweight="bold", va="center", zorder=2)
    ax.text(0.03, 1 - 1.5 * step, sub, fontsize=8.4, color="#555555", va="center", zorder=2)
    for i, rl in enumerate(wrapped):
        ax.text(0.045, 1 - (2.5 + i) * step, rl, fontsize=8.8, fontfamily="monospace",
                va="center", zorder=2)


def _emit_policy_cards(cards: list, out_path: Path, suptitle: str) -> Path:
    wrapped_cards = []
    for (accent, header, sub, lines) in cards:
        w: list[str] = []
        for ln in lines:
            w.extend(_wrap_rule_line(ln))
        wrapped_cards.append((accent, header, sub, w))
    heights = [len(w[3]) + 2 for w in wrapped_cards]
    fig = plt.figure(figsize=(13.5, 0.46 * sum(heights) + 1.2))
    gs = fig.add_gridspec(len(cards), 1, height_ratios=heights, hspace=0.12,
                          top=0.93, bottom=0.04, left=0.015, right=0.985)
    for i, (accent, header, sub, w) in enumerate(wrapped_cards):
        _draw_card(fig.add_subplot(gs[i, 0]), accent, header, sub, w)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.985)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_learned_policies_runs(summary: dict) -> Path:
    best_a, best_b = summary["runA"], summary["runB"]
    cards = [
        (RUN_A_COLOR,
         "Best learned policy — Run A (uniform MCTS, seed 0, 128 sims)",
         f"leaf score {best_a['best_score']:.2f}  ·  solves the B=1 training instance in "
         f"{best_a['best_avg_steps']:.0f} steps  ·  does NOT generalise to B=2 (eval-out solve rate 0)"
         "    [vars: ?b_i:ball, ?r_i:room, ?aux_0:ball]",
         best_a["best_policy_pretty"].split("\n")),
        (RUN_B_COLOR,
         "Degenerate attractor — Run B (uniform MCTS, seed 0, 512 sims) best-found",
         f"leaf score {best_b['best_score']:.2f}  ·  drop is never legal from the start state → "
         "interpret returns None on step 0 → 1 noop → score = 0 + 0.25·0 − 0.01·0 − 0.05·1",
         best_b["best_policy_pretty"].split("\n")),
    ]
    return _emit_policy_cards(
        cards,
        OUT / "02_learned_policies_runs.png",
        "Stage 2 — what uniform-prior MCTS learned (Run A best vs Run B degenerate)",
    )


def fig_learned_policy_hand() -> Path:
    hand = hand_policy()
    hand_lines = [f"ρ_{i + 1}: {r.pretty()}" for i, r in enumerate(hand.rules)]
    cards = [
        (HAND_COLOR,
         "Stage-1 hand policy (reference target — the grammar can express this up to α-renaming)",
         "leaf score 1.18 on B=2 (train solve rate 1.0, 7 steps)  ·  eval-out solve rate 1.0 on "
         "B=3  ·  closed form 4B−1 steps",
         hand_lines),
    ]
    return _emit_policy_cards(
        cards,
        OUT / "02_learned_policies_hand.png",
        "Stage 2 — Stage-1 hand policy (reference target — the grammar can express this up to α-renaming)",
    )


# ---------------------------------------------------------------------------
# Figure 4 — best learned B=1 policy: environment rollout
# ---------------------------------------------------------------------------

class _RecordingEvaluator:
    """Wraps a LiftedLeafEvaluator and keeps the Policy objects of train-solvers."""

    def __init__(self, inner: LiftedLeafEvaluator):
        self.inner = inner
        self.solvers: dict[str, tuple] = {}

    def __call__(self, program):
        score = self.inner(program)
        m = self.inner.metrics_for(program)
        if m["train_solve_rate"] >= 1.0:
            self.solvers[m["policy_pretty"]] = (program, m)
        return score

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _find_best_b1_solver(seed: int = 0, n_episodes: int = 64):
    cfg = LiftedGrammarConfig(max_rules=3)
    sig = gripper_lite_signature()
    train = [GripperLiteEnv(n_balls=1, seed=seed)]
    eval_out = [GripperLiteEnv(n_balls=2, seed=seed)]
    ev = _RecordingEvaluator(LiftedLeafEvaluator(train, train, eval_out))
    net = UniformPolicyValueNet(LiftedDerivationGame(cfg, sig, ev)._max_productions)
    np.random.seed(seed)
    for _ in range(n_episodes):
        game = LiftedDerivationGame(cfg, sig, ev)
        game.reset_wrapper()
        mcts = MCTS(game, net, n_simulations=128, temperature=1.0, c_exploration=1.5)
        while not game.terminated and not game.truncated:
            probs = np.asarray(mcts.perform_simulations(None), dtype=np.float64)
            tot = probs.sum()
            if tot <= 0 or not np.isfinite(tot):
                a = int(np.flatnonzero(game.get_action_mask())[0])
            else:
                a = int(np.random.choice(len(probs), p=probs / tot))
            game.step_wrapper(a)
    if not ev.solvers:
        raise RuntimeError(f"no B=1 solver found in {n_episodes} episodes at seed {seed}")
    return max(ev.solvers.values(), key=lambda t: t[1]["score"])


def _rollout(policy, n_balls: int):
    env = GripperLiteEnv(n_balls=n_balls)
    env.reset()
    states = [env.get_state_atoms()]
    actions, rule_ids = [], []
    for _ in range(env.horizon):
        if env.is_solved():
            break
        out = interpret(policy, env.get_state_atoms(), env.get_goal_atoms(),
                        env.get_objects_by_type(), env.legal_actions(), trace=True)
        if out is None:
            break
        action, rule_idx, _theta = out
        actions.append(action)
        rule_ids.append(rule_idx)
        env.step(action)
        states.append(env.get_state_atoms())
    return env, states, actions, rule_ids


def fig_learned_policy_rollout() -> Path:
    policy, m = _find_best_b1_solver()
    env1, st1, ac1, rid1 = _rollout(policy, 1)
    env2, st2, ac2, rid2 = _rollout(policy, 2)
    bidx = {b: i for i, b in enumerate(env2.balls)}        # stable colours

    def frame_title(i, acts, rids, n_total):
        if i == 0:
            return "step 0 (init)"
        a = acts[i - 1]
        return f"step {i}/{n_total} · ρ{rids[i - 1] + 1}\n{a.pretty()}"

    # row 0: full B=1 rollout (init + 3 steps) + a text panel
    # row 1: 4 representative B=2 frames (init, mid, mid, last) + a text panel
    ncols = max(len(st1), 5)
    fig, axes = plt.subplots(2, ncols, figsize=(2.7 * ncols, 6.0))

    for j in range(ncols):
        axes[0, j].axis("off")
        axes[1, j].axis("off")

    for j, s in enumerate(st1):
        _draw_frame(axes[0, j], s, {b: i for i, b in enumerate(env1.balls)},
                    frame_title(j, ac1, rid1, len(ac1)))
    txt0 = axes[0, ncols - 1] if len(st1) < ncols else None
    if txt0 is not None:
        txt0.text(0.5, 0.5, f"B = 1\nsolved in {len(ac1)} steps\nleaf score {m['score']:.2f}\n\n"
                            "pick(ball_0,room_a)\n→ move(room_a,room_b)\n→ drop(ball_0,room_b)",
                  ha="center", va="center", fontsize=9, fontfamily="monospace",
                  transform=txt0.transAxes,
                  bbox=dict(boxstyle="round,pad=0.5", fc="#eef6ee", ec=HAND_COLOR))

    # B=2: choose ~4 frame indices, always include first and last
    n2 = len(st2)
    if n2 <= 4:
        idxs = list(range(n2))
    else:
        idxs = sorted(set([0, n2 // 3, 2 * n2 // 3, n2 - 1]))
    for col, fi in enumerate(idxs):
        _draw_frame(axes[1, col], st2[fi], bidx, frame_title(fi, ac2, rid2, len(ac2)))
    solved2 = env2.is_solved()
    txt1 = axes[1, ncols - 1]
    txt1.text(0.5, 0.5,
              f"B = 2\n{'solved' if solved2 else 'NOT solved'}\n"
              f"ran {len(ac2)} steps (horizon {env2.horizon})\n\n"
              "after dropping ball_0 the\npick rule fires again on the\nalready-placed ball → loops",
              ha="center", va="center", fontsize=9, fontfamily="monospace",
              transform=txt1.transAxes,
              bbox=dict(boxstyle="round,pad=0.5", fc="#fdeeee", ec=RUN_B_COLOR))

    rules_str = "   ".join(f"ρ{i + 1}: {r.pretty()}" for i, r in enumerate(policy.rules))
    fig.suptitle("Stage 2 — the best learned B=1 policy in the Gripper-lite environment "
                 "(uniform MCTS, seed 0, 128 sims)", fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.01, rules_str, ha="center", va="bottom", fontsize=7.0,
             fontfamily="monospace", color="#333333")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    out = OUT / "02_learned_policy_rollout.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 5 — graded leaf-reward landscape (regenerate)
# ---------------------------------------------------------------------------

def fig_graded_reward(scores_a: dict) -> Path:
    lit = scores_a["num_literals"].astype(float)
    sc = scores_a["score"]
    tsr = scores_a["train_solve_rate"]
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(lit))
    x = lit + jit
    smax, smin = float(sc.max()), float(sc.min())
    n_solv = int(np.sum(sc > 1.0))
    n_dn = int(np.sum(np.isclose(sc, -0.05)))
    solv_lit = lit[sc > 1.0]

    # broken y-axis: ~99.8% of policies sit in [smin, −0.05]; 8 solvers sit at ≈1.22.
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9.4, 6.4), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 2.6], hspace=0.06))
    for ax in (top, bot):
        sc_obj = ax.scatter(x, sc, c=tsr, cmap="RdYlGn", vmin=0.0, vmax=1.0, s=15,
                            alpha=0.75, edgecolor="none")
        ax.grid(alpha=0.3)
    top.set_ylim(1.16, 1.28)
    bot.set_ylim(smin - 0.015, 0.01)
    bot.axhline(-0.05, color="#333333", ls=":", lw=1.0)
    # break marks
    top.spines["bottom"].set_visible(False)
    bot.spines["top"].set_visible(False)
    top.tick_params(bottom=False, labelbottom=False)
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=8, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)
    top.plot([0, 1], [0, 0], transform=top.transAxes, **kw)
    bot.plot([0, 1], [1, 1], transform=bot.transAxes, **kw)

    top.annotate(f"{n_solv} B=1 solvers  (score ≈ {smax:.2f}  =  1.0 solve + 0.25·progress − 0.01·3 steps)",
                 (float(np.median(solv_lit)) if len(solv_lit) else float(np.median(lit)), smax),
                 textcoords="offset points", xytext=(0, -20), ha="center", fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    bot.annotate(f"−0.05 = ⊤ ⇒ drop(?b_0,?r_1): drop never legal → 1 noop  ({n_dn:,} policies)\n"
                 f"the rest of the band ([{smin:.2f}, −0.05]) = useless-action / stall policies "
                 "(extra step & noop penalties)", (float(np.median(lit)), -0.05),
                 textcoords="offset points", xytext=(0, 70), ha="center", fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8))

    bot.set_xlabel("# body literals in the policy (x jittered ±0.18 for visibility)")
    fig.text(0.045, 0.5, "leaf score", va="center", rotation="vertical", fontsize=11)
    top.set_title(f"Run A — graded leaf reward over all {len(sc):,} evaluated lifted policies\n"
                  f"(uniform MCTS, seed 0, 128 sims, ≤3-rule grammar; scores span [{smin:.2f}, {smax:.2f}]; "
                  "broken y-axis)", fontsize=11)
    fig.subplots_adjust(left=0.11, right=0.88, top=0.9, bottom=0.1)
    # colorbar on its own (un-broken) axis so the broken-axis break marks don't bleed onto it
    cax = fig.add_axes([0.905, 0.12, 0.022, 0.74])
    cb = fig.colorbar(sc_obj, cax=cax)
    cb.set_label("train solve rate (B = 1)")
    out = OUT / "02_graded_reward_runA.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = _load_summary()
    scores_a = _load_scores(RUN_A)
    scores_b = _load_scores(RUN_B)
    print(f"wrote {fig_search_tree()}")
    print(f"wrote {fig_uniform_prior(scores_a, scores_b, summary)}")
    print(f"wrote {fig_learned_policies_runs(summary)}")
    print(f"wrote {fig_learned_policy_hand()}")
    print(f"wrote {fig_learned_policy_rollout()}")
    print(f"wrote {fig_graded_reward(scores_a)}")


if __name__ == "__main__":
    main()
