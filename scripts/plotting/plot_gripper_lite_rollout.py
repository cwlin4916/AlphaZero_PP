"""Generate Stage-4 figures for the Gripper-lite results companion.

Produces these PNGs under ``docs/notes/stage4/figures/``:

- ``gripper_lite_domain.png`` — a domain schematic showing the two rooms,
  the robot, the balls, and the goal overlay.
- ``gripper_lite_rollout_B2.png`` — an 8-frame 2×4 panel grid of the
  ``hand_policy()`` rollout on a B=2 instance, captured from
  ``interpret(..., trace=True)`` so the figure cannot drift from code.
- ``gripper_lite_proof_trace_B{2,3}.png`` — the §7 plan-length proof, one
  panel per reachable state: the state schematic plus the four-row
  ρ1–ρ4 binding-set table (✓ fires, with the lex-min binding; ✗ empty,
  with the first failing literal) — every cell recomputed from
  ``find_bindings`` so the figure is generated, never transcribed.

Run from repo root:

    python scripts/plotting/plot_gripper_lite_rollout.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from alphazeropp.instances.gripper_lite.env import GripperLiteEnv
from alphazeropp.instances.gripper_lite.policies import hand_policy
from alphazeropp.synthesis.lifted_dsl import (
    GroundAction,
    LiteralSource,
    RelState,
    Rule,
    Var,
)
from alphazeropp.synthesis.lifted_interpreter import (
    _atom_row_matches,
    _eval_negative_literal,
    _positive_literal_rows,
    find_bindings,
    interpret,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "docs" / "notes" / "stage4" / "figures"

ROOM_COLOR = "#f5f5f5"
ROOM_EDGE = "#444444"
BALL_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")
GOAL_EDGE = "#888888"
ROBOT_COLOR = "#222222"


def _ball_xy(room_x0: float, room_y0: float, room_w: float, room_h: float,
             idx: int, total: int) -> tuple[float, float]:
    pad = 0.18 * room_w
    if total == 1:
        x = room_x0 + room_w / 2
    else:
        x = room_x0 + pad + (room_w - 2 * pad) * idx / max(total - 1, 1)
    y = room_y0 + 0.30 * room_h
    return x, y


def _draw_room(ax, x0: float, y0: float, w: float, h: float, name: str) -> None:
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        linewidth=1.6, edgecolor=ROOM_EDGE, facecolor=ROOM_COLOR,
    )
    ax.add_patch(box)
    ax.text(x0 + 0.04, y0 + h - 0.06, name, fontsize=9, fontfamily="monospace",
            ha="left", va="top", color=ROOM_EDGE)


def _draw_robot(ax, x: float, y: float, carrying: str | None,
                ball_index: dict[str, int],
                show_label: bool = True) -> None:
    ax.plot([x], [y], marker="^", markersize=14, color=ROBOT_COLOR,
            markeredgecolor="black", linestyle="None")
    if carrying is not None:
        idx = ball_index[carrying]
        color = BALL_COLORS[idx % len(BALL_COLORS)]
        ax.plot([x + 0.05], [y + 0.05], marker="o", markersize=8,
                color=color, markeredgecolor="black", linestyle="None")
    if show_label:
        ax.text(x, y - 0.09, "robot", fontsize=7, ha="center", va="top",
                fontfamily="monospace")


def _draw_balls(ax, state: RelState, room_x: dict[str, tuple[float, float, float, float]],
                ball_index: dict[str, int], with_labels: bool = True) -> None:
    by_room: dict[str, list[str]] = {r: [] for r in room_x}
    for b, r in state["at_ball"]:
        by_room.setdefault(r, []).append(b)
    for room, balls in by_room.items():
        balls = sorted(balls)
        x0, y0, w, h = room_x[room]
        for i, b in enumerate(balls):
            color = BALL_COLORS[ball_index[b] % len(BALL_COLORS)]
            bx, by = _ball_xy(x0, y0, w, h, i, max(len(balls), 1))
            ax.plot([bx], [by], marker="o", markersize=14, color=color,
                    markeredgecolor="black", linestyle="None")
            if with_labels:
                ax.text(bx, by - 0.07, b, fontsize=7, ha="center", va="top",
                        fontfamily="monospace")


def _draw_goal_overlay(ax, goal: RelState,
                       room_x: dict[str, tuple[float, float, float, float]],
                       ball_index: dict[str, int]) -> None:
    by_room: dict[str, list[str]] = {r: [] for r in room_x}
    for b, r in goal["at_ball"]:
        by_room.setdefault(r, []).append(b)
    for room, balls in by_room.items():
        balls = sorted(balls)
        x0, y0, w, h = room_x[room]
        for i, b in enumerate(balls):
            color = BALL_COLORS[ball_index[b] % len(BALL_COLORS)]
            bx, by = _ball_xy(x0, y0, w, h, i, max(len(balls), 1))
            ax.plot([bx], [by + 0.16], marker="o", markersize=14,
                    markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=1.5, linestyle="None")
            ax.text(bx, by + 0.16 + 0.07, f"goal: {b}", fontsize=6,
                    ha="center", va="bottom", color=GOAL_EDGE,
                    fontfamily="monospace")


def _draw_move_arrow(ax, room_x: dict[str, tuple[float, float, float, float]]) -> None:
    rooms = list(room_x.keys())
    if len(rooms) < 2:
        return
    a, b = rooms[0], rooms[1]
    ax0, ay0, aw, ah = room_x[a]
    bx0, by0, bw, bh = room_x[b]
    y_mid = (ay0 + ah / 2 + by0 + bh / 2) / 2
    start_x = ax0 + aw + 0.01
    end_x = bx0 - 0.01
    ax.annotate(
        "", xy=(end_x, y_mid), xytext=(start_x, y_mid),
        arrowprops=dict(arrowstyle="<->", linewidth=1.4, color="#666666"),
    )
    ax.text((start_x + end_x) / 2, y_mid + 0.04, "move",
            fontsize=7, ha="center", va="bottom", color="#666666",
            fontfamily="monospace")


def _room_layout(width: float, room_h: float) -> dict[str, tuple[float, float, float, float]]:
    margin = 0.06
    gap = 0.12
    room_w = (width - 2 * margin - gap) / 2
    y0 = 0.18
    return {
        "room_a": (margin, y0, room_w, room_h),
        "room_b": (margin + room_w + gap, y0, room_w, room_h),
    }


def plot_domain() -> Path:
    env = GripperLiteEnv(n_balls=2)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    room_x = _room_layout(width=1.0, room_h=0.62)
    for name, (x0, y0, w, h) in room_x.items():
        _draw_room(ax, x0, y0, w, h, name)

    _draw_move_arrow(ax, room_x)

    ball_index = {b: i for i, b in enumerate(env.balls)}
    state = env.get_state_atoms()
    _draw_balls(ax, state, room_x, ball_index)

    (robot_room_tuple,) = state["at_robot"]
    robot_room = robot_room_tuple[0]
    rx0, ry0, rw, rh = room_x[robot_room]
    _draw_robot(ax, rx0 + 0.18 * rw, ry0 + 0.60 * rh,
                carrying=None, ball_index=ball_index)

    _draw_goal_overlay(ax, env.get_goal_atoms(), room_x, ball_index)

    ax.text(0.5, 0.96,
            "Gripper-lite — domain schematic (B = 2 balls)",
            ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.06,
            "predicates: at_robot(room) · at_ball(ball, room) · "
            "carrying(ball) · handempty()",
            ha="center", va="center", fontsize=7,
            fontfamily="monospace", color="#444444")
    ax.text(0.5, 0.02,
            "actions: move(room, room) · pick(ball, room) · drop(ball, room)",
            ha="center", va="center", fontsize=7,
            fontfamily="monospace", color="#444444")

    out = OUT / "gripper_lite_domain.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _state_robot_room(state: RelState) -> str:
    (room_tuple,) = state["at_robot"]
    return room_tuple[0]


def _state_carrying(state: RelState) -> str | None:
    if not state["carrying"]:
        return None
    (carrying_tuple,) = state["carrying"]
    return carrying_tuple[0]


def _draw_frame(ax, state: RelState, ball_index: dict[str, int],
                title: str) -> None:
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Tighter room band so the title above and the hand-status below are clear.
    margin = 0.06
    gap = 0.08
    room_h = 0.48
    room_w = (1.0 - 2 * margin - gap) / 2
    y0 = 0.20
    room_x = {
        "room_a": (margin, y0, room_w, room_h),
        "room_b": (margin + room_w + gap, y0, room_w, room_h),
    }
    for name, (rx0, ry0, w, h) in room_x.items():
        _draw_room(ax, rx0, ry0, w, h, name)

    carrying = _state_carrying(state)
    on_floor: RelState = {"at_ball": set(state["at_ball"])}
    _draw_balls(ax, on_floor, room_x, ball_index, with_labels=False)

    robot_room = _state_robot_room(state)
    rx0, ry0, rw, rh = room_x[robot_room]
    _draw_robot(ax, rx0 + 0.18 * rw, ry0 + 0.66 * rh,
                carrying=carrying, ball_index=ball_index,
                show_label=False)

    hand = "empty" if state["handempty"] else (f"carrying {carrying}"
                                               if carrying else "—")
    ax.text(0.5, 0.95, title, ha="center", va="top",
            fontsize=8.5, fontweight="bold", fontfamily="monospace")
    ax.text(0.5, 0.10, f"hand: {hand}", ha="center", va="center",
            fontsize=7, fontfamily="monospace", color="#444444")


def plot_rollout(n_balls: int = 2) -> Path:
    env = GripperLiteEnv(n_balls=n_balls)
    policy = hand_policy()
    states: list[RelState] = [env.get_state_atoms()]
    actions: list[GroundAction] = []
    rule_ids: list[int] = []
    for _ in range(env.horizon):
        if env.is_solved():
            break
        out = interpret(
            policy,
            env.get_state_atoms(),
            env.get_goal_atoms(),
            env.get_objects_by_type(),
            env.legal_actions(),
            trace=True,
        )
        if out is None:
            break
        action, rule_idx, _theta = out
        actions.append(action)
        rule_ids.append(rule_idx)
        env.step(action)
        states.append(env.get_state_atoms())

    ball_index = {b: i for i, b in enumerate(env.balls)}
    n_frames = len(states)
    # 2 cols × 4 rows gives 8 panels — exactly fits the B=2 rollout — and keeps
    # the figure portrait so it never overflows a text column.
    ncols = 2
    nrows = (n_frames + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.8 * ncols, 2.6 * nrows))
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]

    for i, (st, ax) in enumerate(zip(states, axes_flat)):
        if i == 0:
            title = "step 0 (init)"
        else:
            a = actions[i - 1]
            rid = rule_ids[i - 1] + 1
            title = f"step {i}  ·  ρ{rid}\n{a.pretty()}"
        _draw_frame(ax, st, ball_index, title)
    # Hide any unused panels (defensive — usually none for B=2 with 8 frames).
    for ax in axes_flat[n_frames:]:
        ax.axis("off")

    fig.suptitle(
        f"Gripper-lite — hand_policy() rollout, B = {n_balls}   "
        f"({n_frames - 1} steps = 4B − 1)",
        y=0.997, fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT / f"gripper_lite_rollout_B{n_balls}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# §7 proof-trace figure: per-state ρ1–ρ4 binding-set table.
#
# ``_diagnose_rule`` replays the *same* pipeline ``interpret`` uses
# (``find_bindings`` → ``bindings[0]``), and when the binding set is empty it
# walks the staged filters of ``find_bindings`` to name the *first* one that
# emptied it. The table is therefore generated from the interpreter, not
# transcribed — it cannot drift from the code the proof in §7 reasons about.
# --------------------------------------------------------------------------

RULE_FIRE_COLOR = "#1a7a1a"
RULE_DEAD_COLOR = "#888888"


def _binding_str(theta: dict[str, str]) -> str:
    return "{" + ", ".join(f"{k}={theta[k]}" for k in sorted(theta)) + "}"


def _lit_under(lit, theta: dict[str, str]) -> str:
    """Render ``lit`` with every variable that ``theta`` binds substituted —
    e.g. ``at_robot(?r)`` with ``{?r: room_a}`` → ``at_robot(room_a)``,
    a goal literal wrapped in ``Goal[...]``."""
    args = [theta.get(a.name, a.name) if isinstance(a, Var) else a for a in lit.atom.args]
    inner = f"{lit.atom.pred}({', '.join(args)})" if args else f"{lit.atom.pred}()"
    return f"Goal[{inner}]" if lit.source is LiteralSource.GOAL else inner


def _diagnose_rule(
    rule: Rule,
    state_atoms: RelState,
    goal_atoms: RelState,
    objects_by_type: dict[str, tuple[str, ...]],
    legal_actions: set[GroundAction],
) -> tuple[bool, str, GroundAction | None]:
    """``(fires, detail, action)`` for one rule at one state.

    ``fires`` mirrors ``find_bindings(...) != []``; on a hit, ``detail`` shows
    the grounded action and the lex-min binding and ``action`` is that
    ``GroundAction``. On a miss, ``detail`` pins the first stage of
    ``find_bindings`` that produced the empty set and ``action`` is ``None``.
    """
    bindings = find_bindings(rule, state_atoms, goal_atoms, objects_by_type, legal_actions)
    if bindings:
        theta = bindings[0]
        action = GroundAction(rule.action.schema, tuple(theta[v.name] for v in rule.action.args))
        return True, f"{action.pretty()}   θ={_binding_str(theta)}", action

    positive = [lit for lit in rule.body if not lit.negated]
    negative = [lit for lit in rule.body if lit.negated]

    # Stage 1 — positive-literal pass (left to right).
    partial: list[dict[str, str]] = [dict()]
    for lit in positive:
        rows = _positive_literal_rows(lit, state_atoms, goal_atoms)
        nxt: list[dict[str, str]] = []
        for theta in partial:
            for row in rows:
                extended = _atom_row_matches(lit.atom, row, theta)
                if extended is not None:
                    nxt.append(extended)
        if not nxt:
            where = "G" if lit.source is LiteralSource.GOAL else "S"
            if len(partial) == 1:
                return False, f"{_lit_under(lit, partial[0])} ∉ {where}", None
            return False, f"no consistent {lit.pretty()} in {where}", None
        partial = nxt

    # Stage 2 — enumerate any still-free declared variables over their type.
    completed: list[dict[str, str]] = []
    for theta in partial:
        unbound = [v for v in rule.vars if v.name not in theta]
        if not unbound:
            completed.append(theta)
            continue
        domains = [objects_by_type.get(v.type_name, ()) for v in unbound]
        if any(len(d) == 0 for d in domains):
            continue
        for tup in product(*domains):
            new_theta = dict(theta)
            for v, obj in zip(unbound, tup):
                new_theta[v.name] = obj
            completed.append(new_theta)
    if not completed:
        return False, "no objects for a free variable", None

    # Stage 3 — negated-literal (closed-world) filter.
    after_neg = [
        th for th in completed
        if all(_eval_negative_literal(lit, th, state_atoms, goal_atoms) for lit in negative)
    ]
    if not after_neg:
        negs = " ∧ ".join(lit.pretty() for lit in negative)
        return False, f"{negs} fails for every binding", None

    # Stage 4 — legality filter (must be the reason if we got here).
    th0 = after_neg[0]
    ga = ", ".join(th0[v.name] for v in rule.action.args)
    return False, f"{rule.action.schema}({ga}) not in legal_actions", None


def _draw_rule_table(ax, rows: list[tuple[bool, str, GroundAction | None]],
                     fired_idx: int | None, footer: str) -> None:
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    ax.text(0.0, 0.96, "binding sets  B_ρ(S, G):", fontsize=7.5, fontweight="bold",
            fontfamily="monospace", ha="left", va="top", color="#222222")
    y_rows = (0.78, 0.60, 0.42, 0.24)
    for j, ((fires, detail, _), y) in enumerate(zip(rows, y_rows)):
        is_fired = fired_idx is not None and j == fired_idx
        if is_fired:
            ax.axhspan(y - 0.085, y + 0.085, xmin=0.0, xmax=1.0,
                       color=RULE_FIRE_COLOR, alpha=0.12, zorder=0)
        mark = "✓" if fires else "✗"
        color = RULE_FIRE_COLOR if fires else RULE_DEAD_COLOR
        ax.text(0.0, y, f"ρ{j + 1}  {mark}  {detail}", fontsize=6.0,
                fontfamily="monospace", ha="left", va="center", color=color,
                fontweight=("bold" if is_fired else "normal"))
    ax.text(0.0, 0.06, footer, fontsize=7, fontfamily="monospace", ha="left",
            va="center", color=(RULE_FIRE_COLOR if fired_idx is not None else "#444444"))


def _rest_tag(i: int, n_balls: int) -> str:
    """If state index ``i`` is one of the §7 ``Rest_k`` configurations, label it."""
    plan_len = 4 * n_balls - 1
    if i == 0:
        return "  = Rest_0  (= s⁰_B)"
    if i >= 3 and (i - 3) % 4 == 0:
        k = (i - 3) // 4 + 1
        return f"  = Rest_{k}" + ("  (solved)" if i == plan_len else "")
    return ""


def plot_proof_trace(n_balls: int = 2) -> Path:
    env = GripperLiteEnv(n_balls=n_balls)
    policy = hand_policy()

    # Each cell: (state, [(fires, detail, action)]×4, fired_idx | None, title, footer).
    cells: list[tuple[RelState, list, int | None, str, str]] = []
    state_idx = 0
    for _ in range(env.horizon):
        if env.is_solved():
            break
        state = env.get_state_atoms()
        goal = env.get_goal_atoms()
        objs = env.get_objects_by_type()
        legal = env.legal_actions()
        rows = [_diagnose_rule(rule, state, goal, objs, legal) for rule in policy.rules]
        fired_idx = next(i for i, (fires, _, _) in enumerate(rows) if fires)
        # Cross-check against the interpreter proper.
        out = interpret(policy, state, goal, objs, legal, trace=True)
        assert out is not None and out[1] == fired_idx, "diagnosis disagrees with interpret"
        action = out[0]
        cells.append((
            state, rows, fired_idx,
            f"S{state_idx}{_rest_tag(state_idx, n_balls)}",
            f"→ ρ{fired_idx + 1} fires:  {action.pretty()}",
        ))
        env.step(action)
        state_idx += 1

    # Terminal cell — solved, every B_ρ empty.
    state = env.get_state_atoms()
    goal = env.get_goal_atoms()
    objs = env.get_objects_by_type()
    legal = env.legal_actions()
    rows = [_diagnose_rule(rule, state, goal, objs, legal) for rule in policy.rules]
    cells.append((
        state, rows, None,
        f"S{state_idx}{_rest_tag(state_idx, n_balls)}",
        "→ solved (G_B ⊆ S); interpret returns None",
    ))

    ball_index = {b: i for i, b in enumerate(env.balls)}
    n_cells = len(cells)
    pairs_per_row = 2
    nrows = (n_cells + pairs_per_row - 1) // pairs_per_row
    fig, axes = plt.subplots(
        nrows, 4,
        figsize=(14.5, 2.2 * nrows),
        gridspec_kw={"width_ratios": [1.0, 1.8, 1.0, 1.8]},
    )
    axes = axes.reshape(nrows, 4)
    used: set[tuple[int, int]] = set()
    for idx, (state, rows, fired_idx, title, footer) in enumerate(cells):
        r, cpair = divmod(idx, pairs_per_row)
        ax_state = axes[r, 2 * cpair]
        ax_table = axes[r, 2 * cpair + 1]
        used.add((r, 2 * cpair))
        used.add((r, 2 * cpair + 1))
        _draw_frame(ax_state, state, ball_index, title)
        _draw_rule_table(ax_table, rows, fired_idx, footer)
    for r in range(nrows):
        for c in range(4):
            if (r, c) not in used:
                axes[r, c].axis("off")

    fig.suptitle(
        f"Gripper-lite — hand_policy() proof trace, B = {n_balls}   "
        f"({4 * n_balls - 1} steps = 4B − 1;  ρ3·ρ2·ρ1 then (ρ4·ρ3·ρ2·ρ1)×{n_balls - 1})",
        y=0.998, fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=2.0, h_pad=1.6)
    out = OUT / f"gripper_lite_proof_trace_B{n_balls}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = plot_domain()
    r = plot_rollout(n_balls=2)
    p2 = plot_proof_trace(n_balls=2)
    p3 = plot_proof_trace(n_balls=3)
    for path in (d, r, p2, p3):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
