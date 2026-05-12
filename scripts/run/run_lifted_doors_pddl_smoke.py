#!/usr/bin/env python3
"""Stage-1 smoke: dispatch the hand-written lifted Doors policy over the relational adapter.

No grammar, no MCTS. Wires ``DoorsPDDLLiteRelationalEnv`` (the relational adapter over
``DoorsPDDLLiteEnv``) -> ``doors_hand_policy()`` -> the generic ``interpret(...)``, rolls out on
the two shipped layouts (D2, D3), prints the per-step trace table, and exits non-zero if either
layout is left unsolved. Used to regenerate the trace tables in ``docs/notes/stage4/01.md``.

Usage:
    python scripts/run/run_lifted_doors_pddl_smoke.py            # both layouts
    python scripts/run/run_lifted_doors_pddl_smoke.py --layout d2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable even without an editable install.
_src = Path(__file__).resolve().parents[2] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from alphazeropp.instances.doors.doors_pddl_lifted import DoorsPDDLLiteRelationalEnv
from alphazeropp.instances.doors.doors_pddl_policies import doors_hand_policy
from alphazeropp.synthesis.lifted_interpreter import interpret

_MAKERS = {"d2": DoorsPDDLLiteRelationalEnv.make_d2, "d3": DoorsPDDLLiteRelationalEnv.make_d3}


def _theta_str(theta: dict[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(theta.items()))


def run_layout(layout: str) -> bool:
    env = _MAKERS[layout]()
    policy = doors_hand_policy()
    print(f"=== Doors {layout.upper()}  (D={env.base.D}, M={env.base.M}, K={env.base.K}, "
          f"goal=loc_{env.base.goal_loc}, horizon={env.horizon}) ===")
    print(f"  goal atoms: {sorted(env.get_goal_atoms()['at_loc'])}")
    print(f"  {'step':>4}  {'rule':>5}  {'action':<22}  theta")
    step = 0
    while not env.is_solved() and step < env.horizon:
        out = interpret(
            policy,
            env.get_state_atoms(), env.get_goal_atoms(),
            env.get_objects_by_type(), env.legal_actions(),
            trace=True,
        )
        if out is None:
            print("  (no rule fired — rollout stalled)")
            break
        action, rule_idx, theta = out
        print(f"  {step:>4}  {'rho' + str(rule_idx + 1):>5}  {action.pretty():<22}  {{{_theta_str(theta)}}}")
        env.step(action)
        step += 1
    solved = env.is_solved()
    print(f"  -> solved={solved}  steps={step}  (closed form for chained-unlock: 2K+1 = {2 * env.base.K + 1})\n")
    return solved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layout", choices=sorted(_MAKERS) + ["all"], default="all")
    args = ap.parse_args(argv)
    layouts = sorted(_MAKERS) if args.layout == "all" else [args.layout]
    ok = all(run_layout(la) for la in layouts)
    if not ok:
        print("FAILURE: at least one layout was left unsolved", file=sys.stderr)
        return 1
    print("OK: all layouts solved by the hand policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
