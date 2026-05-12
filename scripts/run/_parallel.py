"""Fan independent CLI "cells" out over a thread pool of subprocesses.

The lifted-experiment grid drivers (``make_lifted_*``) run a matrix of
independent cells (distinct ``balls × mcts_sims × seed``). Each cell is just a
``python <script> <argv...>`` invocation we block on, so the driver's work is
I/O-bound (waiting on children) — a ``ThreadPoolExecutor`` of size ``jobs`` is
the right fan-out primitive; the CPU-bound work happens inside the child
processes and the OS scheduler does the real parallelism. Largest ``cost_hint``
cells are launched first so the long pole doesn't tail the run.

Sibling-script import (these drivers are run as files, not as a package)::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _parallel import Cell, run_cells, default_jobs
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional


def default_jobs() -> int:
    """One worker per core, minus one for the OS / IO; never below 1."""
    return max(1, (os.cpu_count() or 2) - 1)


@dataclass
class Cell:
    name: str
    argv: list[str]                       # full argv incl. [sys.executable, script, ...]
    cost_hint: float = 1.0                # larger ⇒ scheduled earlier
    env: Optional[dict[str, str]] = None  # extra env vars merged over os.environ


@dataclass
class CellResult:
    name: str
    returncode: int
    tail: str = ""                        # last few KB of stdout+stderr, for failures


def _run_one(cell: Cell) -> CellResult:
    env = {**os.environ, **(cell.env or {})}
    proc = subprocess.run(cell.argv, capture_output=True, text=True, env=env)
    out = (proc.stdout or "") + (proc.stderr or "")
    return CellResult(cell.name, proc.returncode, out[-4000:])


def run_cells(cells: list[Cell], jobs: Optional[int] = None) -> dict[str, int]:
    """Run all *cells*, ``jobs`` at a time (default ``default_jobs()``).

    Returns ``{name: returncode}``. Prints a one-line progress update per cell;
    dumps the tail of any failing cell's output to stderr. Does not raise — the
    caller inspects the return codes.
    """
    if jobs is None:
        jobs = default_jobs()
    jobs = max(1, int(jobs))
    ordered = sorted(cells, key=lambda c: -c.cost_hint)
    print(f"[_parallel] {len(ordered)} cells, {jobs} workers", flush=True)
    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_run_one, c): c for c in ordered}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results[r.name] = r.returncode
            tag = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
            print(f"[_parallel] {i}/{len(ordered)} {r.name}: {tag}", flush=True)
            if r.returncode != 0 and r.tail:
                print(f"--- {r.name} output tail ---\n{r.tail}", file=sys.stderr, flush=True)
    n_fail = sum(1 for v in results.values() if v != 0)
    print(f"[_parallel] done: {len(results) - n_fail} ok, {n_fail} failed", flush=True)
    return results
