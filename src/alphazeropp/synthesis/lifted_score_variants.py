"""Leaf-score variants for lifted-policy landscape analysis (Stage 2.5).

The Stage-2 evaluator scores a complete policy with

    score = solve_rate + 0.25*progress - 0.01*steps - 0.05*noops      ("current")

which is the MCTS reward in :class:`LiftedLeafEvaluator`. This module exposes
four scalar ablations and one lexicographic-tuple ranking so the landscape
script and plots can audit how much of the search structure is an artefact of
the particular reward shape.

All variants take the *aggregate metrics dict* produced by
``LiftedLeafEvaluator.aggregate_metrics_for(policy)`` — keys

    {"solve_rate", "avg_progress", "avg_steps", "num_noops"}

(``train_solve_rate`` is accepted as a fallback for ``solve_rate``). Default
weights match the Stage-2 reward (0.25 / 0.01 / 0.05).

The ``lexicographic`` variant returns a tuple ``(solve_rate, avg_progress,
-avg_steps, -num_noops)`` and is **for ranking only** — do not feed it back as
an MCTS scalar reward unless explicitly converted.
"""

from __future__ import annotations

from typing import Any

SCORE_VARIANT_NAMES: tuple[str, ...] = (
    "current",
    "no_step_penalty",
    "no_noop_penalty",
    "progress_only",
    "lexicographic",
)


def _solve_rate(m: dict[str, Any]) -> float:
    """Tolerate both ``solve_rate`` (aggregate dict) and ``train_solve_rate``
    (the cached ``diag`` dict of :class:`LiftedLeafEvaluator`)."""
    if "solve_rate" in m:
        return float(m["solve_rate"])
    return float(m["train_solve_rate"])


def score_current(
    m: dict[str, Any], *, w_prog: float = 0.25, w_step: float = 0.01, w_noop: float = 0.05
) -> float:
    """The Stage-2 leaf score: ``solve + w_prog*progress - w_step*steps - w_noop*noops``.

    Matches :meth:`LiftedLeafEvaluator._score_from` with the spec weights. This
    is the variant the MCTS engine actually backs up."""
    return (
        _solve_rate(m)
        + w_prog * float(m["avg_progress"])
        - w_step * float(m["avg_steps"])
        - w_noop * float(m["num_noops"])
    )


def score_no_step_penalty(
    m: dict[str, Any], *, w_prog: float = 0.25, w_noop: float = 0.05
) -> float:
    """``current`` minus the step penalty: ``solve + w_prog*progress - w_noop*noops``."""
    return (
        _solve_rate(m)
        + w_prog * float(m["avg_progress"])
        - w_noop * float(m["num_noops"])
    )


def score_no_noop_penalty(
    m: dict[str, Any], *, w_prog: float = 0.25, w_step: float = 0.01
) -> float:
    """``current`` minus the noop penalty: ``solve + w_prog*progress - w_step*steps``."""
    return (
        _solve_rate(m)
        + w_prog * float(m["avg_progress"])
        - w_step * float(m["avg_steps"])
    )


def score_progress_only(m: dict[str, Any], *, w_prog: float = 0.25) -> float:
    """The reward-shape lower bound: ``solve + w_prog*progress``."""
    return _solve_rate(m) + w_prog * float(m["avg_progress"])


def score_lexicographic(m: dict[str, Any]) -> tuple[float, float, float, float]:
    """Ranking-only tuple ``(solve_rate, avg_progress, -avg_steps, -num_noops)``.

    Compares with the natural Python tuple ordering — higher is better at each
    position. **Do not** feed this back as an MCTS scalar reward."""
    return (
        _solve_rate(m),
        float(m["avg_progress"]),
        -float(m["avg_steps"]),
        -float(m["num_noops"]),
    )


def all_score_variants(m: dict[str, Any]) -> dict[str, Any]:
    """Compute every score variant for an aggregate-metrics dict.

    Returns a dict keyed by :data:`SCORE_VARIANT_NAMES`. ``lexicographic`` is a
    tuple; all other entries are floats."""
    return {
        "current": score_current(m),
        "no_step_penalty": score_no_step_penalty(m),
        "no_noop_penalty": score_no_noop_penalty(m),
        "progress_only": score_progress_only(m),
        "lexicographic": score_lexicographic(m),
    }
