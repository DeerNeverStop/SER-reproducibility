"""Pure arithmetic for post-score cross-reporting of frozen last-epoch logits.

This module performs no fitting, file I/O, statistical tests, or interval
estimation.  Each reporting half is excluded from *that direction's* selection.
The opposite direction reuses it for selection; this is cross-reporting, not a
permanently untouched holdout.  A nonnegative two-direction reuse increment is
an argmax identity, not independent evidence of a mechanism.
"""

from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np


CONFIGS = (0, 1, 2, 3)
ROLES = ("seen", "unseen", "test")
DIRECTIONS = (("A_to_B", "A", "B"), ("B_to_A", "B", "A"))
TOLERANCE = 1e-10


def uar(labels, logits):
    """Return percent macro recall, requiring support for every logit class.

    Labels must be integer indices 0..C-1, and logits a finite real N-by-C
    array, C >= 2.  Argmax ties use the lowest class index.  The caller remains
    responsible for enforcing the study's six-class and input-identity schema.
    """
    y = np.asarray(labels)
    z = np.asarray(logits)
    if y.ndim != 1 or y.size == 0 or y.dtype.kind not in "iu":
        raise ValueError("labels must be a nonempty one-dimensional integer array")
    if z.ndim != 2 or z.shape[0] != y.size or z.shape[1] < 2:
        raise ValueError("logits must have shape (len(labels), C), with C >= 2")
    if z.dtype.kind not in "iuf" or not np.isfinite(z).all():
        raise ValueError("logits must contain finite real numbers")
    classes = z.shape[1]
    if np.any(y < 0) or np.any(y >= classes):
        raise ValueError("labels must index the logit columns")
    y = y.astype(np.int64, copy=False)
    pred = np.argmax(z, axis=1)
    matrix = np.bincount(y * classes + pred, minlength=classes * classes)
    matrix = matrix.reshape(classes, classes)
    support = matrix.sum(axis=1)
    if np.any(support == 0):
        raise ValueError("every class must have at least one label")
    if np.all(support == support[0]):
        # Equal class support makes macro recall exactly correct/total.
        # Sum integers first so mathematically tied half scores stay bitwise
        # equal despite different per-class distributions of correct answers.
        return 100.0 * int(np.trace(matrix)) / int(support.sum())
    return float(100.0 * np.mean(np.diag(matrix) / support))


def _keys(value, expected, where):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{where} must have exactly keys {tuple(expected)!r}")


def _score(value, where):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{where} must be a real UAR percentage")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise ValueError(f"{where} must be finite and between 0 and 100")
    return number


def _validate_scores(scores):
    _keys(scores, CONFIGS, "scores")
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) for k in scores):
        raise ValueError("configuration keys must be integers 0, 1, 2, 3")
    validated = {}
    for config in CONFIGS:
        _keys(scores[config], ROLES, f"scores[{config}]")
        validated[config] = {}
        for role in ROLES:
            halves = ("all",) if role == "test" else ("A", "B")
            _keys(scores[config][role], halves, f"scores[{config}][{role}]")
            validated[config][role] = {
                half: _score(scores[config][role][half], f"{config}/{role}/{half}")
                for half in halves
            }
    return validated


def _identity(left, right, name):
    # Explicit exception also enforces the identity under python -O.
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=TOLERANCE):
        raise AssertionError(f"arithmetic identity failed: {name}: {left} != {right}")


def analyze_context(draw, fold, scores):
    """Return (four episodes, two contrasts, two cfg3 controls).

    ``scores[config][role][half]`` contains percentages for all four configs.
    Seen/unseen roles have A/B halves; test has only 'all'.  All configurations
    must refer to the same fixed final training epoch (enforced by the caller).
    Selection uses only the rule's selection-half UAR, with lowest config index
    breaking exact ties.  Comparisons are seen-rule minus unseen-rule in pp.
    """
    for name, value in (("draw", draw), ("fold", fold)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    draw, fold = int(draw), int(fold)
    values = _validate_scores(scores)
    episodes, contrasts, controls = [], [], []
    for direction, selection_half, report_half in DIRECTIONS:
        selected = {}
        for rule in ("seen", "unseen"):
            config = min(CONFIGS, key=lambda c: (-values[c][rule][selection_half], c))
            selection = values[config][rule][selection_half]
            report_seen = values[config]["seen"][report_half]
            report_unseen = values[config]["unseen"][report_half]
            own_report = values[config][rule][report_half]
            test = values[config]["test"]["all"]
            row = {
                "draw": draw, "fold": fold, "direction": direction,
                "rule": rule, "config_index": config,
                "selection_uar": selection,
                "report_seen_uar": report_seen,
                "report_unseen_uar": report_unseen,
                "test_uar": test,
                "reuse_pp": selection - own_report,
                "report_gap_pp": own_report - test,
                "apparent_gap_pp": selection - test,
                "report_exposure_pp": report_seen - report_unseen,
            }
            _identity(row["apparent_gap_pp"], row["reuse_pp"] + row["report_gap_pp"], "episode gap")
            episodes.append(row)
            selected[rule] = row
        seen, unseen = selected["seen"], selected["unseen"]
        contrast = {
            "draw": draw, "fold": fold, "direction": direction,
            "apparent_contrast_pp": seen["apparent_gap_pp"] - unseen["apparent_gap_pp"],
            "reuse_contrast_pp": seen["reuse_pp"] - unseen["reuse_pp"],
            "report_gap_contrast_pp": seen["report_gap_pp"] - unseen["report_gap_pp"],
            "test_contrast_pp": seen["test_uar"] - unseen["test_uar"],
            "config_agreement": int(seen["config_index"] == unseen["config_index"]),
        }
        _identity(
            contrast["apparent_contrast_pp"],
            contrast["reuse_contrast_pp"] + contrast["report_gap_contrast_pp"],
            "contrast decomposition",
        )
        _identity(
            contrast["report_gap_contrast_pp"],
            seen["report_seen_uar"] - unseen["report_unseen_uar"] - contrast["test_contrast_pp"],
            "different selected models' report and test differences",
        )
        contrasts.append(contrast)
        fixed = values[3]
        controls.append({
            "draw": draw, "fold": fold, "direction": direction,
            "fixed_seen_reuse_pp": fixed["seen"][selection_half] - fixed["seen"][report_half],
            "fixed_unseen_reuse_pp": fixed["unseen"][selection_half] - fixed["unseen"][report_half],
            "fixed_report_exposure_pp": fixed["seen"][report_half] - fixed["unseen"][report_half],
            "fixed_test_difference_pp": 0.0,
        })
    for rule in ("seen", "unseen"):
        reuse_mean = sum(row["reuse_pp"] for row in episodes if row["rule"] == rule) / 2.0
        if reuse_mean < -TOLERANCE:
            raise AssertionError(f"argmax two-direction reuse identity failed for {rule}")
        _identity(
            sum(row[f"fixed_{rule}_reuse_pp"] for row in controls) / 2.0,
            0.0, f"fixed {rule} two-direction reuse",
        )
    _identity(
        sum(row["fixed_report_exposure_pp"] for row in controls) / 2.0,
        (values[3]["seen"]["A"] + values[3]["seen"]["B"]
         - values[3]["unseen"]["A"] - values[3]["unseen"]["B"]) / 2.0,
        "fixed two-half report exposure",
    )
    return episodes, contrasts, controls
