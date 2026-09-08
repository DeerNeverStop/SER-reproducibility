"""Frozen, domain-separated seed table for N14R."""
from __future__ import annotations

import hashlib

from .spec import ALL_DRAW_IDS, CELLS, N_FOLDS, SEED_NAMESPACE


def stable_u32(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def split_seed(draw_id: int) -> int:
    _check_draw(draw_id)
    return stable_u32(f"{SEED_NAMESPACE}|outer|cremad|draw={draw_id}")


def inner_seed(draw_id: int, fold: int, cell: str) -> int:
    _check(draw_id, fold, cell)
    rule = "random" if cell == "GR_hpo" else "grouped"
    return stable_u32(f"{SEED_NAMESPACE}|inner|cremad|draw={draw_id}|fold={fold}|rule={rule}")


def train_seed(draw_id: int, fold: int, train_rep: int = 0) -> int:
    """Shared by both cells and all eight configurations within a draw/fold."""
    _check_draw(draw_id)
    if fold not in range(N_FOLDS) or train_rep != 0:
        raise ValueError((draw_id, fold, train_rep))
    return stable_u32(f"{SEED_NAMESPACE}|train|cremad|draw={draw_id}|fold={fold}|rep={train_rep}")


def _check_draw(draw_id: int) -> None:
    if draw_id not in ALL_DRAW_IDS:
        raise ValueError(f"draw_id must be one of {ALL_DRAW_IDS}, got {draw_id}")


def _check(draw_id: int, fold: int, cell: str) -> None:
    _check_draw(draw_id)
    if fold not in range(N_FOLDS) or cell not in CELLS:
        raise ValueError((draw_id, fold, cell))
