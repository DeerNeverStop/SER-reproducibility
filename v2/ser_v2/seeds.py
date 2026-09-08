"""Seed table. Every random state in the program derives from SHA-256 of a fixed key.

Rules (frozen at tag-1):
  split_seed(corpus, r)        -> outer/inner partition draw r
  train_seed(corpus, fold, r)  -> training RNG; NO model, cell, or protocol token, so every
                                  model level and every cell share initialization/batch streams
  inner_seed(corpus, cell, fold, r) -> inner fit/validation split
  draw_seed(corpus, d)         -> CPU Ridge partition draw d
  panel_seed(panel, k)         -> panel subsample k
  hash_order(pulse, id)        -> audit priority order
"""
from __future__ import annotations

from .common import stable_u32, sha256_text

PROGRAM = "SER26"


def split_seed(corpus: str, r: int) -> int:
    return stable_u32(f"{PROGRAM}|split|{corpus}|{r}")


def train_seed(corpus: str, fold: int, r: int) -> int:
    return stable_u32(f"{PROGRAM}|train|{corpus}|{fold}|{r}")


def crossing_train_seed(corpus: str, fold: int, seed_index: int) -> int:
    """Draw x seed crossing: training seed indexed independently of the partition draw."""
    return stable_u32(f"{PROGRAM}|train|{corpus}|{fold}|{seed_index}")


def inner_seed(corpus: str, cell: str, fold: int, r: int) -> int:
    return stable_u32(f"{PROGRAM}|inner|{corpus}|{cell}|{fold}|{r}")


def draw_seed(corpus: str, d: int) -> int:
    return stable_u32(f"{PROGRAM}|draw|{corpus}|{d}")


def panel_seed(panel: str, k: int) -> int:
    return stable_u32(f"{PROGRAM}|panel|{panel}|{k}")


def mech_seed(corpus: str, fold: int, condition: str, r: int) -> int:
    return stable_u32(f"{PROGRAM}|mech|{corpus}|{fold}|{condition}|{r}")


def hash_order_key(pulse_hex: str, canonical_id: str) -> str:
    """Audit priority order: SHA-256(pulse || id); sort ascending."""
    return sha256_text(pulse_hex + "|" + canonical_id)


def bootstrap_seed() -> int:
    return 202609030000
