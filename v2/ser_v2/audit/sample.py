"""Beacon-ordered sequential screening to a fixed n, with the priority order fixed before any
eligibility decision. The NIST beacon pulse is the first pulse after the tag-1 timestamp."""
from __future__ import annotations

from ..common import sha256_text


def priority_order(pulse_hex: str, canonical_ids: list[str]) -> list[str]:
    """Ascending SHA-256(pulse|id); ties impossible for distinct ids."""
    return sorted(canonical_ids, key=lambda cid: sha256_text(f"{pulse_hex}|{cid}"))


def sequential_screen(order: list[str], eligibility: dict[str, str | None], n: int) -> dict:
    """Walk the frozen order; eligibility[cid] is None (eligible) or an E-code. Stop at n eligible.
    Returns the sample, the screened prefix and the per-code counts; time never enters."""
    sample, screened, codes = [], [], {}
    for cid in order:
        if len(sample) >= n:
            break
        screened.append(cid)
        code = eligibility.get(cid, "E6")
        if code is None:
            sample.append(cid)
        else:
            codes[code] = codes.get(code, 0) + 1
    return {"sample": sample, "n_screened": len(screened), "n_eligible": len(sample), "exclusions": codes,
            "eligibility_rate": len(sample) / max(1, len(screened)), "complete": len(sample) >= n}
