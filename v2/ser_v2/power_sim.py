"""Simulated power of every family hypothesis under the actual Holm step-down, using per-speaker
effect sizes and SDs assumed from the 2026 program (P1, A, S0, S3, C). Written into
registry/power_table.json at tag-1; hypotheses with simulated power < 0.5 keep their
confirmatory status only if the registry says so explicitly (a priori demotion is recorded).

    python -m ser_v2.power_sim --out registry/power_table.json [--reps 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

from .stats import holm

# assumed per-speaker mean (pp) and SD (pp) of each contrast, n speakers; sources in comments
ASSUMPTIONS = {
    # Family N
    "N01": (6.0, 5.0, 20),    # SUBESCO-980 scratch RR-GG: S3 ridge ~3 pp; scratch models leak more (B: +5-11 pp at one take)
    "N02": (3.0, 4.0, 20),    # SUBESCO-980 probe RR-GG: S3 primary +3.1 pp, draw SD 1.5; per-speaker SD ~4 (S0 CREMA-D 3.1)
    "N03": (3.5, 4.5, 20),    # outer leakage share (A: O_G/RR-GG ~ 0.56 on RAVDESS)
    "N04": (2.5, 4.0, 20),    # inner leakage (A: I_R +7.3 on RAVDESS with 17.7 total -> ~0.4 share)
    "N05": (2.0, 7.0, 91),    # mech prm-none on CREMA-D: ~28 test items/speaker over 2 rotations, 2 models x 2 r averaged
    "N06": (2.0, 7.0, 91),    # mech spk-none on CREMA-D (C: +1.85 cross-prompt exposure)
    "N07": (3.0, 7.0, 20),    # mech prm-none SUBESCO-full
    "N08": (4.0, 7.0, 20),    # mech spk-none SUBESCO-full
    "N09": (5.0, 7.0, 20),    # sibling increment (B: second take triples the premium)
    "N10": (4.0, 8.0, 70),    # density moderator (RAVDESS 24 spk 17 pp vs CREMA-D 91 spk 2.5 pp suggests large; assume modest)
    "N11": (3.0, 7.0, 24),    # re-exposure RAVDESS (single partition, 2 seeds)
    "N12": (1.0, 4.0, 91),    # re-exposure CREMA-D
    "N13": (2.0, 6.0, 20),    # re-exposure SUBESCO-980
    "N14": (5.0, 8.0, 91),    # HPO validation optimism difference
    "N15": (5.0, 8.0, 24),    # head effect (S0 vs S2 discrepancy 2-5x on RAVDESS)
    # Family R (observed 2026 values)
    "R01": (17.7, 8.0, 24), "R02": (10.0, 7.0, 24), "R03": (7.3, 6.0, 24),
    "R04": (2.8, 3.5, 91), "R05": (1.9, 3.0, 91), "R06": (0.65, 3.0, 91),
    "R07": (4.5, 4.9, 24), "R08": (1.8, 3.1, 91),
    "R09": (12.0, 8.0, 24), "R10": (2.0, 4.0, 91), "R11": (5.0, 6.0, 20),
}
from . import registry as _registry
FAMILIES: dict[str, list[str]] = {}
for _h in _registry.HYPOTHESES:
    if _h["hypothesis_id"] in ASSUMPTIONS:
        FAMILIES.setdefault(_h["family"], []).append(_h["hypothesis_id"])


def simulate(reps: int, seed: int = 20260903, alpha: float = 0.05) -> dict:
    rng = np.random.RandomState(seed)
    hits = {h: 0 for h in ASSUMPTIONS}
    for _ in range(reps):
        for fam, members in FAMILIES.items():
            p = {}
            means = {}
            for h in members:
                mu, sd, n = ASSUMPTIONS[h]
                d = rng.normal(mu, sd, n)
                d = d[d != 0]
                p[h] = float(sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) if len(d) else 1.0
                means[h] = d.mean()
            for h, res in holm(p, alpha).items():
                if res["reject"] and means[h] > 0:
                    hits[h] += 1
    table = {h: {"assumed_mean_pp": ASSUMPTIONS[h][0], "assumed_sd_pp": ASSUMPTIONS[h][1], "n_speakers": ASSUMPTIONS[h][2],
                 "power_holm": hits[h] / reps, "status_if_below_0.5": "demote to estimate" if hits[h] / reps < 0.5 else "keep confirmatory"}
             for h in ASSUMPTIONS}
    return {"reps": reps, "alpha": alpha, "seed": seed, "family_sizes": {k: len(v) for k, v in FAMILIES.items()}, "table": table,
            "note": "assumptions are stated per hypothesis; the registry a_priori_status is decided from this table at tag-1 and never changed after outcomes"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args(argv)
    res = simulate(a.reps)
    a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    for h, t in res["table"].items():
        print(f"{h}: power={t['power_holm']:.2f} ({t['status_if_below_0.5']})")


if __name__ == "__main__":
    main()
