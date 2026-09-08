"""Illustrative precision assumptions, not power estimated from Study II outcomes."""
import argparse
import csv
from math import sqrt
from pathlib import Path
from statistics import NormalDist


def rows():
    z95 = NormalDist().inv_cdf(0.975)
    z80 = NormalDist().inv_cdf(0.8)
    for n in (20, 35, 91):
        for sd in (5, 8, 11, 13, 15):
            yield {"n_assumed_independent_speakers": n, "paired_sd_assumption_pp": sd,
                   "normal_95_halfwidth_pp": round(z95 * sd / sqrt(n), 4),
                   "normal_two_sided_80pct_mde_pp": round((z95 + z80) * sd / sqrt(n), 4),
                   "scope": "illustrative iid paired-normal assumptions; shared-training dependence not captured"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    values = list(rows())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
    print("Wrote 15 explicit-assumption sensitivity rows; no experiment outcomes read.")
