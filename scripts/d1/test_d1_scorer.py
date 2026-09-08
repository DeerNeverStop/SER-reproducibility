"""Synthetic-only tests for score_d1_results.py aggregation functions.

No real D1 predictions exist yet (training has not run), and this suite never
touches the real tree: every input is fabricated in memory with hand-computed
expected values.
"""
import sys
import numpy as np

sys.path.insert(0, r"E:\科研\claudework\07-ser-repro-protocol-audit\tools")
from score_d1_results import (  # noqa: E402
    CHANNELS, PROTOCOLS, SEEDS, actor_uar, collect_actor_uars, aggregate_model,
)

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(("PASS  " if ok else "FAIL  ") + name)
    passed, failed = passed + (1 if ok else 0), failed + (0 if ok else 1)


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def make_predictions(actors, samples_per_actor, uar_by_cell):
    """Fabricate predictions so each (actor, channel, protocol, seed) hits an
    exact target UAR. Each actor gets `samples_per_actor` samples per class 0/1
    (two classes only -> UAR = mean of the two recalls; we set recall of class 0
    to the target and recall of class 1 to the target, giving UAR == target)."""
    actors_of_sample = {}
    idx = 0
    sample_meta = []
    for actor in actors:
        for cls in (0, 1):
            for _ in range(samples_per_actor):
                actors_of_sample[idx] = actor
                sample_meta.append((idx, actor, cls))
                idx += 1
    predictions = {}
    for channel in CHANNELS:
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                cell = {}
                for sample_index, actor, cls in sample_meta:
                    target = uar_by_cell[(actor, channel, protocol, seed)]
                    n_correct = round(target * samples_per_actor)
                    pos_in_class = sum(1 for s, a, c in sample_meta
                                        if a == actor and c == cls and s < sample_index)
                    pred = cls if pos_in_class < n_correct else 1 - cls
                    cell[sample_index] = (cls, pred)
                predictions[(channel, protocol, seed)] = cell
    return predictions, actors_of_sample


def main():
    # --- actor_uar hand-checks ---
    y_true = np.asarray([0, 0, 1, 1])
    check("actor_uar perfect = 1.0", close(actor_uar(y_true, np.asarray([0, 0, 1, 1])), 1.0))
    check("actor_uar all-wrong = 0.0", close(actor_uar(y_true, np.asarray([1, 1, 0, 0])), 0.0))
    check("actor_uar asymmetric recalls average: (1.0 + 0.5)/2",
          close(actor_uar(y_true, np.asarray([0, 0, 1, 0])), 0.75))
    check("actor_uar ignores absent classes",
          close(actor_uar(np.asarray([2, 2]), np.asarray([2, 3])), 0.5))

    # --- pipeline on a 3-actor synthetic panel with known premiums ---
    actors = ["01", "02", "03"]
    spa = 10  # samples per class per actor -> targets in steps of 0.1
    uar_by_cell = {}
    # design: random-UAR fixed at 1.0; groupkfold-UAR set so that
    # P_RG(speech) = 0.2 and P_RG(song) = 0.5 for every actor/seed -> D_mode_RG = 0.3
    # loso-UAR so that P_RL(speech) = 0.1, P_RL(song) = 0.2 -> D_mode_RL = 0.1
    for actor in actors:
        for seed in SEEDS:
            uar_by_cell[(actor, "speech", "random", seed)] = 1.0
            uar_by_cell[(actor, "song", "random", seed)] = 1.0
            uar_by_cell[(actor, "speech", "groupkfold", seed)] = 0.8
            uar_by_cell[(actor, "song", "groupkfold", seed)] = 0.5
            uar_by_cell[(actor, "speech", "loso", seed)] = 0.9
            uar_by_cell[(actor, "song", "loso", seed)] = 0.8
    predictions, actors_of_sample = make_predictions(actors, spa, uar_by_cell)
    uars = collect_actor_uars(predictions, actors_of_sample)
    check("collect: every fabricated cell reproduces its target UAR",
          all(close(uars[k], uar_by_cell[k]) for k in uar_by_cell))

    report = aggregate_model(uars, actors)
    check("aggregate status complete", report["status"] == "complete")
    pe = report["point_estimates"]
    check("P_RG_speech point = 0.2", close(pe["P_RG_speech"], 0.2))
    check("P_RG_song point = 0.5", close(pe["P_RG_song"], 0.5))
    check("D_mode_RG point = 0.3", close(pe["D_mode_RG"], 0.3))
    check("D_mode_RL point = 0.1", close(pe["D_mode_RL"], 0.1))
    check("per-actor table has all 3 actors", set(report["per_actor"]) == set(actors))
    loao = report["leave_one_actor_out"]["D_mode_RG"]
    check("LOAO degenerate (identical actors): min == max == 0.3",
          close(loao["min"], 0.3) and close(loao["max"], 0.3))
    st = report["actor_reweighting"]["D_mode_RG"]
    check("stability range named correctly, flagged not-a-CI",
          st["name"] == "actor_reweighting_95_stability_quantile_range"
          and st["not_a_confidence_interval"] is True)
    check("stability range degenerate here (all actors equal): q2.5 == q97.5 == 0.3",
          close(st["q2_5"], 0.3) and close(st["q97_5"], 0.3))
    check("no p-values are produced", report["p_values"] is None)

    # determinism of the reweighting draws
    report2 = aggregate_model(uars, actors)
    check("reweighting deterministic under frozen seed",
          report2["actor_reweighting"] == report["actor_reweighting"])

    # heterogeneous actors: hand-check panel mean and LOAO spread
    uar2 = dict(uar_by_cell)
    for seed in SEEDS:  # actor 03's song groupkfold premium becomes 0.8 (uar 0.2)
        uar2[("03", "song", "groupkfold", seed)] = 0.2
    pred2, aos2 = make_predictions(actors, spa, uar2)
    rep2 = aggregate_model(collect_actor_uars(pred2, aos2), actors)
    # D_mode_RG: actors 01,02 -> 0.3; actor 03 -> 0.8-0.2=0.6... song premium 1.0-0.2=0.8,
    # speech premium 0.2 -> D=0.6; panel mean = (0.3+0.3+0.6)/3 = 0.4
    check("heterogeneous panel mean D_mode_RG = 0.4", close(rep2["point_estimates"]["D_mode_RG"], 0.4))
    loao2 = rep2["leave_one_actor_out"]["D_mode_RG"]
    # leaving out 03 -> 0.3; leaving out 01 or 02 -> (0.3+0.6)/2 = 0.45
    check("heterogeneous LOAO range = [0.3, 0.45]",
          close(loao2["min"], 0.3) and close(loao2["max"], 0.45))

    # --- fail-closed behaviors ---
    incomplete = {k: v for k, v in uars.items() if k[0] != "02" or k[1] != "song"
                   or k[2] != "loso" or k[3] != 1}
    rep3 = aggregate_model(incomplete, actors)
    check("missing single cell -> incomplete/not_conclusive, no partial pooling",
          rep3["status"] == "incomplete/not_conclusive" and "02/song/loso/s1" in rep3["missing"])

    broken = {k: dict(v) for k, v in predictions.items()}
    victim = next(iter(broken))
    del broken[victim][0]
    try:
        collect_actor_uars(broken, actors_of_sample)
        check("OOF exactly-once violation raises", False)
    except ValueError:
        check("OOF exactly-once violation raises", True)

    # --- real-tree gate refuses without completion+unlock ---
    import subprocess
    proc = subprocess.run(
        [sys.executable, r"E:\科研\claudework\07-ser-repro-protocol-audit\tools\score_d1_results.py"],
        capture_output=True, text=True)
    check("main() refuses real tree without completion manifest (exit 1)",
          proc.returncode == 1 and "REFUSED" in proc.stdout)

    print(f"\n{passed} PASS, {failed} FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
