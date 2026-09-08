"""Split generator. Every partition in the program is produced here from the seed table
and checked by pre-fit assertions. Nothing downstream may construct its own split.

Cells (outer letter first): RR, RG, GR, GG.
  R outer = StratifiedKFold(5, shuffle, split_seed) ignoring speaker
  G outer = StratifiedGroupKFold(5, shuffle, split_seed, groups=speaker)
  R inner = stratified 25% of outer train allowing speaker overlap
  G inner = GroupShuffleSplit 25% by speaker, retried up to 100 offsets until every class
            is present on both sides (P1 contract)

Other partitions: sentence-grouped (SG), take-grouped, LOSO / LOSO-sub, the mechanism
block (speaker-overlap x prompt-overlap checkerboard with matched training size), panels
(CREMA-D-24, CREMA-D-91-matched, SUBESCO 1400/700 panels, SUBESCO-980) and the P1 frozen
splits used for reconciliation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)

from . import seeds
from .common import canonical_json, sha256_text, read_csv

N_FOLDS = 5
INNER_FRACTION = 0.25
CELLS = ("RR", "RG", "GR", "GG")
MECH_CONDITIONS = ("none", "spk", "prm", "both", "both_sib", "spk_half_h1", "spk_half_h2")


class SplitError(RuntimeError):
    pass


@dataclass
class Fold:
    fold: int
    train: np.ndarray          # outer train (fit + val)
    test: np.ndarray           # outer test
    fit: np.ndarray
    val: np.ndarray
    meta: dict = field(default_factory=dict)

    def hashes(self) -> dict:
        return {
            "outer_sha256": sha256_text(canonical_json({"train": self.train.tolist(), "test": self.test.tolist()})),
            "inner_sha256": sha256_text(canonical_json({"fit": self.fit.tolist(), "val": self.val.tolist()})),
        }


# ----------------------------------------------------------------------------- primitives

def _sorted(a) -> np.ndarray:
    return np.sort(np.asarray(a, dtype=np.int64))


def outer_random(y: np.ndarray, seed: int, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(y), dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(_sorted(tr), _sorted(te)) for tr, te in skf.split(idx, y)]


def outer_grouped(y: np.ndarray, groups: np.ndarray, seed: int, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(y), dtype=np.int64)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(_sorted(tr), _sorted(te)) for tr, te in sgkf.split(idx, y, groups)]


def inner_random(outer_train: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    fit, val = train_test_split(outer_train, test_size=INNER_FRACTION, stratify=y[outer_train], random_state=seed)
    return _sorted(fit), _sorted(val), 0


def inner_grouped(outer_train: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int,
                  all_labels: set[int], max_offsets: int = 100) -> tuple[np.ndarray, np.ndarray, int]:
    for offset in range(max_offsets):
        state = (seed + offset) & 0xFFFFFFFF
        gss = GroupShuffleSplit(n_splits=1, test_size=INNER_FRACTION, random_state=state)
        lf, lv = next(gss.split(outer_train, y[outer_train], groups[outer_train]))
        fit, val = _sorted(outer_train[lf]), _sorted(outer_train[lv])
        if set(y[fit].tolist()) == all_labels and set(y[val].tolist()) == all_labels:
            return fit, val, offset
    raise SplitError("inner_split_infeasible")


# ----------------------------------------------------------------------------- CTRL cells

def ctrl_cell_folds(cell: str, y: np.ndarray, groups: np.ndarray, corpus_level: str, r: int,
                    subset: np.ndarray | None = None) -> list[Fold]:
    """Folds for one cell on one replicate. `subset` restricts the population (panels)."""
    if cell not in CELLS:
        raise ValueError(cell)
    all_labels = set(np.unique(y).tolist())
    base_idx = np.arange(len(y), dtype=np.int64) if subset is None else _sorted(subset)
    y_s, g_s = y[base_idx], groups[base_idx]
    sseed = seeds.split_seed(corpus_level, r)
    folds_local = outer_random(y_s, sseed) if cell[0] == "R" else outer_grouped(y_s, g_s, sseed)
    out = []
    for k, (tr_l, te_l) in enumerate(folds_local):
        tr, te = base_idx[tr_l], base_idx[te_l]
        iseed = seeds.inner_seed(corpus_level, cell, k, r)
        if cell[1] == "R":
            fit, val, attempt = inner_random(tr, y, iseed)
        else:
            fit, val, attempt = inner_grouped(tr, y, groups, iseed, all_labels)
        f = Fold(k, tr, te, fit, val, {"cell": cell, "corpus_level": corpus_level, "r": r,
                                       "split_seed": sseed, "inner_seed": iseed, "inner_attempt": attempt})
        assert_fold(f, y, groups, outer_exclusive=(cell[0] == "G"), inner_exclusive=(cell[1] == "G"),
                    labels=all_labels)
        out.append(f)
    assert_exactly_once([f.test for f in out], base_idx)
    return out


# ----------------------------------------------------------------------------- other outer protocols

def sentence_grouped_folds(y: np.ndarray, sentences: np.ndarray, seed: int, n_folds: int = N_FOLDS):
    return outer_grouped(y, sentences, seed, n_folds)


def take_grouped_folds(y: np.ndarray, cells: np.ndarray, seed: int, n_folds: int = N_FOLDS):
    """groups = speaker x sentence x label cell, speakers mixed across folds, sibling takes never split."""
    return outer_grouped(y, cells, seed, n_folds)


def loso_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(groups), dtype=np.int64)
    return [(idx[groups != s], idx[groups == s]) for s in sorted(np.unique(groups).tolist())]


def loso_sub_folds(groups: np.ndarray, seed: int, train_fraction: float = 0.8) -> list[tuple[np.ndarray, np.ndarray]]:
    """LOSO with the training speakers hash-subsampled to `train_fraction` (size-matched to 5-fold)."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(groups), dtype=np.int64)
    out = []
    for s in sorted(np.unique(groups).tolist()):
        others = sorted(set(np.unique(groups).tolist()) - {s})
        keep = sorted(rng.choice(others, size=max(1, int(round(train_fraction * len(others)))), replace=False).tolist())
        out.append((idx[np.isin(groups, keep)], idx[groups == s]))
    return out


# ----------------------------------------------------------------------------- panels

def _hash_rank(items: list[str], salt: str) -> list[str]:
    return sorted(items, key=lambda x: sha256_text(f"{salt}|{x}"))


def panel_speakers(groups: np.ndarray, sex: np.ndarray, n_speakers: int, panel: str, k: int) -> np.ndarray:
    """Hash-select `n_speakers`, stratified by sex when the sex column is populated."""
    speakers = sorted(np.unique(groups).tolist())
    sex_of = {s: str(sex[groups == s][0]) for s in speakers}
    strata: dict[str, list[str]] = {}
    for s in speakers:
        strata.setdefault(sex_of[s] or "unknown", []).append(s)
    chosen: list[str] = []
    total = len(speakers)
    remaining = n_speakers
    keys = sorted(strata)
    for j, key in enumerate(keys):
        pool = _hash_rank(strata[key], f"{panel}|{k}|{key}")
        quota = remaining if j == len(keys) - 1 else int(round(n_speakers * len(pool) / total))
        chosen.extend(pool[:quota])
        remaining -= quota
    chosen = chosen[:n_speakers]
    return _sorted(np.where(np.isin(groups, chosen))[0])


def panel_utterance_matched(y: np.ndarray, groups: np.ndarray, target_n: int, panel: str, k: int) -> np.ndarray:
    """Keep all speakers but subsample utterances per speaker (stratified by label) to ~target_n."""
    speakers = sorted(np.unique(groups).tolist())
    total = len(y)
    keep: list[int] = []
    for s in speakers:
        idx_s = np.where(groups == s)[0]
        n_s = max(len(np.unique(y[idx_s])), int(round(target_n * len(idx_s) / total)))
        labels = sorted(np.unique(y[idx_s]).tolist())
        per_label = {lab: [int(i) for i in idx_s[y[idx_s] == lab]] for lab in labels}
        ranked = {lab: sorted(v, key=lambda i: sha256_text(f"{panel}|{k}|{s}|{i}")) for lab, v in per_label.items()}
        take: list[int] = []
        # round-robin across labels keeps the label mix proportional
        while len(take) < n_s and any(ranked.values()):
            for lab in labels:
                if ranked[lab] and len(take) < n_s:
                    take.append(ranked[lab].pop(0))
        keep.extend(take)
    # exact trim to target_n: remove last-ranked items from the largest speakers, round-robin
    if len(keep) > target_n:
        by_spk: dict[str, list[int]] = {}
        for i in keep:
            by_spk.setdefault(str(groups[i]), []).append(i)
        for s in by_spk:
            by_spk[s].sort(key=lambda i: sha256_text(f"{panel}|{k}|trim|{i}"))
        while len(keep) > target_n:
            s = max(by_spk, key=lambda x: (len(by_spk[x]), x))
            drop = by_spk[s].pop()
            keep.remove(drop)
    return _sorted(keep)


def panel_one_take_per_cell(cells: np.ndarray, takes: np.ndarray, salt: str, n_takes: int = 1,
                            cell_subset: set[str] | None = None) -> np.ndarray:
    """Select `n_takes` takes per speaker x sentence x label cell by hash order."""
    keep: list[int] = []
    by_cell: dict[str, list[int]] = {}
    for i, c in enumerate(cells.tolist()):
        if cell_subset is None or c in cell_subset:
            by_cell.setdefault(c, []).append(i)
    for c, members in by_cell.items():
        ranked = sorted(members, key=lambda i: sha256_text(f"{salt}|{c}|{takes[i]}"))
        keep.extend(ranked[:n_takes])
    return _sorted(keep)


def select_cells(cells: np.ndarray, n_cells: int, salt: str) -> set[str]:
    return set(_hash_rank(sorted(set(cells.tolist())), salt)[:n_cells])


# ----------------------------------------------------------------------------- mechanism block

def _mass_balanced_groups(items: list[str], mass: dict[str, int], n_groups: int, salt: str) -> dict[str, int]:
    """Greedy assignment: items in descending mass (ties by hash) go to the lightest group."""
    order = sorted(items, key=lambda x: (-mass[x], sha256_text(f"{salt}|{x}")))
    load = [0] * n_groups
    assign: dict[str, int] = {}
    for it in order:
        g = min(range(n_groups), key=lambda j: (load[j], j))
        assign[it] = g
        load[g] += mass[it]
    return assign


def _balanced_groups(items: list[str], n_groups: int, salt: str, strata: dict[str, str] | None = None) -> dict[str, int]:
    """Round-robin assignment of items to n_groups in hash order, within strata when given."""
    assign: dict[str, int] = {}
    if strata:
        buckets: dict[str, list[str]] = {}
        for it in items:
            buckets.setdefault(strata.get(it, ""), []).append(it)
        offset = 0
        for key in sorted(buckets):
            for j, it in enumerate(_hash_rank(buckets[key], f"{salt}|{key}")):
                assign[it] = (j + offset) % n_groups
            offset += len(buckets[key])
    else:
        for j, it in enumerate(_hash_rank(items, salt)):
            assign[it] = j % n_groups
    return assign


def _stratified_fill(pool: np.ndarray, y: np.ndarray, n: int, seed_text: str) -> np.ndarray:
    """Hash-ordered subsample of `pool` of size n, proportional by label."""
    if n >= len(pool):
        return _sorted(pool)
    labels = sorted(np.unique(y[pool]).tolist())
    ranked = {lab: sorted([int(i) for i in pool[y[pool] == lab]], key=lambda i: sha256_text(f"{seed_text}|{i}"))
              for lab in labels}
    out: list[int] = []
    while len(out) < n:
        progressed = False
        for lab in labels:
            if ranked[lab] and len(out) < n:
                out.append(ranked[lab].pop(0))
                progressed = True
        if not progressed:
            break
    return _sorted(out)


def mechanism_block(y: np.ndarray, speakers: np.ndarray, sentences: np.ndarray, cells: np.ndarray,
                    takes: np.ndarray, sex: np.ndarray, corpus_level: str, r: int,
                    conditions: tuple[str, ...] = MECH_CONDITIONS, n_groups: int = N_FOLDS,
                    include_siblings_condition: bool = True, rotations: tuple[int, ...] = (0, 1)) -> dict[str, list[Fold]]:
    """Speaker-overlap x prompt-overlap checkerboard with matched training size.

    Fold (i, rot): test cell = speakers in speaker-group i AND sentences in prompt-group
    (i + rot) mod n_groups, one take per speaker x sentence x label cell (siblings held out of
    training except under both_sib). With rotations (0, 1) every speaker is tested on two prompt
    groups, doubling the per-speaker test items; fold index = rot * n_groups + i.
    Every condition trains on exactly N_train utterances: its full exposure set plus a
    stratified hash-ordered fill from the both-exclusive pool. Inner validation is a
    speaker-grouped 25% of the both-exclusive pool, identical across conditions.
    """
    speakers_u = sorted(np.unique(speakers).tolist())
    sentences_u = sorted(np.unique(sentences).tolist())
    if len(sentences_u) < n_groups:
        raise SplitError(f"mechanism block needs >= {n_groups} sentences; got {len(sentences_u)}")
    sex_of = {s: str(sex[speakers == s][0]) for s in speakers_u}
    spk_group = _balanced_groups(speakers_u, n_groups, f"mech-spk|{corpus_level}|{r}", sex_of)
    sent_mass = {s: int((sentences == s).sum()) for s in sentences_u}
    prm_group = _mass_balanced_groups(sentences_u, sent_mass, n_groups, f"mech-prm|{corpus_level}|{r}")
    spk_g = np.asarray([spk_group[s] for s in speakers.tolist()])
    prm_g = np.asarray([prm_group[s] for s in sentences.tolist()])
    all_labels = set(np.unique(y).tolist())
    idx = np.arange(len(y), dtype=np.int64)
    active = tuple(c for c in conditions if not (c == "both_sib" and not include_siblings_condition))
    out: dict[str, list[Fold]] = {c: [] for c in active}
    fold_specs = [(rot * n_groups + i, i, (i + rot) % n_groups) for rot in rotations for i in range(n_groups)]
    for fold_id, i, j in fold_specs:
        in_cell = (spk_g == i) & (prm_g == j)
        cell_idx = idx[in_cell]
        test = panel_one_take_per_cell(cells[cell_idx], takes[cell_idx], f"mech-test|{corpus_level}|{r}|{fold_id}")
        test = _sorted(cell_idx[test])
        siblings = _sorted(np.setdiff1d(cell_idx, test))
        bb_pool = idx[(spk_g != i) & (prm_g != j)]
        iseed = seeds.mech_seed(corpus_level, fold_id, "inner", r)
        fill_pool, val, attempt = inner_grouped(bb_pool, y, speakers, iseed, all_labels)
        n_train = len(fill_pool)
        test_speakers = sorted(set(speakers[cell_idx].tolist()))
        halves = _hash_rank(test_speakers, f"mech-half|{corpus_level}|{r}|{i}")   # same halves in both rotations
        h1, h2 = set(halves[: len(halves) // 2]), set(halves[len(halves) // 2:])
        spk_exposure = idx[(spk_g == i) & (prm_g != j)]
        prm_exposure = idx[(spk_g != i) & (prm_g == j)]
        for cond in active:
            if cond == "none":
                exposure = np.asarray([], dtype=np.int64)
            elif cond == "spk":
                exposure = spk_exposure
            elif cond == "prm":
                exposure = prm_exposure
            elif cond == "both":
                exposure = np.union1d(spk_exposure, prm_exposure)
            elif cond == "both_sib":
                exposure = np.union1d(np.union1d(spk_exposure, prm_exposure), siblings)
            elif cond == "spk_half_h1":
                exposure = spk_exposure[np.isin(speakers[spk_exposure], sorted(h1))]
            elif cond == "spk_half_h2":
                exposure = spk_exposure[np.isin(speakers[spk_exposure], sorted(h2))]
            else:
                raise ValueError(cond)
            if len(exposure) > n_train:
                raise SplitError(f"exposure larger than matched training size ({cond}, fold {fold_id})")
            fill = _stratified_fill(fill_pool, y, n_train - len(exposure), f"mech-fill|{corpus_level}|{r}|{fold_id}|{cond}")
            fit = _sorted(np.union1d(exposure, fill))
            train = _sorted(np.union1d(fit, val))
            f = Fold(fold_id, train, test, fit, val, {"speaker_group": i, "prompt_group": j,
                "condition": cond, "corpus_level": corpus_level, "r": r, "n_train_fit": int(len(fit)),
                "n_exposure": int(len(exposure)), "n_fill": int(len(fill)), "n_val": int(len(val)),
                "n_test": int(len(test)), "n_siblings_held_out": int(len(siblings)) if cond != "both_sib" else 0,
                "test_speakers": test_speakers, "exposed_speakers": sorted(set(speakers[exposure].tolist()) & set(test_speakers)),
                "unexposed_test_speakers": sorted(set(test_speakers) - set(speakers[exposure].tolist())),
                "inner_attempt": attempt,
            })
            assert_mechanism_fold(f, y, speakers, sentences, cells, cond, test_speakers, set(sentences[cell_idx].tolist()),
                                  siblings, n_train)
            out[cond].append(f)
    # cross-condition identity checks and exactly-once test cover within each rotation
    for k in range(len(fold_specs)):
        tests = {c: out[c][k].test.tobytes() for c in out}
        vals = {c: out[c][k].val.tobytes() for c in out}
        fits = {c: len(out[c][k].fit) for c in out}
        if len(set(tests.values())) != 1 or len(set(vals.values())) != 1 or len(set(fits.values())) != 1:
            raise SplitError(f"mechanism fold {k}: test/val/train-size not identical across conditions")
    first = next(iter(out.values()))
    all_tests = np.concatenate([f.test for f in first])
    if len(np.unique(all_tests)) != len(all_tests):
        raise SplitError("mechanism test cells overlap across folds")
    return out


# ----------------------------------------------------------------------------- P1 frozen splits

def p1_frozen_folds(split_csv: Path, manifest_rows: list[dict]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Map P1's outer test assignments (random, groupkfold) onto v2 manifest indices by relative path."""
    pos = {r["relative_path"]: int(r["sample_index"]) for r in manifest_rows}
    folds: dict[str, dict[int, list[int]]] = {}
    missing = 0
    for row in read_csv(split_csv):
        p = row["relative_path"]
        if p not in pos:
            missing += 1
            continue
        folds.setdefault(row["protocol"], {}).setdefault(int(row["outer_fold"]), []).append(pos[p])
    out = {}
    all_idx = np.arange(len(manifest_rows), dtype=np.int64)
    for protocol, by_fold in folds.items():
        fl = []
        for k in sorted(by_fold):
            te = _sorted(by_fold[k])
            fl.append((np.setdiff1d(all_idx, te), te))
        out[protocol] = fl
    out["_missing_paths"] = missing
    return out


def p1_inner_split(corpus: str, protocol: str, outer_fold: int, seed: int, outer_train: np.ndarray,
                   y: np.ndarray, groups: np.ndarray, n_labels: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Byte-identical re-implementation of P1's inner split rule (key 'P1-inner|...')."""
    from .common import stable_u32
    base = stable_u32(f"P1-inner|{corpus}|{protocol}|{outer_fold}|{seed}")
    all_labels = set(range(n_labels))
    if protocol == "random":
        fit, val = train_test_split(outer_train, test_size=0.25, stratify=y[outer_train], random_state=base)
        return _sorted(fit), _sorted(val), base, 0
    for offset in range(100):
        state = (base + offset) & 0xFFFFFFFF
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=state)
        lf, lv = next(gss.split(outer_train, y[outer_train], groups[outer_train]))
        fit, val = _sorted(outer_train[lf]), _sorted(outer_train[lv])
        if set(y[fit].tolist()) == all_labels and set(y[val].tolist()) == all_labels:
            return fit, val, base, offset
    raise SplitError("inner_split_infeasible")


# ----------------------------------------------------------------------------- assertions

def assert_fold(f: Fold, y: np.ndarray, groups: np.ndarray, outer_exclusive: bool, inner_exclusive: bool,
                labels: set[int]) -> None:
    if np.intersect1d(f.train, f.test).size:
        raise SplitError("outer train/test overlap")
    if np.intersect1d(f.fit, f.val).size or np.intersect1d(f.fit, f.test).size or np.intersect1d(f.val, f.test).size:
        raise SplitError("inner fit/val/test overlap")
    if not np.array_equal(_sorted(np.concatenate([f.fit, f.val])), _sorted(f.train)):
        raise SplitError("fit + val must partition outer train")
    if outer_exclusive and set(groups[f.train]) & set(groups[f.test]):
        raise SplitError("speaker overlap across an exclusive outer boundary")
    if inner_exclusive and set(groups[f.fit]) & set(groups[f.val]):
        raise SplitError("speaker overlap across an exclusive inner boundary")
    if set(y[f.fit].tolist()) != labels or set(y[f.val].tolist()) != labels:
        raise SplitError("inner split missing a class")
    if set(y[f.test].tolist()) != labels:
        f.meta["test_missing_classes"] = sorted(labels - set(y[f.test].tolist()))


def assert_exactly_once(tests: list[np.ndarray], population: np.ndarray) -> None:
    cat = np.concatenate(tests)
    if len(np.unique(cat)) != len(cat) or not np.array_equal(np.sort(cat), np.sort(population)):
        raise SplitError("outer test folds are not an exactly-once cover of the population")


def assert_mechanism_fold(f: Fold, y, speakers, sentences, cells, cond, test_speakers, test_sentences,
                          siblings, n_train) -> None:
    if len(f.fit) != n_train:
        raise SplitError(f"{cond}: training size {len(f.fit)} != matched {n_train}")
    if np.intersect1d(f.fit, f.test).size or np.intersect1d(f.val, f.test).size:
        raise SplitError(f"{cond}: test leaks into training")
    if set(speakers[f.val]) & set(test_speakers) or set(sentences[f.val]) & test_sentences:
        raise SplitError(f"{cond}: inner validation touches test speakers or prompts")
    fit_spk, fit_sent = set(speakers[f.fit].tolist()), set(sentences[f.fit].tolist())
    if cond in ("none", "prm") and fit_spk & set(test_speakers):
        raise SplitError(f"{cond}: test speaker present in training")
    if cond in ("none", "spk", "spk_half_h1", "spk_half_h2") and fit_sent & test_sentences:
        raise SplitError(f"{cond}: test prompt present in training")
    if cond != "both_sib" and np.intersect1d(f.fit, siblings).size:
        raise SplitError(f"{cond}: sibling take leaked into training")
    if cond == "spk_half_h1" or cond == "spk_half_h2":
        exposed = set(f.meta["exposed_speakers"])
        if not (0 < len(exposed) < len(test_speakers)):
            raise SplitError(f"{cond}: half exposure must expose a strict subset of test speakers")


def speaker_overlap_fraction(train: np.ndarray, test: np.ndarray, groups: np.ndarray) -> float:
    ts = set(groups[test].tolist())
    return len(ts & set(groups[train].tolist())) / max(1, len(ts))
