import numpy as np
import pytest

from ser_v2 import corpora, seeds, splits


def arrays(corpus, n_spk, n_sent, n_takes):
    return corpora.manifest_arrays(corpora.synthetic_manifest(corpus, n_spk, n_sent, n_takes))


def test_seed_table_is_deterministic_and_model_free():
    assert seeds.split_seed("cremad", 0) == seeds.split_seed("cremad", 0)
    assert seeds.split_seed("cremad", 0) != seeds.split_seed("cremad", 1)
    assert seeds.train_seed("cremad", 3, 1) == seeds.crossing_train_seed("cremad", 3, 1)
    assert 0 <= seeds.split_seed("ravdess", 2) < 2 ** 32


@pytest.mark.parametrize("cell", splits.CELLS)
def test_ctrl_cells_respect_boundaries(cell):
    A = arrays("cremad", 20, 12, 1)
    folds = splits.ctrl_cell_folds(cell, A["y"], A["speaker"], "cremad", 0)
    assert len(folds) == 5
    tests = np.concatenate([f.test for f in folds])
    assert len(np.unique(tests)) == len(A["y"]) == len(tests)
    for f in folds:
        ov = splits.speaker_overlap_fraction(f.train, f.test, A["speaker"])
        assert ov == (0.0 if cell[0] == "G" else 1.0)
        inner_ov = set(A["speaker"][f.fit]) & set(A["speaker"][f.val])
        assert (len(inner_ov) == 0) == (cell[1] == "G")
        assert set(A["y"][f.fit]) == set(A["y"][f.val]) == set(np.unique(A["y"]))
        assert len(f.fit) + len(f.val) == len(f.train)


def test_ctrl_split_is_reproducible():
    A = arrays("ravdess", 24, 2, 4)
    a = splits.ctrl_cell_folds("GG", A["y"], A["speaker"], "ravdess", 1)
    b = splits.ctrl_cell_folds("GG", A["y"], A["speaker"], "ravdess", 1)
    assert all(np.array_equal(x.test, y.test) and np.array_equal(x.fit, y.fit) for x, y in zip(a, b))
    c = splits.ctrl_cell_folds("GG", A["y"], A["speaker"], "ravdess", 2)
    assert not all(np.array_equal(x.test, y.test) for x, y in zip(a, c))


def test_mechanism_block_invariants():
    A = arrays("subesco", 20, 10, 5)
    block = splits.mechanism_block(A["y"], A["speaker"], A["sentence"], A["cell"], A["take"], A["sex"], "subesco_full", 0)
    assert set(block) == set(splits.MECH_CONDITIONS)
    for i in range(5):
        tests = {c: block[c][i].test.tobytes() for c in block}
        assert len(set(tests.values())) == 1
        sizes = {len(block[c][i].fit) for c in block}
        assert len(sizes) == 1
        none, spk, prm, sib = block["none"][i], block["spk"][i], block["prm"][i], block["both_sib"][i]
        test_spk = set(A["speaker"][none.test]); test_sent = set(A["sentence"][none.test])
        assert not (set(A["speaker"][none.fit]) & test_spk) and not (set(A["sentence"][none.fit]) & test_sent)
        assert set(A["speaker"][spk.fit]) >= test_spk and not (set(A["sentence"][spk.fit]) & test_sent)
        assert set(A["sentence"][prm.fit]) >= test_sent and not (set(A["speaker"][prm.fit]) & test_spk)
        sib_cells = set(A["cell"][none.test])
        assert not (set(A["cell"][none.fit]) & sib_cells) and (set(A["cell"][sib.fit]) & sib_cells)
        h1, h2 = block["spk_half_h1"][i].meta["exposed_speakers"], block["spk_half_h2"][i].meta["exposed_speakers"]
        assert set(h1).isdisjoint(h2) and set(h1) | set(h2) == test_spk
    all_tests = np.concatenate([block["none"][i].test for i in range(5)])
    assert len(np.unique(all_tests)) == len(all_tests)


def test_mechanism_block_rejects_too_few_sentences():
    A = arrays("ravdess", 24, 2, 4)
    with pytest.raises(splits.SplitError):
        splits.mechanism_block(A["y"], A["speaker"], A["sentence"], A["cell"], A["take"], A["sex"], "ravdess", 0)


def test_panels():
    A = arrays("subesco", 20, 10, 5)
    p = splits.panel_speakers(A["speaker"], A["sex"], 8, "p", 0)
    assert len(set(A["speaker"][p])) == 8 and len({"M", "F"} & set(A["sex"][p])) == 2
    m = splits.panel_utterance_matched(A["y"], A["speaker"], len(p), "m", 0)
    assert len(m) == len(p) and len(set(A["speaker"][m])) == 20
    one = splits.panel_one_take_per_cell(A["cell"], A["take"], "s")
    assert len(one) == 20 * 10 * 7
    cells = splits.select_cells(A["cell"], 700, "c")
    two = splits.panel_one_take_per_cell(A["cell"], A["take"], "c", 2, cells)
    assert len(two) == 1400 and len(set(A["cell"][two])) == 700


def test_take_and_sentence_grouping():
    A = arrays("subesco", 20, 10, 5)
    for tr, te in splits.take_grouped_folds(A["y"], A["cell"], 7):
        assert not (set(A["cell"][tr]) & set(A["cell"][te]))
        assert set(A["speaker"][tr]) & set(A["speaker"][te])   # speakers mixed across folds
    B = arrays("cremad", 20, 12, 1)
    for tr, te in splits.sentence_grouped_folds(B["y"], B["sentence"], 7):
        assert not (set(B["sentence"][tr]) & set(B["sentence"][te]))


def test_loso_sub_size_matches():
    A = arrays("cremad", 20, 12, 1)
    for tr, te in splits.loso_sub_folds(A["speaker"], 3):
        assert len(set(A["speaker"][te])) == 1 and len(set(A["speaker"][tr])) == round(0.8 * 19)


def test_hygiene_rules():
    rows = corpora.synthetic_manifest("cremad", 3, 2, 1)
    rows[0]["sha256"] = rows[1]["sha256"]                     # same label? make sure conflicting
    rows[1]["label"] = "sad" if rows[0]["label"] != "sad" else "happy"
    rows[2]["sha256"] = rows[3]["sha256"] = "dup"; rows[3]["label"] = rows[2]["label"]
    kept, log = corpora.apply_hygiene("cremad", rows)
    reasons = {d["relative_path"]: d["reason"][:2] for d in log["dropped"]}
    assert reasons[rows[0]["relative_path"]] == "H1" and reasons[rows[1]["relative_path"]] == "H1"
    assert sum(1 for r in log["dropped"] if r["reason"].startswith("H2")) == 1
    assert [r["sample_index"] for r in kept] == list(range(len(kept)))


def test_filename_parsers():
    assert corpora.parse_filename("ravdess", "03-01-05-02-01-02-07.wav")["label"] == "angry"
    assert corpora.parse_filename("ravdess", "03-02-05-02-01-02-07.wav") is None          # song excluded
    assert corpora.parse_filename("cremad", "1040_ITH_SAD_X.wav")["label"] == "sad"
    assert corpora.parse_filename("subesco", "F_01_OISHI_S_1_ANGRY_4.wav")["speaker"] == "F01"
