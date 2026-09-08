"""
evaluate.py
===========
Honest evaluation for SER, and the single most important improvement:
speaker normalization.

It compares three settings on the SAME cached features:

  (A) speaker-DEPENDENT   -- random train/test split. The same actor can appear
      in both train and test, so the score is optimistic.
  (B) speaker-INDEPENDENT -- GroupKFold by actor: whole speakers are held out,
      so the test speakers are never seen in training. This is the honest number
      and is usually LOWER than (A).
  (C) speaker-INDEPENDENT + per-speaker normalization -- z-score each speaker's
      features using only that speaker's own utterances (label-free, so no
      leakage), then evaluate speaker-independently. This removes much of the
      per-speaker baseline (pitch/loudness/timbre) and usually recovers accuracy.

Runs off the feature cache, so it's fast (no re-extraction) as long as
features_cache.npz already exists from train.py.

Usage:
    python evaluate.py data --scheme ravdess
"""
import argparse

import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from dataset import build_dataset


def fixed_svm():
    """One fixed SVM everywhere, so only the split / normalization changes."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")),
    ])


def per_speaker_normalize(X, groups):
    """Z-score features within each speaker. Uses feature values only (no labels)."""
    Xn = X.astype(np.float64).copy()
    for g in np.unique(groups):
        mask = groups == g
        mu = Xn[mask].mean(axis=0)
        sd = Xn[mask].std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xn[mask] = (Xn[mask] - mu) / sd
    return Xn


def eval_speaker_dependent(X, y):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)
    model = fixed_svm().fit(Xtr, ytr)
    return accuracy_score(yte, model.predict(Xte))


def eval_speaker_independent(X, y, groups, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    pred = cross_val_predict(fixed_svm(), X, y, groups=groups, cv=gkf, n_jobs=-1)
    return pred, accuracy_score(y, pred)


def main():
    ap = argparse.ArgumentParser(description="Speaker-independent SER evaluation.")
    ap.add_argument("data_dir")
    ap.add_argument("--scheme", default="ravdess", choices=["ravdess", "folder"])
    ap.add_argument("--cache", default="features_cache.npz")
    args = ap.parse_args()

    X, y, groups, _ = build_dataset(args.data_dir, args.scheme, args.cache or None)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    print(f"\n{X.shape[0]} samples, {n_groups} speakers, {len(le.classes_)} classes.")
    print(f"Speaker-independent CV uses GroupKFold with {n_splits} folds.\n")

    # (A) speaker-dependent
    acc_a = eval_speaker_dependent(X, y_enc)

    # (B) speaker-independent
    pred_b, acc_b = eval_speaker_independent(X, y_enc, groups, n_splits)

    # (C) speaker-independent + per-speaker normalization
    Xn = per_speaker_normalize(X, groups)
    pred_c, acc_c = eval_speaker_independent(Xn, y_enc, groups, n_splits)

    print("=" * 56)
    print("Accuracy comparison")
    print("=" * 56)
    print(f"(A) speaker-dependent  (random split) : {acc_a:.3f}   <- optimistic")
    print(f"(B) speaker-independent(GroupKFold)   : {acc_b:.3f}   <- honest")
    print(f"(C) (B) + per-speaker normalization   : {acc_c:.3f}   <- improved")
    print("=" * 56)

    print("\nDetailed report for the honest + normalized setting (C):\n")
    print(classification_report(y_enc, pred_c, target_names=le.classes_, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print("labels:", list(le.classes_))
    print(confusion_matrix(y_enc, pred_c))


if __name__ == "__main__":
    main()
