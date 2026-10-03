from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import load_samples  # noqa: E402
from src.model import build_pipeline  # noqa: E402
from src.text_utils import preprocess  # noqa: E402

SEED = 42
def main() -> None:
    samples = load_samples()
    texts = np.array([s.text for s in samples], dtype=object)
    labels = np.array([s.label for s in samples])
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    print(f"train={len(X_tr)} test={len(X_te)}\n")

    for variant in ("logreg", "svm", "nb"):
        pipe = build_pipeline(variant)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred)
        print(f"{variant:7s} acc={acc:.4f} f1={f1:.4f}")

    meta = np.array(
        [f"{s.subject} {s.speaker}" for s in samples], dtype=object
    )
    M_tr, M_te, ym_tr, ym_te = train_test_split(
        meta, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    pipe = build_pipeline("logreg")
    pipe.fit(M_tr, ym_tr)
    print(f"metadata-only acc={accuracy_score(ym_te, pipe.predict(M_te)):.4f}")


def grid() -> None:
    """Small grid over the classifier head and regularisation strength."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    samples = load_samples()
    texts = np.array([s.text for s in samples], dtype=object)
    labels = np.array([s.label for s in samples])
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    def features(word_ngram, use_char, min_df):
        parts = [
            (
                "word",
                TfidfVectorizer(
                    preprocessor=preprocess,
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']{2,}\b",
                    ngram_range=word_ngram,
                    min_df=min_df,
                    sublinear_tf=True,
                    max_features=60000,
                ),
            )
        ]
        if use_char:
            parts.append(
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        sublinear_tf=True,
                        max_features=80000,
                    ),
                )
            )
        return FeatureUnion(parts)

    results = []
    for word_ngram in [(1, 1), (1, 2)]:
        for min_df in (1, 2, 3):
            for clf_name, clf in (
                ("logreg-C1", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
                ("logreg-C4", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", random_state=SEED)),
                ("sgd-mh", SGDClassifier(loss="modified_huber", alpha=1e-4, max_iter=3000, random_state=SEED)),
                ("svm-cal", CalibratedClassifierCV(LinearSVC(C=0.3, random_state=SEED), cv=3)),
                ("nb", ComplementNB(alpha=0.3)),
            ):
                for use_char in (True, False):
                    pipe = Pipeline(
                        [
                            ("f", features(word_ngram, use_char, min_df)),
                            ("clf", clf),
                        ]
                    )
                    pipe.fit(X_tr, y_tr)
                    pred = pipe.predict(X_te)
                    acc = accuracy_score(y_te, pred)
                    f1 = f1_score(y_te, pred)
                    results.append((acc, f1, word_ngram, min_df, clf_name, use_char))
                    print(
                        f"acc={acc:.4f} f1={f1:.4f} ngram={word_ngram} "
                        f"min_df={min_df} clf={clf_name} char={use_char}"
                    )

    print("\n--- top 8 by accuracy ---")
    for row in sorted(results, reverse=True)[:8]:
        print(
            f"acc={row[0]:.4f} f1={row[1]:.4f} ngram={row[2]} min_df={row[3]} "
            f"clf={row[4]} char={row[5]}"
        )


if __name__ == "__main__":
    if "--grid" in sys.argv:
        grid()
    else:
        main()