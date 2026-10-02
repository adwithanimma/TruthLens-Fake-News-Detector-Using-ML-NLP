"""TruthLens CLI: build the dataset, train the model, or run a prediction.

Usage:
    python train.py              # train with the default logistic regression pipeline
    python train.py --variant svm
    python train.py --predict "some article text"
    python train.py --url https://example.com/article
"""

from __future__ import annotations

import argparse
import json
import sys

from src.dataset import PREPARED_CSV, build, load_samples, summarize
from src.extractor import ExtractionError, extract_article
from src.model import METRICS_PATH, get_predictor, train


def cmd_build() -> int:
    samples = build()
    print(summarize(samples))
    print(f"Written: {PREPARED_CSV}")
    return 0


def cmd_train(variant: str) -> int:
    print(f"Training TruthLens with variant={variant}")
    if not PREPARED_CSV.exists() and not load_samples():
        print("No LIAR source files found; building dataset first...")
        build()
    payload = train(variant=variant)
    metrics = payload["metrics"]
    print("\nModel metrics")
    print("--------------")
    if "accuracy" in metrics:
        print(f"accuracy      : {metrics['accuracy']}")
        print(f"macro-ish f1  : {metrics['f1']}")
        print(f"fake  P/R     : {metrics['precision_fake']} / {metrics['recall_fake']}")
        print(f"real  P/R     : {metrics['precision_real']} / {metrics['recall_real']}")
        print(f"confusion     : {metrics['confusion_matrix']}")
    print(f"model         : {METRICS_PATH.parent}")
    return 0


def cmd_predict(text: str) -> int:
    predictor = get_predictor()
    result = predictor.predict(text)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def cmd_predict_url(url: str) -> int:
    try:
        article = extract_article(url)
    except ExtractionError as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1
    print(f"Title  : {article.title}")
    print(f"Source : {article.source}")
    print(f"Words  : {article.word_count}\n")
    print(json.dumps(get_predictor().predict(article.text).to_dict(), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TruthLens fake news detector")
    parser.add_argument("--variant", choices=["logreg", "svm", "nb"], default="logreg")
    parser.add_argument("--build-only", action="store_true", help="Only build the dataset CSV")
    parser.add_argument("--predict", metavar="TEXT", help="Predict on raw text")
    parser.add_argument("--url", metavar="URL", help="Extract a URL then predict")
    args = parser.parse_args()

    if args.predict:
        return cmd_predict(args.predict)
    if args.url:
        return cmd_predict_url(args.url)
    if args.build_only:
        return cmd_build()
    return cmd_train(args.variant)


if __name__ == "__main__":
    raise SystemExit(main())