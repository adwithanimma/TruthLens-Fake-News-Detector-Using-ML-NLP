"""Dataset preparation for TruthLens.

Source: LIAR — "Liar, Liar Pants on Fire" (Wang, ACL 2017), 12.8k PolitiFact
statements with six graded truthfulness labels.

LIAR labels are collapsed to a binary target:

    FAKE (0) <- pants-fire, false, barely-true
    REAL (1) <- true, mostly-true, half-true

Each training example is the statement text joined to its surrounding context
(venue / speech excerpt), which gives the model article-length prose to work
with rather than a single bare sentence.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PREPARED_CSV = DATA_DIR / "news_dataset.csv"

LABEL_REAL = 1
LABEL_FAKE = 0

# LIAR label -> binary label
LABEL_MAP = {
    "pants-fire": LABEL_FAKE,
    "false": LABEL_FAKE,
    "barely-true": LABEL_FAKE,
    "half-true": LABEL_REAL,
    "mostly-true": LABEL_REAL,
    "true": LABEL_REAL,
}

SPLIT_FILES = {
    "train": DATA_DIR / "liar_train.tsv",
    "valid": DATA_DIR / "liar_valid.tsv",
    "test": DATA_DIR / "liar_test.tsv",
}

MIN_WORDS = 8


@dataclass
class Sample:
    text: str
    label: int
    split: str
    liar_label: str
    subject: str = ""
    speaker: str = ""


def _parse_tsv(path: Path, split: str) -> list[Sample]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Download the LIAR dataset from "
            "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip "
            "and place train.tsv / valid.tsv / test.tsv in data/."
        )

    samples: list[Sample] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) < 14:
                continue
            liar_label = row[1].strip().lower()
            if liar_label not in LABEL_MAP:
                continue
            statement, subject, speaker, context = (
                row[2].strip(),
                row[3].strip(),
                row[4].strip(),
                row[13].strip(),
            )
            text = f"{statement} {context}".strip()
            if len(text.split()) < MIN_WORDS:
                continue
            samples.append(
                Sample(
                    text=text,
                    label=LABEL_MAP[liar_label],
                    split=split,
                    liar_label=liar_label,
                    subject=subject,
                    speaker=speaker,
                )
            )
    return samples


def load_samples() -> list[Sample]:
    """Load every LIAR split as prepared binary samples."""
    samples: list[Sample] = []
    for split, path in SPLIT_FILES.items():
        samples.extend(_parse_tsv(path, split))
    return samples


def write_dataset(samples: list[Sample], path: Path = PREPARED_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label", "split", "liar_label", "subject", "speaker"])
        for s in samples:
            writer.writerow([s.text, s.label, s.split, s.liar_label, s.subject, s.speaker])
    return path


def build(seed: int = 42, shuffle: bool = True) -> list[Sample]:
    """Parse LIAR, shuffle, and persist the combined dataset."""
    samples = load_samples()
    if shuffle:
        random.Random(seed).shuffle(samples)
    write_dataset(samples)
    return samples


def summarize(samples: list[Sample]) -> str:
    by_split: dict[str, list[Sample]] = {}
    for s in samples:
        by_split.setdefault(s.split, []).append(s)
    lines = [f"Total samples: {len(samples)}"]
    for split in ("train", "valid", "test"):
        group = by_split.get(split, [])
        real = sum(1 for s in group if s.label == LABEL_REAL)
        lines.append(f"  {split:5s} n={len(group):5d}  real={real:5d}  fake={len(group) - real:5d}")
    return "\n".join(lines)


if __name__ == "__main__":
    prepared = build()
    print(summarize(prepared))
    print(f"Written: {PREPARED_CSV}")