# TruthLens — Fake News Detector using ML & NLP

Classifies news content as **REAL** or **FAKE** and returns a confidence score.
Paste article text or hand over a news URL; the Flask API extracts the readable
body, cleans it, vectorises it with TF-IDF, and runs a logistic regression model
trained on the LIAR fact-checking benchmark.

## How it works

```
article text or URL
        │
        ▼
  extraction (BeautifulSoup)   ← URL mode only: strips nav, script, ads
        │
        ▼
  cleaning (src/text_utils.py) ← lowercase, strip URLs + punctuation, drop stopwords
        │
        ▼
  TF-IDF (word 1-2 grams)      ← 60k max features, sublinear TF
        │
        ▼
  LogisticRegression (C=1.0)   ← calibrated probability
        │
        ▼
  { label, confidence, scores }
```

## Setup

```bash
pip install -r requirements.txt
```

The trained model is committed in `models/`, so the app runs immediately after
`pip install` — no training step required.

The LIAR dataset is **not** committed. It is distributed for research use only
(Wang, ACL 2017), so fetch it yourself before retraining:

```bash
# Download https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
# and extract train.tsv / valid.tsv / test.tsv into data/
```

`python train.py` then rebuilds `data/news_dataset.csv` and the model from those
files. Both generated files are gitignored.

Train the model (writes `models/truthlens_model.joblib` + `models/metrics.json`):

```bash
python train.py                # logistic regression (default)
python train.py --variant svm  # calibrated LinearSVC
python train.py --variant nb   # ComplementNB baseline
```

Other CLI modes:

```bash
python train.py --build-only              # rebuild data/news_dataset.csv
python train.py --predict "article text"  # classify a string
python train.py --url https://site/story  # extract a URL, then classify
```

Run the server:

```bash
python app.py                          # http://127.0.0.1:5000
python app.py --port 8080 --debug
```

If the port is already served by another process, TruthLens picks the next free
one and prints the URL to use:

```
WARNING Port 5000 is already held by another program.
WARNING TruthLens is starting on port 5001 instead.
WARNING Open http://127.0.0.1:5001/
```

Use the port from that line — the page and the API must come from the same port.
A stale tab pointing at a different port answers `/api/*` with 404, and the
status pill will read "Wrong port — API not found".

Port availability is resolved by asking the OS which addresses actually have a
listener, via `psutil`. A `bind()` probe is not sufficient on Windows: binding a
wildcard address succeeds even when the port is already served, and binding a
specific address succeeds alongside another process's wildcard bind. A bind
probe therefore reports a contended port as free, and `localhost` — which may
resolve to `::1` first — then reaches the other server instead of this one.

## API

| Method | Endpoint        | Body            | Returns                          |
| ------ | --------------- | --------------- | -------------------------------- |
| GET    | `/`             | –               | Web interface                    |
| GET    | `/api/health`   | –               | Service + model load status      |
| GET    | `/api/model-info` | –             | Training variant and metrics     |
| POST   | `/api/predict`  | `{"text": str}` | Verdict, confidence, word count  |
| POST   | `/api/analyze-url` | `{"url": str}` | Extracted article + verdict      |

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The transport authority said all services would return to normal by Friday after engineers replaced damaged track near the estuary. Commuters faced delays of up to forty minutes while crews worked overnight."}'
```

```json
{
  "input_type": "text",
  "result": {
    "label": "REAL",
    "is_fake": false,
    "confidence": 0.6312,
    "confidence_percent": "63.1%",
    "scores": { "fake": 0.3688, "real": 0.6312 },
    "word_count": 44
  }
}
```

URL mode adds an `article` block with `title`, `source`, `published`, `image`,
`word_count`, `excerpt`, and the first few `paragraphs`.

Errors return `{"error": "..."}` with status 400 (bad input, extraction failure),
503 (model missing), or 500.

## Model performance

LIAR labels are collapsed to binary: `pants-fire / false / barely-true` → FAKE,
`true / mostly-true / half-true` → REAL. 12,764 samples, 80/20 stratified split:

| variant          | accuracy | F1    | fake P/R      | real P/R      |
| ---------------- | -------- | ----- | ------------- | ------------- |
| logreg (shipped) | 0.6255   | 0.697 | 0.60 / 0.44   | 0.64 / 0.77   |
| svm (calibrated) | 0.6224   | 0.700 | —             | —             |
| nb               | 0.6283   | 0.674 | —             | —             |

Defaults came from the grid in `experiments/sweep.py`. Reproduce with
`python experiments/sweep.py --grid`.

These numbers are honest but modest, and that is expected: LIAR statements are
short political claims, so surface TF-IDF features carry limited signal. A text-only
model scoring near 0.63 is in line with published LIAR baselines. The model leans
toward REAL, so treat a low confidence score as "inconclusive" rather than a verdict.

## Tests

```bash
python -m pytest tests -q
```

45 tests covering preprocessing, extraction parsing and error paths, API status
codes, input validation, probability consistency, static asset integrity, and
port-conflict detection. The suite skips itself if no trained model is present.

`experiments/check_render.py` is a separate live check: it replays the frontend's
percentage math against a running server and fails if any displayed value exceeds
100% or the two probability bars stop summing to 100%.

```bash
python app.py &
python experiments/check_render.py http://localhost:5000   # or whatever port it printed
```

Static assets are served with a cache-busting `?v=` parameter derived from the
newest file under `static/`, so editing the JS or CSS and reloading always picks up
the change instead of running a cached copy.

## Layout

```
app.py                  Flask REST API + entrypoint
train.py                training / prediction CLI
src/dataset.py          LIAR parsing, label mapping, CSV build
src/model.py            pipeline construction, training, Predictor
src/extractor.py        URL fetch + boilerplate-stripped article extraction
src/text_utils.py       cleaning and stopword removal
templates/index.html    web interface
static/css/style.css    dark/light theme
static/js/app.js        tab state, fetch, result rendering
experiments/sweep.py    model selection grid
experiments/check_render.py  live frontend percentage sanity check
tests/                  pytest suite
```

## Limits

- Fetches only server-rendered HTML. Paywalled and JavaScript-rendered pages
  return an extraction error rather than a wrong verdict.
- Many publishers block automated requests (401/403), surfaced as a clear error.
- No metadata, author, or image-forensics signals. Stylistic cues only.
- For research and education. Always verify a headline against primary sources.
