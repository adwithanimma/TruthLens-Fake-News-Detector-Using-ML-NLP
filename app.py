"""TruthLens Flask REST API.

Endpoints:
    GET  /                web interface
    GET  /api/health      service + model status
    GET  /api/model-info  training variant and evaluation metrics
    POST /api/predict     classify article text  {"text": "..."}
    POST /api/analyze-url classify a news URL     {"url": "..."}
"""

from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from src.extractor import ExtractionError, extract_article
from src.model import get_predictor

ROOT = Path(__file__).resolve().parent

MIN_WORDS = 20
MAX_CHARS = 20000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("truthlens")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
CORS(app, resources={r"/api/*": {"origins": "*"}})


def _asset_version() -> str:
    """Timestamp of the newest static/template file, for cache-busting.

    A browser that keeps a previously-loaded script in memory keeps running the
    old code after a file is edited. Tagging asset URLs with this value changes
    the URL whenever anything under static/ changes, so a normal reload always
    pulls the current file.
    """
    candidates = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix in {".css", ".js"}
        and (ROOT / "static") in p.parents
    ]
    newest = max((p.stat().st_mtime for p in candidates), default=0.0)
    return str(int(newest))


app.jinja_env.globals["asset_version"] = _asset_version


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400, **extra: Any):
        super().__init__(message)
        self.message = message
        self.status = status
        self.extra = extra


@app.errorhandler(ApiError)
def _handle_api_error(exc: ApiError):
    payload = {"error": exc.message, **exc.extra}
    return jsonify(payload), exc.status


@app.errorhandler(404)
def _handle_404(_):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(500)
def _handle_500(exc):
    log.exception("Unhandled error: %s", exc)
    return jsonify({"error": "Internal server error."}), 500


def _json_body() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("Request body must be a JSON object.")
    return data


def _validate_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise ApiError("No article text provided.")
    if len(text) > MAX_CHARS:
        raise ApiError(
            f"Text is too long ({len(text)} characters). Limit is {MAX_CHARS}."
        )
    if len(text.split()) < MIN_WORDS:
        raise ApiError(
            f"Text is too short ({len(text.split())} words). "
            f"Paste at least {MIN_WORDS} words for a reliable result."
        )
    return text


def _predict_or_error(text: str) -> dict[str, Any]:
    try:
        return get_predictor().predict(text).to_dict()
    except FileNotFoundError as exc:
        raise ApiError(str(exc), status=503) from exc
    except ValueError as exc:
        raise ApiError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Prediction failed")
        raise ApiError(f"Prediction failed: {exc}", status=500) from exc


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    predictor = get_predictor()
    loaded = predictor.is_loaded
    if not loaded:
        try:
            predictor.load()
            loaded = True
        except FileNotFoundError:
            loaded = False
    return jsonify(
        {
            "status": "ok" if loaded else "model_missing",
            "model_loaded": loaded,
            "min_words": MIN_WORDS,
            "max_chars": MAX_CHARS,
        }
    )


@app.get("/api/model-info")
def model_info():
    predictor = get_predictor()
    payload = predictor.metrics or {}
    return jsonify(
        {
            "ready": predictor.is_loaded or predictor.path.exists(),
            "variant": payload.get("variant"),
            "trained_at_samples": payload.get("trained_at_samples"),
            "label_map": payload.get("label_map"),
            "metrics": payload.get("metrics", {}),
        }
    )


@app.post("/api/predict")
def predict():
    text = _validate_text(_json_body().get("text", ""))
    result = _predict_or_error(text)
    return jsonify(
        {
            "input_type": "text",
            "characters": len(text),
            "preview": text[:280] + ("..." if len(text) > 280 else ""),
            "result": result,
        }
    )


@app.post("/api/analyze-url")
def analyze_url():
    url = _json_body().get("url", "")
    try:
        article = extract_article(url)
    except ExtractionError as exc:
        raise ApiError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("URL extraction crashed")
        raise ApiError(f"Could not analyze this URL: {exc}", status=500) from exc

    result = _predict_or_error(article.text)
    return jsonify(
        {
            "input_type": "url",
            "article": {
                "url": article.url,
                "title": article.title,
                "source": article.source,
                "published": article.published,
                "image": article.image,
                "word_count": article.word_count,
                "excerpt": article.text[:600] + ("..." if len(article.text) > 600 else ""),
                "paragraphs": article.paragraphs[:6],
            },
            "result": result,
        }
    )


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TruthLens API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    port = args.port
    if not args.debug and not _port_is_free(port, args.host):
        chosen = None
        for candidate in range(port + 1, port + 21):
            if _port_is_free(candidate, args.host):
                chosen = candidate
                break
        if chosen is None:
            log.error("Port %d is in use and no free port was found nearby.", port)
            raise SystemExit(1)
        log.warning("Port %d is already in use by another program.", port)
        log.warning("TruthLens is starting on port %d instead.", chosen)
        log.warning("Open http://%s:%d/ - do not use the busy port.", args.host, chosen)
        port = chosen

    # Bind IPv6 too when possible: some browsers resolve "localhost" to ::1
    # first, and an IPv4-only bind makes that fall through to a different
    # server on the same port, which answers 404 for every /api call.
    app.run(host=args.host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()