"""API tests for the TruthLens Flask app. Run: python -m pytest tests -q"""

from __future__ import annotations

import pytest

from app import app
from src.model import get_predictor

LONG_REAL = (
    "In a televised address on Tuesday evening, the finance ministry confirmed that "
    "the national deficit had narrowed for the third consecutive quarter, citing "
    "audited figures released by the treasury. Analysts had expected the improvement, "
    "though the pace was slower than the projection published in the spring budget."
)

LONG_FAKE = (
    "Scientists confirm Earth is definitively flat and every photo from the space "
    "station was staged, according to a viral post. The post claims a whistleblower "
    "engineer proved the curvature models were invented to mislead the public."
)


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Skip the whole module when no trained model is on disk."""
    predictor = get_predictor()
    try:
        predictor.load()
    except FileNotFoundError:
        pytest.skip("Model not found. Run: python train.py")


@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


# ---------- health / meta ----------

def test_health_reports_model_loaded(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["model_loaded"] is True


def test_model_info_exposes_metrics(client):
    res = client.get("/api/model-info")
    assert res.status_code == 200
    assert "metrics" in res.get_json()


def test_index_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"TruthLens" in res.data


# ---------- /api/predict ----------

def test_predict_returns_verdict_and_scores(client):
    res = client.post("/api/predict", json={"text": LONG_REAL})
    assert res.status_code == 200
    body = res.get_json()
    assert body["input_type"] == "text"
    result = body["result"]
    assert result["label"] in {"REAL", "FAKE"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["scores"]) == {"fake", "real"}
    assert result["is_fake"] is (result["label"] == "FAKE")


def test_predict_scores_are_consistent(client):
    res = client.post("/api/predict", json={"text": LONG_FAKE})
    result = res.get_json()["result"]
    assert abs(result["scores"]["real"] + result["scores"]["fake"] - 1.0) < 0.01
    assert result["confidence"] == pytest.approx(
        max(result["scores"]["real"], result["scores"]["fake"]), abs=0.01
    )


def test_predict_rejects_empty_text(client):
    res = client.post("/api/predict", json={"text": "   "})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_predict_rejects_short_text(client):
    res = client.post("/api/predict", json={"text": "totally fake news"})
    assert res.status_code == 400
    assert "too short" in res.get_json()["error"].lower()


def test_predict_rejects_oversized_text(client):
    res = client.post("/api/predict", json={"text": "word " * 6000})
    assert res.status_code == 400
    assert "too long" in res.get_json()["error"].lower()


def test_predict_rejects_non_json_body(client):
    res = client.post("/api/predict", data="plain text", content_type="text/plain")
    assert res.status_code == 400


# ---------- /api/analyze-url ----------

def test_analyze_url_rejects_invalid_url(client):
    res = client.post("/api/analyze-url", json={"url": "not-a-url"})
    assert res.status_code == 400
    assert "invalid url" in res.get_json()["error"].lower()


def test_analyze_url_rejects_empty_url(client):
    res = client.post("/api/analyze-url", json={"url": ""})
    assert res.status_code == 400


def test_analyze_url_rejects_ftp_scheme(client):
    res = client.post("/api/analyze-url", json={"url": "ftp://example.com/a.txt"})
    assert res.status_code == 400


def test_analyze_url_unreachable_host(client):
    res = client.post(
        "/api/analyze-url",
        json={"url": "https://this-domain-should-not-exist-9f2a1c.test/story"},
    )
    assert res.status_code == 400
    assert "could not fetch" in res.get_json()["error"].lower()