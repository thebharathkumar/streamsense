"""FastAPI endpoint tests using a freshly-exported ONNX model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamsense.export.onnx_export import export_to_onnx  # noqa: E402
from streamsense.models.fusion import MultimodalLateFusion  # noqa: E402

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

T = 512


@pytest.fixture(scope="module")
def served_app(tmp_path_factory):
    """Export a tiny ONNX model and point the FastAPI app at it."""
    tmp = tmp_path_factory.mktemp("serve")
    torch.manual_seed(0)
    model = MultimodalLateFusion(num_classes=12)
    model.eval()
    onnx_path = tmp / "test_model.onnx"
    export_to_onnx(model, onnx_path, window_samples=T, opset=17, dynamic_batch=True)

    os.environ["STREAMSENSE_MODEL_PATH"] = str(onnx_path)
    os.environ["STREAMSENSE_CONFIG"] = str(REPO_ROOT / "configs" / "default.yaml")

    # Import after env is set so the InferenceState picks them up.
    import importlib

    import streamsense.serve.app as app_module
    importlib.reload(app_module)
    app_module.state = app_module.InferenceState()
    app_module.state.load()
    return app_module.app


@pytest.fixture
def synthetic_window():
    rng = np.random.default_rng(0)
    return {
        "sampling_rate_hz": 100,
        "hand": {"channels": rng.standard_normal((9, T)).astype(float).tolist()},
        "chest": {"channels": rng.standard_normal((9, T)).astype(float).tolist()},
        "ankle": {"channels": rng.standard_normal((9, T)).astype(float).tolist()},
        "hr": {"values": rng.standard_normal(T).astype(float).tolist()},
    }


def test_health(served_app):
    client = TestClient(served_app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["expected_window_samples"] == 512
    assert body["expected_sampling_rate_hz"] == 100


def test_predict_happy_path(served_app, synthetic_window):
    client = TestClient(served_app)
    r = client.post("/predict", json=synthetic_window)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "predicted_label" in body
    assert isinstance(body["predicted_index"], int)
    assert 0 <= body["predicted_index"] < 12
    assert len(body["top3"]) == 3
    probs = [t["probability"] for t in body["top3"]]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs, reverse=True)
    assert body["inference_ms"] > 0.0


def test_predict_rejects_wrong_window_length(served_app, synthetic_window):
    bad = dict(synthetic_window)
    bad["hand"] = {"channels": [row[:100] for row in synthetic_window["hand"]["channels"]]}
    bad["chest"] = {"channels": [row[:100] for row in synthetic_window["chest"]["channels"]]}
    bad["ankle"] = {"channels": [row[:100] for row in synthetic_window["ankle"]["channels"]]}
    bad["hr"] = {"values": synthetic_window["hr"]["values"][:100]}
    client = TestClient(served_app)
    r = client.post("/predict", json=bad)
    assert r.status_code == 400


def test_predict_rejects_wrong_sampling_rate(served_app, synthetic_window):
    bad = dict(synthetic_window)
    bad["sampling_rate_hz"] = 50
    client = TestClient(served_app)
    r = client.post("/predict", json=bad)
    assert r.status_code == 400


def test_predict_rejects_wrong_channel_count(served_app, synthetic_window):
    bad = dict(synthetic_window)
    bad["hand"] = {"channels": synthetic_window["hand"]["channels"][:6]}
    client = TestClient(served_app)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # Pydantic validation error
