"""FastAPI inference server for streamsense-har.

Loads an ONNX model at startup (int8 by default) and serves /predict and
/health. Single-threaded ORT session for predictable latency.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf

from .schemas import HealthResponse, PredictRequest, PredictResponse, TopK


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class InferenceState:
    """Holds the loaded ORT session and label mapping."""

    def __init__(self) -> None:
        self.cfg_path = os.environ.get("STREAMSENSE_CONFIG", "configs/default.yaml")
        self.cfg = OmegaConf.load(self.cfg_path)

        env_path = os.environ.get("STREAMSENSE_MODEL_PATH")
        self.model_path = Path(env_path or self.cfg.serve.model_path)
        self.expected_samples = int(self.cfg.serve.expect_window_samples)
        self.expected_rate = int(self.cfg.serve.expect_sampling_rate_hz)
        self.model_name = "uninitialized"
        self.session: ort.InferenceSession | None = None
        self.class_names: list[str] = []
        self.input_names: list[str] = []

    def load(self) -> None:
        if not self.model_path.exists():
            # Defer raising; /health can still report status.
            return
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.model_name = self.model_path.name
        activity_names = OmegaConf.to_container(self.cfg.data.activity_names, resolve=True)
        self.class_names = [activity_names[a] for a in self.cfg.data.activity_ids]


state = InferenceState()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(title="streamsense-har", version="0.1.0", lifespan=_lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.session is not None else "model_not_loaded",
        model_loaded=state.session is not None,
        model_path=str(state.model_path),
        expected_window_samples=state.expected_samples,
        expected_sampling_rate_hz=state.expected_rate,
    )


def _to_feeds(req: PredictRequest) -> dict[str, np.ndarray]:
    T = state.expected_samples
    hand = np.asarray(req.hand.channels, dtype=np.float32).reshape(1, 9, T)
    chest = np.asarray(req.chest.channels, dtype=np.float32).reshape(1, 9, T)
    ankle = np.asarray(req.ankle.channels, dtype=np.float32).reshape(1, 9, T)
    hr = np.asarray(req.hr.values, dtype=np.float32).reshape(1, 1, T)
    feeds = {"hand": hand, "chest": chest, "ankle": ankle, "hr": hr}
    # Filter to whatever the model actually expects (e.g. baseline only takes chest).
    return {k: v for k, v in feeds.items() if k in state.input_names}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if state.session is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded at {state.model_path}")

    try:
        req.validate_consistency(state.expected_samples, state.expected_rate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    feeds = _to_feeds(req)

    t0 = time.perf_counter()
    outputs = state.session.run(None, feeds)
    t1 = time.perf_counter()

    logits = outputs[0]
    probs = _softmax(logits)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = state.class_names[pred_idx]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [TopK(label=state.class_names[i], probability=float(probs[i])) for i in top3_idx]

    return PredictResponse(
        predicted_label=pred_label,
        predicted_index=pred_idx,
        top3=top3,
        inference_ms=(t1 - t0) * 1000.0,
        model_name=state.model_name,
    )
