# Multi-stage build for the streamsense-har FastAPI inference server.
# Stage 1 installs build deps + Python packages into a venv; stage 2 copies
# the venv and source into a slim runtime image without compilers.

# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only what the server needs at runtime: numpy + onnxruntime +
# fastapi + uvicorn + pydantic + omegaconf. No torch/lightning in the image.
RUN pip install --no-cache-dir \
        "numpy>=1.26,<2.1" \
        "onnxruntime>=1.17" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "pydantic>=2.6" \
        "omegaconf>=2.3"

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    STREAMSENSE_CONFIG="/app/configs/default.yaml" \
    STREAMSENSE_MODEL_PATH="/app/artifacts/onnx/multimodal_late_fusion_fold0_int8.onnx"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY src /app/src
COPY configs /app/configs
COPY artifacts/onnx /app/artifacts/onnx

# Run as a non-root user.
RUN useradd --create-home --uid 10001 streamsense && \
    chown -R streamsense:streamsense /app
USER streamsense

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3); \
        sys.exit(0 if r.status==200 else 1)" || exit 1

CMD ["uvicorn", "streamsense.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
