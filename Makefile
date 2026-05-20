# streamsense-har Makefile
# All targets assume a venv has been activated, or rely on python from PATH.

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

CONFIG ?= configs/default.yaml
FOLD   ?= 1
SUBSET ?= 0

.PHONY: help install install-dev download prepare train train-fast \
        evaluate export quantize benchmark serve docker test lint format clean

help:
	@echo "Common targets:"
	@echo "  install        Install runtime dependencies"
	@echo "  install-dev    Install dev dependencies (pytest, ruff, black)"
	@echo "  download       Fetch PAMAP2 raw zip into data/raw/"
	@echo "  prepare        Window, normalize, and persist processed arrays"
	@echo "  train          Train one fold with the default config"
	@echo "  train-fast     Short 5-epoch sanity run"
	@echo "  evaluate       Evaluate best checkpoint, write report"
	@echo "  export         Export best checkpoint to ONNX"
	@echo "  quantize       Static int8 quantization of the ONNX model"
	@echo "  benchmark      Latency benchmark (fp32, fp16, int8)"
	@echo "  serve          Run the FastAPI inference server locally"
	@echo "  docker         Build the inference Docker image"
	@echo "  test           Run pytest -q"
	@echo "  lint           ruff + black --check"
	@echo "  format         ruff --fix + black"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

download:
	$(PYTHON) scripts/download_pamap2.py

prepare:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG)

train:
	$(PYTHON) scripts/train.py --config $(CONFIG) --fold $(FOLD)

train-fast:
	$(PYTHON) scripts/train.py --config $(CONFIG) --fold $(FOLD) --max-epochs 5 --fast

evaluate:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --fold $(FOLD)

export:
	$(PYTHON) scripts/export_onnx.py --config $(CONFIG) --fold $(FOLD)

quantize:
	$(PYTHON) scripts/quantize_onnx.py --config $(CONFIG)

benchmark:
	$(PYTHON) scripts/benchmark_latency.py --config $(CONFIG)

serve:
	uvicorn streamsense.serve.app:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker build -f docker/Dockerfile -t streamsense-har:latest .

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests scripts
	black --check src tests scripts

format:
	ruff check --fix src tests scripts
	black src tests scripts

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
