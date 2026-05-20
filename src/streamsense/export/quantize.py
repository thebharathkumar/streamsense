"""ONNX Runtime static int8 quantization.

We feed a calibration subset of real validation windows so the activation
ranges are realistic. Calibration runs on CPU since this is a one-time
offline step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


class MultimodalCalibReader(CalibrationDataReader):
    """CalibrationDataReader over numpy arrays for the multimodal model."""

    def __init__(
        self,
        hand: np.ndarray,
        chest: np.ndarray,
        ankle: np.ndarray,
        hr: np.ndarray,
        batch_size: int = 8,
    ) -> None:
        self.hand = hand.astype(np.float32)
        self.chest = chest.astype(np.float32)
        self.ankle = ankle.astype(np.float32)
        self.hr = hr.astype(np.float32)
        self.batch_size = batch_size
        self._iter = iter(self._gen())

    def _gen(self):
        n = self.hand.shape[0]
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield {
                "hand": self.hand[start:end],
                "chest": self.chest[start:end],
                "ankle": self.ankle[start:end],
                "hr": self.hr[start:end],
            }

    def get_next(self):
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = iter(self._gen())


def quantize_int8(
    fp32_path: Path,
    int8_path: Path,
    calib_reader: CalibrationDataReader,
) -> Path:
    """Run ORT static int8 quantization with the supplied calibration data."""
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return int8_path


def convert_fp16(fp32_path: Path, fp16_path: Path) -> Path:
    """Convert a float32 ONNX model to float16 in-place via onnxconverter-common."""
    import onnx
    from onnxconverter_common import float16  # type: ignore

    fp16_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
    onnx.save(model_fp16, str(fp16_path))
    return fp16_path
