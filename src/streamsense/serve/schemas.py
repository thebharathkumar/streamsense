"""Pydantic schemas for the inference API.

Inputs:
  - sampling_rate_hz: must match the model's expected rate.
  - window_samples: must match the model's expected window length.
  - hand/chest/ankle: 9 channels x window_samples float arrays.
  - hr: window_samples floats (single channel).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class IMUWindow(BaseModel):
    """One IMU window as channels-first (9, T)."""

    channels: list[list[float]] = Field(
        ..., description="Channel-major 9 x T floats (accel xyz, gyro xyz, mag xyz)."
    )

    @field_validator("channels")
    @classmethod
    def _check_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 9:
            raise ValueError(f"IMU window must have 9 channels, got {len(v)}")
        lengths = {len(row) for row in v}
        if len(lengths) != 1:
            raise ValueError(f"All channels must share length, got {lengths}")
        return v


class HRWindow(BaseModel):
    """Heart rate window as a 1D list of length T."""

    values: list[float] = Field(..., description="Heart rate samples (length T).")


class PredictRequest(BaseModel):
    """Inference request schema."""

    sampling_rate_hz: int = Field(..., description="Must equal server.expect_sampling_rate_hz.")
    hand: IMUWindow
    chest: IMUWindow
    ankle: IMUWindow
    hr: HRWindow

    def window_samples(self) -> int:
        return len(self.hand.channels[0])

    def validate_consistency(self, expected_samples: int, expected_rate: int) -> None:
        if self.sampling_rate_hz != expected_rate:
            raise ValueError(
                f"sampling_rate_hz mismatch: got {self.sampling_rate_hz}, "
                f"expected {expected_rate}"
            )
        T = self.window_samples()
        if T != expected_samples:
            raise ValueError(
                f"window length mismatch: got {T}, expected {expected_samples}"
            )
        for name, imu in [("hand", self.hand), ("chest", self.chest), ("ankle", self.ankle)]:
            for c, row in enumerate(imu.channels):
                if len(row) != T:
                    raise ValueError(f"{name} channel {c} has length {len(row)} != {T}")
        if len(self.hr.values) != T:
            raise ValueError(f"hr length {len(self.hr.values)} != {T}")


class TopK(BaseModel):
    """One entry in a top-k prediction list."""

    label: str
    probability: Annotated[float, Field(ge=0.0, le=1.0)]


class PredictResponse(BaseModel):
    """Inference response schema."""

    predicted_label: str
    predicted_index: int
    top3: list[TopK]
    inference_ms: float
    model_name: str


class HealthResponse(BaseModel):
    """Liveness / readiness response."""

    status: str
    model_loaded: bool
    model_path: str
    expected_window_samples: int
    expected_sampling_rate_hz: int
