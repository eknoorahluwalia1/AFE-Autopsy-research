"""Analog high-pass, anti-alias low-pass, and sampling.

Analog filters are approximated with Butterworth designs applied on a dense
"analog" grid, then resampled to the ADC rate.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def highpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 1) -> np.ndarray:
    """High-pass filter (0.05 to 1 Hz in the sweep). Causal, as in hardware."""
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return signal.sosfilt(sos, x)


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Anti-alias low-pass (40 to 150 Hz, order 1 to 4)."""
    if cutoff_hz >= fs / 2:
        raise ValueError("cutoff must be below Nyquist")
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    return signal.sosfilt(sos, x)


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Sample the analog signal at ``fs_out`` (linear interpolation, no extra filtering)."""
    if fs_out > fs_in:
        raise ValueError("fs_out must not exceed the analog grid rate")
    n_out = int(np.floor(len(x) * fs_out / fs_in))
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, x)
