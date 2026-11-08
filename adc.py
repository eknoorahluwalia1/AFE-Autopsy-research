"""ADC quantizer. Ideal uniform mid-tread quantizer with clipping."""

from __future__ import annotations

import numpy as np


def quantize(x: np.ndarray, bits: int, full_scale: float) -> np.ndarray:
    """Quantize ``x`` to ``bits`` over [-full_scale, +full_scale].

    Values outside the range clip. Returns floats on the quantizer grid so
    downstream code does not need to know the code width.
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")
    if full_scale <= 0:
        raise ValueError("full_scale must be positive")
    levels = 2**bits
    lsb = 2 * full_scale / levels
    clipped = np.clip(x, -full_scale, full_scale - lsb)
    return np.round(clipped / lsb) * lsb


def sqnr_theory_db(bits: int) -> float:
    """Ideal SQNR for a full-scale sine: 6.02 N + 1.76 dB."""
    return 6.02 * bits + 1.76
