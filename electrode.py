"""Electrode / skin model and mains (50/60 Hz) injection.

Mains couples to the body as a common-mode voltage. If the two electrode
impedances differ, part of it becomes a *differential* voltage that the
instrumentation amp cannot reject. That is why mismatch and CMRR are
modeled together (hypothesis H1).
"""

from __future__ import annotations

import numpy as np


def inject_mains(
    n: int,
    fs: float,
    amplitude_v: float,
    freq_hz: float = 60.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Return a common-mode mains tone of ``n`` samples at sample rate ``fs``."""
    if amplitude_v == 0:
        return np.zeros(n)
    t = np.arange(n) / fs
    return amplitude_v * np.sin(2 * np.pi * freq_hz * t + phase)


def mains_to_differential(
    cm_voltage: np.ndarray,
    z_a_ohm: float,
    z_b_ohm: float,
    z_input_ohm: float = 1e9,
) -> np.ndarray:
    """Convert a common-mode voltage to a differential one via impedance mismatch.

    Each electrode forms a divider with the amp input impedance. The
    differential term is ``cm * (Zb - Za) / (Zin + Z)`` to first order, which
    is zero when the electrodes match and grows with the mismatch.
    """
    if z_a_ohm < 0 or z_b_ohm < 0 or z_input_ohm <= 0:
        raise ValueError("impedances must be non-negative and Zin positive")
    mismatch = abs(z_b_ohm - z_a_ohm)
    return cm_voltage * mismatch / (z_input_ohm + max(z_a_ohm, z_b_ohm))
