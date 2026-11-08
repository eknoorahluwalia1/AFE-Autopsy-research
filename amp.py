"""Instrumentation amplifier: gain, input-referred noise, finite CMRR.

Parameter ranges should be justified from public datasheets (AD8232, INA333,
ADS1292R). Record the justification in ``docs/datasheets.md``.
"""

from __future__ import annotations

import numpy as np


def cmrr_db_to_ratio(cmrr_db: float) -> float:
    """Common-mode rejection ratio in dB to a linear voltage ratio."""
    return 10.0 ** (cmrr_db / 20.0)


def instrumentation_amp(
    diff: np.ndarray,
    common_mode: np.ndarray,
    gain: float = 1000.0,
    cmrr_db: float = 100.0,
    noise_uvrms: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply gain, leak a fraction of the common-mode input, and add noise.

    ``out = gain * (diff + cm / CMRR + noise)``.
    White noise is a simplification; flag it as a realism limit in the paper.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    leak = common_mode / cmrr_db_to_ratio(cmrr_db)
    noise = rng.normal(0.0, noise_uvrms * 1e-6, size=diff.shape) if noise_uvrms > 0 else 0.0
    return gain * (diff + leak + noise)
