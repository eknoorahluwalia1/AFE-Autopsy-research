"""The full simulated signal chain.

Clean ECG -> electrode/skin -> mains + common-mode -> instrumentation amp ->
analog high-pass -> anti-alias low-pass -> sampling -> ADC quantizer.
The optional digital denoiser and the classifier live in ``models/``.

Analog and DSP agree on this one interface. ML consumes whatever it outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .adc import quantize
from .amp import instrumentation_amp
from .electrode import inject_mains, mains_to_differential
from .filters import highpass, lowpass, resample


@dataclass(frozen=True)
class AFEConfig:
    """One point in the sweep. Defaults are a near-ideal chain (see tests)."""

    analog_fs: float = 1000.0  # dense grid standing in for continuous time
    gain: float = 1.0
    cmrr_db: float = 120.0
    noise_uvrms: float = 0.0
    mains_amp_v: float = 0.0
    mains_hz: float = 60.0
    z_a_ohm: float = 1e3
    z_b_ohm: float = 1e3
    hp_cutoff_hz: float | None = None
    lp_cutoff_hz: float | None = None
    lp_order: int = 2
    fs_out: float = 500.0
    adc_bits: int = 16
    full_scale_v: float = 5e-3  # input-referred, mV-scale ECG
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def run_chain(ecg_v: np.ndarray, cfg: AFEConfig) -> np.ndarray:
    """Push a clean ECG (volts, sampled at ``cfg.analog_fs``) through the AFE."""
    rng = np.random.default_rng(cfg.seed)
    n = len(ecg_v)

    cm = inject_mains(n, cfg.analog_fs, cfg.mains_amp_v, cfg.mains_hz)
    diff = ecg_v + mains_to_differential(cm, cfg.z_a_ohm, cfg.z_b_ohm)
    x = instrumentation_amp(diff, cm, cfg.gain, cfg.cmrr_db, cfg.noise_uvrms, rng)

    if cfg.hp_cutoff_hz is not None:
        x = highpass(x, cfg.analog_fs, cfg.hp_cutoff_hz)
    if cfg.lp_cutoff_hz is not None:
        x = lowpass(x, cfg.analog_fs, cfg.lp_cutoff_hz, cfg.lp_order)

    x = resample(x, cfg.analog_fs, cfg.fs_out)
    return quantize(x, cfg.adc_bits, cfg.full_scale_v * cfg.gain)
