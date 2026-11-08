import numpy as np

from afe.filters import highpass, lowpass


def _gain(fn, f_hz: float, fs: float = 1000.0, **kw) -> float:
    t = np.arange(int(20 * fs)) / fs
    x = np.sin(2 * np.pi * f_hz * t)
    y = fn(x, fs, **kw)
    s = slice(len(x) // 2, None)  # skip transient
    return np.sqrt(np.mean(y[s] ** 2)) / np.sqrt(np.mean(x[s] ** 2))


def test_lowpass_minus3db_at_cutoff():
    g = _gain(lowpass, 100.0, cutoff_hz=100.0, order=2)
    assert abs(g - 1 / np.sqrt(2)) < 0.03


def test_lowpass_passes_dc_band_and_blocks_high():
    assert _gain(lowpass, 5.0, cutoff_hz=100.0, order=2) > 0.99
    assert _gain(lowpass, 400.0, cutoff_hz=100.0, order=2) < 0.1


def test_highpass_minus3db_at_cutoff():
    g = _gain(highpass, 0.5, cutoff_hz=0.5, order=1)
    assert abs(g - 1 / np.sqrt(2)) < 0.05


def test_highpass_blocks_dc():
    y = highpass(np.ones(20000), 1000.0, 0.5)
    assert abs(y[-1]) < 1e-3
