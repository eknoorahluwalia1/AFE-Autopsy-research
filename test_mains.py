import numpy as np

from afe.electrode import inject_mains, mains_to_differential


def test_mains_tone_at_exact_frequency():
    fs, n = 1000.0, 10000
    for f in (50.0, 60.0):
        x = inject_mains(n, fs, 1e-3, f)
        spec = np.abs(np.fft.rfft(x))
        peak_hz = np.argmax(spec) * fs / n
        assert abs(peak_hz - f) < 0.2


def test_matched_electrodes_reject_mains():
    cm = inject_mains(1000, 1000.0, 1e-3, 60.0)
    assert np.max(np.abs(mains_to_differential(cm, 1e3, 1e3))) == 0.0


def test_mismatch_increases_differential_pickup():
    cm = inject_mains(1000, 1000.0, 1e-3, 60.0)
    lo = np.max(np.abs(mains_to_differential(cm, 1e3, 5e3)))
    hi = np.max(np.abs(mains_to_differential(cm, 1e3, 1e5)))
    assert hi > lo > 0
