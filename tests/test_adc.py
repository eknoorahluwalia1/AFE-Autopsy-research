import numpy as np

from afe.adc import quantize, sqnr_theory_db


def _sqnr_db(bits: int) -> tuple[float, float]:
    """Measured SQNR and the theory value corrected for the sine's amplitude."""
    n = 2**16
    amp = 1.0 - 2.0 / 2**bits  # largest sine that never clips on this quantizer
    t = np.arange(n)
    x = amp * np.sin(2 * np.pi * 0.0123456789 * t)  # non-harmonic: no tonal artifacts
    err = quantize(x, bits, 1.0) - x
    measured = 10 * np.log10(np.mean(x**2) / np.mean(err**2))
    return measured, sqnr_theory_db(bits) + 20 * np.log10(amp)


def test_sqnr_matches_6p02n_plus_1p76():
    for bits in (6, 8, 10, 12):
        measured, theory = _sqnr_db(bits)
        assert abs(measured - theory) < 1.0


def test_clipping():
    q = quantize(np.array([-10.0, 10.0]), 8, 1.0)
    assert q[0] == -1.0
    assert q[1] < 1.0


def test_high_resolution_is_near_identity():
    x = np.sin(np.linspace(0, 6, 1000)) * 0.5
    assert np.max(np.abs(quantize(x, 16, 1.0) - x)) < 1e-4
