import numpy as np

from afe.chain import AFEConfig, run_chain


def _toy_ecg(fs: float = 1000.0, seconds: float = 10.0) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    # Slow wave plus a sharper component, ~1 mV peak. A stand-in, not real ECG.
    return 1e-3 * (0.6 * np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.sin(2 * np.pi * 12 * t))


def test_ideal_chain_reproduces_clean_signal():
    """Zero noise + 16 bits + high sample rate must reproduce the clean input."""
    ecg = _toy_ecg()
    out = run_chain(ecg, AFEConfig(fs_out=500.0, adc_bits=16, full_scale_v=5e-3))
    ref = ecg[:: int(1000 / 500)][: len(out)]
    assert np.max(np.abs(out - ref)) < 5e-4 * 1e-3 * 10  # well under 1% of peak


def test_fewer_bits_means_more_error():
    ecg = _toy_ecg()
    ref = lambda o: ecg[::2][: len(o)]  # noqa: E731
    errs = []
    for bits in (6, 10, 16):
        out = run_chain(ecg, AFEConfig(adc_bits=bits))
        errs.append(np.sqrt(np.mean((out - ref(out)) ** 2)))
    assert errs[0] > errs[1] > errs[2]


def test_chain_is_deterministic_per_seed():
    ecg = _toy_ecg()
    cfg = AFEConfig(noise_uvrms=10.0, seed=3)
    assert np.array_equal(run_chain(ecg, cfg), run_chain(ecg, cfg))


def test_low_cmrr_with_mismatch_hurts_more_than_high_cmrr():
    ecg = _toy_ecg()
    kw = dict(mains_amp_v=1e-2, z_b_ohm=1e5, fs_out=500.0)
    ref = ecg[::2]
    def err(db):
        o = run_chain(ecg, AFEConfig(cmrr_db=db, **kw))
        return np.sqrt(np.mean((o - ref[: len(o)]) ** 2))
    assert err(60.0) > err(120.0)
