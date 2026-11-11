import numpy as np
from src import preprocess as pp
from src import features as ft
from src.models import lopo_splits

FS = 125


def synthetic(hr=75, ptt_s=0.25, seconds=60, seed=0):
    """Synthetic ECG spikes + PPG pulses delayed by a known transit time."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * FS) / FS
    period = 60 / hr
    beats = np.arange(0.5, seconds - 1, period)
    ecg = np.zeros_like(t)
    ppg = np.zeros_like(t)
    for b in beats:
        ecg += np.exp(-((t - b) ** 2) / (2 * 0.01 ** 2))
        on = b + ptt_s
        x = t - on
        ppg += np.where(x > 0, (x / 0.15) * np.exp(1 - x / 0.15), 0)  # foot at onset, peak 150 ms later
    return ppg + 0.01 * rng.standard_normal(len(t)), ecg + 0.01 * rng.standard_normal(len(t)), beats


def test_filters_keep_length_and_remove_drift():
    ppg, _, _ = synthetic()
    drift = ppg + 5 * np.linspace(0, 1, len(ppg))
    f = pp.filter_ppg(drift)
    assert len(f) == len(ppg)
    assert abs(np.mean(f[:FS * 5]) - np.mean(f[-FS * 5:])) < 0.5


def test_beat_detection_and_ptt():
    ppg, ecg, beats = synthetic(hr=75, ptt_s=0.25)
    pk, feet = pp.ppg_beats(pp.filter_ppg(ppg))
    assert abs(len(pk) - len(beats)) <= 2
    r = pp.r_peaks(pp.filter_ecg(ecg))
    assert abs(len(r) - len(beats)) <= 2
    f, _ = ft.window_features(ppg, ecg)
    assert abs(f["hr_bpm"] - 75) < 2
    # band-pass shifts the detected foot slightly; PTT must still be near 250 ms
    assert 200 < f["ptt_ms"] < 300
    f2, _ = ft.window_features(*synthetic(hr=75, ptt_s=0.30)[:2])
    assert f2["ptt_ms"] > f["ptt_ms"] + 30  # longer delay -> longer PTT


def test_flat_window_rejected():
    ok, why = pp.window_checks(np.ones(7500), np.random.randn(7500), np.full(7500, 10.0))
    assert not ok and why == "ppg_flat"
    ok, why = pp.window_checks(np.random.randn(7500), np.random.randn(7500), np.full(7500, 80.0) + np.random.randn(7500))
    assert not ok and why == "icp_out_of_range"


def test_lopo_no_patient_in_train_and_test():
    groups = np.array(["a"] * 5 + ["b"] * 3 + ["c"] * 4)
    n = 0
    for pid, tr, te in lopo_splits(groups):
        assert set(groups[te]) == {pid}
        assert pid not in set(groups[tr])
        assert len(tr) + len(te) == len(groups)
        n += 1
    assert n == 3
