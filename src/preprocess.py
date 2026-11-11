"""Filtering, beat detection and window quality checks (125 Hz MIMIC-III).

PPG band-pass 0.5-8 Hz, ECG band-pass 5-30 Hz (4th-order Butterworth,
zero-phase filtfilt), per-window z-score, ECG R-peaks, PPG systolic peaks and
feet, beat quality by template correlation, and window rejection rules.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FS = 125
ICP_MIN, ICP_MAX = -5.0, 60.0


def bandpass(x, lo, hi, fs=FS, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    # MATLAB/Octave filtfilt pads 3*(nfilt-1) samples; SciPy's default is 3*nfilt.
    # Use the MATLAB length so both implementations give the same output.
    return filtfilt(b, a, x, padlen=3 * (max(len(a), len(b)) - 1))


def zscore(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 0 else x * 0.0


def filter_ppg(x):
    return zscore(bandpass(x, 0.5, 8.0))


def filter_ecg(x):
    return zscore(bandpass(x, 5.0, 30.0))


def r_peaks(ecg_f, fs=FS):
    """R-peaks on the filtered ECG: peaks of |ecg| with 0.3 s refractory period.
    Polarity is chosen from the larger tail so inverted leads still work."""
    sig = ecg_f if np.percentile(ecg_f, 99.5) >= -np.percentile(ecg_f, 0.5) else -ecg_f
    pk, _ = find_peaks(sig, height=np.percentile(sig, 90) * 0.5 + 0.5, distance=int(0.3 * fs))
    return pk


def ppg_beats(ppg_f, fs=FS):
    """Systolic peaks and the preceding foot (minimum) of each PPG pulse."""
    pk, _ = find_peaks(ppg_f, distance=int(0.33 * fs), prominence=0.5)
    feet = []
    for i, p in enumerate(pk):
        lo = pk[i - 1] if i > 0 else max(0, p - int(0.8 * fs))
        seg = ppg_f[lo:p]
        feet.append(lo + int(np.argmin(seg)) if len(seg) else -1)
    feet = np.array(feet, dtype=int)
    ok = feet >= 0
    return pk[ok], feet[ok]


def beat_quality(ppg_f, feet, fs=FS):
    """Pearson correlation of each beat (foot to foot, resampled to 64 points)
    with the window's median beat template. Returns array of correlations."""
    if len(feet) < 4:
        return np.array([])
    beats = []
    for a, b in zip(feet[:-1], feet[1:]):
        if not (int(0.33 * fs) <= b - a <= int(1.5 * fs)):
            beats.append(None); continue
        seg = ppg_f[a:b]
        beats.append(np.interp(np.linspace(0, len(seg) - 1, 64), np.arange(len(seg)), seg))
    good = [b for b in beats if b is not None]
    if len(good) < 3:
        return np.zeros(len(beats))
    tmpl = np.median(np.stack(good), axis=0)
    return np.array([np.corrcoef(b, tmpl)[0, 1] if b is not None else 0.0 for b in beats])


def window_checks(ppg_raw, ecg_raw, icp_raw):
    """Return (ok, reason). Rejects NaN gaps, flat lines, clipping, ICP out of range."""
    for name, x in (("ppg", ppg_raw), ("ecg", ecg_raw), ("icp", icp_raw)):
        if np.mean(np.isnan(x)) > 0.01:
            return False, f"{name}_nan"
    p = np.nan_to_num(ppg_raw)
    if np.std(p) < 1e-3:
        return False, "ppg_flat"
    if np.std(np.nan_to_num(ecg_raw)) < 1e-3:
        return False, "ecg_flat"
    # clipping: >5% of samples at the window max or min
    if max(np.mean(p >= p.max() - 1e-6), np.mean(p <= p.min() + 1e-6)) > 0.05:
        return False, "ppg_clipped"
    icp = icp_raw[~np.isnan(icp_raw)]
    m = float(np.mean(icp))
    if not (ICP_MIN <= m <= ICP_MAX) or icp.min() < -20 or icp.max() > 100:
        return False, "icp_out_of_range"
    if np.std(icp) < 0.05:
        return False, "icp_flat"  # EVD clamped / transducer off
    return True, "ok"


def fill_nan(x):
    x = np.asarray(x, dtype=float)
    bad = np.isnan(x)
    if bad.any() and (~bad).any():
        x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return x
