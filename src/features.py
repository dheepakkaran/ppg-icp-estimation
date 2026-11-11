"""Beat-level features per 60 s window; window value = median over good beats.

See FEATURES.md for the physiological reason behind each feature.
"""
import numpy as np
from . import preprocess as pp

FS = pp.FS
FEATURES = ["ptt_ms", "ptt_peak_ms", "pulse_amp", "rise_time_ms", "width50_ms",
            "hr_bpm", "ibi_sd_ms", "harm2_ratio", "harm3_ratio", "n_good_beats", "beat_quality"]
SQI_MIN = 0.9  # beat template correlation threshold


def ptt(r_pk, feet, ppg_pk, fs=FS):
    """For each R-peak, the first PPG foot (and peak) 100-500 ms later."""
    out_f, out_p = [], []
    for r in r_pk:
        f = feet[(feet > r + 0.1 * fs) & (feet < r + 0.5 * fs)]
        if len(f):
            out_f.append((f[0] - r) / fs * 1000)
            p = ppg_pk[(ppg_pk > f[0]) & (ppg_pk < f[0] + 0.5 * fs)]
            if len(p):
                out_p.append((p[0] - r) / fs * 1000)
    return np.array(out_f), np.array(out_p)


def harmonic_ratios(ppg_f, hr_hz, fs=FS):
    """Spectral power at the 2nd and 3rd harmonic of the heart rate over power at
    the fundamental (sum of |FFT| in +-0.1 Hz bands)."""
    spec = np.abs(np.fft.rfft(ppg_f * np.hanning(len(ppg_f))))
    fr = np.fft.rfftfreq(len(ppg_f), 1 / fs)
    band = lambda f0: spec[(fr > f0 - 0.1) & (fr < f0 + 0.1)].sum()
    h1 = band(hr_hz)
    if h1 <= 0:
        return np.nan, np.nan
    return band(2 * hr_hz) / h1, band(3 * hr_hz) / h1


def window_features(ppg_raw, ecg_raw, fs=FS):
    """Returns dict of features (NaN when not computable) and the filtered PPG."""
    ppg_f = pp.filter_ppg(pp.fill_nan(ppg_raw))
    ecg_f = pp.filter_ecg(pp.fill_nan(ecg_raw))
    pk, feet = pp.ppg_beats(ppg_f, fs)
    q = pp.beat_quality(ppg_f, feet, fs)
    f = dict.fromkeys(FEATURES, np.nan)
    f["n_good_beats"] = 0
    if len(q) == 0:
        return f, ppg_f
    good = np.flatnonzero(q >= SQI_MIN)  # beat i spans feet[i] -> feet[i+1]
    f["n_good_beats"] = len(good)
    f["beat_quality"] = float(np.median(q))
    if len(good) < 10:
        return f, ppg_f
    amps, rises, widths = [], [], []
    for i in good:
        a, b = feet[i], feet[i + 1]
        p = pk[(pk > a) & (pk < b)]
        if not len(p):
            continue
        p = p[0]
        amp = ppg_f[p] - ppg_f[a]
        amps.append(amp)
        rises.append((p - a) / fs * 1000)
        half = ppg_f[a] + amp / 2
        above = np.flatnonzero(ppg_f[a:b] >= half)
        widths.append((above[-1] - above[0]) / fs * 1000 if len(above) else np.nan)
    gfeet = feet[good]
    ibi = np.diff(feet)[good[good < len(feet) - 1]] / fs * 1000
    f["pulse_amp"] = float(np.median(amps)) if amps else np.nan
    f["rise_time_ms"] = float(np.median(rises)) if rises else np.nan
    f["width50_ms"] = float(np.nanmedian(widths)) if widths else np.nan
    f["hr_bpm"] = float(60000 / np.median(ibi))
    f["ibi_sd_ms"] = float(np.std(ibi))
    f["harm2_ratio"], f["harm3_ratio"] = harmonic_ratios(ppg_f, f["hr_bpm"] / 60, fs)
    r = pp.r_peaks(ecg_f, fs)
    pf, ppk = ptt(r, gfeet, pk, fs)
    # ECG-derived HR must agree with PPG HR, otherwise the R-peaks are unreliable
    if len(r) > 10:
        hr_ecg = 60000 / np.median(np.diff(r) / fs * 1000)
        if abs(hr_ecg - f["hr_bpm"]) < 10 and len(pf) >= 10:
            f["ptt_ms"] = float(np.median(pf))
            f["ptt_peak_ms"] = float(np.median(ppk)) if len(ppk) >= 10 else np.nan
    return f, ppg_f
