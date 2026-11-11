# Features (one row per 60 s window, value = median over good beats)

Signals: finger PPG (MIMIC-III `PLETH`, an optical transmission photoplethysmogram) and ECG lead II, both 125 Hz.
Code: `src/features.py`. Beats are kept only if their shape correlates ≥ 0.9 with the window's median beat (`src/preprocess.py: beat_quality`), and a window needs ≥ 10 good beats.

| Feature | How it is computed | Why it could move with ICP (physiological reason) |
|---|---|---|
| `ptt_ms` — pulse transit time to foot | ECG R-peak → next PPG foot (minimum before the systolic upstroke), 100–500 ms window; only when ECG HR and PPG HR agree within 10 bpm | PTT shortens when arterial pressure and wall stiffness rise (Moens–Korteweg: pulse wave velocity grows with stiffness). Rising ICP triggers the Cushing response (higher arterial pressure, sympathetic tone), so PTT is expected to **fall** as ICP rises. It also contains the pre-ejection period (R-peak to aortic valve opening), which varies with contractility; that is a known confounder. |
| `ptt_peak_ms` | ECG R-peak → PPG systolic peak | Same pathway as `ptt_ms`, but includes the rise time, so it also reflects how fast the peripheral bed fills. |
| `pulse_amp` | systolic peak − foot on the band-passed, z-scored PPG | After per-window normalization this is a pulse shape measure (peak height relative to the window's spread), not absolute blood volume. Raw finger PPG amplitude depends on probe gain and perfusion; sympathetic vasoconstriction during raised ICP lowers peripheral pulsatility. |
| `rise_time_ms` | foot → systolic peak | Longer rise = more compliant vessels or slower ejection; stiffer, more constricted vessels steepen the upstroke. Intracranial compliance itself is not seen at the finger; this tracks systemic vascular tone, which ICP changes can drive. |
| `width50_ms` | pulse width at 50 % of amplitude | Wider pulses come with more wave reflection and lower vascular tone; narrow pulses with constriction. |
| `hr_bpm` | 60 000 / median foot-to-foot interval | The Cushing reflex includes heart rate change (bradycardia late); heart rate is also a confounder for PTT and pulse width, so it is kept as a feature. |
| `ibi_sd_ms` | SD of foot-to-foot intervals | Beat-to-beat variability reflects autonomic balance, which is disturbed in brain injury and with raised ICP. |
| `harm2_ratio`, `harm3_ratio` | FFT power in ±0.1 Hz bands around 2× and 3× the heart rate, over power at the fundamental | Harmonic content summarizes pulse shape (sharpness, dicrotic notch, reflected wave) without needing to detect the notch; cerebral blood volume and vascular tone changes alter the reflected wave. |

Quality columns kept in `results/features.csv` but **not** used as model inputs: `n_good_beats`, `beat_quality` (median template correlation).

## Honest limits
- Finger PPG sees systemic circulation, not the brain. Any link to ICP must go through systemic responses (arterial pressure, sympathetic tone), which also change for many non-ICP reasons. Forehead near-infrared (NIR) PPG, used by several non-invasive ICP devices, is not in this data.
- PTT direction is checked in `results/metrics.json` → `ptt_direction` (pooled and within-patient Spearman rho with mean ICP).
