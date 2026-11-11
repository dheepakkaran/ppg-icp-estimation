"""Compare the MATLAB/Octave outputs with Python on the same 20 windows."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np
from src import preprocess as pp

X = np.loadtxt("results/matlab_windows.csv", delimiter=",")
F = np.loadtxt("results/matlab_filtered.csv", delimiter=",")
py = np.stack([pp.filter_ppg(x) for x in X])
peaks_m = {}
for ln in open("results/matlab_peaks.csv"):
    i, rest = ln.split(",", 1)
    peaks_m[int(i)] = [int(v) for v in rest.split()]
from scipy.signal import find_peaks
match, tot_py, tot_m = 0, 0, 0
for i, y in enumerate(py):
    p_all, _ = find_peaks(y, distance=int(0.33 * pp.FS), prominence=0.5)
    m = np.array(peaks_m[i])
    tot_py += len(p_all); tot_m += len(m)
    match += len(np.intersect1d(p_all, m))
out = {"engine": open("results/matlab_engine.txt").read().strip(),
       "windows": int(len(X)),
       "filtered_max_abs_diff": float(np.max(np.abs(F - py))),
       # filtfilt edge padding differs between SciPy and MATLAB/Octave; interior = drop 2 s each end
       "filtered_max_abs_diff_interior": float(np.max(np.abs(F - py)[:, 250:-250])),
       "peaks_python": int(tot_py), "peaks_matlab": int(tot_m), "peaks_identical_index": int(match)}
json.dump(out, open("results/matlab_check.json", "w"), indent=2)
print(json.dumps(out, indent=1))
