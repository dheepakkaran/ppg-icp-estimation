"""Demo figures for README and docs/index.html, drawn only from committed results/ files.

Run: uv run --with matplotlib python scripts/make_demo_figures.py
"""
import sys, pathlib, numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[1]; out = R/"docs/assets"; out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R))
from src import preprocess as pp
from scipy.signal import find_peaks
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
  "axes.edgecolor":"#737373","axes.labelcolor":"#3d3d3d","xtick.color":"#737373","ytick.color":"#737373"})
X = np.loadtxt(R/"results/matlab_windows.csv", delimiter=",")
F = np.loadtxt(R/"results/matlab_online/matlab_filtered.csv", delimiter=",")
i = 0; x = X[i]; y = pp.filter_ppg(x); fs = pp.FS
pk, _ = find_peaks(y, distance=int(0.33*fs), prominence=0.5)
pm = [int(v) for v in open(R/"results/matlab_online/matlab_peaks.csv").readlines()[i].split(",",1)[1].split()]
n = 10*fs; t = np.arange(len(x))/fs
fig, ax = plt.subplots(2,1, figsize=(9,5.2), sharex=True, dpi=150)
ax[0].plot(t[:n], x[:n], color="#0a0a0a", lw=1); ax[0].set_ylabel("raw PLETH")
ax[0].set_title("Window 0 of 20: raw finger PPG (125 Hz), first 10 s", loc="left", fontsize=10, color="#0a0a0a")
ax[1].plot(t[:n], y[:n], color="#0a0a0a", lw=1, label="Python: 0.5–8 Hz Butterworth, zero-phase, z-scored")
ax[1].plot(t[:n], F[i][:n], color="#a3a3a3", lw=3, alpha=.6, zorder=0, label="MATLAB R2026b output (same window)")
k = pk[pk<n]; ax[1].plot(t[k], y[k], "o", ms=5, mfc="white", mec="#0a0a0a", label=f"systolic peaks (full 60 s window: Python {len(pk)}, MATLAB {len(pm)}, {len(np.intersect1d(pk,pm))} on the same sample)")
ax[1].set_ylabel("filtered (z)"); ax[1].set_xlabel("seconds")
ax[1].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5,-0.32), ncol=2)
diff = np.max(np.abs(F[i]-y)); print("win0 diff", diff, len(pk), len(pm), len(np.intersect1d(pk, pm)))
fig.tight_layout(); fig.savefig(out/"demo-ppg-filter-peaks.png", facecolor="white"); plt.close(fig)

o = pd.read_csv(R/"results/oof_predictions.csv")
fig, ax = plt.subplots(1,2, figsize=(9,3.9), dpi=150)
hi = o.ich==1
ax[0].scatter(o.icp_mean[~hi], o.cls_gboost[~hi], s=9, c="#a3a3a3", lw=0, label=f"ICP ≤ 20 ({(~hi).sum()} windows)")
ax[0].scatter(o.icp_mean[hi], o.cls_gboost[hi], s=14, c="#0a0a0a", lw=0, label=f"ICP > 20 ({hi.sum()} windows)")
ax[0].axvline(20, color="#737373", lw=.8, ls="--")
ax[0].set_xlabel("measured mean ICP per window (mmHg)"); ax[0].set_ylabel("held-out P(ICP > 20)")
ax[0].set_title("Gradient boosting, patient left out", loc="left", fontsize=10)
ax[0].legend(frameon=False, fontsize=8.5, loc="upper right")
ax[1].scatter(o.icp_mean, o.reg_gboost, s=9, c="#0a0a0a", lw=0, alpha=.7, label="gradient boosting")
ax[1].scatter(o.icp_mean, o.reg_baseline_mean, s=9, c="#a3a3a3", lw=0, label="predict-the-mean")
lim = [-5, 40]; ax[1].plot(lim, lim, color="#737373", lw=.8, ls="--"); ax[1].set_xlim(lim)
ax[1].set_xlabel("measured mean ICP (mmHg)"); ax[1].set_ylabel("held-out predicted ICP (mmHg)")
ax[1].set_title("Mean ICP: held-out MAE 7.14 vs 5.04", loc="left", fontsize=10)
ax[1].legend(frameon=False, fontsize=8.5, loc="upper left")
from sklearn.metrics import roc_auc_score
fig.tight_layout(); fig.savefig(out/"demo-heldout-predictions.png", facecolor="white"); plt.close(fig)
print("mae", (o.reg_gboost-o.icp_mean).abs().mean(), (o.reg_baseline_mean-o.icp_mean).abs().mean(), "auc", roc_auc_score(o.ich, o.cls_gboost))
