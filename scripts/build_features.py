"""data/raw/<subject>.npz -> results/features.csv (one row per kept window),
results/quality.csv (every window with its keep/reject reason),
data/processed/cnn.npz (filtered PPG of kept windows, 25 Hz, for the CNN),
results/matlab_windows.csv (20 raw PPG windows for the MATLAB/Octave check)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pathlib import Path
import numpy as np
import pandas as pd
from src import preprocess as pp
from src import features as ft

RAW, PROC = Path("data/raw"), Path("data/processed")


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    rows, qrows, cnn_x, cnn_meta, mwin = [], [], [], [], []
    for f in sorted(RAW.glob("p*.npz")):
        subj = f.stem
        with np.load(f) as z:  # decompress each array once per subject
            d = {k: z[k] for k in ("ppg", "ecg", "icp", "seg", "start")}
        for i in range(len(d["start"])):
            ppg, ecg, icp = d["ppg"][i], d["ecg"][i], d["icp"][i]
            ok, why = pp.window_checks(ppg, ecg, icp)
            feats = None
            if ok:
                feats, ppg_f = ft.window_features(ppg, ecg)
                if feats["n_good_beats"] < 10:
                    ok, why = False, "few_good_beats"
            qrows.append({"subject": subj, "seg": d["seg"][i], "start": int(d["start"][i]),
                          "kept": ok, "reason": why})
            if not ok:
                continue
            icp_m = float(np.nanmean(icp))
            rows.append({"subject": subj, "seg": d["seg"][i], "start": int(d["start"][i]),
                         "icp_mean": icp_m, "ich": int(icp_m > 20), "ich22": int(icp_m > 22), **feats})
            cnn_x.append(ppg_f.reshape(-1, 5).mean(1).astype(np.float32))  # 125 -> 25 Hz
            if len(mwin) < 20 and len(rows) % 7 == 1:
                mwin.append(pp.fill_nan(ppg))
    feats = pd.DataFrame(rows)
    feats.to_csv("results/features.csv", index=False)
    pd.DataFrame(qrows).to_csv("results/quality.csv", index=False)
    np.savez_compressed(PROC / "cnn.npz", x=np.stack(cnn_x), subject=feats.subject.values,
                        y=feats.icp_mean.values, ich=feats.ich.values)
    np.savetxt("results/matlab_windows.csv", np.stack(mwin), delimiter=",", fmt="%.6f")
    print(f"kept {len(feats)} of {len(qrows)} windows from {feats.subject.nunique()} subjects")


if __name__ == "__main__":
    main()
