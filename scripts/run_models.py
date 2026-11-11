"""Leave-one-patient-out models -> results/metrics.json + results/oof_predictions.csv."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from src import models as M
from src.features import FEATURES

MIN_KEPT = 10  # a patient enters LOPO only with >= 10 kept windows
MODEL_FEATURES = [f for f in FEATURES if f not in ("n_good_beats", "beat_quality")]


def mae(y, p):
    return float(mean_absolute_error(y, p))


def main():
    T0 = time.time()
    df = pd.read_csv("results/features.csv")
    keep = df.groupby("subject").subject.transform("size") >= MIN_KEPT
    df = df[keep].reset_index(drop=True)
    X = df[MODEL_FEATURES].to_numpy(float); y = df.icp_mean.to_numpy(); c = df.ich.to_numpy()
    g = df.subject.to_numpy()
    out = {"n_patients": int(df.subject.nunique()), "n_windows": int(len(df)), "min_kept_windows_per_patient": MIN_KEPT,
           "patients": sorted(df.subject.unique().tolist()),
           "pct_windows_icp_gt20": float(100 * c.mean()), "features": MODEL_FEATURES,
           "validation": "leave-one-patient-out; held-out = pooled out-of-patient predictions; "
                         "CI = 95% bootstrap over patients (1000 resamples); training = mean over folds",
           "regression": {}, "classification_icp_gt20": {}, "classification_icp_gt22": {}}
    oof = df[["subject", "seg", "start", "icp_mean", "ich"]].copy()
    for name, make in M.regressors().items():
        p, tr = M.run_lopo(X, y, g, make, "reg")
        oof[f"reg_{name}"] = p
        print(f"reg {name} {time.time()-T0:.0f}s", flush=True)
        out["regression"][name] = {"mae_heldout": mae(y, p), "mae_heldout_ci95": M.bootstrap_ci(g, y, p, mae),
                                    "mae_training": tr}
    for key, target in (("classification_icp_gt20", c), ("classification_icp_gt22", df.ich22.to_numpy())):
        for name, make in M.classifiers().items():
            p, tr = M.run_lopo(X, target, g, make, "cls")
            print(f"{key} {name} {time.time()-T0:.0f}s", flush=True)
            if key.endswith("20"):
                oof[f"cls_{name}"] = p
            out[key][name] = {"auroc_heldout": M.safe_auc(target, p),
                              "auroc_heldout_ci95": M.bootstrap_ci(g, target, p, M.safe_auc),
                              "auroc_training": tr}
    # PTT direction: pooled and within-patient Spearman rho vs mean ICP
    d = df.dropna(subset=["ptt_ms"])
    within = {s: float(spearmanr(x.ptt_ms, x.icp_mean)[0]) for s, x in d.groupby("subject")
              if len(x) >= 10 and x.icp_mean.std() > 0}
    out["ptt_direction"] = {
        "n_windows_with_ptt": int(len(d)),
        "pooled_spearman_rho": float(spearmanr(d.ptt_ms, d.icp_mean)[0]),
        "within_patient_spearman_rho": within,
        "median_within_patient_rho": float(np.median(list(within.values()))) if within else None,
        "n_patients_negative_rho": int(sum(v < 0 for v in within.values())),
    }
    # CNN on raw filtered PPG
    try:
        from src.cnn import run_cnn_lopo
        z = np.load("data/processed/cnn.npz", allow_pickle=True)
        zx = z["x"][keep.to_numpy()]
        assert (z["subject"][keep.to_numpy()] == g).all()
        reg, prob, tr_out, secs = run_cnn_lopo(zx, y.astype(np.float32), c, g)
        tr_mae = np.mean([mae(y[tr], r) for tr, r, _ in tr_out])
        tr_auc = np.nanmean([M.safe_auc(c[tr], p) for tr, _, p in tr_out])
        out["regression"]["cnn1d"] = {"mae_heldout": mae(y, reg), "mae_heldout_ci95": M.bootstrap_ci(g, y, reg, mae),
                                      "mae_training": float(tr_mae)}
        out["classification_icp_gt20"]["cnn1d"] = {"auroc_heldout": M.safe_auc(c, prob),
                                                  "auroc_heldout_ci95": M.bootstrap_ci(g, c, prob, M.safe_auc),
                                                  "auroc_training": float(tr_auc)}
        out["cnn_train_seconds"] = secs
        oof["reg_cnn1d"], oof["cls_cnn1d"] = reg, prob
    except TimeoutError as e:
        out["cnn_note"] = str(e)
    oof.to_csv("results/oof_predictions.csv", index=False)
    json.dump(out, open("results/metrics.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("n_patients", "n_windows", "pct_windows_icp_gt20")}, indent=1))
    for k in ("regression", "classification_icp_gt20"):
        for n, v in out[k].items():
            print(k, n, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    print("PTT", out["ptt_direction"]["pooled_spearman_rho"], out["ptt_direction"]["median_within_patient_rho"])


if __name__ == "__main__":
    main()
