"""Leave-one-patient-out evaluation: baseline, ridge/logistic, gradient boosting."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


def lopo_splits(groups):
    """Yield (held_out_patient, train_idx, test_idx); a patient is never in both."""
    groups = np.asarray(groups)
    for pid in np.unique(groups):
        te = np.flatnonzero(groups == pid)
        tr = np.flatnonzero(groups != pid)
        assert not set(groups[tr]) & set(groups[te])
        yield pid, tr, te


class MeanBaseline:
    def fit(self, X, y):
        self.m = float(np.mean(y)); return self
    def predict(self, X):
        return np.full(len(X), self.m)
    def predict_proba(self, X):
        return np.column_stack([1 - np.full(len(X), self.m), np.full(len(X), self.m)])


def regressors():
    return {
        "baseline_mean": lambda: MeanBaseline(),
        "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "gboost": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, random_state=SEED),
    }


def classifiers():
    return {
        "baseline_mean": lambda: MeanBaseline(),
        "logistic": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        "gboost": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                                         class_weight="balanced", random_state=SEED),
    }


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def bootstrap_ci(groups, y, pred, metric, n=1000, seed=SEED):
    """95% CI by resampling patients (not windows) with replacement."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups); pids = np.unique(groups)
    idx = {p: np.flatnonzero(groups == p) for p in pids}
    vals = []
    for _ in range(n):
        s = np.concatenate([idx[p] for p in rng.choice(pids, len(pids))])
        v = metric(y[s], pred[s])
        if not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def run_lopo(X, y, groups, make, kind):
    """Pooled held-out predictions + mean training-fold score."""
    oof = np.full(len(y), np.nan); train_scores = []
    for pid, tr, te in lopo_splits(groups):
        m = make().fit(X[tr], y[tr])
        if kind == "reg":
            oof[te] = m.predict(X[te])
            train_scores.append(mean_absolute_error(y[tr], m.predict(X[tr])))
        else:
            if len(np.unique(y[tr])) < 2:
                oof[te] = np.mean(y[tr]); continue
            oof[te] = m.predict_proba(X[te])[:, 1]
            train_scores.append(safe_auc(y[tr], m.predict_proba(X[tr])[:, 1]))
    return oof, float(np.nanmean(train_scores)) if train_scores else float("nan")
