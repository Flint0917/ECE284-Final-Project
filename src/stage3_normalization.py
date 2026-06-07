"""
Stage 3: Normalization Comparison

Compares five normalization strategies using LinearSVC and the six Stage 2
handcrafted features, under TWO patient-aware CV schemes plus a pooled estimate:

  - LOSO (13 folds): kept for accuracy/F1 and the pairwise-ranking story. Its
    per-fold AUC is degenerate (each fold has exactly 1 lesional + 1 non-lesional
    test trace, so fold AUC is binary 0/1) -> reported as "pairwise ranking", not AUC.
  - GroupKFold (n=5): 2-3 patients per fold (4-6 test traces) so per-fold AUC is
    non-degenerate. This is the headline discriminability metric.
  - Pooled-LOSO AUC: one AUC over all 26 LOSO out-of-fold scores. Uses every
    sample but carries a mild negative bias (Airola et al. 2011), so it reads
    slightly below GroupKFold; the two together bracket the true discriminability.

Why this matters for the project: the motivation is generalization under data
scarcity (N=13). The degenerate LOSO AUC is a direct symptom of that scarcity;
GroupKFold + pooled-LOSO recover an honest discriminability estimate without
pretending the 2-sample folds give a real ROC.

Methods:
  - trace z-score: normalize each 30-point trace independently before feature
    extraction. This forces feature mean=0 and std=1 for every trace.
  - trace min-max: rescale each 30-point trace independently to [0, 1] before
    feature extraction. This forces range=1 for every non-constant trace.
  - feature z-score: extract raw features first, then fit StandardScaler on
    training-fold features only and transform train/test features.
  - robust scaler: same fold-safe feature-level approach using RobustScaler.
  - training-only patient baseline normalization: learn one scalar baseline per
    training patient inside each fold. Test patients use their learned training
    baseline when available, otherwise the training global mean fallback.

Outputs:
  results/stage3_normalization.png
  results/stage3_normalization_results.csv
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_patient_map(txt_path):
    mapping = {}
    with open(txt_path) as f:
        for line in f:
            if "//" not in line:
                continue
            left, right = line.split("//", 1)
            try:
                mapping[right.strip()] = int(left.strip())
            except ValueError:
                continue
    return mapping


patient_map = load_patient_map(DATA / "PatientNumbering.txt")

les = pd.read_csv(DATA / "lesional_sensor.csv", index_col=0).ffill().bfill()
non = pd.read_csv(DATA / "nonlesional_sensor.csv", index_col=0).ffill().bfill()

records = []
for sid in les.columns:
    records.append({"patient": sid, "label": 1, "trace": les[sid].values.astype(float)})
for sid in non.columns:
    records.append({"patient": sid, "label": 0, "trace": non[sid].values.astype(float)})

df = pd.DataFrame(records)
df["paper_num"] = df["patient"].map(patient_map)
df = df.sort_values(["paper_num", "label"], ignore_index=True)

t30 = np.arange(30)

FEATURE_NAMES = ["mean", "std", "median", "range", "slope", "skewness"]


def extract_features(trace):
    sk = float(skew(trace))
    return [
        trace.mean(),
        trace.std(),
        np.median(trace),
        trace.max() - trace.min(),
        np.polyfit(t30, trace, 1)[0],
        0.0 if np.isnan(sk) else sk,
    ]


def trace_zscore(trace):
    mu, sigma = trace.mean(), trace.std()
    return (trace - mu) / sigma if sigma > 0 else trace - mu


def trace_minmax(trace):
    lo, hi = trace.min(), trace.max()
    return (trace - lo) / (hi - lo) if hi > lo else np.zeros_like(trace)


def fit_patient_baseline_normalizer(X_train, patient_train_ids, fallback="global_mean"):
    """
    Learns normalization statistics from training data only.

    X_train: np.ndarray, shape (n_train_traces, trace_length)
    patient_train_ids: array-like, shape (n_train_traces,)
    fallback:
        "global_mean" = mean of all training trace values
        "median_patient_baseline" = median of training patient baselines

    Returns:
        baselines: dict mapping patient_id -> scalar baseline
        fallback_baseline: scalar
    """
    X_train = np.asarray(X_train, dtype=float)
    patient_train_ids = np.asarray(patient_train_ids)

    baselines = {}
    for pid in np.unique(patient_train_ids):
        mask = patient_train_ids == pid
        baselines[pid] = np.nanmean(X_train[mask])

    if fallback == "global_mean":
        fallback_baseline = np.nanmean(X_train)
    elif fallback == "median_patient_baseline":
        fallback_baseline = np.nanmedian(list(baselines.values()))
    else:
        raise ValueError(f"Unknown fallback: {fallback}")

    return baselines, fallback_baseline


def transform_patient_baseline_normalizer(X, patient_ids, baselines, fallback_baseline):
    """
    Applies training-learned patient baselines.
    For unseen patients, uses the training-learned fallback baseline.
    """
    X = np.asarray(X, dtype=float)
    patient_ids = np.asarray(patient_ids)
    X_norm = X.copy()

    for i, pid in enumerate(patient_ids):
        baseline = baselines.get(pid, fallback_baseline)
        X_norm[i] = X_norm[i] - baseline

    return X_norm


def check_training_patient_means(X_train_norm, patient_train_ids, atol=1e-10):
    patient_train_ids = np.asarray(patient_train_ids)
    for pid in np.unique(patient_train_ids):
        patient_mean = np.nanmean(X_train_norm[patient_train_ids == pid])
        if not np.isclose(patient_mean, 0.0, atol=atol):
            raise AssertionError(
                f"Training patient {pid} has nonzero normalized mean: {patient_mean}"
            )


# Precompute non-fold-dependent feature matrices. The feature-level scalers are
# still fit inside each fold below.
X_traces = np.vstack(df["trace"].values)
X_raw = np.array([extract_features(row["trace"]) for _, row in df.iterrows()])
X_trace_zscore = np.array([
    extract_features(trace_zscore(row["trace"])) for _, row in df.iterrows()
])
X_trace_minmax = np.array([
    extract_features(trace_minmax(row["trace"])) for _, row in df.iterrows()
])

groups = df["patient"].values
labels = df["label"].values

def transform_fold(method, train_idx, test_idx):
    """Return (X_train, X_test) feature matrices for one CV fold.

    Every scaler/baseline is fit on the TRAIN indices only, so the same function
    is leakage-free for both the LOSO and the GroupKFold splits.
    """
    if method == "trace z-score":
        return X_trace_zscore[train_idx], X_trace_zscore[test_idx]
    if method == "trace min-max":
        return X_trace_minmax[train_idx], X_trace_minmax[test_idx]
    if method == "feature z-score":
        scaler = StandardScaler()
        return scaler.fit_transform(X_raw[train_idx]), scaler.transform(X_raw[test_idx])
    if method == "robust scaler":
        scaler = RobustScaler()
        return scaler.fit_transform(X_raw[train_idx]), scaler.transform(X_raw[test_idx])

    # training-only patient baseline normalization
    patient_train_ids = groups[train_idx]
    patient_test_ids = groups[test_idx]
    baselines, fallback_baseline = fit_patient_baseline_normalizer(
        X_traces[train_idx], patient_train_ids, fallback="global_mean",
    )
    X_train_norm = transform_patient_baseline_normalizer(
        X_traces[train_idx], patient_train_ids, baselines, fallback_baseline,
    )
    X_test_norm = transform_patient_baseline_normalizer(
        X_traces[test_idx], patient_test_ids, baselines, fallback_baseline,
    )
    check_training_patient_means(X_train_norm, patient_train_ids)  # leakage guard
    X_train_features = np.array([extract_features(t) for t in X_train_norm])
    X_test_features = np.array([extract_features(t) for t in X_test_norm])
    scaler = StandardScaler()
    return scaler.fit_transform(X_train_features), scaler.transform(X_test_features)


def run_cv(method, splitter):
    """Run one CV scheme for one normalization method.

    Returns per-fold accuracy/F1/AUC lists plus the pooled out-of-fold AUC
    (a single ROC computed over all held-out scores pooled across folds).
    """
    accs, f1s, aucs = [], [], []
    pooled_scores, pooled_labels = [], []
    for train_idx, test_idx in splitter.split(np.zeros(len(df)), labels, groups):
        X_tr, X_te = transform_fold(method, train_idx, test_idx)
        y_tr, y_te = labels[train_idx], labels[test_idx]

        clf = LinearSVC(C=1.0, max_iter=5000, random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_score = clf.decision_function(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_te, y_score))
        pooled_scores.extend(np.asarray(y_score).tolist())
        pooled_labels.extend(np.asarray(y_te).tolist())

    pooled_auc = roc_auc_score(pooled_labels, pooled_scores)
    return accs, f1s, aucs, pooled_auc


logo = LeaveOneGroupOut()
gkf = GroupKFold(n_splits=5)

METHODS = [
    "trace z-score",
    "trace min-max",
    "feature z-score",
    "robust scaler",
    "training-only patient baseline normalization",
]
results = {}

for method in METHODS:
    loso_acc, loso_f1, loso_rank, loso_pooled_auc = run_cv(method, logo)
    gkf_acc, gkf_f1, gkf_auc, _ = run_cv(method, gkf)

    results[method] = {
        "loso_acc_mean": np.mean(loso_acc), "loso_acc_std": np.std(loso_acc),
        "loso_f1_mean": np.mean(loso_f1), "loso_f1_std": np.std(loso_f1),
        "loso_rank_mean": np.mean(loso_rank), "loso_rank_std": np.std(loso_rank),
        "loso_pooled_auc": loso_pooled_auc,
        "gkf_acc_mean": np.mean(gkf_acc), "gkf_acc_std": np.std(gkf_acc),
        "gkf_f1_mean": np.mean(gkf_f1), "gkf_f1_std": np.std(gkf_f1),
        "gkf_auc_mean": np.mean(gkf_auc), "gkf_auc_std": np.std(gkf_auc),
    }

print(f"\n{'Method':<46}{'LOSO acc':>16}{'GKF acc':>16}"
      f"{'GKF AUC':>16}{'pooled-LOSO':>14}{'LOSO rank':>13}")
print("-" * 121)
for m, r in results.items():
    print(
        f"{m:<46}"
        f"{r['loso_acc_mean']:.3f}+/-{r['loso_acc_std']:.3f}   "
        f"{r['gkf_acc_mean']:.3f}+/-{r['gkf_acc_std']:.3f}   "
        f"{r['gkf_auc_mean']:.3f}+/-{r['gkf_auc_std']:.3f}   "
        f"{r['loso_pooled_auc']:.3f}        "
        f"{r['loso_rank_mean']:.3f}"
    )
print("\nNote: 'LOSO rank' = per-fold LOSO AUC = pairwise ranking accuracy "
      "(degenerate: 1 lesional + 1 non-lesional per fold).")
print("      'GKF AUC' is the headline discriminability metric; 'pooled-LOSO' "
      "is the secondary pooled estimate.")

pd.DataFrame([
    {"normalization": method, **{k: round(v, 4) for k, v in metrics.items()}}
    for method, metrics in results.items()
]).to_csv(RESULTS / "stage3_normalization_results.csv", index=False)
print(f"\nSaved -> {RESULTS / 'stage3_normalization_results.csv'}")

short_labels = [
    "trace\nz-score",
    "trace\nmin-max",
    "feature\nz-score",
    "robust\nscaler",
    "patient\nbaseline",
]
methods_list = list(METHODS)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

gkf_aucs  = [results[m]["gkf_auc_mean"]  for m in methods_list]
gkf_errs  = [results[m]["gkf_auc_std"]   for m in methods_list]
loso_accs = [results[m]["loso_acc_mean"] for m in methods_list]
loso_errs = [results[m]["loso_acc_std"]  for m in methods_list]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

# Top: GKF AUC
bars1 = ax1.bar(short_labels, gkf_aucs, yerr=gkf_errs, capsize=4,
                color=colors, edgecolor="white", width=0.55)
ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
ax1.set_ylabel("GroupKFold AUC")
ax1.set_title("Normalization comparison — discriminability\n(GroupKFold n=5, honest metric)")
ax1.set_ylim(0.3, 1.05)
for bar, v in zip(bars1, gkf_aucs):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
             ha="center", va="bottom", fontsize=8)

# Bottom: LOSO accuracy
bars2 = ax2.bar(short_labels, loso_accs, yerr=loso_errs, capsize=4,
                color=colors, edgecolor="white", width=0.55)
ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
ax2.set_ylabel("LOSO accuracy")
ax2.set_title("Normalization comparison — threshold accuracy\n(LOSO 13-fold, primary generalization test)")
ax2.set_ylim(0.3, 1.05)
for bar, v in zip(bars2, loso_accs):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
             ha="center", va="bottom", fontsize=8)

plt.tight_layout()
fig.savefig(RESULTS / "stage3_normalization.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {RESULTS / 'stage3_normalization.png'}")
