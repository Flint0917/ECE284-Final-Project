"""
Stage 2: Normalization Comparison

Compares three normalization strategies as preprocessing for LinearSVC
classification under LOSO cross-validation:
  - z-score:       zero-mean, unit-variance per trace
  - min-max:       rescale each trace to [0, 1]
  - patient-level: subtract each patient's mean across both of their traces

LOSO (Leave-One-Subject-Out): 13 folds; each fold trains on 24 traces
(12 patients × 2 labels) and tests on 2 traces (the held-out patient).
This prevents the model from ever training and testing on the same patient,
simulating real-world deployment on a new individual.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import LinearSVC

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Patient map and data loading
# ---------------------------------------------------------------------------
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

les = pd.read_csv(DATA / "lesional_sensor.csv",    index_col=0).ffill()
non = pd.read_csv(DATA / "nonlesional_sensor.csv", index_col=0).ffill()

# One row per trace: (patient ID, label, 30-point array)
records = []
for sid in les.columns:
    records.append({"patient": sid, "label": 1, "trace": les[sid].values.astype(float)})
for sid in non.columns:
    records.append({"patient": sid, "label": 0, "trace": non[sid].values.astype(float)})

df = pd.DataFrame(records)
df["paper_num"] = df["patient"].map(patient_map)
df = df.sort_values(["paper_num", "label"], ignore_index=True)

# ---------------------------------------------------------------------------
# Normalization functions (applied per trace)
# ---------------------------------------------------------------------------
def zscore(trace):
    """Subtract trace mean and divide by trace std → mean=0, std=1."""
    mu, sigma = trace.mean(), trace.std()
    return (trace - mu) / sigma if sigma > 0 else trace - mu

def minmax(trace):
    """Rescale trace to [0, 1]."""
    lo, hi = trace.min(), trace.max()
    return (trace - lo) / (hi - lo) if hi > lo else np.zeros_like(trace)

# Patient-level: subtract patient's mean computed across BOTH labels.
# Applied before the LOSO loop because it only uses each patient's own data
# (not cross-patient information), so no leakage occurs.
pl_traces = {}
for pid, group in df.groupby("patient"):
    patient_mean = np.concatenate(group["trace"].values).mean()
    for idx in group.index:
        pl_traces[idx] = df.at[idx, "trace"] - patient_mean

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
t = np.arange(30)   # time axis [0..29] for slope

FEATURE_NAMES = ["mean", "std", "median", "range", "slope", "skewness"]

def extract_features(trace):
    """
    Compress a 30-point trace into 6 summary statistics.
    These replace raw signal values as input to the SVM — hand-crafted
    features work well on small datasets where CNNs tend to overfit.
    """
    return [
        trace.mean(),
        trace.std(),
        np.median(trace),
        trace.max() - trace.min(),          # range: peak-to-peak amplitude
        np.polyfit(t, trace, 1)[0],         # slope: linear trend (polyfit deg 1)
        float(skew(trace)),                 # skewness: distribution asymmetry
    ]

def build_X(df, method):
    rows = []
    for idx, row in df.iterrows():
        if method == "z-score":
            trace = zscore(row["trace"])
        elif method == "min-max":
            trace = minmax(row["trace"])
        else:
            trace = pl_traces[idx]
        rows.append(extract_features(trace))
    return np.array(rows)

# ---------------------------------------------------------------------------
# LOSO cross-validation with LinearSVC
# ---------------------------------------------------------------------------
logo   = LeaveOneGroupOut()
groups = df["patient"].values
labels = df["label"].values

METHODS = ["z-score", "min-max", "patient-level"]
results = {}

for method in METHODS:
    X = build_X(df, method)
    accs, f1s, aucs = [], [], []

    for train_idx, test_idx in logo.split(X, labels, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]

        clf = LinearSVC(C=1.0, max_iter=5000, random_state=42)
        clf.fit(X_tr, y_tr)

        y_pred  = clf.predict(X_te)
        # decision_function returns signed distance to hyperplane — higher → class 1
        y_score = clf.decision_function(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))
        # roc_auc_score requires both classes in y_te; each fold has 1 of each, so this holds
        aucs.append(roc_auc_score(y_te, y_score))

    results[method] = {
        "acc_mean": np.mean(accs),  "acc_std": np.std(accs),
        "f1_mean":  np.mean(f1s),   "f1_std":  np.std(f1s),
        "auc_mean": np.mean(aucs),  "auc_std": np.std(aucs),
    }

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print(f"\n{'Method':<16} {'Accuracy':>16} {'F1-Score':>16} {'AUC-ROC':>16}")
print("-" * 68)
for m, r in results.items():
    print(f"{m:<16}  {r['acc_mean']:.3f} ± {r['acc_std']:.3f}   "
          f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}   "
          f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}")

# ---------------------------------------------------------------------------
# Bar chart: x-axis = evaluation metrics, grouped bars = normalization methods
# ---------------------------------------------------------------------------
METRIC_LABELS = ["Accuracy", "F1-Score", "AUC-ROC"]
METRIC_KEYS   = ["acc", "f1", "auc"]
METHOD_COLORS = ["#f14040", "#1a6fdf", "#37ad6b"]

x     = np.arange(len(METRIC_LABELS))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

for i, (method, color) in enumerate(zip(METHODS, METHOD_COLORS)):
    means  = [results[method][f"{k}_mean"] for k in METRIC_KEYS]
    stds   = [results[method][f"{k}_std"]  for k in METRIC_KEYS]
    offset = (i - 1) * width
    ax.bar(x + offset, means, width, yerr=stds,
           label=method, color=color, capsize=4,
           error_kw={"elinewidth": 1.2, "ecolor": "black"})

ax.set_xlabel("Metric", fontsize=14, fontweight="bold")
ax.set_ylabel("Score", fontsize=14, fontweight="bold")
ax.set_title("Normalization Comparison — LinearSVC under LOSO",
             fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(METRIC_LABELS, fontsize=13)
ax.set_ylim(0, 1.2)
ax.legend(fontsize=11, title="Normalization", title_fontsize=11)
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

out = RESULTS / "stage2_normalization.png"
plt.savefig(out, dpi=150)
print(f"\nSaved → {out}")
