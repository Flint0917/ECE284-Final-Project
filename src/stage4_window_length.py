"""
Stage 4: Expanded Signal Length Study

Classifier: LinearSVC (C=1.0) under LOSO; 13 patients, one held out per fold.

Outputs:
  results/stage4_window_length.png   3-column figure (Acc / F1 / AUC vs duration)
  results/stage4_window_length.csv   per-window LOSO summary
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*moment calculation.*")

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

les = pd.read_csv(DATA / "lesional_sensor.csv",    index_col=0).ffill().bfill()
non = pd.read_csv(DATA / "nonlesional_sensor.csv", index_col=0).ffill().bfill()

records = []
for sid in les.columns:
    records.append({"patient": sid, "label": 1, "trace": les[sid].values.astype(float)})
for sid in non.columns:
    records.append({"patient": sid, "label": 0, "trace": non[sid].values.astype(float)})

df = pd.DataFrame(records)
df["paper_num"] = df["patient"].map(patient_map)
df = df.sort_values(["paper_num", "label"], ignore_index=True)

# ---------------------------------------------------------------------------
# Window families
# ---------------------------------------------------------------------------
WINDOW_FAMILIES = {
    "Prefix (0→N)":    [(0, 5), (0, 10), (0, 15), (0, 20), (0, 25), (0, 30)],
    "Post-5s (5→N)":   [(5, 10), (5, 15), (5, 20), (5, 25), (5, 30)],
    "Post-10s (10→N)": [(10, 15), (10, 20), (10, 25), (10, 30)],
}
FAMILY_COLORS  = {
    "Prefix (0→N)":    "#f14040",
    "Post-5s (5→N)":   "#1a6fdf",
    "Post-10s (10→N)": "#37ad6b",
}
FAMILY_MARKERS = {"Prefix (0→N)": "o", "Post-5s (5→N)": "s", "Post-10s (10→N)": "^"}
FAMILY_STYLES  = {"Prefix (0→N)": "-", "Post-5s (5→N)": "--", "Post-10s (10→N)": ":"}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def zscore_seg(seg):
    mu, sigma = seg.mean(), seg.std()
    return (seg - mu) / sigma if sigma > 0 else seg - mu

def extract_features(seg):
    t  = np.arange(len(seg))
    sk = float(skew(seg))
    return [
        seg.mean(),
        seg.std(),
        np.median(seg),
        seg.max() - seg.min(),
        np.polyfit(t, seg, 1)[0],
        0.0 if np.isnan(sk) else sk,   # constant segment → symmetric → skewness = 0
    ]

def build_X_window(df, start, end):
    rows = []
    for _, row in df.iterrows():
        seg = zscore_seg(row["trace"][start:end])
        rows.append(extract_features(seg))
    return np.array(rows)

# ---------------------------------------------------------------------------
# LOSO cross-validation across all windows
# ---------------------------------------------------------------------------
logo   = LeaveOneGroupOut()
groups = df["patient"].values
labels = df["label"].values

csv_rows       = []
family_results = {}

for family_name, windows in WINDOW_FAMILIES.items():
    family_results[family_name] = {}
    print(f"\n{family_name}")
    print(f"  {'Window':>8}  {'Accuracy':>18}  {'F1':>18}  {'AUC':>18}")
    print(f"  {'-'*70}")

    for start, end in windows:
        duration = end - start
        X = build_X_window(df, start, end)
        accs, f1s, aucs = [], [], []

        for train_idx, test_idx in logo.split(X, labels, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = labels[train_idx], labels[test_idx]

            clf = LinearSVC(C=1.0, max_iter=5000, random_state=42)
            clf.fit(X_tr, y_tr)

            y_pred  = clf.predict(X_te)
            y_score = clf.decision_function(X_te)

            accs.append(accuracy_score(y_te, y_pred))
            f1s.append(f1_score(y_te, y_pred, zero_division=0))
            aucs.append(roc_auc_score(y_te, y_score))

        r = {
            "acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "f1_mean":  np.mean(f1s),  "f1_std":  np.std(f1s),
            "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
        }
        family_results[family_name][duration] = r

        print(f"  {start:2d}-{end:2d} s   "
              f"{r['acc_mean']:.3f} +/- {r['acc_std']:.3f}   "
              f"{r['f1_mean']:.3f} +/- {r['f1_std']:.3f}   "
              f"{r['auc_mean']:.3f} +/- {r['auc_std']:.3f}")

        csv_rows.append({
            "family": family_name, "start": start, "end": end, "duration": duration,
            **{k: round(v, 4) for k, v in r.items()},
        })

pd.DataFrame(csv_rows).to_csv(RESULTS / "stage4_window_length.csv", index=False)
print(f"\nSaved -> {RESULTS / 'stage4_window_length.csv'}")

# ---------------------------------------------------------------------------
# Figure: 3 columns (Accuracy, F1, AUC) vs window duration
# ---------------------------------------------------------------------------
METRICS = [("acc", "Accuracy"), ("f1", "F1-Score"), ("auc", "AUC-ROC")]

fig, axes = plt.subplots(3, 1, figsize=(6, 13), sharex=True)

for ax, (mkey, mlabel) in zip(axes, METRICS):
    for family_name, dur_dict in family_results.items():
        durations = sorted(dur_dict.keys())
        means = [dur_dict[d][f"{mkey}_mean"] for d in durations]
        stds  = [dur_dict[d][f"{mkey}_std"]  for d in durations]

        ax.plot(
            durations, means,
            color=FAMILY_COLORS[family_name],
            linestyle=FAMILY_STYLES[family_name],
            marker=FAMILY_MARKERS[family_name],
            linewidth=2, markersize=7,
            label=family_name,
        )
        ax.fill_between(
            durations,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=FAMILY_COLORS[family_name],
            alpha=0.12,
        )

    ax.set_title(mlabel, fontsize=13, fontweight="bold")
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (mean +/- std, 13 LOSO folds)", fontsize=10, fontweight="bold")
    # chance-level reference line
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(True, linestyle="--", alpha=0.4)

axes[-1].set_xlabel("Window Duration (s)", fontsize=11, fontweight="bold")

# single shared legend below all subplots
handles, leg_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, leg_labels, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

plt.suptitle(
    "Stage 4 - Expanded Window Length Study\n"
    "(LinearSVC, z-score per segment, LOSO cross-validation)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig(RESULTS / "stage4_window_length.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {RESULTS / 'stage4_window_length.png'}")
plt.close()
