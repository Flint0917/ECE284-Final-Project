"""
Stage 2: Feature Extraction and Importance

Extracts 6 hand-crafted features from each raw trace and
ranks them by Random Forest feature importance (mean decrease in impurity).

Output: results/stage2_features.png — two-panel figure:
  Left:  Feature importance bar chart (Random Forest, all 26 samples)
  Right: Heatmap of raw-derived feature values per trace (standardized for display),
         rows ordered by patient number, y-axis labels coloured by label.
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
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

les = pd.read_csv(DATA / "lesional_sensor.csv",    index_col=0).ffill()
non = pd.read_csv(DATA / "nonlesional_sensor.csv", index_col=0).ffill()

records = []
for sid in les.columns:
    records.append({"patient": sid, "label": 1, "trace": les[sid].values.astype(float)})
for sid in non.columns:
    records.append({"patient": sid, "label": 0, "trace": non[sid].values.astype(float)})

df = pd.DataFrame(records)
df["paper_num"] = df["patient"].map(patient_map)
# Sort: patient 1→13, lesional (1) before non-lesional (0) within each patient
df = df.sort_values(["paper_num", "label"], ascending=[True, False], ignore_index=True)

t = np.arange(30)

FEATURE_NAMES = ["Mean", "Std", "Median", "Range", "Slope", "Skewness"]

def extract_features(trace):
    return [
        trace.mean(),
        trace.std(),
        np.median(trace),
        trace.max() - trace.min(),
        np.polyfit(t, trace, 1)[0],
        float(skew(trace)),
    ]

# Raw traces required — per-trace z-score forces mean/std constant, making them useless as features.
X = np.array([extract_features(row["trace"]) for _, row in df.iterrows()])
y = df["label"].values

# Random Forest importance (N=26 → illustrative, not definitive).
# Mean decrease in impurity = how much each feature reduces node uncertainty on average.
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

importances    = rf.feature_importances_
importance_ord = np.argsort(importances)[::-1]   # descending rank

print("\nFeature importances (Random Forest, N=26 samples):")
for rank, i in enumerate(importance_ord):
    print(f"  {rank + 1}. {FEATURE_NAMES[i]:<10}  {importances[i]:.4f}")

# ---------------------------------------------------------------------------
# Standardize feature matrix for heatmap display only
# ---------------------------------------------------------------------------
X_std = StandardScaler().fit_transform(X)

row_labels = [
    f"P{row['paper_num']} ({'L' if row['label'] == 1 else 'NL'})"
    for _, row in df.iterrows()
]
heatmap_df = pd.DataFrame(X_std, index=row_labels, columns=FEATURE_NAMES)
tick_colors = ["#f14040" if lbl == 1 else "#1a6fdf" for lbl in df["label"].values]

fig = plt.figure(figsize=(7, 13))
gs  = gridspec.GridSpec(2, 1, height_ratios=[2.0, 1], hspace=0.45, figure=fig)

ax_hm = fig.add_subplot(gs[0])
sns.heatmap(
    heatmap_df,
    ax=ax_hm,
    cmap="RdBu_r",
    center=0,
    linewidths=0.4,
    linecolor="white",
    cbar_kws={"label": "Standardized feature value", "shrink": 0.75},
    yticklabels=True,
)
ax_hm.set_aspect("auto")  # prevent seaborn from centering axes with blank gaps
ax_hm.set_title(
    "Feature Values per Trace — raw features standardised for display\n"
    "(red y-labels = Lesional, blue = Non-Lesional)",
    fontsize=11, fontweight="bold",
)
ax_hm.set_xlabel("Feature", fontsize=12, fontweight="bold")
ax_hm.set_ylabel("")
ax_hm.tick_params(axis="x", labelsize=11, rotation=25)
for tick, color in zip(ax_hm.get_yticklabels(), tick_colors):
    tick.set_color(color)
    tick.set_fontsize(8.5)

patches = [
    mpatches.Patch(color="#f14040", label="Lesional"),
    mpatches.Patch(color="#1a6fdf", label="Non-Lesional"),
]
ax_hm.legend(handles=patches, fontsize=10, loc="upper right",
             bbox_to_anchor=(1.32, 1.0))

ax_imp = fig.add_subplot(gs[1])
bar_cols = ["#f14040" if i == importance_ord[0] else "#515151"
            for i in range(len(FEATURE_NAMES))]
ax_imp.bar(
    [FEATURE_NAMES[i] for i in importance_ord],
    importances[importance_ord],
    color=[bar_cols[i] for i in importance_ord],
    edgecolor="white",
)
ax_imp.set_ylabel("Mean Decrease in Impurity", fontsize=12, fontweight="bold")
ax_imp.set_title("Feature Importance (Random Forest)", fontsize=13, fontweight="bold")
ax_imp.grid(True, axis="y", linestyle="--", alpha=0.5)
ax_imp.tick_params(axis="x", labelsize=11, rotation=20)

fig.suptitle("Stage 2 — Feature Extraction (raw traces)",
             fontsize=14, fontweight="bold")
gs.tight_layout(fig, rect=[0, 0, 1, 0.96])

out = RESULTS / "stage2_features.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
