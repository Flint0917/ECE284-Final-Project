"""
Stage 5: Evaluation Expansion

Complements LOSO with a GroupKFold sensitivity analysis and a per-patient
prediction table for the intra-individual tracking comparison.

  LOSO (primary):      13 folds, one patient per fold. Each fold tests exactly
                       1 lesional + 1 non-lesional trace, so fold-level AUC equals
                       the correct-ranking indicator (1 if les_score > non_les_score).
                       Do NOT interpret AUC=1.00 as clinical validation; it means all
                       13 patients were correctly ranked on the proof-of-concept dataset.

  GroupKFold (n=5):    2-3 patients per test fold -> 4-6 samples per fold. Larger test
                       sets yield a more graded per-fold AUC, but N=13 patients total
                       means this remains a proof-of-concept sensitivity check.

  Within-patient ranking accuracy: fraction of LOSO folds where the model scores
                       the lesional trace higher than the non-lesional trace for the
                       same patient. Mirrors Todorov's intra-individual tracking claim.

Normalization: training-only patient baseline normalization. Each fold learns
               patient baselines from training traces only; unseen test patients
               use the training global mean fallback. Feature StandardScaler is
               also fit on training features only.
Classifier: LinearSVC (C=1.0, max_iter=5000, random_state=42).

Outputs:
  results/stage5_evaluation.png
  results/stage5_loso_per_patient.csv
  results/stage5_metrics.csv
  results/fold_safe_patient_baseline_log.csv
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
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
# Feature extraction and fold-safe normalization helpers
# ---------------------------------------------------------------------------
t30 = np.arange(30)

def extract_features(trace):
    sk = float(skew(trace))
    return [
        trace.mean(),
        trace.std(),
        np.median(trace),
        trace.max() - trace.min(),
        np.polyfit(t30, trace, 1)[0],
        0.0 if np.isnan(sk) else sk,   # constant segment → skewness = 0 by convention
    ]

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


def prepare_fold_features(train_idx, test_idx):
    X_train_raw = X_traces[train_idx]
    X_test_raw = X_traces[test_idx]
    patient_train_ids = groups[train_idx]
    patient_test_ids = groups[test_idx]

    baselines, fallback_baseline = fit_patient_baseline_normalizer(
        X_train_raw,
        patient_train_ids,
        fallback="global_mean",
    )
    X_train_norm = transform_patient_baseline_normalizer(
        X_train_raw,
        patient_train_ids,
        baselines,
        fallback_baseline,
    )
    X_test_norm = transform_patient_baseline_normalizer(
        X_test_raw,
        patient_test_ids,
        baselines,
        fallback_baseline,
    )
    check_training_patient_means(X_train_norm, patient_train_ids)

    X_train_features = np.array([extract_features(trace) for trace in X_train_norm])
    X_test_features = np.array([extract_features(trace) for trace in X_test_norm])

    scaler = StandardScaler()
    X_train_features_scaled = scaler.fit_transform(X_train_features)
    X_test_features_scaled = scaler.transform(X_test_features)

    baseline_rows = []
    for pid in np.unique(patient_test_ids):
        used_patient_specific = pid in baselines
        baseline_rows.append({
            "patient_id": pid,
            "used_patient_specific_baseline": used_patient_specific,
            "baseline_value": baselines.get(pid, fallback_baseline),
        })

    return X_train_features_scaled, X_test_features_scaled, baseline_rows


X_traces = np.vstack(df["trace"].values)
y      = df["label"].values
groups = df["patient"].values

# ---------------------------------------------------------------------------
# 1. LOSO — primary evaluation, capturing per-patient scores for the table
# ---------------------------------------------------------------------------
logo = LeaveOneGroupOut()

loso_accs, loso_f1s, loso_aucs = [], [], []
loso_pooled_scores, loso_pooled_labels = [], []   # for the pooled-LOSO ROC curve
per_patient_rows = []
baseline_log_rows = []

for fold_id, (train_idx, test_idx) in enumerate(logo.split(X_traces, y, groups), start=1):
    X_train_features_scaled, X_test_features_scaled, baseline_rows = prepare_fold_features(
        train_idx,
        test_idx,
    )
    for baseline_row in baseline_rows:
        baseline_log_rows.append({"cv_method": "LOSO", "fold": fold_id, **baseline_row})

    clf = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    clf.fit(X_train_features_scaled, y[train_idx])

    y_pred  = clf.predict(X_test_features_scaled)
    y_score = clf.decision_function(X_test_features_scaled)
    y_te    = y[test_idx]

    fold_acc = accuracy_score(y_te, y_pred)
    loso_accs.append(fold_acc)
    loso_f1s.append(f1_score(y_te, y_pred, zero_division=0))
    loso_aucs.append(roc_auc_score(y_te, y_score))
    loso_pooled_scores.extend(y_score.tolist())
    loso_pooled_labels.extend(y_te.tolist())

    # Identify which test sample is lesional and which is non-lesional
    les_score = non_les_score = None
    les_pred = non_les_pred = None
    patient_id = paper_num = None
    patient_baseline = baseline_rows[0]   # LOSO: exactly one test patient per fold

    for local_i, global_i in enumerate(test_idx):
        row = df.iloc[global_i]
        if row["label"] == 1:
            les_score  = float(y_score[local_i])
            les_pred   = int(y_pred[local_i])
            patient_id = row["patient"]
            paper_num  = int(row["paper_num"])
        else:
            non_les_score = float(y_score[local_i])
            non_les_pred  = int(y_pred[local_i])

    per_patient_rows.append({
        "patient_id":      patient_id,
        "paper_num":       paper_num,
        "les_score":       round(les_score, 4),
        "non_les_score":   round(non_les_score, 4),
        "les_pred":        les_pred,
        "non_les_pred":    non_les_pred,
        "correct_ranking": les_score > non_les_score,
        "fold_acc":        fold_acc,
        "used_patient_specific_baseline": patient_baseline["used_patient_specific_baseline"],
        "baseline_value": round(patient_baseline["baseline_value"], 4),
    })

per_patient_df = (
    pd.DataFrame(per_patient_rows)
    .sort_values("paper_num", ignore_index=True)
)

loso_res = {
    "acc_mean": np.mean(loso_accs), "acc_std": np.std(loso_accs),
    "f1_mean":  np.mean(loso_f1s),  "f1_std":  np.std(loso_f1s),
    "auc_mean": np.mean(loso_aucs), "auc_std": np.std(loso_aucs),
}
n_correct = int(per_patient_df["correct_ranking"].sum())

print("\nLOSO Results (training-only patient baseline normalization, LinearSVC, full 30 s, 13 folds):")
print(f"  Accuracy : {loso_res['acc_mean']:.3f} +/- {loso_res['acc_std']:.3f}")
print(f"  F1-Score : {loso_res['f1_mean']:.3f} +/- {loso_res['f1_std']:.3f}")
print(f"  AUC-ROC  : {loso_res['auc_mean']:.3f} +/- {loso_res['auc_std']:.3f}")
print(f"\nWithin-patient correct ranking: {n_correct}/13 patients")
print("  (LOSO fold-level AUC equals correct_ranking: each fold has exactly 1L + 1NL)")

# ---------------------------------------------------------------------------
# 2. GroupKFold — sensitivity analysis only, not a replacement for LOSO
# ---------------------------------------------------------------------------
gkf = GroupKFold(n_splits=5)

gkf_accs, gkf_f1s, gkf_aucs = [], [], []
gkf_pooled_scores, gkf_pooled_labels = [], []   # for the pooled-GroupKFold ROC curve

for fold_id, (train_idx, test_idx) in enumerate(gkf.split(X_traces, y, groups), start=1):
    X_train_features_scaled, X_test_features_scaled, baseline_rows = prepare_fold_features(
        train_idx,
        test_idx,
    )
    for baseline_row in baseline_rows:
        baseline_log_rows.append({"cv_method": "GroupKFold", "fold": fold_id, **baseline_row})

    clf = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    clf.fit(X_train_features_scaled, y[train_idx])

    y_pred  = clf.predict(X_test_features_scaled)
    y_score = clf.decision_function(X_test_features_scaled)
    y_te    = y[test_idx]

    gkf_accs.append(accuracy_score(y_te, y_pred))
    gkf_f1s.append(f1_score(y_te, y_pred, zero_division=0))
    gkf_aucs.append(roc_auc_score(y_te, y_score))
    gkf_pooled_scores.extend(y_score.tolist())
    gkf_pooled_labels.extend(y_te.tolist())

gkf_res = {
    "acc_mean": np.mean(gkf_accs), "acc_std": np.std(gkf_accs),
    "f1_mean":  np.mean(gkf_f1s),  "f1_std":  np.std(gkf_f1s),
    "auc_mean": np.mean(gkf_aucs), "auc_std": np.std(gkf_aucs),
}

print("\nGroupKFold (n_splits=5) Sensitivity Analysis:")
print(f"  Accuracy : {gkf_res['acc_mean']:.3f} +/- {gkf_res['acc_std']:.3f}")
print(f"  F1-Score : {gkf_res['f1_mean']:.3f} +/- {gkf_res['f1_std']:.3f}")
print(f"  AUC-ROC  : {gkf_res['auc_mean']:.3f} +/- {gkf_res['auc_std']:.3f}")

baseline_log_df = pd.DataFrame(baseline_log_rows)
print("\nTest baselines used:")
for _, row in baseline_log_df.iterrows():
    print(
        f"  {row['cv_method']} fold {int(row['fold']):02d}, patient {row['patient_id']}: "
        f"patient_specific={row['used_patient_specific_baseline']}, "
        f"baseline={row['baseline_value']:.4f}"
    )

# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------
per_patient_df.to_csv(RESULTS / "stage5_loso_per_patient.csv", index=False)

pd.DataFrame([
    {"method": "LOSO",       **{k: round(v, 4) for k, v in loso_res.items()}},
    {"method": "GroupKFold", **{k: round(v, 4) for k, v in gkf_res.items()}},
]).to_csv(RESULTS / "stage5_metrics.csv", index=False)
baseline_log_df.to_csv(RESULTS / "fold_safe_patient_baseline_log.csv", index=False)

print(f"\nSaved -> {RESULTS / 'stage5_loso_per_patient.csv'}")
print(f"Saved -> {RESULTS / 'stage5_metrics.csv'}")
print(f"Saved -> {RESULTS / 'fold_safe_patient_baseline_log.csv'}")

# ---------------------------------------------------------------------------
# Figure: (top-left) per-fold metric distributions as box plots,
#         (top-right) pooled ROC curves, (bottom) per-patient table.
# Box plots replace bar+error-bar so the wide LOSO fold spread is visible;
# the ROC curve shows discriminability as a curve, not a single AUC bar.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 11))
gs  = plt.GridSpec(2, 2, height_ratios=[1, 1.15], hspace=0.32, wspace=0.22, figure=fig)

# ---- Top-left: per-fold distributions (box + points) -----------------------
ax_box = fig.add_subplot(gs[0, 0])
box_series = [
    ("LOSO\nacc", loso_accs, "#f14040"),
    ("LOSO AUC\n(rank, degen.)", loso_aucs, "#bdbdbd"),
    ("GKF\nacc", gkf_accs, "#1a6fdf"),
    ("GKF\nAUC", gkf_aucs, "#37ad6b"),
]
_jit = np.random.RandomState(0)
bp = ax_box.boxplot([s[1] for s in box_series], widths=0.55, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.4),
                    flierprops=dict(marker="", markersize=0))
for patch, (_, _, c) in zip(bp["boxes"], box_series):
    patch.set_facecolor(c); patch.set_alpha(0.55)
for i, (_, vals, _) in enumerate(box_series):
    xj = np.full(len(vals), i + 1) + (_jit.rand(len(vals)) - 0.5) * 0.18
    ax_box.scatter(xj, vals, s=20, color="#333333", alpha=0.65, zorder=3)
ax_box.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax_box.set_xticklabels([s[0] for s in box_series], fontsize=10)
ax_box.set_ylim(-0.05, 1.1)
ax_box.set_ylabel("Score", fontsize=12, fontweight="bold")
ax_box.set_title("Per-fold distributions (LOSO 13 folds, GKF 5 folds)\n"
                 "LOSO AUC is all 1.0 — the degenerate 1L+1NL metric",
                 fontsize=11, fontweight="bold")
ax_box.grid(True, axis="y", linestyle="--", alpha=0.4)

# ---- Top-right: pooled ROC curves ------------------------------------------
ax_roc = fig.add_subplot(gs[0, 1])
for labels_, scores_, color, name in [
    (loso_pooled_labels, loso_pooled_scores, "#f0a030", "pooled-LOSO"),
    (gkf_pooled_labels, gkf_pooled_scores, "#37ad6b", "pooled-GroupKFold"),
]:
    fpr, tpr, _ = roc_curve(labels_, scores_)
    ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC {roc_auc_score(labels_, scores_):.3f})")
ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.9)
ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)
ax_roc.set_xlabel("False positive rate", fontsize=12, fontweight="bold")
ax_roc.set_ylabel("True positive rate", fontsize=12, fontweight="bold")
ax_roc.set_title("Pooled out-of-fold ROC (LinearSVC)", fontsize=11, fontweight="bold")
ax_roc.legend(fontsize=10, loc="lower right")
ax_roc.grid(True, linestyle="--", alpha=0.4)

# ---- Bottom: per-patient LOSO prediction table -----------------------------
ax_tbl = fig.add_subplot(gs[1, :])
ax_tbl.axis("off")

col_labels = ["Patient", "L-Score", "NL-Score", "L-Pred", "NL-Pred", "L>NL?", "Fold Acc"]
table_data  = []
cell_colors = []

for _, row in per_patient_df.iterrows():
    l_pred_str  = "L"   if row["les_pred"]    == 1 else "NL"
    nl_pred_str = "NL"  if row["non_les_pred"] == 0 else "L"
    rank_str    = "Yes" if row["correct_ranking"]  else "No"
    acc_str     = f"{row['fold_acc']:.0%}"

    table_data.append([
        f"P{row['paper_num']}",
        f"{row['les_score']:+.3f}",
        f"{row['non_les_score']:+.3f}",
        l_pred_str,
        nl_pred_str,
        rank_str,
        acc_str,
    ])

    row_bg = "#d4edda" if row["correct_ranking"] else "#f8d7da"
    cell_colors.append([row_bg] * len(col_labels))

tbl = ax_tbl.table(
    cellText=table_data,
    colLabels=col_labels,
    cellColours=cell_colors,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.05, 1.5)

# Dark header row
for col in range(len(col_labels)):
    cell = tbl[0, col]
    cell.set_facecolor("#2c2c2c")
    cell.set_text_props(color="white", fontweight="bold")

ax_tbl.set_title(
    "LOSO Per-Patient Predictions\n"
    "(green row = L-score > NL-score, red row = inverted ranking)",
    fontsize=11, fontweight="bold",
)

plt.suptitle(
    "Stage 5 - Evaluation Expansion (training-only patient baseline normalization)",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.savefig(RESULTS / "stage5_evaluation.png", dpi=150, bbox_inches="tight")
print(f"Saved -> {RESULTS / 'stage5_evaluation.png'}")
plt.close()
