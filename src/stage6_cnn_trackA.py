"""
Stage 6 — Track A: from-scratch 1D CNN vs the SVM baseline.

Goal (tied to motivation Q2, generalization under data scarcity): test whether an
end-to-end 1D CNN, learning directly from the raw 30-point trace, can match the
handcrafted-feature SVM under the SAME patient-aware evaluation. The CNN receives
strictly MORE information than the SVM's 6 summary features, so if it still loses,
the cause is the small-N learning problem, not an information deficit.

Design decisions (and why):
  - Input: raw 30-pt trace, standardized per fold with the TRAINING global mean/std
    (fold-safe). Standardizes scale for stable optimization without destroying the
    within-trace shape the CNN is meant to learn.
  - Architecture: 2x Conv1d (8->16 ch, k=3) -> global average pool -> dropout -> FC(2).
    A "reasonable small CNN" a student would actually try — small enough to run, not
    artificially crippled, so any overfitting is honest.
  - Two variants: (1) from scratch, (2) from scratch + jittering augmentation
    (training-only Gaussian noise, sigma on the standardized/unit-variance scale).
    Directly tests whether augmentation mitigates small-N overfitting.
  - Evaluation: the SAME schemes as the SVM — LOSO (acc/F1 + pairwise ranking),
    GroupKFold n=5 (headline AUC), pooled-LOSO AUC.
  - Multi-seed: a CNN on ~24 training samples is highly seed-sensitive, so a single
    run could show "overfits and loses" or "ties SVM" by luck. We run 5 seeds, with a
    FRESH model per fold per seed, and base every conclusion on the seed-averaged result.
  - Report train accuracy explicitly: distinguishes OVERFIT (train high, test low) from
    UNDERFIT (both low). Both support "small data is the bottleneck" but are different
    stories; we name the mechanism after seeing the gap.

Track B (transfer learning) is triggered only if the seed-averaged CNN is meaningfully
worse than the SVM AND shows a large train-test gap (pre-registered below).

Outputs:
  results/stage6_cnn_trackA.png
  results/stage6_cnn_trackA_metrics.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import skew  # noqa: F401  (kept for parity with other stages)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# --- SVM reference (Stage 3, feature z-score = chosen primary normalization) ----
SVM_REF = {"loso_acc": 0.769, "gkf_acc": 0.750, "gkf_auc": 0.822, "pooled_loso_auc": 0.781}

# --- Pre-registered Track B trigger (decided BEFORE running) --------------------
TRACKB_AUC_MARGIN = 0.05   # CNN GKF-AUC must be > this far below SVM's to "lose"
TRACKB_GAP_MIN = 0.20      # train-test accuracy gap considered "large" (overfit)

SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 300
LR = 1e-3
DROPOUT = 0.3
JITTER_SIGMA = 0.2         # on the standardized (unit-variance) input scale


# ---------------------------------------------------------------------------
# Data loading (identical convention to the other stages)
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

X_traces = np.vstack(df["trace"].values)          # (26, 30)
labels = df["label"].values
groups = df["patient"].values


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TinyCNN(nn.Module):
    """2x Conv1d -> global average pool -> dropout -> linear(2)."""

    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)           # -> (batch, 16, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):                              # x: (batch, 1, 30)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)                   # (batch, 16)
        return self.fc(self.drop(x))


def standardize(X_tr, X_te):
    """Fold-safe: standardize by the TRAINING global mean/std over all trace values."""
    mu, sd = X_tr.mean(), X_tr.std()
    sd = sd if sd > 0 else 1.0
    return (X_tr - mu) / sd, (X_te - mu) / sd


def to_tensor(X):
    return torch.tensor(X, dtype=torch.float32).unsqueeze(1)   # (n, 1, 30)


def train_one(X_tr, y_tr, jitter):
    """Train a fresh TinyCNN full-batch; returns the trained model and train accuracy."""
    model = TinyCNN()                                  # fresh weights per fold
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    xb = to_tensor(X_tr)
    yb = torch.tensor(y_tr, dtype=torch.long)

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        xin = xb + torch.randn_like(xb) * JITTER_SIGMA if jitter else xb
        loss = loss_fn(model(xin), yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = accuracy_score(y_tr, model(xb).argmax(1).numpy())
    return model, train_acc


def predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(to_tensor(X))
        prob1 = torch.softmax(logits, dim=1)[:, 1].numpy()   # P(lesional) as the score
        pred = logits.argmax(1).numpy()
    return pred, prob1


def run_cv(splitter, jitter):
    """One CV scheme, one variant. Returns dict of fold-level metrics + pooled AUC + train acc."""
    accs, f1s, aucs, train_accs = [], [], [], []
    rank_hits, n_folds = 0, 0
    pooled_scores, pooled_labels = [], []

    for train_idx, test_idx in splitter.split(np.zeros(len(df)), labels, groups):
        X_tr, X_te = standardize(X_traces[train_idx], X_traces[test_idx])
        y_tr, y_te = labels[train_idx], labels[test_idx]

        model, tr_acc = train_one(X_tr, y_tr, jitter)
        pred, score = predict(model, X_te)

        accs.append(accuracy_score(y_te, pred))
        f1s.append(f1_score(y_te, pred, zero_division=0))
        aucs.append(roc_auc_score(y_te, score))
        train_accs.append(tr_acc)
        pooled_scores.extend(score.tolist())
        pooled_labels.extend(y_te.tolist())

        # pairwise ranking only meaningful for LOSO (exactly 1 L + 1 NL per fold)
        if len(test_idx) == 2 and set(y_te) == {0, 1}:
            les_score = score[np.where(y_te == 1)[0][0]]
            nls_score = score[np.where(y_te == 0)[0][0]]
            rank_hits += int(les_score > nls_score)
            n_folds += 1

    return {
        "acc": np.mean(accs), "f1": np.mean(f1s), "auc": np.mean(aucs),
        "train_acc": np.mean(train_accs),
        "pooled_auc": roc_auc_score(pooled_labels, pooled_scores),
        "rank_frac": (rank_hits / n_folds) if n_folds else np.nan,
    }


# ---------------------------------------------------------------------------
# Run: 5 seeds x 2 variants x {LOSO, GroupKFold}
# ---------------------------------------------------------------------------
logo = LeaveOneGroupOut()
gkf = GroupKFold(n_splits=5)
VARIANTS = [("CNN-scratch", False), ("CNN-jitter", True)]

# seed_results[variant] = list of per-seed summary dicts
seed_results = {name: [] for name, _ in VARIANTS}

for seed in SEEDS:
    for name, jitter in VARIANTS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        loso = run_cv(logo, jitter)
        torch.manual_seed(seed)          # reset so GKF is comparable across variants
        np.random.seed(seed)
        gk = run_cv(gkf, jitter)
        seed_results[name].append({
            "loso_acc": loso["acc"], "loso_f1": loso["f1"],
            "loso_train_acc": loso["train_acc"], "loso_rank": loso["rank_frac"],
            "pooled_loso_auc": loso["pooled_auc"],
            "gkf_acc": gk["acc"], "gkf_auc": gk["auc"],
        })


def agg(name):
    """Mean +/- std across seeds for one variant."""
    d = pd.DataFrame(seed_results[name])
    return {f"{c}_mean": d[c].mean() for c in d.columns} | {f"{c}_std": d[c].std(ddof=0) for c in d.columns}


summary = {name: agg(name) for name, _ in VARIANTS}

# ---------------------------------------------------------------------------
# Print + verdict
# ---------------------------------------------------------------------------
print(f"\nSVM reference (feature z-score): LOSO acc {SVM_REF['loso_acc']:.3f} | "
      f"GKF acc {SVM_REF['gkf_acc']:.3f} | GKF AUC {SVM_REF['gkf_auc']:.3f} | "
      f"pooled-LOSO AUC {SVM_REF['pooled_loso_auc']:.3f}")
print(f"\n{'Variant':<14}{'train acc':>14}{'LOSO acc':>14}{'GKF acc':>14}"
      f"{'GKF AUC':>16}{'pooled-LOSO':>14}{'LOSO rank':>12}  (mean +/- std over 5 seeds)")
print("-" * 112)
for name, _ in VARIANTS:
    s = summary[name]
    print(
        f"{name:<14}"
        f"{s['loso_train_acc_mean']:.3f}+/-{s['loso_train_acc_std']:.3f} "
        f"{s['loso_acc_mean']:.3f}+/-{s['loso_acc_std']:.3f} "
        f"{s['gkf_acc_mean']:.3f}+/-{s['gkf_acc_std']:.3f} "
        f"{s['gkf_auc_mean']:.3f}+/-{s['gkf_auc_std']:.3f} "
        f"{s['pooled_loso_auc_mean']:.3f}+/-{s['pooled_loso_auc_std']:.3f} "
        f"{s['loso_rank_mean']:.3f}+/-{s['loso_rank_std']:.3f}"
    )

scratch = summary["CNN-scratch"]
gap = scratch["loso_train_acc_mean"] - scratch["loso_acc_mean"]
auc_loses = scratch["gkf_auc_mean"] < (SVM_REF["gkf_auc"] - TRACKB_AUC_MARGIN)
big_gap = gap > TRACKB_GAP_MIN
mechanism = ("OVERFIT (train high, test low)" if scratch["loso_train_acc_mean"] > 0.85
             else "UNDERFIT (train also low)" if scratch["loso_train_acc_mean"] < 0.7
             else "mixed")

print(f"\nCNN-scratch train-test acc gap = {gap:.3f}  ->  mechanism: {mechanism}")
print(f"CNN-scratch GKF-AUC {scratch['gkf_auc_mean']:.3f} vs SVM {SVM_REF['gkf_auc']:.3f} "
      f"(margin {TRACKB_AUC_MARGIN}) -> loses to SVM: {auc_loses}")
print(f"Large train-test gap (> {TRACKB_GAP_MIN}): {big_gap}")
print(f"\n>>> Track B (transfer learning) triggered: {auc_loses and big_gap} "
      f"[pre-registered: AUC loses AND large gap]")

# ---------------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------------
rows = []
for name, _ in VARIANTS:
    s = summary[name]
    rows.append({
        "model": name,
        "train_acc": f"{s['loso_train_acc_mean']:.3f}+/-{s['loso_train_acc_std']:.3f}",
        "loso_acc": f"{s['loso_acc_mean']:.3f}+/-{s['loso_acc_std']:.3f}",
        "loso_f1": f"{s['loso_f1_mean']:.3f}+/-{s['loso_f1_std']:.3f}",
        "gkf_acc": f"{s['gkf_acc_mean']:.3f}+/-{s['gkf_acc_std']:.3f}",
        "gkf_auc": f"{s['gkf_auc_mean']:.3f}+/-{s['gkf_auc_std']:.3f}",
        "pooled_loso_auc": f"{s['pooled_loso_auc_mean']:.3f}+/-{s['pooled_loso_auc_std']:.3f}",
        "loso_rank": f"{s['loso_rank_mean']:.3f}+/-{s['loso_rank_std']:.3f}",
    })
rows.append({"model": "SVM (feature z-score, ref)", "train_acc": "-",
             "loso_acc": f"{SVM_REF['loso_acc']:.3f}", "loso_f1": "-",
             "gkf_acc": f"{SVM_REF['gkf_acc']:.3f}", "gkf_auc": f"{SVM_REF['gkf_auc']:.3f}",
             "pooled_loso_auc": f"{SVM_REF['pooled_loso_auc']:.3f}", "loso_rank": "-"})
pd.DataFrame(rows).to_csv(RESULTS / "stage6_cnn_trackA_metrics.csv", index=False)

# ---------------------------------------------------------------------------
# Figure: (A) train vs test acc (overfit gap), (B) discriminability vs SVM
# ---------------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))
_jit = np.random.RandomState(0)


def _seed_vals(name, key):
    return [d[key] for d in seed_results[name]]


def _box_strip(ax, series, ref_line, ref_label, ylabel, title, ylim):
    """series: list of (label, values, color). Box over seeds + individual points."""
    positions = np.arange(len(series))
    bp = ax.boxplot([s[1] for s in series], positions=positions, widths=0.5,
                    patch_artist=True, medianprops=dict(color="black", linewidth=1.3),
                    flierprops=dict(marker="", markersize=0))
    for patch, (_, _, c) in zip(bp["boxes"], series):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    for i, (_, vals, _) in enumerate(series):
        xj = np.full(len(vals), i) + (_jit.rand(len(vals)) - 0.5) * 0.16
        ax.scatter(xj, vals, s=24, color="#333333", alpha=0.7, zorder=3)
    ax.axhline(ref_line, color="#1a6fdf", linestyle="--", linewidth=1.5, label=ref_label)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xticks(positions); ax.set_xticklabels([s[0] for s in series], fontsize=10)
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(True, axis="y", linestyle="--", alpha=0.4)


# Panel A: train vs test accuracy across 5 seeds
seriesA = []
for n, _ in VARIANTS:
    seriesA.append((f"{n}\ntrain", _seed_vals(n, "loso_train_acc"), "#9b59b6"))
    seriesA.append((f"{n}\ntest", _seed_vals(n, "loso_acc"), "#f14040"))
_box_strip(axA, seriesA, SVM_REF["loso_acc"], f"SVM LOSO acc ({SVM_REF['loso_acc']:.3f})",
           "Accuracy", "Train vs test accuracy across 5 seeds\n(small train-test gap = no overfit)",
           (0, 1.15))

# Panel B: discriminability across 5 seeds
seriesB = []
for n, _ in VARIANTS:
    seriesB.append((f"{n}\nGKF AUC", _seed_vals(n, "gkf_auc"), "#37ad6b"))
    seriesB.append((f"{n}\npooled", _seed_vals(n, "pooled_loso_auc"), "#f0a030"))
_box_strip(axB, seriesB, SVM_REF["gkf_auc"], f"SVM GKF AUC ({SVM_REF['gkf_auc']:.3f})",
           "AUC-ROC", "Discriminability across 5 seeds: CNN vs SVM", (0, 1.0))

plt.suptitle("Stage 6 Track A - From-scratch 1D CNN vs SVM (per-seed distributions, 5 seeds)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(RESULTS / "stage6_cnn_trackA.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> {RESULTS / 'stage6_cnn_trackA.png'}")
print(f"Saved -> {RESULTS / 'stage6_cnn_trackA_metrics.csv'}")
plt.close()
