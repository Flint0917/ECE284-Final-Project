"""
Stage 6 — Track B: self-supervised transfer learning from WESAD EDA.

Pretraining task: train a 1D convolutional autoencoder (encoder = same two Conv1d
layers as Track A's TinyCNN) on EDA windows from the WESAD dataset (15 subjects,
wrist EDA at 4 Hz, downsampled to 1 Hz to match the IDC sequence length of 30 s).
The pretraining objective is signal reconstruction (MSE) — purely self-supervised;
no WESAD stress labels are used. This avoids source-target task mismatch when the
downstream task (lesional/non-lesional classification) differs from stress detection
(Fawaz et al., ECML 2019).

Transfer: the pretrained encoder's weights are frozen, then a fresh Global-Average-
Pool → Dropout(0.3) → Linear(16, 2) head is trained on the 26 IDC traces using the
same LOSO + GroupKFold evaluation protocol as Track A.

Framing: Track A (from-scratch CNN) already ties/beats the SVM (GKF AUC 0.871 vs
0.822) without overfitting. Track B asks: does domain-adjacent pretraining push
further still, or does it make no difference because the discriminative signal (mean
capacitance level per site) is too simple to benefit from richer temporal
initialization? Either outcome is informative.

Expected result: given that the useful "feature" is the trace mean — something GAP
directly averages — pretrained temporal features are unlikely to matter much.
Transfer learning's advantage is typically on *shape/waveform* discrimination; here
the task is essentially a level comparison, so Track B is predicted to tie Track A.

Outputs:
  results/stage6_cnn_trackB.png    — GKF AUC bar chart + pretraining loss curve
  results/stage6_cnn_trackB_metrics.csv
"""

import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WESAD_DIR = DATA / "wesad"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# --- References from Track A (5-seed averages) ----------------------------------
SVM_REF   = {"gkf_auc": 0.822, "pooled_loso_auc": 0.781, "loso_acc": 0.769}
CNN_SCRATCH = {"gkf_auc": 0.871, "pooled_loso_auc": 0.808, "loso_acc": 0.769}
CNN_JITTER  = {"gkf_auc": 0.889, "pooled_loso_auc": 0.815, "loso_acc": 0.777}

SEEDS = [42, 43, 44, 45, 46]
PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 300
LR_PRETRAIN = 1e-3
LR_FINETUNE = 1e-3
DROPOUT = 0.3
BATCH_SIZE = 64
WESAD_EDA_HZ = 4        # E4 wristband EDA sampling rate
TARGET_HZ = 1           # IDC sensor rate; downsample WESAD to match
WINDOW_S = 30           # seconds; matches the IDC trace length


# ---------------------------------------------------------------------------
# WESAD data loading
# ---------------------------------------------------------------------------

def unzip_wesad():
    """Unzip the Kaggle download if pkl files not yet extracted."""
    zip_path = WESAD_DIR / "wesad-wearable-stress-affect-detection-dataset.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"WESAD zip not found at {zip_path}.\n"
            "Run: kaggle datasets download orvile/wesad-wearable-stress-affect-detection-dataset "
            f"--path {WESAD_DIR}"
        )
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(WESAD_DIR)
    print("Extraction complete.")


def load_wesad_eda_windows():
    """
    Return a numpy array of shape (N, WINDOW_S) containing z-scored WESAD EDA
    windows. EDA is downsampled from WESAD_EDA_HZ to TARGET_HZ by block-averaging,
    then segmented into non-overlapping WINDOW_S-sample windows.

    Expects .pkl files anywhere under WESAD_DIR with the standard WESAD structure:
      data['signal']['wrist']['EDA']  — 1-D array at WESAD_EDA_HZ Hz
    """
    pkl_files = sorted(WESAD_DIR.rglob("S*.pkl"))
    if not pkl_files:
        unzip_wesad()
        pkl_files = sorted(WESAD_DIR.rglob("S*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"No S*.pkl files found under {WESAD_DIR} after extraction. "
            "Check the Kaggle dataset contents."
        )

    downsample_factor = WESAD_EDA_HZ // TARGET_HZ   # 4
    all_windows = []

    for fp in pkl_files:
        with open(fp, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        # Navigate to wrist EDA (shape: (n_samples,) or (n_samples, 1))
        eda = raw["signal"]["wrist"]["EDA"].ravel().astype(np.float32)

        # Downsample: reshape to blocks of `downsample_factor`, take mean
        n_complete = (len(eda) // downsample_factor) * downsample_factor
        eda = eda[:n_complete].reshape(-1, downsample_factor).mean(axis=1)
        # eda is now at TARGET_HZ (1 Hz)

        # Segment into non-overlapping WINDOW_S-sample windows
        n_windows = len(eda) // WINDOW_S
        for i in range(n_windows):
            w = eda[i * WINDOW_S : (i + 1) * WINDOW_S]
            # Skip constant windows (e.g., sensor dropout)
            if w.std() < 1e-6:
                continue
            all_windows.append(w)

    windows = np.vstack(all_windows)   # (N, WINDOW_S)

    # Global z-score across ALL windows (not per-window — preserves relative levels)
    mu = windows.mean()
    sd = windows.std()
    sd = sd if sd > 0 else 1.0
    windows = (windows - mu) / sd

    print(f"WESAD EDA windows: {len(pkl_files)} subjects, {windows.shape[0]} windows "
          f"of {WINDOW_S} s at {TARGET_HZ} Hz  (global mean={mu:.4f} uS, std={sd:.4f} uS)")
    return windows


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TinyEncoder(nn.Module):
    """Shared encoder: same Conv1d stack as Track A's TinyCNN.features."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
        )

    def forward(self, x):          # (batch, 1, T)
        return self.layers(x)      # (batch, 16, T)


class TinyAE(nn.Module):
    """Autoencoder for self-supervised pretraining on WESAD EDA."""

    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        # Mirror decoder: Conv1d (not transposed) keeps sequence length fixed.
        # Using Conv1d with same padding mirrors the encoder exactly, which is
        # appropriate because no pooling reduces T in the encoder.
        self.decoder = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8,  1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class TransferCNN(nn.Module):
    """Pretrained encoder (frozen) + new GAP + Dropout + Linear head."""

    def __init__(self, encoder: TinyEncoder, dropout=DROPOUT):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(16, 2)

    def forward(self, x):                      # (batch, 1, 30)
        feat = self.encoder(x)                 # (batch, 16, 30)
        feat = self.pool(feat).squeeze(-1)     # (batch, 16)
        return self.fc(self.drop(feat))


# ---------------------------------------------------------------------------
# IDC data loading (same as Track A)
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

X_traces = np.vstack(df["trace"].values)   # (26, 30)
labels    = df["label"].values
groups    = df["patient"].values


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def pretrain_ae(windows: np.ndarray, seed: int):
    """
    Train a TinyAE on WESAD EDA windows and return the pretrained encoder.
    windows: (N, WINDOW_S) numpy array, already z-scored.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ae = TinyAE()
    opt = torch.optim.Adam(ae.parameters(), lr=LR_PRETRAIN)
    loss_fn = nn.MSELoss()

    X = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)   # (N, 1, 30)
    n = len(X)
    losses = []

    ae.train()
    for epoch in range(PRETRAIN_EPOCHS):
        # Mini-batch shuffle
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb = X[idx]
            opt.zero_grad()
            recon = ae(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        losses.append(epoch_loss / n)

    ae.eval()
    return ae.encoder, losses


# ---------------------------------------------------------------------------
# Fine-tuning helpers (same fold-safe protocol as Track A)
# ---------------------------------------------------------------------------

def standardize(X_tr, X_te):
    mu, sd = X_tr.mean(), X_tr.std()
    sd = sd if sd > 0 else 1.0
    return (X_tr - mu) / sd, (X_te - mu) / sd


def to_tensor(X):
    return torch.tensor(X, dtype=torch.float32).unsqueeze(1)   # (n, 1, 30)


def finetune_head(encoder: TinyEncoder, X_tr, y_tr):
    """Freeze encoder; train only the new GAP+FC head."""
    model = TransferCNN(encoder)
    for param in model.encoder.parameters():
        param.requires_grad = False
    head_params = list(model.pool.parameters()) + \
                  list(model.fc.parameters())
    opt = torch.optim.Adam(head_params, lr=LR_FINETUNE)
    loss_fn = nn.CrossEntropyLoss()
    xb = to_tensor(X_tr)
    yb = torch.tensor(y_tr, dtype=torch.long)

    model.train()
    for _ in range(FINETUNE_EPOCHS):
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = accuracy_score(y_tr, model(xb).argmax(1).numpy())
    return model, train_acc


def predict(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(to_tensor(X))
        prob1 = torch.softmax(logits, dim=1)[:, 1].numpy()
        pred  = logits.argmax(1).numpy()
    return pred, prob1


def run_cv(splitter, encoder):
    """LOSO or GroupKFold evaluation with a given pretrained encoder."""
    accs, f1s, aucs, train_accs = [], [], [], []
    rank_hits, n_folds = 0, 0
    pooled_scores, pooled_labels = [], []

    for train_idx, test_idx in splitter.split(np.zeros(len(df)), labels, groups):
        X_tr, X_te = standardize(X_traces[train_idx], X_traces[test_idx])
        y_tr, y_te = labels[train_idx], labels[test_idx]

        model, tr_acc = finetune_head(encoder, X_tr, y_tr)
        pred, score   = predict(model, X_te)

        accs.append(accuracy_score(y_te, pred))
        f1s.append(f1_score(y_te, pred, zero_division=0))
        aucs.append(roc_auc_score(y_te, score))
        train_accs.append(tr_acc)
        pooled_scores.extend(score.tolist())
        pooled_labels.extend(y_te.tolist())

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
# Main: pretrain + evaluate over 5 seeds
# ---------------------------------------------------------------------------

print("Loading WESAD EDA windows ...")
wesad_windows = load_wesad_eda_windows()

logo = LeaveOneGroupOut()
gkf  = GroupKFold(n_splits=5)

seed_results = []
pretrain_loss_curves = []   # one curve per seed for the figure

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    print(f"  Pretraining autoencoder on {len(wesad_windows)} WESAD windows ...")
    encoder, loss_curve = pretrain_ae(wesad_windows, seed)
    pretrain_loss_curves.append(loss_curve)
    print(f"  Final pretrain MSE: {loss_curve[-1]:.5f}")

    # Reset seed before each CV scheme so head initializations are comparable
    # across LOSO/GKF (matches Track A's seed protocol)
    torch.manual_seed(seed)
    np.random.seed(seed)
    loso = run_cv(logo, encoder)

    torch.manual_seed(seed)
    np.random.seed(seed)
    gk = run_cv(gkf, encoder)

    seed_results.append({
        "loso_acc":         loso["acc"],
        "loso_f1":          loso["f1"],
        "loso_train_acc":   loso["train_acc"],
        "loso_rank":        loso["rank_frac"],
        "pooled_loso_auc":  loso["pooled_auc"],
        "gkf_acc":          gk["acc"],
        "gkf_auc":          gk["auc"],
    })
    print(f"  LOSO acc {loso['acc']:.3f}  GKF AUC {gk['auc']:.3f}  "
          f"pooled-LOSO {loso['pooled_auc']:.3f}  rank {loso['rank_frac']:.3f}")

df_res = pd.DataFrame(seed_results)
means  = df_res.mean()
stds   = df_res.std(ddof=0)

print(f"\n{'Metric':<22}{'Mean':>8}{'Std':>8}")
print("-" * 40)
for col in df_res.columns:
    print(f"{col:<22}{means[col]:>8.3f}{stds[col]:>8.3f}")

# ---------------------------------------------------------------------------
# Save metrics CSV
# ---------------------------------------------------------------------------
rows = [
    {
        "model": "CNN-transfer (Track B)",
        "train_acc":        f"{means['loso_train_acc']:.3f}+/-{stds['loso_train_acc']:.3f}",
        "loso_acc":         f"{means['loso_acc']:.3f}+/-{stds['loso_acc']:.3f}",
        "loso_f1":          f"{means['loso_f1']:.3f}+/-{stds['loso_f1']:.3f}",
        "gkf_acc":          f"{means['gkf_acc']:.3f}+/-{stds['gkf_acc']:.3f}",
        "gkf_auc":          f"{means['gkf_auc']:.3f}+/-{stds['gkf_auc']:.3f}",
        "pooled_loso_auc":  f"{means['pooled_loso_auc']:.3f}+/-{stds['pooled_loso_auc']:.3f}",
        "loso_rank":        f"{means['loso_rank']:.3f}+/-{stds['loso_rank']:.3f}",
    },
    {"model": "CNN-jitter (Track A)",  "train_acc": "-",
     "loso_acc": "0.777", "loso_f1": "-",
     "gkf_acc": "-", "gkf_auc": "0.889", "pooled_loso_auc": "0.815", "loso_rank": "-"},
    {"model": "CNN-scratch (Track A)", "train_acc": "-",
     "loso_acc": "0.769", "loso_f1": "-",
     "gkf_acc": "-", "gkf_auc": "0.871", "pooled_loso_auc": "0.808", "loso_rank": "-"},
    {"model": "SVM (feature z-score)", "train_acc": "-",
     "loso_acc": "0.769", "loso_f1": "-",
     "gkf_acc": "-", "gkf_auc": "0.822", "pooled_loso_auc": "0.781", "loso_rank": "-"},
]
pd.DataFrame(rows).to_csv(RESULTS / "stage6_cnn_trackB_metrics.csv", index=False)
print(f"\nSaved -> {RESULTS / 'stage6_cnn_trackB_metrics.csv'}")

# ---------------------------------------------------------------------------
# Figure: GKF AUC comparison + pretraining loss
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(5, 7))

# Top: GKF AUC bar chart (all four models)
ax = axes[0]
models     = ["SVM", "CNN-scratch", "CNN-jitter", "CNN-transfer"]
gkf_aucs   = [
    SVM_REF["gkf_auc"],
    CNN_SCRATCH["gkf_auc"],
    CNN_JITTER["gkf_auc"],
    means["gkf_auc"],
]
gkf_errs = [0, 0, 0, stds["gkf_auc"]]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

bars = ax.bar(models, gkf_aucs, yerr=gkf_errs, capsize=4,
              color=colors, edgecolor="white", width=0.55)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
ax.set_ylabel("GroupKFold AUC")
ax.set_title("Discriminability comparison\n(GroupKFold n=5, honest metric)")
ax.set_ylim(0.4, 1.02)
ax.legend(fontsize=8)
for bar, v in zip(bars, gkf_aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8)

# Bottom: pretraining loss curves (one per seed, + mean)
ax2 = axes[1]
loss_arr = np.array(pretrain_loss_curves)   # (5, PRETRAIN_EPOCHS)
epochs_x = np.arange(1, PRETRAIN_EPOCHS + 1)
for curve in loss_arr:
    ax2.plot(epochs_x, curve, color="#999999", alpha=0.4, linewidth=0.8)
ax2.plot(epochs_x, loss_arr.mean(axis=0), color="#C44E52", linewidth=1.5,
         label="Mean (5 seeds)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE reconstruction loss")
ax2.set_title(f"WESAD EDA autoencoder pretraining\n({len(wesad_windows)} windows, {PRETRAIN_EPOCHS} epochs)")
ax2.legend(fontsize=8)

plt.suptitle("Stage 6 Track B — Self-supervised transfer from WESAD EDA", fontsize=11)
plt.tight_layout()
fig.savefig(RESULTS / "stage6_cnn_trackB.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {RESULTS / 'stage6_cnn_trackB.png'}")
