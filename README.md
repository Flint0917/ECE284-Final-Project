# Atopic Dermatitis Monitoring with IDC Wearable Sensors

ECE284 final project. Proof-of-concept analysis of interdigitated-capacitive (IDC)
sensor traces for distinguishing lesional from non-lesional skin in atopic
dermatitis (AD), under data scarcity (N = 13 patients).

**Dataset:** Todorov et al., *Wearable System Using Printed Interdigitated Capacitive
Sensor for Monitoring Atopic Dermatitis in Patients* (IEEE Sensors Journal, 2025).
13 patients, 30-second capacitance traces sampled at 1 Hz (30 points/trace),
lesional/non-lesional labels, with Corneometer references.

> This is a course proof-of-concept, **not clinical validation**. All reported
> numbers are on N = 13 with binary 2-sample LOSO folds.

## Repository layout

| Path | Contents |
|---|---|
| `data/` | Sensor + Corneometer CSVs, patient ID map (`PatientNumbering.txt`) |
| `WearSystemIDCforAD/` | Original raw dataset (read-only) |
| `src/` | One Python script per pipeline stage |
| `results/` | Figures (`.png`) and metric tables (`.csv`) produced by the scripts |

## Pipeline

| Script | Stage | What it does | Output |
|---|---|---|---|
| `src/stage1_replication.py` | 1 | Reproduces Todorov Fig. 9(a): per-patient lesional vs non-lesional mean capacitance | `stage1_replication.png` |
| `src/stage2_features.py` | 2 | Extracts 6 features (mean, std, median, range, slope, skewness) per trace; Random Forest importance heatmap | `stage2_features.png` |
| `src/stage3_normalization.py` | 3 | Compares five normalization strategies under LOSO + GroupKFold; GroupKFold AUC is the headline metric | `stage3_normalization.png`, `stage3_normalization_results.csv` |
| `src/stage4_window_length.py` | 4 | Window-length sweep — prefix (0→N) and post-settling (5→N, 10→N) windows | `stage4_window_length.png`, `stage4_window_length.csv` |
| `src/stage5_evaluation.py` | 5 | LinearSVC under LOSO + GroupKFold; per-patient ranking table | `stage5_evaluation.png`, `stage5_metrics.csv`, `stage5_loso_per_patient.csv` |
| `src/stage6_cnn_trackA.py` | 6A | From-scratch TinyCNN (5 seeds, jitter variant) vs SVM under LOSO + GroupKFold | `stage6_cnn_trackA.png`, `stage6_cnn_trackA_metrics.csv` |
| `src/stage6_cnn_trackB.py` | 6B | Self-supervised EDA autoencoder pretraining (WESAD) → frozen encoder fine-tuned on IDC traces | `stage6_cnn_trackB.png`, `stage6_cnn_trackB_metrics.csv` |

All normalization and feature scaling is fit **inside each CV fold on training
patients only** to prevent identity leakage; held-out patients use the training
global-mean fallback.

## Setup & run

```bash
conda create -n ece284 python=3.11
conda activate ece284
pip install numpy pandas scipy scikit-learn matplotlib seaborn torch
# run any stage; each reads from data/ and writes to results/
python src/stage1_replication.py
```

Stage 6B additionally requires WESAD EDA data. Download the WESAD dataset
(available on Kaggle / UCI) and place the `.pkl` subject files under
`data/wesad/WESAD/S*/S*_E4_Data/EDA.csv` or adjust the path in `stage6_cnn_trackB.py`.

`random_state=42` throughout for reproducibility.

## Key results

- **Stage 1:** Non-lesional capacitance exceeds lesional for all 13 patients; per-patient
  ordering matches Todorov Fig. 9(a). Absolute baselines vary ~10 pF across patients,
  making a population-wide threshold unreliable.
- **Stage 3:** Feature z-score and RobustScaler tie (GKF AUC 0.822, LOSO acc 0.769);
  patient-baseline centering gives correct paired ranking for all 13 patients but lower
  threshold accuracy (0.654). Feature z-score used in later stages.
- **Stage 4:** Short prefix windows score highest (0–5 s AUC 0.923), but this is a
  **data-quality artifact** — S-02, S-03, S-04 have genuine signal only in the first ~10 s;
  later samples are forward-fill constants. Not a clinical claim about window duration.
- **Stage 5:** LOSO ranks the lesional site correctly for **13/13** held-out patients
  (degenerate 2-sample-per-fold metric); threshold accuracy 0.654. GroupKFold AUC 0.817
  is the honest discriminability estimate. Supports intra-individual tracking, not a
  population-wide threshold.
- **Stage 6A:** TinyCNN (466 parameters, Global Average Pooling) matches and slightly
  edges the SVM — GKF AUC 0.871 (scratch) / 0.889 (jitter) vs 0.822 (SVM). Train–test
  gap is only 0.077 (no overfitting). The CNN and SVM converge because the dominant
  signal is per-site mean capacitance, which GAP trivially captures.
- **Stage 6B:** Self-supervised pretraining on 2,889 WESAD EDA windows (autoencoder MSE,
  no stress labels), frozen encoder fine-tuned on IDC traces. GKF AUC 0.827 — above SVM
  but below CNN-scratch. Frozen EDA encoder does not improve over learning directly from
  IDC traces; EDA–IDC modality gap limits transfer benefit at N = 26.

## Status

All six stages are complete.

| Stage | Status |
|---|---|
| 1 — Replication | Complete |
| 2 — Feature extraction | Complete |
| 3 — Normalization comparison | Complete |
| 4 — Window length experiment | Complete |
| 5 — Evaluation expansion (LOSO + GroupKFold) | Complete |
| 6A — From-scratch 1D CNN | Complete |
| 6B — Self-supervised transfer learning (WESAD EDA) | Complete |

## AI usage disclosure

Claude (Claude Code) was used for concept explanation, literature search, code guidance,
markdown documentation, and result interpretation. Research questions, scope decisions, and
scientific conclusions were made by the author. All AI-generated code was reviewed, tested,
and modified before use.
