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
| `supplements/` | Project brief, concept glossary, progress log |

## Pipeline

| Script | Stage | What it does | Output |
|---|---|---|---|
| `src/stage1_replication.py` | 1 | Reproduces Todorov Fig. 9(a): per-patient lesional vs non-lesional mean capacitance | `stage1_replication.png` |
| `src/stage2_features.py` | 2 | Extracts 6 features (mean, std, median, range, slope, skewness) per trace; Random Forest importance | `stage2_features.png` |
| `src/stage3_normalization.py` | 3 | Compares feature z-score, RobustScaler, and fold-safe patient-baseline centering under LOSO | `stage3_normalization.png` |
| `src/stage4_window_length.py` | 4 | Window-length sweep — prefix (0→N) and post-settling (5→N, 10→N) windows | `stage4_window_length.{png,csv}` |
| `src/stage5_evaluation.py` | 5 | LinearSVC under LOSO + GroupKFold; per-patient ranking table | `stage5_evaluation.png`, `stage5_metrics.csv`, `stage5_loso_per_patient.csv` |

All normalization and feature scaling is fit **inside each CV fold on training
patients only** to prevent identity leakage; held-out patients use the training
global-mean fallback.

## Setup & run

```bash
conda create -n ece284 python=3.11
conda activate ece284
pip install numpy pandas scipy scikit-learn matplotlib seaborn
# run any stage; each reads from data/ and writes to results/
python src/stage1_replication.py
```

`random_state=42` throughout for reproducibility.

## Key results

- **Stage 1:** non-lesional capacitance exceeds lesional for all 13 patients; per-patient
  ordering matches Todorov Fig. 9(a).
- **Stage 3:** feature z-score and RobustScaler tie (acc 0.769, F1 0.744); patient-baseline
  centering gives perfect paired ranking but lower accuracy (acc 0.654).
- **Stage 4:** short prefix windows score highest, but this is a **data-quality artifact** —
  several patients have genuine signal only in the first ~10 s; later samples are
  backfill-propagated constants. Not a clinical claim.
- **Stage 5:** LOSO ranks the lesional site above non-lesional for **13/13** held-out
  patients (AUC 1.000, a degenerate 2-sample-per-fold metric); accuracy 0.654 because a
  single global threshold still mislabels several folds. GroupKFold (multi-sample folds)
  gives the more realistic AUC 0.817. Supports intra-individual tracking, not a
  population-wide threshold.

## Status & scope

Stages 1–5 (core) are complete. Classifier is **LinearSVC only**; the planned 1D CNN was
not implemented — at N = 26 traces an end-to-end CNN is not trainable, and the intended
approach (pretrain on a larger 1D time-series corpus, then frozen-layer transfer learning)
is described as future work. Optional stages (CNN + jittering augmentation,
feature–Corneometer correlation) are not implemented.

## AI usage disclosure

Claude (Claude Code) was used for concept explanation, literature search, code guidance,
markdown documentation, and result interpretation. Research questions, scope decisions, and
scientific conclusions were made by the author. All AI-generated code was reviewed, tested,
and modified before use.
