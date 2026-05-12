# Claude Code Project Brief — ECE284 Final Project

## Project

Atopic Dermatitis (AD) monitoring using IDC wearable sensor data.
Dataset: Todorov et al. — 13 patients, 30-second capacitance traces at 1 Hz,
lesional / non-lesional labels, with Corneometer and TEWL references.

Framing: proof-of-concept under data scarcity, not clinical validation.

## Folder Layout

- @data/ — sensor and reference CSVs (read these directly to infer format)
- @WearSystemIDCforAD/ — original raw dataset; do not modify
- @src/ — all Python scripts go here
- @results/ — all figures and metric tables go here
- @base knowledge/ — project briefs, glossary, progress log

## Pipeline Stages

1. **Replication** — reproduce Todorov Fig. 9(a) lesional vs non-lesional mean capacitance
2. **Normalization comparison** — z-score, min-max, patient-level
3. **Feature extraction** — mean, std, median, range, slope, skewness per trace
4. **Window length experiment** — 5, 10, 15, 20, 25, 30 sec
5. **Classifier comparison** — SVM vs 1D CNN under LOSO cross-validation
6. **(Optional) Data augmentation** — jittering on CNN training set
7. **(Optional) Feature–Corneometer correlation**

## Evaluation

- LOSO cross-validation across 13 patients (13 folds)
- Metrics: accuracy, F1-score, AUC-ROC — reported as mean ± std across folds

## Coding Standards

- Python 3.11, conda environment `ece284`
- Stack: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, torch
- Clean code with clear comments
- Handle NaN by forward-filling within each patient column
- Reproducibility: set `random_state=42` everywhere applicable
- Every script saves its figure(s) to @results/ and prints a metrics summary
- One script per stage in @src/ (e.g. `stage1_replication.py`, `stage2_normalization.py`)

## Version Control

GitHub repo: https://github.com/Flint0917/ECE284-Final-Project

- Commit after each stage runs successfully
- Commit messages: `stage<N>: <short description>` (e.g. `stage2: add normalization comparison`)
- Push to `main` after each working commit
- Include @results/ figures in commits so progress is visible
- Add a `.gitignore` covering: `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, conda env folders

## How to Help

- Read data files directly to infer format; don't ask me about column names
- Explain ML concepts in code comments when they appear (LOSO, AUC-ROC, etc.) since I'm an undergrad in a grad course
- Flag scope creep — if a request goes beyond the current stage, say so
- Don't write prose for the report; that's my job. Code + comments only.
- Don't suggest paid tools

## AI Usage Disclosure

Claude Code is part of the documented methodology. Keep code attributable —
don't silently restructure files I wrote myself without flagging it.