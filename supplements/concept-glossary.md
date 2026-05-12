# Concept Glossary — ECE284 Project

Reference document for technical concepts used throughout the project. Written for an undergrad audience — explanations build from basics.

---

## Cross-Validation Methods

### Random Train/Test Split
Randomly assigns 80% of all recordings to training and 20% to testing.

**Problem for this project:** With multiple recordings from the same patients, recordings from Patient 5 may end up in both training and testing. The model is then tested on someone it has already "seen," which inflates accuracy and isn't clinically meaningful.

### LOSO (Leave-One-Subject-Out) Cross-Validation
Train on all patients except one, test on the held-out patient, then rotate. With 13 patients, this means 13 rounds — each patient gets a turn as the test case. Final result is the average across all 13 rounds.

**Why it matters:** Simulates how the model would perform on a genuinely new patient. This is the clinically relevant scenario. **This is the method used in this project.**

**Dataset-specific note:** With 13 patients and 2 traces per patient (one lesional, one non-lesional), each LOSO fold has exactly 24 training samples and 2 test samples. AUC-ROC per fold is therefore binary (either 0 or 1), so the mean AUC across folds equals the fraction of folds where the model correctly ranked the lesional score above the non-lesional. Accuracy per fold can only be 0%, 50%, or 100%. This high variance is expected and is inherent to the small dataset, not a code error.

---

## Classifier Models

### SVM (Support Vector Machine)
Takes hand-crafted features (mean, standard deviation, range, slope of the signal) and draws the best possible boundary line separating lesional from non-lesional in feature space.

- **Strengths:** Simple, reliable on small datasets
- **Weaknesses:** Limited to linear or kernel-based boundaries
- **Role in project:** Baseline classifier

### GBM (Gradient Boosting Machine)
Also takes hand-crafted features, but instead of one boundary builds many small decision trees. Each tree corrects the mistakes of the previous one. Final prediction combines all trees.

- **Strengths:** Generally outperforms SVM on small/medium datasets, handles nonlinear patterns
- **Weaknesses:** More hyperparameters to tune
- **Role in project:** Main model
- **Field evidence:** The nocturnal scratch paper found GBM beat CNN on a similarly small dataset, attributing this to GBM's efficiency with limited data

### 1D CNN (1D Convolutional Neural Network)
Learns features directly from the raw 30-point signal without manual feature engineering. A convolutional filter slides along the signal looking for patterns automatically.

- **Strengths:** Can discover patterns humans would miss
- **Weaknesses:** Needs large datasets to generalize; with N=13 will likely overfit
- **Role in project:** Exploratory model — finding that CNN underperforms simpler models is itself a meaningful result

---

## Preprocessing Methods

### Normalization (Three Methods to Compare)

**Z-score normalization:** Subtract the mean and divide by standard deviation. Each signal ends up with mean 0 and standard deviation 1.
- Good for comparing across patients with different baseline capacitance values
- **Experimental result (Stage 2):** Accuracy 0.538 ± 0.237, AUC 0.769 ± 0.421. Near-chance accuracy because z-scoring destroys the mean and std features by construction — both become constants (0 and 1) for every trace, leaving only median, range, slope, and skewness as informative.

**Min-max scaling:** Rescale each signal to a 0–1 range.
- Simpler than z-score
- Sensitive to outliers (one extreme value distorts the entire scale)
- **Experimental result (Stage 2):** Accuracy 0.462 ± 0.308, AUC 0.462 ± 0.499. Performs at chance. Additionally destroys the range feature (always = 1 after scaling) and is unstable on the near-flat 30-point traces.

**Patient-level normalization:** Subtract each patient's own mean across all their measurements (both lesional and non-lesional traces combined).
- Removes inter-patient differences entirely
- Focuses analysis on within-patient lesional vs. non-lesional contrast
- **Experimental result (Stage 2):** Accuracy 1.000 ± 0.000, AUC 1.000 ± 0.000. Perfect LOSO separation. After subtracting the patient baseline, the lesional/non-lesional difference (non-lesional is consistently ~2–4 pF higher) becomes a clean positive/negative mean offset that a linear SVM separates perfectly.
- **Key implication:** The sensor's discriminative signal lives almost entirely in the per-patient baseline offset, not in cross-patient absolute values. This corroborates Todorov's finding that no population-wide threshold exists — the sensor is most useful for intra-individual tracking.

### Feature Extraction (For SVM and GBM)
From each 30-point trace, extract summary statistics:
- **Mean** — average capacitance
- **Standard deviation** — variability
- **Median** — middle value (robust to outliers)
- **Range** — max minus min
- **Slope** — linear trend over the window (computed via `np.polyfit` degree 1)
- **Skewness** — asymmetry of the distribution

These six numbers replace the raw signal as input to SVM/GBM.

**Experimental result (Stage 3, Random Forest importance, z-score norm):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | Range | 0.306 |
| 2 | Slope | 0.288 |
| 3 | Skewness | 0.205 |
| 4 | Median | 0.201 |
| 5 | Std | 0.000 |
| 6 | Mean | 0.000 |

Mean and Std score zero because z-score normalization makes them constants for every trace. Range and Slope being top-ranked indicates the traces are not flat — there is detectable amplitude spread and linear drift that differs between lesional and non-lesional skin. For Stage 7 (feature–Corneometer correlation), use raw or patient-level features rather than z-score, since Mean and Std will be uninformative under z-score.

### Data Augmentation (Addresses Small-N Problem)

**Jittering:** Add small random Gaussian noise to each signal. Creates slightly varied versions of each trace.

**Window slicing:** Take shorter sub-windows from the 30-second trace. Already part of the window length experiment — doubles as augmentation.

**Scaling:** Multiply the signal by a small random factor (e.g., 0.9–1.1) to simulate sensor gain variation.

**Tuning caution:** Too much augmentation destroys signal; too little doesn't help. Magnitude must be tuned during experiments.

---

## Evaluation Metrics

**Accuracy:** Fraction of predictions that are correct. Simple but can be misleading if classes are imbalanced.

**F1-score:** Harmonic mean of precision and recall. Better than accuracy when classes are imbalanced.

**AUC-ROC:** Area under the receiver operating characteristic curve. Measures the model's ability to rank lesional examples above non-lesional ones. Threshold-independent.

**For this project:** report all three for every model.

---

## Standardization in AD Wearables (Field-Level Concept)

Two distinct standardization problems exist in this field:

### 1. Clinical Outcome Standardization
No single agreed-upon score exists for validating wearables against AD. Different studies use different references:
- **SCORAD** — Severity Scoring of Atopic Dermatitis
- **EASI** — Eczema Area and Severity Index
- **Corneometer** — clinical hydration measurement (used in this project's dataset)
- **TEWL** — transepidermal water loss (subset of this project's dataset)

Even EASI shows only moderate inter-observer reliability.

### 2. Sensor / Pipeline Standardization
Different studies use different sensors, preprocessing, and evaluation methods, making published accuracy numbers incomparable across studies.

### This Project's Contribution
Does not *solve* standardization, but *characterizes* it by:
- Explicitly comparing three normalization methods
- Using consistent LOSO evaluation
- Aligning features against Corneometer as clinical reference
- Discussing what a standardized protocol would require

---

## Why Not Cross-Dataset Comparison

The IDC sensor (this project) measures **skin capacitance** — hydration and barrier function. The nocturnal scratch dataset measures **wrist motion** — scratching behavior. These are fundamentally different phenomena with different reference standards, so direct signal-level comparison would not be scientifically valid.

The scratch paper is still useful as **parallel evidence** in Related Work — it independently found GBM outperforms CNN on a similarly small AD dataset.
