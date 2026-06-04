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

### GroupKFold Cross-Validation (Sensitivity Analysis)
GroupKFold is another patient-aware validation method. It still uses patient ID as the group label, so recordings from the same patient cannot be split between training and testing.

**Difference from LOSO:** LOSO holds out exactly one patient per fold. GroupKFold holds out several patients per fold. With 13 patients, a 5-fold GroupKFold split usually tests on about 2-3 patients per fold, or 4-6 traces. This makes fold-level AUC less binary than LOSO because each test fold contains more than one lesional/non-lesional pair.

**Role in project:** Secondary sensitivity analysis, not a replacement for LOSO. LOSO remains the primary evaluation because it directly tests unseen-patient generalization one patient at a time. GroupKFold is added to check whether the main trend is stable when the held-out test set contains multiple patients.

**Experimental result (Stage 5, fold-safe patient-baseline centering — current code):** GroupKFold (n=5) gave Accuracy 0.700 ± 0.113, F1 0.651 ± 0.146, AUC 0.817 ± 0.092. The LOSO "AUC" for the same setup is a degenerate 1.000 (= pairwise ranking 13/13; see the two-dimensions note below). GroupKFold's AUC is meaningful because each fold holds 4–6 samples, whereas LOSO's per-fold AUC is binary. GroupKFold accuracy std (0.113) is lower than LOSO's (0.231) because larger test folds reduce fold-to-fold variance. *(Earlier drafts of this file quoted 0.611 / 0.769 from a superseded z-score Stage 5 run; corrected Session 7 to match `results/stage5_metrics.csv`.)*

### Per-Patient Prediction Table
A per-patient table reports what happened in each held-out LOSO fold instead of hiding everything inside a mean score.

Recommended columns:
- Patient ID
- True lesional score
- True non-lesional score
- Predicted lesional label
- Predicted non-lesional label
- Correct ranking? \(lesional score higher than non-lesional score, if lesional is coded as the positive class\)
- Fold accuracy

**Why it matters:** In this dataset, each LOSO fold has only two test samples, so fold-level AUC is either 0 or 1. The per-patient table makes the result more transparent by showing exactly which patients were ranked correctly or incorrectly.

### Pooled Out-of-Fold AUC (Optional Reporting)
Pooled out-of-fold AUC collects the prediction scores from all LOSO test folds, then computes one AUC across all 26 held-out predictions.

**Benefit:** It avoids reporting only binary fold-level AUC values.

**Experimental result (Stage 5, patient-baseline):** pooled-LOSO AUC = **0.775** over all 26 held-out scores — far more informative than the degenerate per-fold 1.000, and consistent with GroupKFold's 0.817 (verified by `src/verify_loso_auc.py`).

**Statistical caveat (Airola et al. 2011):** pooling LOO/LOSO predictions carries a mild *negative* bias, while averaging across folds is unbiased but high-variance at small N. This is why pooled-LOSO (0.775) reads slightly below GroupKFold-averaged (0.817); the true discriminability is plausibly ~0.82.

**Caution:** It compares scores across patients, while Todorov et al. argue that the IDC sensor is most useful for intra-individual tracking because no single population-wide threshold was found. Therefore, pooled out-of-fold AUC can be useful as a secondary summary, but it should not replace the per-patient interpretation.

### LOSO vs GroupKFold vs Pooled — Two Independent Dimensions

A common confusion: LOSO, GroupKFold, and "pooled AUC" are **not** three options on one axis. There are two separate dimensions.

**Dimension A — how patients are split (CV scheme):** LOSO holds out 1 patient/fold (2 test traces); GroupKFold n=5 holds out 2–3 patients/fold (4–6 test traces). Both keep a patient's traces together.

**Dimension B — when the AUC is computed (aggregation):** *per-fold-then-average* (one AUC inside each fold, then mean) vs *pooled* (collect all test scores, compute one AUC).

Combining them explains every AUC in this project:

| Combination | Samples per AUC | Result | Usable? |
|---|---|---|---|
| LOSO + per-fold average | 2 (1L+1NL) | 1.000 | ✗ degenerate — AUC can only be 0 or 1 |
| LOSO + pooled | 26 | 0.775 | ✓ same split, better aggregation |
| GroupKFold + per-fold average | 4–6 | 0.817 | ✓ bigger folds, AUC not degenerate |

The misleading 1.000 comes specifically from the **LOSO + per-fold** combination, not from LOSO itself. Fixing *either* dimension (pool the scores, or use bigger GroupKFold folds) recovers the true ~0.78–0.82.

**Why n=5 for GroupKFold:** folds ≤ #patients (13). 5 is a bias–variance compromise — ~11 patients train (enough data) while each test fold holds 4–6 traces (AUC non-degenerate). 10 folds would shrink some test folds back to 2 samples (degenerate again); 2 folds would starve training. 5- and 10-fold are also the conventional ML defaults.

**per-fold vs pooled — which to use:** per-fold averaging gives a usable ±std when folds are large (most common in papers); pooling is the standard/necessary choice for LOO/LOSO where per-fold is degenerate. They are different estimators, not statistically equal (Forman & Scholz 2010; Airola et al. 2011).

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
- **Role in project:** ~~Main model~~ **Dropped** (see progress log) — redundant with SVM given only 6 features and N=13
- **Field evidence:** The nocturnal scratch paper found GBM beat CNN on a similarly small dataset, attributing this to GBM's efficiency with limited data

### 1D CNN (1D Convolutional Neural Network)
Learns features directly from the raw 30-point signal without manual feature engineering. A convolutional filter slides along the signal looking for patterns automatically.

- **Strengths:** Can discover patterns humans would miss
- **Weaknesses:** Large over-parameterized CNNs need lots of data and overfit at small N. *Small* CNNs with Global Average Pooling + dropout are strongly regularized and can generalize even at small N (see result below).
- **Role in project (Track A — built, Session 7):** TinyCNN (2× Conv1d 8→16, GAP, dropout 0.3, FC), trained from scratch on the 26 IDC traces under LOSO + GroupKFold, 5 seeds, with a jittering variant.
  - **Result:** the CNN did NOT overfit (train 0.846 vs test 0.769) and matched/beat the SVM (GroupKFold AUC 0.871 vs 0.822). GAP averages the trace, so the CNN rediscovers the mean-capacitance-level feature — the SVM's #1 feature — and the two tie because the discriminative signal is the simple per-site level, not subtle shape. The CNN sees strictly more information than the 6 handcrafted features yet only matches the SVM → signal simplicity, not an information deficit.
  - **Track B — transfer learning: NOT pursued.** The pre-registered condition (overfit AND lose to SVM) did not fire, so transfer learning was unnecessary. Retained as described methodology / future work (see next entry).

### Self-Supervised Transfer Learning (CNN Track B — described, not run)

*Not implemented:* Track A already generalized (CNN ≈ SVM, no overfit), so transfer learning was not needed on this dataset. Kept here as methodology / future work for a larger or harder dataset where a from-scratch CNN would overfit. The reasoning still earns oral credit ("here's what I would do, and why it wasn't necessary here").

**Self-supervised pretraining:** instead of human labels, the model learns from the signal itself — e.g. reconstruct a compressed window (autoencoder) or forecast its later points from earlier ones. The conv layers thereby learn a general "what 1D skin-electrical signals look like" feature extractor, with no dependence on any external dataset's labels.

**Transfer:** freeze the pretrained conv layers and retrain only the final layer(s) on the 26 IDC traces. Because most layers stay fixed, far less target data is needed — directly addressing N=26.

**Source dataset:** WESAD EDA (skin conductance, 4 Hz; window to 30 pts to match the IDC trace length) is the primary candidate — most domain-similar. UCR/UEA archive is the methodological-standard alternative (Fawaz et al. 2018). No public skin-capacitance-on-AD dataset exists besides Todorov, so a *similar* 1D signal must be used; self-supervised pretraining avoids the source–target *task* mismatch that label-based transfer would risk.

**Claim discipline:** frame as *mitigating* data scarcity, not *solving* it. The real benchmark is the SVM baseline, not the (expectedly weak) from-scratch CNN; report whichever ordering actually occurs.

---

## Preprocessing Methods

### Normalization (Three Methods to Compare)

**Z-score (feature-level):** Extract 6 raw features first, then inside each LOSO fold fit a StandardScaler on the training-fold feature matrix and apply it to both train and test. Normalizes each feature column across training patients (not per-trace), so mean and std features remain informative.
- **Experimental result (Stage 3):** Accuracy 0.769 ± 0.249, F1 0.744 ± 0.350, AUC 0.846 ± 0.361.
- Contrast with old trace-level z-score: per-trace z-scoring forced each trace to mean=0, std=1 by construction, making those two features trivially constant and destroying their discriminative value.

**Robust scaler (feature-level):** Same fold-safe approach as z-score, but uses RobustScaler (subtracts median, divides by IQR). More resistant to the large inter-patient absolute baseline shifts that dominate this dataset.
- **Experimental result (Stage 3):** Accuracy 0.769 ± 0.249, F1 0.744 ± 0.350, AUC 0.846 ± 0.361. Identical to z-score in this dataset — with only 13 patients and 6 features, the IQR-based centering offers no practical advantage over mean/std centering.

**Training-only patient baseline normalization:** Learn one scalar baseline per patient using training traces only inside each CV fold. Training traces subtract their own learned patient baseline. Test traces subtract that patient's learned training baseline if the patient appears in the training fold; otherwise, as in LOSO, they subtract the training global mean fallback.
- Removes training-patient baseline offsets without using the held-out patient's own traces
- Leakage-free: LOSO test patients use only a fallback baseline computed from the 12 training patients
- **Experimental result (Stage 3):** Accuracy 0.654 ± 0.231, F1 0.462 ± 0.444, AUC 1.000 ± 0.000. This should not be called generic "patient-level normalization"; it is fold-safe patient-baseline centering.
- **Key implication:** The sensor's paired lesional/non-lesional ranking survives fold-safe centering, but threshold accuracy remains limited for unseen patients. Do not compute a held-out patient's baseline from their own paired traces unless clearly labeling that as oracle patient-level normalization/intra-patient calibration.

### Feature Extraction (For SVM and GBM)
From each 30-point trace, extract summary statistics:
- **Mean** — average capacitance
- **Standard deviation** — variability
- **Median** — middle value (robust to outliers)
- **Range** — max minus min
- **Slope** — linear trend over the window (computed via `np.polyfit` degree 1)
- **Skewness** — asymmetry of the distribution

These six numbers replace the raw signal as input to SVM/GBM.

**Experimental result (Stage 2, Random Forest importance, raw traces):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | Mean | 0.257 |
| 2 | Median | 0.229 |
| 3 | Slope | 0.179 |
| 4 | Std | 0.136 |
| 5 | Range | 0.118 |
| 6 | Skewness | 0.081 |

All six features carry non-zero importance because features are extracted from raw (un-normalized) traces. Mean and Median being top-ranked reflects the consistent 2–5 pF non-lesional capacitance advantage seen in Stage 1. Slope being #3 indicates detectable linear drift within traces that differs between lesional and non-lesional skin. For Stage 7 (feature–Corneometer correlation), use raw or fold-safe patient-baseline centered features since all features remain informative on raw traces.

### Window Length Experiment
The window length experiment asks whether the full 30-second acquisition is necessary. Todorov et al. showed that 30-second measurements were stable and repeatable, but that does not prove that 30 seconds are required for classification.

Two window families should be tested:

**Prefix windows:** Use the beginning of the trace.
- 0-5 s
- 0-10 s
- 0-15 s
- 0-20 s
- 0-25 s
- 0-30 s

**Question answered:** If the user starts recording immediately after wearing the sensor, how much time is needed?

**Post-settling windows:** Skip the early skin-sensor contact period before extracting the window.
- 5-10 s, 5-15 s, 5-20 s, 5-25 s, 5-30 s
- 10-15 s, 10-20 s, 10-25 s, 10-30 s

**Question answered:** If the first 5-10 seconds contain contact-settling drift, can a shorter stable segment still preserve the lesional/non-lesional difference?

**Experimental result (Stage 4):** Short prefix windows (0–5, 0–10 s) achieved the highest performance (~73% accuracy, AUC 0.923–0.846). Performance degraded as prefix length increased. Post-settling windows performed worse than equal-duration prefix windows at every matched duration; post-10s windows were near-random.

**Data quality confound discovered:** At least three patients have genuine data in only the first ~10 seconds; remaining timesteps are backfill-propagated constants. Post-settling windows for these patients produce uninformative constant feature vectors (range, slope, skewness all collapse). This means the expected interpretation rules below cannot be applied cleanly — the dataset needs to be inspected for trace completeness before the settling vs. duration question can be answered.

**Intended interpretation (blocked by confound):**
- Prefix ≈ post-settling at same duration → settling is not the bottleneck.
- Post-settling >> prefix at same duration → early contact dynamics limit short windows.
- Only long windows work → the IDC offset needs time to develop, not just stabilize.

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

**LOSO-specific caution:** Because each LOSO fold has only two test samples, fold-level AUC is binary. Interpret mean LOSO AUC as the fraction of held-out patients whose lesional/non-lesional pair was ranked correctly.

**For this project:** report accuracy, F1-score, and AUC-ROC for every model. Also include a per-patient prediction table for LOSO and a GroupKFold sensitivity result to make the small-N limitation explicit.

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

---

## Key References (verified Session 7)

- **Todorov et al. (2025)** — *Wearable System Using Printed Interdigitated Capacitive Sensor for Monitoring Atopic Dermatitis in Patients*, IEEE Sensors Journal. Primary dataset.
- **Forman & Scholz (2010)** — *Apples-to-apples in cross-validation studies: pitfalls in classifier performance measurement*, ACM SIGKDD Explorations 12(1):49–57. DOI 10.1145/1882471.1882479. Pooling-vs-averaging of cross-validated metrics.
- **Airola et al. (2011)** — *An experimental comparison of cross-validation techniques for estimating the area under the ROC curve*, Comput. Stat. Data Anal. 55(4):1828–1844 (conference version: *A comparison of AUC estimators in small-sample studies*, PMLR v8, 2010). Small-sample AUC bias; pooling negative bias; leave-pair-out. *Verify exact pages before citing.*
- **Fawaz et al. (2018)** — *Transfer learning for time series classification*, IEEE Int. Conf. Big Data (arXiv 1811.01533). CNN transfer for TSC; source–target similarity governs success.
- **Iwana & Uchida (2021)** — *An empirical survey of data augmentation for time series classification with neural networks*, PLOS ONE. Jittering augmentation (Track A).
