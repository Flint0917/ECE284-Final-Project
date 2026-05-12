# Project Brief — ECE284

## Overview

**Topic:** Atopic Dermatitis (AD) monitoring using IDC wearable sensor data
**Course:** ECE284 (graduate-level health tech / biomedical AI), taken as undergrad
**Framing:** Proof-of-concept under data scarcity, NOT clinical validation

---

## Dataset

**Source:** Todorov et al., "Wearable System Using Printed Interdigitated Capacitive Sensor for
Monitoring Atopic Dermatitis in Patients" (IEEE Sensors Journal, 2025), University of Southampton

**Contents:**
- 13 patients
- 30-second traces sampled at 1 Hz (30 data points per trace)
- Lesional / non-lesional labels
- Corneometer measurements (clinical reference)
- TEWL measurements (subset of patients only)

---

## Motivation (Aligned with Methods — TA Requirement)

Practical deployment of IDC sensors for AD monitoring depends on two underexplored
questions:

1. **How short can the measurement window be** while still distinguishing lesional from
   non-lesional skin? This affects patient wearability, compliance, and hardware design.
2. **Can classification models trained on small AD cohorts generalize to unseen patients?**
   Todorov et al. showed 100% within-patient sensitivity using simple thresholding, but
   explicitly noted that a single population-wide threshold could not be defined — suggesting
   the sensor is most useful for intra-individual tracking. This project uses ML under
   LOSO cross-validation to quantify this generalization gap directly.

These two questions are concrete, testable with the available dataset, and map directly to
every stage of the methodology below.

---

## Project Goals

### Core (Must Complete)

1. **Replication** — reproduce Todorov et al. Figure 9(a): lesional vs. non-lesional mean
   capacitance per patient
2. **Normalization comparison** — compare z-score, min-max, and patient-level normalization
   as preprocessing choices for downstream classification
3. **Feature extraction** — extract mean, std, median, range, slope, skewness from each trace
4. **Window length experiment** — test 5, 10, 15, 20, 25, 30 sec windows; motivated by
   patient compliance, real-world wear feasibility, and hardware constraints
5. **Classifier comparison under LOSO + within-patient** — SVM vs. 1D CNN under both
   LOSO (cross-patient) and leave-one-recording-out within-patient evaluation; the gap
   between the two directly quantifies Todorov's intra-individual tracking claim

### Optional (Add if Time Permits)

6. **Data augmentation** — jittering only (Gaussian noise on training set); test whether it
   improves CNN LOSO generalization; reference: Iwana & Uchida, PLOS ONE 2021
7. **ML feature vs. Corneometer correlation** — if Stage 5 features show good discriminative
   power, test whether those same features correlate with Corneometer values as a secondary
   clinical validation; only meaningful if Stage 5 produces interpretable results

---

## Project Pipeline

```
Raw IDC signal (30 points per trace, 13 patients)
              ↓
   Stage 1: Replication (reproduce Todorov Figure 9a)
              ↓
   Stage 2: Normalization comparison (z-score, min-max, patient-level)
              ↓
   Stage 3: Feature extraction (mean, std, median, range, slope, skewness)
              ↓
   Stage 4: Window length experiment (5, 10, 15, 20, 25, 30 sec)
              ↓
   Stage 5: SVM vs 1D CNN — LOSO (cross-patient)
                          — within-patient leave-one-recording-out
            → key output: cross-patient generalization gap
              ↓
   [Optional] Stage 6: CNN + jittering augmentation vs. CNN baseline
              ↓
   [Optional] Stage 7: ML features vs. Corneometer correlation
              ↓
   Final Report Discussion: standardization gap as field-level implication
   (not a standalone stage — 1–2 paragraphs in Discussion section)
```

---

## Evaluation

- **Method:** Leave-One-Subject-Out (LOSO) cross-validation across 13 patients
- **Secondary:** leave-one-recording-out within-patient (for Stage 5 comparison)
- **Metrics:** accuracy, F1-score, AUC-ROC for every model and every condition
- **Reporting:** mean ± std across folds

---

## Classifier Decisions

| Classifier | Role | Rationale |
|---|---|---|
| SVM | Baseline | Robust under small N; handcrafted features; well understood |
| 1D CNN | Main exploratory | End-to-end on raw signal; tests raw vs. feature paradigm |
| ~~GBM~~ | ~~Dropped~~ | With only 6 features and 13 patients, expected to perform similarly to SVM; removed to keep comparison focused and save implementation time |

**The SVM vs. CNN contrast is intentional:** it represents two paradigms — handcrafted
feature classifiers vs. end-to-end signal learning — and the gap between them under small-N
LOSO is itself an informative result.

---

## Required Figures

| Figure | Stage | Content |
|---|---|---|
| 1 | Pipeline | System diagram: raw signal → features → classifiers → evaluation |
| 2 | Stage 1 | Replication of Todorov Fig 9(a): lesional vs. non-lesional per patient |
| 3 | Stage 2 | Bar chart: 3 normalization methods × accuracy |
| 4 | Stage 4 | Line plot: accuracy vs. window length (5–30 sec) |
| 5 | Stage 5 | Grouped bar: LOSO vs. within-patient accuracy for SVM and CNN |

*Optional figures (add if stages completed):*
- Figure 6: Jittering augmentation vs. baseline CNN (Stage 6)
- Figure 7: Feature-Corneometer scatter plot (Stage 7)

---

## Deliverables

| Deliverable | Length | Notes |
|---|---|---|
| Project Update | 2 pages, 1–2 figures | Must show progress, not plans |
| Final Report | 7–10 pages | ACM Large 2-column format |
| GitHub Repo | — | With README |
| Final Oral Assessment | In-person | 15% of grade |

**Final Report sections:** Motivation, Related Work, Project Aim, Methodology, Results,
Discussion, Conclusion

---

## Timeline (Revised from Week 6)

| Week | Target |
|---|---|
| Week 6 (now) | Environment setup + data exploration + Stage 1 replication |
| Week 7 | Stage 2 normalization + Stage 3 features + Stage 4 window length |
| Week 8 | Stage 5 SVM + CNN under LOSO + within-patient comparison |
| Week 9 | Optional stages if time; Discussion write-up; finalize report + oral prep |
