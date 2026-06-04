# Progress Log — ECE284 Project

---

## Scope Decisions

| Decision | Outcome |
|---|---|
| Cross-dataset comparison (scratch paper) | Dropped — different modality; kept as Related Work only |
| Multi-dataset IDC comparison | Dropped — no second public IDC-on-AD-patients dataset exists |
| GBM classifier | Dropped — redundant with SVM given 6 features + 13 patients |
| Stage 8 standardization (standalone) | Dropped — becomes 1–2 paragraphs in Discussion |
| LOSO cross-validation | Adopted — prevents patient identity leakage |
| Within-patient evaluation | Added to Stage 5 — quantifies Todorov's intra-individual tracking claim |
| Motivation framing | Corrected — two testable questions (window length + generalization gap), not broad field problems |
| Stage 6 augmentation | Optional — jittering only; add if Stages 1–5 complete |
| Stage 7 feature-Corneometer | Optional, results-dependent — only if Stage 5 features are interpretable |
| Per-patient prediction table | Added — makes LOSO results interpretable despite binary fold-level AUC |
| GroupKFold sensitivity analysis | Added — keeps patient grouping intact while testing whether results are stable with multi-patient test folds |
| Signal length extension | Added — compare prefix windows against post-settling windows to separate short-duration effects from early contact-settling drift |
| Fold-safe patient-baseline centering fix | Corrected — baselines are learned after each CV split from training patients only; unseen test patients use the training global mean fallback |
| Stage 2/3 reorder | Feature extraction moved to Stage 2; normalization comparison moved to Stage 3 — features must be characterized before choosing which normalization preserves them |
| Min-max → RobustScaler | Min-max replaced with RobustScaler (median/IQR); z-score and robust scaler now applied feature-level (not trace-level) inside LOSO fold |
| Primary normalization = feature z-score | Resolved (Session 7). Feature z-score is the primary method (higher acc 0.769 / F1 0.744); patient-baseline kept as the intra-individual-tracking variant. Patient-baseline's LOSO "AUC 1.000" is a degenerate metric, not a real advantage — see AUC clarification below |
| 1D CNN reactivated as two-track | Adopted (Session 7). CNN is back in scope, primarily for the oral exam. Track A = from-scratch 1D CNN on N=26 (LOSO + jittering) to demonstrate small-N overfitting; Track B = self-supervised transfer learning (pretrain a 1D CNN on a larger external 1D signal, freeze conv layers, retrain only the final layer(s) on the 26 IDC traces). Motivated by TA transfer-learning advice |
| CNN pretraining via self-supervision | Adopted (Session 7). Pretrain via signal reconstruction (autoencoder) or forecasting on *unlabeled* EDA windows, NOT external stress labels — sidesteps source-target task mismatch (Fawaz 2018) |
| CNN pretraining source dataset | Candidates identified (Session 7). WESAD EDA (skin conductance, most domain-similar) as primary; UCR/UEA archive as methodological-standard alternative. Confirmed no public skin-capacitance-on-AD dataset exists besides Todorov, so transfer must use a *similar* (not identical) 1D signal |
| CNN claim framing | Decided (Session 7). CNN positioned as a data-scarcity *mitigation* attempt: from-scratch CNN failure diagnoses the bottleneck; transfer learning tests a remedy. Claim must be "mitigate/narrow," NOT "solve" scarcity. True benchmark is the SVM baseline, not the (expectedly weak) from-scratch CNN |
| Track A outcome → Track B dropped | Resolved by data (Session 7). Track A CNN did NOT overfit and matched/beat the SVM (GKF AUC 0.871 vs 0.822); pre-registered Track B trigger did not fire, so Track B is not pursued. New framing: a small well-regularized CNN (GAP + dropout) rediscovers the mean-level feature and ties the SVM at N=26 — end-to-end learning needs no transfer learning when the discriminative signal is this simple. Stronger and more honest than the original "CNN fails" narrative |

---

## Stage Status

| Stage | Priority | Status |
|---|---|---|
| 1. Replication (Todorov Fig 9a) | Core | **Complete** — results/stage1_replication.png |
| 2. Feature extraction (mean, std, median, range, slope, skewness) | Core | **Complete** — results/stage2_features.png |
| 3. Normalization comparison (z-score, robust scaler, fold-safe patient-baseline centering) | Core | **Complete (corrected)** — results/stage3_normalization.png |
| 4. Window length experiment (prefix + post-settling windows) | Core | **Complete** — results/stage4_window_length.png, results/stage4_window_length.csv |
| 5. Evaluation expansion — LOSO + GroupKFold + within-patient | Core | **Complete** — results/stage5_evaluation.png, results/stage5_loso_per_patient.csv, results/stage5_metrics.csv |
| 6a. 1D CNN — Track A (from-scratch + jittering) | Core | **Complete** — results/stage6_cnn_trackA.png; CNN ties/beats SVM, NO overfit |
| 6b. 1D CNN — Track B (self-supervised transfer learning) | Optional | **Not pursued** — Track A's pre-registered trigger (overfit AND loses to SVM) did not fire; retained as described methodology / future work |
| 7. ML features vs. Corneometer correlation | Optional, results-dependent | Not started |

---

## Experimental Findings

### Stage 1 — Replication
- Non-lesional capacitance is consistently higher than lesional for all 13 patients (range ~40–55 pF).
- Patient-level separation is clear visually; population-wide threshold would fail because baseline shifts vary widely across patients (Patient 1 ≈ 40 pF vs. Patient 4 ≈ 49 pF lesional).
- Replicates the direction and per-patient ordering of Todorov Fig 9(a). ✓

### Stage 2 — Feature Importance (Random Forest, raw traces, N=26)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Mean | 0.2572 |
| 2 | Median | 0.2288 |
| 3 | Slope | 0.1788 |
| 4 | Std | 0.1360 |
| 5 | Range | 0.1180 |
| 6 | Skewness | 0.0812 |

All six features carry non-zero importance because features are extracted from raw (un-normalized) traces. Mean and Median being top-ranked reflects the 2–5 pF non-lesional capacitance advantage visible in Stage 1. Slope at #3 indicates detectable linear drift that differs between lesional and non-lesional skin.

### Stage 3 — Normalization (LinearSVC, feature-level, LOSO) — Corrected

| Method | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| z-score (feature-level) | 0.769 ± 0.249 | 0.744 ± 0.350 | 0.846 ± 0.361 |
| robust scaler (feature-level) | 0.769 ± 0.249 | 0.744 ± 0.350 | 0.846 ± 0.361 |
| training-only patient baseline normalization | 0.654 ± 0.231 | 0.462 ± 0.444 | **1.000 ± 0.000** |

**Design change:** min-max replaced with RobustScaler (median/IQR). Z-score and robust scaler are now feature-level (StandardScaler/RobustScaler fit on training-fold features inside LOSO loop), not trace-level. This preserves all six features as informative — old per-trace z-score forced mean=0 and std=1 for every trace, making those features constant and useless.

**Bug fixed:** The fold-safe implementation now splits first, learns patient baselines only from training patients, and uses the training global mean fallback for unseen LOSO test patients. It does not compute a held-out patient's baseline from that patient's own traces.

Key insight: Training-only patient baseline normalization gives the strongest paired LOSO ranking (AUC 1.000), but threshold accuracy remains lower (0.654). Feature-level z-score and robust scaler perform equally; the small N means IQR-based centering offers no practical advantage over mean/std centering.

### Stage 4 — Window Length (LinearSVC, z-score per segment, LOSO)

**Prefix windows (0→N):**

| Window | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| 0–5 s | **0.731 ± 0.317** | 0.692 ± 0.402 | **0.923 ± 0.266** |
| 0–10 s | **0.731 ± 0.317** | 0.641 ± 0.443 | 0.846 ± 0.361 |
| 0–15 s | 0.615 ± 0.288 | 0.615 ± 0.366 | 0.769 ± 0.421 |
| 0–20 s | 0.577 ± 0.385 | 0.538 ± 0.444 | 0.692 ± 0.462 |
| 0–25 s | 0.577 ± 0.331 | 0.564 ± 0.401 | 0.462 ± 0.499 |
| 0–30 s | 0.538 ± 0.237 | 0.410 ± 0.396 | 0.769 ± 0.421 |

**Post-5s windows (5→N):**

| Window | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| 5–10 s | 0.731 ± 0.249 | 0.615 ± 0.431 | 0.846 ± 0.361 |
| 5–15 s | 0.577 ± 0.180 | 0.410 ± 0.396 | 0.615 ± 0.487 |
| 5–20 s | 0.500 ± 0.196 | 0.385 ± 0.366 | 0.308 ± 0.462 |
| 5–25 s | 0.346 ± 0.231 | 0.205 ± 0.308 | 0.308 ± 0.462 |
| 5–30 s | 0.231 ± 0.317 | 0.128 ± 0.308 | 0.154 ± 0.361 |

**Post-10s windows (10→N):**

| Window | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| 10–15 s | 0.269 ± 0.249 | 0.205 ± 0.308 | 0.115 ± 0.211 |
| 10–20 s | 0.500 ± 0.196 | 0.385 ± 0.366 | 0.500 ± 0.480 |
| 10–25 s | 0.192 ± 0.243 | 0.103 ± 0.241 | 0.077 ± 0.266 |
| 10–30 s | 0.346 ± 0.303 | 0.179 ± 0.336 | 0.192 ± 0.369 |

**Key finding — data quality confound:** Short prefix windows (0–5, 0–10) perform best (~73% acc, AUC 0.923). Performance degrades as window grows; post-settling windows are worse than equal-duration prefix windows at every duration; post-10s windows are near-random. Root cause: several patients (at minimum S-02, S-03, S-04) have genuine data only in the first ~10 seconds; the rest are backfill-propagated constants. Post-10s segments for these patients produce uninformative constant feature vectors. This confounds the settling analysis — the experiment cannot cleanly answer the settling vs. duration question without first identifying which patients have complete traces.

**Implication for report:** Frame as a data-quality finding. Do not claim that short windows are better clinically — the result is an artifact of trace-length variation in the dataset.

### Stage 5 — Evaluation Expansion (LinearSVC, training-only patient baseline normalization, full 30 s)

| Method | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| LOSO — 13 folds (primary) | 0.654 ± 0.231 | 0.462 ± 0.444 | 1.000 ± 0.000 |
| GroupKFold n=5 (sensitivity) | 0.700 ± 0.113 | 0.651 ± 0.146 | 0.817 ± 0.092 |

**Within-patient correct ranking:** 13/13 patients (100%).

Key observations:
- LOSO AUC 1.000 = correct-ranking fraction 13/13 = 1.000. These are numerically equal because each LOSO fold has exactly 1L + 1NL, making fold-level AUC the binary "lesional ranked above non-lesional" indicator.
- GroupKFold AUC 0.817 is more conservative — computed over 4–6 samples per fold (multiple patient pairs). This is the more realistic discriminability estimate.
- GroupKFold accuracy std is lower (0.113 vs. 0.231 for LOSO) because larger test folds reduce fold-to-fold volatility; LOSO AUC std is 0.000 only because every fold ranks correctly.
- 13/13 correct rankings support Todorov's intra-individual tracking claim: without any test-patient data during training, the model ranks the higher-capacitance (lesional) site above the non-lesional site for every held-out patient. Accuracy is only 0.654 because a single global decision threshold still mislabels traces in ~5 folds — consistent with Todorov's finding that a population-wide threshold cannot be defined.
- Do not claim AUC = 1.000 as clinical validation. It is a proof-of-concept paired-ranking result on N=13 with binary LOSO fold evaluation.
- "Within-patient correct ranking 13/13" is the LOSO ranking count, NOT a separate leave-one-recording-out evaluation. With only 1 lesional + 1 non-lesional trace per patient, true within-patient LORO is not possible; the brief's third evaluation should be reframed accordingly in the report.

### AUC Verification — instructor's leakage concern (Session 7)

Instructor flagged on the progress update: *"I suspect the AUC is an error. Double check it is not computed on training data or something."* Verified with `src/verify_loso_auc.py`:

- **No leakage.** In 0/13 LOSO folds did the held-out patient use its own baseline — every held-out patient gets the *training* global-mean fallback. Normalizer, StandardScaler, and SVM are all fit on training patients only.
- **Scored on test data.** Every fold's AUC comes from `decision_function(X_test)` vs held-out labels, never training data.
- **The "1.000" is a degenerate metric, not a real AUC.** Each LOSO fold has exactly 1 lesional + 1 non-lesional test sample, so per-fold `roc_auc_score` can only return 0.0 or 1.0. "AUC 1.000" = mean of 13 binary ranking indicators = **pairwise ranking accuracy 13/13**.
- **Genuine discriminability:** pooled-LOSO AUC = **0.775** (one ROC over all 26 test scores); GroupKFold AUC = **0.817**. These agree, and the pooled value sitting slightly below GroupKFold matches the known *negative bias of pooling* in LOO/LOSO (Airola et al. 2011).
- **Acc-vs-AUC gap is evidence against leakage:** accuracy 0.654 ≪ "AUC" 1.000. Under oracle leakage, accuracy would also be ≈1.0.
- **History (be candid with instructor):** an earlier version *did* have a real leakage bug (held-out patient's own baseline), since fixed by moving baseline fitting inside the fold.

**Project-wide reporting fix:** ALL per-fold LOSO AUCs (Stages 3, 4, 5) are degenerate 1-vs-1 ranking fractions. Report LOSO as "pairwise ranking accuracy N/13" and use GroupKFold AUC (or pooled-LOSO AUC) as the real discriminability metric.

### Stage 3 — Evaluation expansion (Session 7)

Stage 3 now reports, per normalization method, three views: LOSO (acc/F1 + pairwise ranking), GroupKFold n=5 (headline AUC), and pooled-LOSO AUC. Key numbers (LinearSVC):

| Method | LOSO acc | GKF acc | **GKF AUC** | pooled-LOSO | LOSO rank (degenerate) |
|---|---|---|---|---|---|
| feature z-score | 0.769 | 0.750 | **0.822** | 0.781 | 0.846 |
| robust scaler | 0.769 | 0.750 | 0.822 | 0.793 | 0.846 |
| patient baseline | 0.654 | 0.700 | 0.817 | 0.775 | 1.000 |
| trace z-score | 0.538 | 0.583 | 0.611 | 0.615 | 0.769 |
| trace min-max | 0.462 | 0.433 | 0.517 | 0.379 | 0.462 |

Confirms the normalization decision: feature z-score (GKF AUC 0.822) ties patient-baseline (0.817) on the honest metric while keeping higher accuracy — patient-baseline's "AUC 1.000" was a degenerate-LOSO artifact, gone under GroupKFold.

### Stage 6 Track A — from-scratch 1D CNN vs SVM (Session 7)

5 seeds, fresh model per fold per seed; same LOSO + GroupKFold + pooled-LOSO as the SVM. TinyCNN (2× Conv1d 8→16, GAP, dropout 0.3, FC).

| Model | train acc | LOSO test acc | GKF AUC | pooled-LOSO |
|---|---|---|---|---|
| CNN-scratch | 0.846 | 0.769 | 0.871 | 0.808 |
| CNN-jitter | 0.844 | 0.777 | 0.889 | 0.815 |
| SVM (ref, feature z-score) | — | 0.769 | 0.822 | 0.781 |

**Surprising, honest finding (contradicts the expected "CNN overfits at N=26"):** the CNN does NOT overfit (train–test gap only 0.076, far below the pre-registered 0.20) and does NOT lose to the SVM — its GroupKFold AUC (0.871) edges *above* the SVM (0.822). Mechanism: the small architecture + Global Average Pooling + dropout is strongly regularized; GAP effectively averages the trace, so the CNN rediscovers the "mean capacitance level" — the same #1 feature the SVM uses. They tie because the discriminative signal is the simple per-site level, not subtle waveform shape. Jittering barely helps (nothing to fix). The CNN sees strictly *more* information than the 6 handcrafted features yet only matches the SVM → confirms signal simplicity, not an information deficit.

**Decision:** Track B (transfer learning) NOT pursued — pre-registered trigger (overfit AND loses to SVM) did not fire. Honest contract honored.

---

## Open Questions

- ~~Replication target: reproduce Todorov Figure 9(a) — confirm per-patient ordering matches~~ ✓ Confirmed
- TEWL: restrict to Corneometer only; note TEWL as limitation ✓ Decision made
- ~~Stage 4: run both prefix and post-settling windows~~ ✓ Complete — see data quality confound above
- ~~Stage 5: add GroupKFold sensitivity and per-patient table~~ ✓ Complete
- Trace completeness: identify which patients have fewer than 30 genuine timesteps; needed before Stage 4 settling interpretation is valid
- ~~Normalization primary method (TA question on progress update)~~ ✓ Resolved (Session 7) — feature z-score primary; patient-baseline as tracking variant
- ~~CNN Track A architecture~~ ✓ Done (Session 7) — TinyCNN (2× Conv1d 8→16, GAP, dropout) + jitter variant; ties/beats SVM, no overfit
- ~~CNN Track B~~ Not pursued (Session 7) — data didn't motivate it; remains described as future work for a larger/harder dataset (self-supervised pretrain on WESAD EDA, freeze conv, retrain final layer)
- Stage 7 (optional): All features are informative on raw traces. If Stage 7 is attempted, use raw or fold-safe patient-baseline centered features for correlation with Corneometer
- ~~GitHub README: draft before Week 9~~ ✓ Created (Session 7) — `README.md`, `CLAUDE.md`
- Oral assessment format: slides, demo, or both?

---

## AI Usage Disclosure

**Claude did:** explained ML concepts, searched literature, corrected over-interpretation of TA feedback, suggested within-patient evaluation, updated project files, wrote and debugged stage scripts (Stages 1–5), identified normalization leakage bug, identified backfill confound in Stage 4.

**Author did:** chose topic and dataset, wrote original proposal, made all scope decisions, judged cross-dataset comparison as methodologically unsound, decided augmentation is optional, reviewed all code before use.

**Report wording:**
> "AI tools (Claude) were used for concept explanation, literature search, code guidance, and result interpretation. Experimental design, research questions, and scientific conclusions were determined by the author. All AI-generated code was reviewed, tested, and modified."

---

## Session Notes

| Session | Summary |
|---|---|
| 1 | Initial scope; adopted LOSO; rejected scratch dataset comparison; defined pipeline |
| 2 (Week 6) | Dropped GBM + Stage 8; added within-patient eval; corrected motivation; confirmed no second IDC dataset; updated project files |
| 3 (Week 7) | Ran Stages 1–3. Confirmed replication of Fig 9(a). Early oracle patient-level normalization appeared to achieve perfect LOSO separation but was later identified as leakage; z-score was near-chance. Feature importance shows Mean/Std useless after z-score; Range and Slope are top discriminators. |
| 4 (Week 7 update) | Added per-patient LOSO table and GroupKFold sensitivity analysis. Expanded Stage 4 into prefix vs. post-settling windows. |
| 5 (Week 8) | Completed Stages 4 and 5. Fixed normalization leakage by moving baseline fitting inside each CV fold. Identified backfill confound in Stage 4 (several patients have <10 genuine timesteps). |
| 6 (Week 8 update) | Reordered stages 2/3 (features before normalization). Replaced min-max with RobustScaler. Switched z-score and robust scaler to feature-level (fold-safe StandardScaler/RobustScaler). All six features now non-zero importance on raw traces (Mean=0.257, Median=0.229, Slope=0.179). Fold-safe patient-baseline centering gives LOSO AUC 1.000 but threshold accuracy 0.654. |
| 7 (Week 9) | Created `CLAUDE.md` + `README.md`. Verified instructor's AUC concern: no leakage; the "1.000" is a degenerate 1-vs-1-per-fold metric (= ranking 13/13); real discriminability is GroupKFold 0.817 / pooled-LOSO 0.775 (`src/verify_loso_auc.py`). Resolved normalization (z-score primary). Reactivated 1D CNN as two tracks (from-scratch baseline + self-supervised transfer learning) for the oral; identified WESAD EDA + UCR/UEA as pretraining sources. Removed duplicate `per_patient_prediction_table.csv`. Corrected stale LOSO/GroupKFold numbers in progress-log and concept-glossary. Did NOT edit submitted `progress_update.md`. Verified refs: Forman & Scholz 2010, Airola et al. 2011, Fawaz 2018. **Then:** (Task 1) expanded Stage 3 to report GroupKFold n=5 + pooled-LOSO AUC alongside LOSO. (Task 2) built `src/stage6_cnn_trackA.py` — from-scratch TinyCNN (5 seeds) ties/beats SVM with no overfit, so (Task 3) Track B was NOT triggered per the pre-registered condition. |
