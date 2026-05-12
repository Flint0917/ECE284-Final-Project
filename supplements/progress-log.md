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

---

## Stage Status

| Stage | Priority | Status |
|---|---|---|
| 1. Replication (Todorov Fig 9a) | Core | **Complete** — results/stage1_replication.png |
| 2. Normalization comparison (z-score, min-max, patient-level) | Core | **Complete** — results/stage2_normalization.png |
| 3. Feature extraction (mean, std, median, range, slope, skewness) | Core | **Complete** — results/stage3_features.png |
| 4. Window length experiment (5–30 sec) | Core | Not started |
| 5. SVM vs 1D CNN — LOSO + within-patient | Core | Not started |
| 6. CNN + jittering augmentation | Optional | Not started |
| 7. ML features vs. Corneometer correlation | Optional, results-dependent | Not started |

---

## Experimental Findings

### Stage 1 — Replication
- Non-lesional capacitance is consistently higher than lesional for all 13 patients (range ~40–55 pF).
- Patient-level separation is clear visually; population-wide threshold would fail because baseline shifts vary widely across patients (Patient 1 ≈ 40 pF vs. Patient 4 ≈ 49 pF lesional).
- Replicates the direction and per-patient ordering of Todorov Fig 9(a). ✓

### Stage 2 — Normalization (LinearSVC, LOSO)

| Method | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| z-score | 0.538 ± 0.237 | 0.410 ± 0.396 | 0.769 ± 0.421 |
| min-max | 0.462 ± 0.308 | 0.308 ± 0.402 | 0.462 ± 0.499 |
| patient-level | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |

Key insight: Patient-level normalization achieves perfect LOSO separation. After removing each patient's baseline, the lesional vs. non-lesional contrast becomes a consistent positive/negative mean offset that a linear boundary splits perfectly. Z-score degrades to near-chance because it destroys the mean and std features by construction (both become constants: 0 and 1). Min-max performs at chance — it additionally destroys the range feature (always = 1 after scaling) and is unstable when the 30-point trace has very low variance.

**Implication for report:** The result is not "patient-level is always best" — it's that the sensor's discriminative signal lives almost entirely in the per-patient baseline offset. Cross-patient classification (z-score/min-max) is genuinely hard, which corroborates Todorov's own finding that no population-wide threshold exists.

### Stage 3 — Feature Importance (Random Forest, z-score norm, N=26)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Range | 0.3062 |
| 2 | Slope | 0.2876 |
| 3 | Skewness | 0.2053 |
| 4 | Median | 0.2008 |
| 5 | Std | 0.0000 |
| 6 | Mean | 0.0000 |

Mean and Std score zero — both are constants after z-score normalization (0 and 1 respectively for every trace). The remaining four features (Range, Slope, Skewness, Median) carry all discriminative power. **Slope and Range being top-ranked suggests the traces are not flat** — there is a detectable linear drift and amplitude spread that differs between lesional and non-lesional skin.

---

## Open Questions

- ~~Replication target: reproduce Todorov Figure 9(a) — confirm per-patient ordering matches~~ ✓ Confirmed
- TEWL: restrict to Corneometer only; note TEWL as limitation
- CNN architecture: specify Week 8 (likely 2–3 Conv1D + GAP + dropout)
- Statistical reporting: mean ± std across LOSO folds ✓ Implemented in Stage 2
- Stage 4: should window experiment use z-score or patient-level norm? Patient-level gives perfect separation even at 30 s — consider using z-score to show a more meaningful degradation curve across window lengths
- Stage 7 (optional): Mean and Std are zero-importance under z-score. If Stage 7 is attempted, use raw or patient-level features instead
- GitHub README: draft before Week 9
- Oral assessment format: slides, demo, or both?

---

## AI Usage Disclosure

**Claude did:** explained ML concepts, searched literature, corrected over-interpretation of TA feedback, suggested within-patient evaluation, updated project files.

**Author did:** chose topic and dataset, wrote original proposal, made all scope decisions, judged cross-dataset comparison as methodologically unsound, decided augmentation is optional.

**Report wording:**
> "AI tools (Claude) were used for concept explanation, literature search, code guidance, and result interpretation. Experimental design, research questions, and scientific conclusions were determined by the author. All AI-generated code was reviewed, tested, and modified."

---

## Session Notes

| Session | Summary |
|---|---|
| 1 | Initial scope; adopted LOSO; rejected scratch dataset comparison; defined pipeline |
| 2 (Week 6) | Dropped GBM + Stage 8; added within-patient eval; corrected motivation; confirmed no second IDC dataset; updated project files |
| 3 (Week 7) | Ran Stages 1–3. Confirmed replication of Fig 9(a). Key finding: patient-level norm achieves perfect LOSO separation; z-score near-chance. Feature importance shows Mean/Std are useless after z-score; Range and Slope are top discriminators. |
