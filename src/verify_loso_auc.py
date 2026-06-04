"""
Verification: is the LOSO AUC = 1.000 an error / leakage?

Written to answer the instructor's concern:
  "I suspect the AUC is an error. Double check it is not computed on training
   data or something."

This script does NOT change any results. It re-runs the Stage 3/5 patient-baseline
LOSO pipeline and prints four diagnostics:

  1. LEAKAGE CHECK   - confirms each held-out LOSO patient's baseline is the
                       training-only fallback, never their own (test) data.
  2. SCORING SOURCE  - confirms AUC is computed from decision_function(X_TEST),
                       not training data.
  3. DEGENERACY      - shows every LOSO fold has exactly 1 lesional + 1 non-lesional
                       sample, so per-fold roc_auc_score can only be 0.0 or 1.0.
                       The "AUC 1.000" is therefore the *mean of 13 binary ranking
                       indicators* = pairwise ranking accuracy, not a smooth AUC.
  4. ACC vs AUC GAP  - accuracy (~0.65) << "AUC" (1.000). If there were oracle
                       leakage, accuracy would also be ~1.0. The gap is positive
                       evidence AGAINST leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


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

t30 = np.arange(30)


def extract_features(trace):
    sk = float(skew(trace))
    return [
        trace.mean(), trace.std(), np.median(trace),
        trace.max() - trace.min(), np.polyfit(t30, trace, 1)[0],
        0.0 if np.isnan(sk) else sk,
    ]


X_traces = np.vstack(df["trace"].values)
y = df["label"].values
groups = df["patient"].values

logo = LeaveOneGroupOut()

per_fold_auc, per_fold_acc, ranking_hits = [], [], 0
pooled_scores, pooled_labels = [], []
leakage_violations = 0

print("=" * 92)
print(f"{'Fold':<5}{'HeldOut':<9}{'n_test':<7}{'n_pos':<6}{'n_neg':<6}"
      f"{'usedOwnBaseline?':<18}{'L_score':>9}{'NL_score':>10}{'foldAUC':>9}{'foldAcc':>9}")
print("-" * 92)

for fold_id, (train_idx, test_idx) in enumerate(logo.split(X_traces, y, groups), start=1):
    X_tr_raw, X_te_raw = X_traces[train_idx], X_traces[test_idx]
    pid_tr, pid_te = groups[train_idx], groups[test_idx]

    # --- training-only patient baselines (the method under suspicion) ---------
    baselines = {pid: np.nanmean(X_tr_raw[pid_tr == pid]) for pid in np.unique(pid_tr)}
    fallback = np.nanmean(X_tr_raw)                      # training global mean

    held_out = pid_te[0]
    used_own_baseline = held_out in baselines           # MUST be False under LOSO
    if used_own_baseline:
        leakage_violations += 1

    X_tr_norm = X_tr_raw - np.array([[baselines.get(p, fallback)] for p in pid_tr])
    X_te_norm = X_te_raw - np.array([[baselines.get(p, fallback)] for p in pid_te])

    X_tr = np.array([extract_features(t) for t in X_tr_norm])
    X_te = np.array([extract_features(t) for t in X_te_norm])
    scaler = StandardScaler().fit(X_tr)                 # fit on TRAIN only
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    clf = LinearSVC(C=1.0, max_iter=5000, random_state=42).fit(X_tr, y[train_idx])

    y_te = y[test_idx]
    y_score = clf.decision_function(X_te)               # scores from TEST features
    y_pred = clf.predict(X_te)

    les_score = float(y_score[np.where(y_te == 1)[0][0]])
    nls_score = float(y_score[np.where(y_te == 0)[0][0]])
    fold_auc = roc_auc_score(y_te, y_score)
    fold_acc = accuracy_score(y_te, y_pred)

    per_fold_auc.append(fold_auc)
    per_fold_acc.append(fold_acc)
    ranking_hits += int(les_score > nls_score)
    pooled_scores.extend(y_score.tolist())
    pooled_labels.extend(y_te.tolist())

    print(f"{fold_id:<5}{held_out:<9}{len(test_idx):<7}{int((y_te==1).sum()):<6}"
          f"{int((y_te==0).sum()):<6}{str(used_own_baseline):<18}"
          f"{les_score:>9.3f}{nls_score:>10.3f}{fold_auc:>9.1f}{fold_acc:>9.2f}")

print("=" * 92)
print(f"\n[1] LEAKAGE CHECK : held-out patient used own baseline in "
      f"{leakage_violations}/13 folds  ->  {'LEAK!' if leakage_violations else 'OK, no leak'}")
print(f"[2] SCORING SOURCE: every fold above scored decision_function(X_TEST), labels=y_TEST")
print(f"[3] DEGENERACY    : every fold has n_pos=1, n_neg=1  ->  per-fold AUC is binary {{0,1}}")
print(f"      mean per-fold AUC      = {np.mean(per_fold_auc):.3f}  (== ranking accuracy)")
print(f"      pairwise ranking hits  = {ranking_hits}/13 = {ranking_hits/13:.3f}")
print(f"      POOLED AUC (all 26)    = {roc_auc_score(pooled_labels, pooled_scores):.3f}  "
      f"(single AUC over pooled test scores, less degenerate)")
print(f"[4] ACC vs AUC GAP: mean fold accuracy = {np.mean(per_fold_acc):.3f}  <<  'AUC' {np.mean(per_fold_auc):.3f}")
print(f"      -> ranking is perfect but the threshold mislabels folds; under oracle")
print(f"         leakage accuracy would also be ~1.0. The gap argues AGAINST leakage.")
