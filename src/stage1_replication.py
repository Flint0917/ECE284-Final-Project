"""
Stage 1: Replication of Todorov et al. Figure 9(a)

Reproduces the lesional vs. non-lesional mean capacitance per patient as a
grouped bar chart with ±1 std error bars across the 30-second trace window.

Original (Fig9a_ScriptForPlotting.py) used a scatter/errorbar style; this
replication uses grouped bars as requested while preserving the same colour
scheme, axis labels, and data treatment.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — ROOT is the project directory, one level above src/
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Patient mapping: study ID (S-XX) → paper patient number (1–13)
# File format:  "N // S-XX"  (one entry per line; header line is skipped)
# ---------------------------------------------------------------------------
def load_patient_map(txt_path: Path) -> dict:
    mapping = {}
    with open(txt_path) as f:
        for line in f:
            if "//" not in line:
                continue
            left, right = line.split("//", 1)
            try:
                paper_num = int(left.strip())
                study_id  = right.strip()
                mapping[study_id] = paper_num
            except ValueError:
                continue   # skip the header row "Patient Number in Paper // ..."
    return mapping

patient_map = load_patient_map(DATA / "PatientNumbering.txt")

# Sort study IDs by their paper number so x-axis runs 1 → 13 left-to-right
study_ids  = sorted(patient_map, key=lambda sid: patient_map[sid])
paper_nums = [patient_map[sid] for sid in study_ids]

# ---------------------------------------------------------------------------
# Load sensor CSVs
# Shape: (30 rows × 13 patient columns); Time (s) column becomes the index
# ---------------------------------------------------------------------------
les = pd.read_csv(DATA / "lesional_sensor.csv",    index_col=0)
non = pd.read_csv(DATA / "nonlesional_sensor.csv", index_col=0)

# Forward-fill NaN within each patient column (project standard)
les = les.ffill()
non = non.ffill()

# ---------------------------------------------------------------------------
# Per-patient statistics across the 30 time points
# mean() / std() operate column-wise (axis=0) → one value per patient
# std = temporal variability of capacitance within the single 30-s trace
# ---------------------------------------------------------------------------
les_mean = les.mean()
les_std  = les.std()
non_mean = non.mean()
non_std  = non.std()

# Extract in paper-number order
lm = [les_mean[sid] for sid in study_ids]
ls = [les_std[sid]  for sid in study_ids]
nm = [non_mean[sid] for sid in study_ids]
ns = [non_std[sid]  for sid in study_ids]

# ---------------------------------------------------------------------------
# Grouped bar chart — two bars per patient, error bars = ±1 std
# ---------------------------------------------------------------------------
x     = np.arange(len(paper_nums))
width = 0.38

# Colours taken from Todorov's OriginPro palette (red=lesional, blue=non-lesional)
LESIONAL_COLOR    = "#f14040"
NONLESIONAL_COLOR = "#1a6fdf"
ERR_KW = dict(elinewidth=1.2, ecolor="black", capsize=4)

fig, ax = plt.subplots(figsize=(13, 6))

ax.bar(x - width / 2, lm, width, yerr=ls,
       color=LESIONAL_COLOR,    label="Lesional",     error_kw=ERR_KW)
ax.bar(x + width / 2, nm, width, yerr=ns,
       color=NONLESIONAL_COLOR, label="Non-Lesional", error_kw=ERR_KW)

ax.set_xlabel("Patient Number",               fontsize=14, fontweight="bold")
ax.set_ylabel("Mean Sensor Capacitance (pF)", fontsize=14, fontweight="bold")
ax.set_title(
    "Sensor Capacitance Values: Lesional vs Non-Lesional Skin of Patients",
    fontsize=15, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(paper_nums, fontsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.legend(fontsize=12)
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
ax.set_ylim(bottom=40)

plt.tight_layout()
out = RESULTS / "stage1_replication.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")

# ---------------------------------------------------------------------------
# Metrics summary printed to console
# ---------------------------------------------------------------------------
header = f"{'Patient':>8}  {'Les mean (pF)':>14}  {'Les std':>8}  {'Non mean (pF)':>14}  {'Non std':>8}"
print("\n" + header)
print("-" * len(header))
for i, sid in enumerate(study_ids):
    print(f"{paper_nums[i]:>8}  {lm[i]:>14.3f}  {ls[i]:>8.3f}  {nm[i]:>14.3f}  {ns[i]:>8.3f}")
