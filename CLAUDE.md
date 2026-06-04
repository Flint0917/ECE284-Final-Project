# CLAUDE.md — ECE284 Final Project

Live status: @supplements/progress-log.md

## Project
Atopic Dermatitis (AD) monitoring from IDC wearable sensor data (Todorov et al.,
IEEE Sensors 2025). 13 patients, 30-pt traces @1 Hz, lesional/non-lesional labels.
Framing: **proof-of-concept under data scarcity, not clinical validation.**

## Layout
`src/` scripts (one per stage) · `results/` figures + metric CSVs ·
`WearSystemIDCforAD/` raw data (read-only) · `supplements/` briefs, glossary, log.

## Standards
Python 3.11 / conda `ece284`. numpy, pandas, sklearn, scipy, matplotlib, torch.
`random_state=42` everywhere. Forward-fill NaN within patient column.
Every script saves figure(s) to `results/` and prints a metrics summary.
Git: commit per working stage, message `stage<N>: <desc>`, push to `main`.

## How to help
- Read data files directly to infer format — don't ask about columns.
- Explain ML concepts (LOSO, AUC-ROC) in code comments; I'm an undergrad in a grad course.
- **Be open to scope changes** — data is the major limit. If I reference other
  work/ideas, search and verify it fits *this* dataset before integrating.
- Flag scope creep. Write code + comments, **not report prose** (my job).
- No paid tools. Keep code attributable; flag any restructuring of my own code.

## Deadline
Final report due **2026-06-05**: 7–10 pp, ACM Large 2-col, sections Motivation /
Related Work / Project Aim / Methodology / Results / Discussion / Conclusion.
Graded on clarity, judicious visuals, logic+data-backed claims. Avoid whitespace/
oversized-image formatting penalties. Include GitHub repo + README.
