# Project Memory — ML Assignment 2

Last updated: 2026-10-18

## Project setup
- Project: ML_Assignment_2
- Environment: macOS, VS Code, Python 3.8.10, virtual environment `.venv`
- Dataset: scikit-learn Olivetti Faces (`fetch_olivetti_faces`)
- Dataset structure: 40 identities, 10 images per identity, images are 64×64
- Assignment split: 6 train / 2 validation / 2 test images per identity
- Assignment constraints mentioned in the project history: raw flattened pixels (no PCA), implement bagging/boosting yourself (DecisionTreeClassifier is allowed), robustness test masks centered rows/columns 24–39 (16×16), clean training data only
- Expected deliverables: two-column CVPR-style PDF (up to 8 pages excluding references), GitHub repository with code/config/logs/figures/environment/commands/final hash

## Confirmed progress from the pasted conversation
- Git repository branch: `main`
- Last confirmed commit shown in the conversation: `58f7419 Add experiment result plots`
- Plot script: `src/plot_results.py`
- Figures generated under `figures/`:
  - `q1_accuracy.png`
  - `q2_tuning.png`
  - `q3_accuracy.png`
  - `q3_variance.png`
- Q3 implementation was changed to use **within-identity bootstrap sampling**: for each identity, sample that identity's six training examples with replacement.
- Q3 script: `src/q3_bias_variance.py`
- Q3 log: `logs/q3_bias_variance.json`
- Q3 was rerun and the plots were regenerated successfully after the sampling change.
- The pasted conversation does **not** confirm that the corrected Q3 code/log/figures have been committed or pushed to GitHub.

## Latest Q3 output recorded in the conversation
30 bootstraps per depth; seed 26504.

| Max depth | Mean individual test accuracy | Std. dev. | Majority-vote test accuracy | Multiclass variance | Aggregate error proxy |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0450 | 0.00764 | 0.0250 | 0.10683 | 0.9750 |
| 2 | 0.06208 | 0.01270 | 0.0375 | 0.14228 | 0.9625 |
| 3 | 0.0875 | 0.01407 | 0.0500 | 0.12122 | 0.9500 |
| 4 | 0.10083 | 0.01328 | 0.0500 | 0.15208 | 0.9500 |
| Unlimited | 0.4850 | 0.04824 | 0.7875 | 0.62103 | 0.2125 |

**Interpretation caution:** `aggregate_error_bias_proxy` is an aggregate error proxy, not a formal statistical bias estimate. The results are unusually different between individual-model mean accuracy (48.5%) and majority-vote accuracy (78.75%) for unlimited depth, so verify the implementation, label alignment, split logic, and metric calculation before making strong claims in the report.

## Immediate next steps
1. In the project terminal, inspect repository status:
   ```bash
   git status
   ```
2. Review the Q3 changes and confirm the expected files exist:
   ```bash
   git diff -- src/q3_bias_variance.py logs/q3_bias_variance.json figures/q3_accuracy.png figures/q3_variance.png
   git status --short
   ```
   If the files are already staged/committed, `git diff` may show nothing; use `git log -5 --oneline` and `git show --stat HEAD`.
3. If the corrected files are still uncommitted, commit them:
   ```bash
   git add src/q3_bias_variance.py logs/q3_bias_variance.json figures/q3_accuracy.png figures/q3_variance.png
   git commit -m "Correct Q3 within-identity bootstrap analysis"
   ```
4. Verify:
   ```bash
   git log -1 --oneline
   git status
   ```
5. If a GitHub remote is configured, check it:
   ```bash
   git remote -v
   ```
   Push only after confirming the correct remote/repository:
   ```bash
   git push -u origin main
   ```
6. Verify Q1/Q2/Q3 results, assignment compliance, reproducibility, and report claims before final submission.
7. Keep this file updated whenever an important decision, result, commit hash, or next action changes.

## Resume prompt for a future ChatGPT session
I am continuing my ML_Assignment_2 project. First read `PROJECT_MEMORY.md`, then ask me for current `git status`, `git log -5 --oneline`, and any relevant code/output before assuming the repository state. Help me continue from the recorded next steps, preserve the assignment constraints, and do not claim anything is committed or pushed until the terminal output confirms it.
