# Submissions

Every AIcrowd submission of this project, newest first. Score = final-layer MSE x max(0.1, FLOPs / 2^41), averaged over the hidden suite (lower is better).

| id | date (UTC) | tag | score | final-layer MSE | multiplier | status | file | description |
|---|---|---|---|---|---|---|---|---|
| 333482 | 2026-10-02 07:58 | s1_v34 | 4.9848e-09 | 2.1320e-08 | 0.2338 | graded | [333482_s1_v34/estimator.py](333482_s1_v34/estimator.py) | V29 cumulant-K3 + dead-ReLU pruning of the K3 legs (V30), C/B~0.232 |
| 333364 | 2026-10-01 16:00 | v29lam1 | 5.3936e-09 | 2.1584e-08 | 0.2499 | graded | - | 504aldo V29 with lambda-table scale 0.95 -> 1.00 (file from an earlier sandbox, not kept) |
| 333363 | 2026-10-01 15:56 | v30valve | - | - | - | failed | - | V29 + kappa4 channel attempt; failed: leftover mmap/tempfile helper code = prohibited file (file not kept) |
