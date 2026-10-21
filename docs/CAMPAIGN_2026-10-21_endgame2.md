# Campaign 2026-10-21 — endgame session 3 (grader-degradation day)

Target: user directive "1.40e-9, realistic way, 5 shots one by one, push to GitHub".
1.40e-9 = J2W's #2 score (oliver_chen #1 at 1.30e-9, MSE 1.10e-8). Both sit in the
climber-class regime (MSE 1.10–1.36e-8 at C/B ≈ 0.106) that no public technology reaches:
the public frontier (ALG058 lineage) carries MSE ≈ 1.94–1.99e-8, and every measured knob
has MSE-elasticity ≥ 0.93 per FLOP%. The honest ceiling for public tech today was
~2.87–2.90e-9 (entered the 2.9 display block); the ladder below pursued exactly that.

## Shots fired today (team koushik_rudra / Nanoloom, quota 10/day)

| # | id | tag | content | result |
|---|---|---|---|---|
| 1 | 337532 | s32refire | s27 + FUSE_P (343→1) | graded **9.35e-2** — FUSE_P numerically broken at full width; axis closed with grader evidence |
| 2 | 337545 | s36ahub6 | s27 + main young-hub Strassen depth 6 | graded **2.9577e-9** (MSE 1.98535e-8, util 0.14898) — then team best |
| 3 | 337551 | s38hub7 | s36a + hub depth 7 | graded **2.9575029e-9** = TEAM BEST (hub depth exhausted: util −0.003%, wall PASS) |
| 4 | 337561 | s39ytr6 | s36a + young-transport trio depth 6 | FAILED smoke (16:44Z) |
| 5 | 337569 | s41cpre6 | 337551 + C_pre sym family depth 6 | FAILED smoke (17:35Z) |
| 6 | 337581 | s46ctl | CONTROL: byte-identical 337551 | FAILED smoke (18:19Z) ⇒ **grader-side breakage proven** |
| 7 | 337589 | s47tauL | 337551 + X_DEAD_TAU_L 12-15:2.7 (wall-negative) | FAILED smoke (18:47Z) ⇒ lineage-specific breakage, wall direction irrelevant |

## The degradation timeline (evidence)

- 16:23Z — 337551 (byte-identical to the later control) GRADES 2.9575e-9, wall PASS.
- 16:44Z — s39 fails smoke. 16:50/16:58Z — other teams grade fine.
- 17:15/17:19/17:25Z — other teams grade fine (lighter engines).
- 17:35Z — s41 fails. 18:19Z — CONTROL (same bytes as the 16:23 PASS) fails.
- 18:47Z — s47 (wall-REDUCING vs the base) fails.
⇒ Not wall, not content: the grader environment/pipeline breaks on OUR lineage
(byte-identical bytes went PASS → FAIL between 16:23 and 18:19) while other engines
continue grading. No organizer notice posted; API/UI expose no error detail
(tracebacks organizer-only until results release). Diagnosis channels exhausted.

## What was built and staged (validated, ready to fire if the pipeline revives)

- `scratch/s43ytr6s1` — site-1-only young-transport depth 6 (half-dose of s39; isolates
  the WC/kc family; +2-3 ms wall) — the best remaining upside (~2.925-2.935e-9).
- `scratch/s44stack` — CPRE6 + site-1 ytr6 stack (~2.905-2.925e-9).
- `scratch/s42ytr7` — trio depth 7 (shelved: assumes ytr6 is healthy AND wall margin).
- `scratch/s48s27taul` — lightest-wall rescue (337079 content + tau_L 12-15:2.7).
- Patch tools (assert-once, guarded engagement, kill-switch env in-file):
  `scripts/mk_s39ytr6.py` (site selector + smin wrap args), `scripts/mk_cpre6.py`,
  `scripts/mk_s47taul.py`; headers `scripts/s4[0-8]_header.txt`.

## Endgame protocol (quota 8/10 used, 2 left, deadline 23:59Z)

- Nomination auto-fallback pair = two best graded = **337551 + 337545** (both
  deterministic ALG058 config variants, generalization-safe for the private re-eval).
- Hold the last 2 shots. At ~21:45Z fire s43 as the pipeline probe (upside content);
  s44 only if s43 grades clean by ~22:35Z. If the pipeline stays broken, the standing
  best 2.9575e-9 (display 3.0e-9, rank ~42) stands.

## Standing result of the day

Team best improved 2.9663e-9 → **2.9575029e-9** (337551); rank 42→43 on a tie-block
shuffle (▼3 on the 18:11 board). The 2.9-block entry (≤ 2.949e-9) was one healthy
grading window away (s39's −2.0..−2.9% util was modeled, not yet grader-measured).
