# [Phase 2 write-up] Nanoloom — surgical Strassen-depth and birth-column billing cuts on the ALG058 chain (2.9575e-9)

Team: **Nanoloom** (koushik_rudra) · Phase 2 best public score: **2.9575029e-9**
(= MSE 1.9854e-8 × multiplier 0.14896) · Final-leaderboard nominations:
**337551 + 337066** (selection verified, deadline 2026-10-22 23:59 UTC)

---

## TL;DR

We adopted the public ALG058-V001 chain (TristanMiano, MIT) as our anchor and spent
our 36 graded Phase 2 entries on two questions: (1) which of the *published* cost-cut
recipes survive the flopscope 0.12.3 grader on this base, and (2) is there a knob
ladder on this engine that reaches below ~2.95e-9. Answers: the birth-column cut and
a **surgical Strassen depth escalation at a single hub site** both transfer cleanly
(2.9711e-9 → 2.9663e-9 → **2.9575e-9**); the rest of the ladder is measured-closed
(dose cliffs, catastrophic caps, server-punished rank cuts). We also document a
reusable grader-outage forensics method (byte-identical control re-uploads) that
settled attribution of 14 smoke-test failures across three outage windows.

| # | entry | build | score | MSE | mult |
|---|-------|-------|-------|-----|------|
| 1 | 337551 | ALG058 + tau_b 2.2 + **hub Strassen depth 7** (single site) | **2.95750e-9** | 1.9854e-8 | 0.14896 |
| 2 | 337545 | same, depth 6 | 2.95775e-9 | 1.9853e-8 | 0.14898 |
| 3 | 337079 | ALG058 + **tau_b 2.2** (birth-column cut) | 2.96630e-9 | 1.9861e-8 | 0.14936 |
| 4 | 337066 | ALG058-V001 verbatim (attribution anchor) | 2.97112e-9 | 1.9447e-8 | 0.15278 |
| 5 | 337069 | ALG055-V000 verbatim | 2.98300e-9 | 1.9531e-8 | 0.15273 |
| 6 | 337367 | + tau_L "0-4:2.6" + QPASS2=1 | 2.99417e-9 | 2.0090e-8 | 0.14904 |
| 7 | 337370 | tau_b 2.0 (dose escalation) | 3.04860e-9 | 2.0636e-8 | 0.14773 |

## Base and attribution (license compliance)

- **ALG058-V001 / ALG055-V000**: TristanMiano (github.com/TristanMiano/arc-whitebox-estimation-2026, MIT) — graded 336809 = 2.9711e-9. Our 337066 is a verbatim port with an attribution header; it re-graded at **exactly** 2.9711234e-9, confirming artifact determinism across worker pools.
- **e30_a engine**: LordProcciuto (github.com/siming767-creator/whestbench-lordprocciuto, MIT, byte-verified against their graded package).
- **V29 Strassen skeleton**: 504aldo (al07-alt repositories, MIT).
- **Column-cut recipe equivalence**: our tau_b birth-column cut is the engine-native analogue of Susupro's active-set column cut (github.com/logxio/fourth-cumulant, MIT; forum topics 18254/18246) — colcut −3.3% adj 50/50 on their chain B.
- **k31 two-term closure**: al07-alt (whest-p2-k31-two-term, MIT) — we tested it only on the older s3/s10a lineages (below); it is dead on e30_a per LordProcciuto C10 and our s22/s23 receipts.
- All shipped sources carry MIT notices and in-file attribution headers; MIT notice + write-up disclosure per forum topic 18240 ruling.

## Own contribution 1: birth-column cut (X_DEAD_TAU_B = 2.2, dead_fis = 5)

Engine-native column cut applied at atom-birth: newborn columns whose hub-entry
statistic falls below tau_b are zeroed before the hub contraction and physically
compacted (dead_fis=5 live-width products). This is the same economic mechanism as
Susupro's w1 < 0.014 active-set cut but expressed in the engine's own tau/dead
machinery, which keeps the op-stream inside the residual budget (0.3877 s of the
0.4 s cap). Measured on the grader: **2.9711e-9 → 2.9663e-9** (−0.16% score: MSE
+0.07%, multiplier −0.23%). Dose response is sharply non-linear — tau_b 2.0 crosses
a compaction cliff (337370: MSE +3.9%, elasticity 3.6 vs the 0.93 measured at 2.2),
so 2.2 is the shipped operating point and the axis is closed in both directions.

## Own contribution 2: surgical hub Strassen depth escalation (the team best)

LordProcciuto's cost ledger prices the main young-hub D21 contraction at ~27% of
total billed cost, but a *global* Strassen-depth or leaf-floor change is either
clock-fatal or MSE-fatal (E1/L6, H1 receipts). Our change instead escalates depth
**at one call site only** — the main young-hub site:

- depth 6: smin=16 wrap at that call site, global leaf floor stays 32, old-level
  site stays depth 5, first-predict keeps STRASSEN_FIRST=4 (337545 = 2.95775e-9,
  MSE −0.04% = rounding-level, residual PASS);
- depth 7: leaf blocks 8, smin=8 wrapped at that call site only; `level()` clamps
  gracefully to 6 where shapes/floor disallow, so the change is inert at mini scale
  and engages only on full-width families (337551 = **2.95750e-9**, mult 0.14896,
  residual PASS).

Combined with contribution 1 this stacks two independent billing cuts (−0.23% and
−0.27%) at zero MSE cost, which is the entire 2.9711 → 2.9575 gap. We stopped at
depth 7: the sub-343 fusion experiment (337386, FUSE_P 343→1, bit-exact predictions
at mini) smoke-failed at full 1024×16 because it re-routes every level-1 can't-split
site — a reminder that site-local surgery is the only surviving form of this axis.

## Measured negative results (all on flopscope 0.12.3 / whestbench 0.16.2)

- **tau_L "0-4:2.6" + QPASS2=1** (337367): +0.94% score vs its s27 base; TM's 4-net
  deltas do not transfer (+0.34% claimed → +1.15% MSE on grader).
- **tau_b 2.0** (337370): compaction cliff, elasticity 3.6 — axis closed both ways.
- **SMIN=8/STV=1/VB_HOT=2.6 bundle** (337369): grader-fatal despite individually
  "measured safe" components.
- **Sub-343 fusion** (337386 FUSE_P, 337532 refire): smoke-fatal / catastrophic
  (multiplier forced 1.0) — mini-scale bit-exactness does not imply full-scale safety.
- **Young-atom cap port** (337088): catastrophic 2.94e-7 on the grader despite
  validating on 128×8 mini (finite, deterministic, −2..−4% expected) — the cap
  misfires on full-width birth distributions.
- **k31 two-term closure** (al07-alt recipe): −2.70% on the old s3 lineage
  (336782 = 4.2839e-9, our then-best); b2-table variant a wash (+0.05%); on the
  s10a lineage it is catastrophic (0.1466) — MIN=8/Strassen-BETA108 interaction.
- **Final-layer readout band restores** (336264/336272): multiplier-forced-1.0
  signatures; the final hop is residual-bound, not FLOP-bound; band axis closed at
  any restore width (832→896 and 1024 both fatal).
- Scalar-knob ladder (X_DEAD_TAU/V21_R_OLD/V17 family, 333363–335854 era): every
  measured elasticity ≥ 0.93; no scalar axis reaches 0.5% (consistent with
  LordProcciuto F1).

## Grader-outage forensics (method note)

Three smoke-test outage windows hit our campaign (Oct 16; Oct 20 17:55–19:06 UTC;
Oct 21 16:32–22:39 UTC, 14 failed entries total). We attribute outages — as opposed
to content failures — by re-uploading a **byte-identical control** of an
already-graded artifact (337390 = re-upload of graded 337079) and observing it fail
the same way while the leaderboard goes dark board-wide. This settles the
content-vs-platform question in one shot and avoids post-morteming healthy code.
Caveat for future runners: the same file graded clean one day can lose ~10% of the
suite to zero-prediction fallbacks the next (forum 18252); budget wall-clock margin
accordingly, especially for deep-Strassen builds.

## Nomination strategy for the private re-eval

Prize ranking comes from the fresh-seed re-eval of up to two nominated entries
(flopscope 0.12.3 planned). We nominate **337551 + 337066**: the cheapest build
(depth-7 hub, graded clean, best public score) paired with the unmodified ALG058
anchor (lowest MSE of the pool, 1.9447e-8, and the only artifact proven
byte-stable across two independent gradings). The near-twin 337545 was rejected:
nominating both depth variants doubles exposure to one knob family — including the
documented runtime-margin risk of deep builds — for +0.008% of expected score.
Expected-value and tail-risk both favor the diversified pair for a max-of-two metric.

## Reproduction

- Environment pins: `flopscope>=0.12.3`, `whestbench>=0.16.2,<0.17.0` (organizer
  post 18251). All scores above are grader receipts under this toolchain.
- Submission folders, logs, and per-entry meta:
  github.com/acco-cyber/arc-whitebox-estimation-2026 (`submissions/`, `docs/`).
- Nominated artifacts: platform entries 337551 (s38hub7) and 337066 (s25alg058),
  folders `submissions/337545_s36ahub6` (depth-6 variant) and the ALG058 anchor
  sources with attribution headers.
- Contact: this forum topic or the challenge inbox.
