# Top-10 push — Phase 2 endgame (2026-10-12)

State at writing: **Nanoloom rank 55 / ~300, best entry 4.0449e-9** (submission 334627,
final-layer MSE 1.9654e-8, C/B 0.2058). Top-10 cutoff = **1.9e-9** (rank 9–10, tied pairs).
Phase 2 ends **2026-10-17 23:59 UTC**.

## The score formula decides the endgame

    score = final-layer MSE x max(0.1, C / 2^41)

The multiplier **floors at 0.1**. The #1 team (J2W) sits at C/B = 0.1065 — just above the
floor — with MSE 1.36e-8. Below C/B = 0.1 every further FLOP cut is free, so the endgame is:
**drive utilisation to the floor, then minimise MSE subject to that budget.**

| rank | team | score | MSE | C/B |
|---|---|---|---|---|
| 1 | J2W | 1.4e-9 | 1.36e-8 | 0.1065 |
| 9–10 | (cutoff) | 1.9e-9 | ~1.44–1.57e-8 | ~0.12–0.13 |
| 55 | **Nanoloom** | 4.0e-9 | 1.965e-8 | 0.2058 |

## What the sweeps said (mini ids 0–2, flopscope-metered, deltas vs the s3 config at equal cap)

Net-negative (MSE penalty exceeds the FLOP gain at C/B ≈ 0.2):
- R_OLD 384→320 (−4.9% C/B, +7.1% MSE), →256 (−9.5%, +35%)
- R_OLD2 224→160 (−3.0%, +5.1%), →128 (−4.4%, +17.3%)
- AGE_OLD 4→3 (−8.6%, +12.8%), R_FB 8→0 (−3.0%, +8.2%)
- S_OUT early-heavy trims (−11..−25% C/B, +216..+2465% MSE — catastrophic)
- final-layer band 256 (+13.4% MSE), stale online mean-correction (+113%)
- λ scale 0.90/0.925/0.975/1.00 and V25_BETA 0/1.08: all within ±0.3% of the shipped optimum

Net-positive / clean:
- **Strassen combo-buffer cap MAX_EL 12M → larger**: −4.3% C/B per +67% cap locally at
  *identical* MSE. The cap is pure memory policy — the grader allows 8 GiB.
- Strassen depth 6 with leaf minimum 16: level 6 needs all sides ≥ 1024; f32 block-sum
  rounding grows ~1.4× per level (MSE unchanged to 4 digits at L5 per the V28 note).
- R_FB 4: −1.3% C/B, +1.1% MSE — a wash, useful only as a rounding-budget hedge.

## The 5 submissions (all validated with `whest validate` before upload)

| # | dir | id | knobs baked vs s3 | thesis |
|---|---|---|---|---|
| s4 | sub/s4 | 335627 | `MAX_EL=32M` | pure memory unlock — the one measured-clean lever |
| s5 | sub/s5 | — | + `STRASSEN=6`, `MIN=16` | L6 on the full-width transports at the 32M cap |
| s6 | sub/s6 | — | s5 + `R_FB=4` | −1.3% FLOPs; hedges the L6 rounding budget |
| s7 | sub/s7 | — | s5 + late-heavy `S_OUT` | read-out rows trimmed only at layers ≥ 11 |
| s8 | sub/s8 | — | s5 + RFB4 + LATE | aggressive frontier bracket |

Expected outcome: C/B −2..−10% at MSE ±1% → entries in the 3.7–4.0e-9 band (rank ~45–55).
The honest read: knob-tuning cannot reach 1.9e-9 from 4.0e-9; the top-10 regime requires the
utilisation floor strategy (C/B → 0.1) plus ~30% lower final-layer MSE than the V29 cumulant-K3
lineage currently delivers — i.e. a better estimator, not a better-tuned one. The sweeps above
map exactly which directions are closed.

## Reproducing

1. `whest-starterkit` environment (`flopscope==0.12.1`, `whestbench==0.16.1`), Python 3.11+.
2. Public mini split: HF `aicrowd/arc-whestbench-public-2026` → `tools/pq2npz.py` → npz.
3. `python tools/run_est.py --est sub/s4/estimator.py --ids 0,1,2 --warm 1`
   (on ≤4 GiB machines add `V37_MAX_EL=2500000`; MSE is exact at any cap).
