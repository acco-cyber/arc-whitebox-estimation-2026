# Research 2026-10-20 (evening) — the FUSE_P billing discovery + shot s32

## Mission
User target: 2.10e-9 (= -29.2% vs s27 = 337079 = 2.9663e-9). Honest math: rank-10-12
cutoff is 1.9e-9 and the campaign doc already showed that is closed for public tech;
2.10e-9 needs climber-class MSE (1.35-1.45e-8) or mult 0.1057 at our MSE - both out of
same-day reach. The reachable frontier today = bit-exact billing cuts on the s27 base.

## Fresh per-callsite billing probe (mini 128x8, flopscope 0.12.1 unit-weights)
`runs/probe_sites.py` re-run against the graded s27 file (337079). Top sites:
matmul leaf 39.5% + combine 27.1% (Strassen recursion), metered page touches ~5.9%
(load-bearing for the 0.4 s residual cap - do not touch), PK2 assembly 2.8% (RELOJ
multiply+multiply+sum; the einsum alternative bills MORE and is NOT bit-exact - LP knew),
then a long tail.

## LP writeup cross-check (scratch/hunt/lordprocciuto_e30a/WRITEUP.md)
- E9 micro-savings row (the origin of our "E9 reserve" line): LP measured each item
  <= 0.6% FLOPs and REJECTED all: "below noise or paid in operations" - the grader has
  an operation/clock currency (~7 us/op, 120 s server clock, ~6,000 spare ops). E9 is
  op-positive => CLOSED (do not implement).
- E1 (L6): -1.74% FLOPs but +3,700 ops -> clock 119.3/120 + 20% nets failed => fatal.
- C3 (born-hub tail): all K score-negative. CLOSED.

## The discovery: V28_STRASSEN_FUSE_P 343 -> 1
The fused per-product leaf path runs the SAME per-element add-tree and matmul operands
on batched slabs - a billing-only change. Measured (kitvenv, unit weights):

| scale | config | FLOPs | ops | wall | predictions |
|---|---|---|---|---|---|
| 128x8 seed 1000 | s27 (343) | 490,096,397 | - | 0.568 s | sha 021c31c947fe486f |
| 128x8 seed 1000 | FUSE_P=1 | 470,893,325 (-3.9%) | - | 0.584 s | sha 021c31c947fe486f (IDENTICAL) |
| 128x8 seed 1001 | s27 / FUSE_P=1 | 459,600,909 / 456,218,637 (-0.7%) | - | - | 1a0805239728879e (IDENTICAL) |
| 256x8 seed 1000 | s27 (343) | 3,127,268,813 | 3,377 | 1.37 s | sha ce8b1d9a550dd3e1 |
| 256x8 seed 1000 | FUSE_P=1 | 2,992,847,309 (-4.3%) | 3,456 (+2.3%) | 1.34 s (-2%) | sha ce8b1d9a550dd3e1 (IDENTICAL) |

Bit-exact at both scales (predictions SHA-256 identical), billed FLOPs -3.9..-4.3%,
ops +2.3% (vs LP's ~6,000-op margin => trivial), wall -2% at 256. LP shipped 343 with
the comment "fused per-product leaf when the leaf batch has >= this many blocks" - the
shallow levels (batch < 343 blocks) never fused in ANY graded run of this lineage.

## Rejected variants (measured at mini, single-knob each)
- AUD_IND2=0 (4-operand einsum PK2): FLOPs +3.9% AND predictions change. Double-rejected.
- AUD_ZS=0 (skip zero slices): FLOPs +1.8%. Rejected (slab geometry beats slice skip).
- RELOJ=0: inert at mini (pmode 0). Unknown at full scale - not touched (wall risk).
- E9 micros / tau_b 2.4 / MIN 16 / L6 / MAX_EL: see LP writeup + graded history. Closed.

## Shot A: s32 = 337386 (tag s32fuse1)
est/v67fuse1.py = graded s27 (ALG058 + X_DEAD_TAU_B=2.2) with V28_STRASSEN_FUSE_P
default 343 -> 1 (kill-switch restores s27). Pre-flight on the final artifact: FLOPs
exactly 470,893,325 / 456,218,637 and hashes 021c31c947fe486f / 1a0805239728879e.
Expected: MSE flat at 1.9861e-8 (score_secondary must equal s27's to all digits),
mult ~0.1427-0.1449, score ~2.84-2.88e-9. Verification signal: score_secondary.

## Honest ceiling note
2.10e-9 remains out of reach of this engine family today: bit-exact billing cuts sum
to ~-4..-6% at best, every MSE-costing knob has elasticity >= 0.93, and the MSE side
(learned correctors, miscusi refit) is blocked by the 3 GB sandbox. The two shots
today defend/improve the best entry; the knob ladder + this fusion axis is now the
whole measured frontier.
