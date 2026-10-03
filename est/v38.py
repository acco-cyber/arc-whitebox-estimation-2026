"""K3-simple factored cumulant propagation + MEMORYLESS KAPPA4 REGENERATION (V17)
+ D21 FEEDBACK THIN LEGS (V18, 2026-09-02, F69)
+ FINAL-LAYER TRIM (V19, 2026-09-03)
+ RESIDUAL-TIME FUSION (V20, 2026-09-03)
+ AGE-GATED SHARED BASIS FOR OLD SOURCES (V21, 2026-09-03, F66 option C / F72)
+ PRE-ACTIVATION LAMBDA TABLE (V22, 2026-09-03, F68 prefit)
+ NESTED SECOND TIER FOR THE OLDEST SOURCES (V24, 2026-09-03, F72 tiers)
+ ADAPTIVE PER-MLP LAMBDA (V25, 2026-09-03)
+ POOLED BATCHED STRASSEN-WINOGRAD PRODUCTS (V26-V28, 2026-09-05, F79/F80)
+ NEWBORN + COVARIANCE RIDING THE TRANSPORT FAMILY, BLOCK-SYMMETRIC C_PRE (V29, 2026-09-06).

V29: the dense young legs are pre-scaled by the wick w1 at the wick stage, so the
transport family multiplies by the raw W and the newborn's A leg (born as w1 * C_off) and
the post-ReLU covariance C ride along as two extra slots; C_pre = (W C) W^T is assembled
from the three block products of its 2x2 partition (one Strassen family), layer 0 uses the
aliased Gram w32^T w32, and the trimmed last layer reads var from (W C) * W.

V25: the regenerated kappa4 off-diagonal coefficient is no longer a frozen per-layer
table but lam_l = LAM[l] * ((mean(dG)/mean(var)) / REF_R[l])^BETA, evaluated online at the
layer from the transported kappa4 diagonal dG and the pre-activation variance (both already
computed there). Fitted offline on public dumps 0-7: the per-MLP LS lambdas track this
ratio with log-log slope ~1 at every layer and the unexplained spread drops from 3-10% to
1-2%. Kill-switch: V25_BETA=0 -> V24 values.

V24: sources older than AGE_OLD2 transports are re-confined to a rank-R_OLD2 sub-basis U
(r1 x r2) INSIDE the shared tier-1 basis Qc:  A_s = Qc U FA2_s.  Legs are formed from
QU = Qc U (r2/n units per leg), the dslice contraction of tier-2 sources is lifted by U^T
into the tier-1 inner product (one trailing Qc^T per layer, unchanged), U rotates with the
tier-1 factors at each rebuild, and the oldest tier-1 member moves into tier 2 in factor
space right after each rebuild (range finder on the r1 x r1 core, no n-space work). Lean
reference (nested tiers 4:384 + 7:256, dumps 0/1): 2.155e-8 vs 2.164e-8 = free.
Kill-switch: V24_AGE_OLD2=0 -> V22 op stream.

V21: sources older than AGE_OLD transports are confined to one shared rank-R_OLD basis
Q (n x r): A_s = Q FA_s, P_s = Q FP_s with static factors. Per old source-layer the
dense legs are formed from the factors (2 n^2 r) and the two dslice contractions go
through the factors (2 n^2 r) instead of 4 dense n^3 units. One source joins the old
group per layer; at a join the basis is rebuilt by a one-pass randomized range finder on
the weighted leg Gram of old + joiner (sketch = a slice of the layer weight), the old
factors are rotated into the new basis and the Gram core S (r x r) is updated. The lean
reference (g_regen_probe.py "...+agerank:3:384") measures +8.4% raw on dumps 0/1 for a
~25% cost cut. Kill-switch: V21_NO_CONFINE=1 -> exact V20 op stream.

V20: exact, bit-identical values; only the client-side op stream changes. The nonlin
term products are written with out= into one persistent (T, n, n) buffer instead of
~30 fresh (n,n) multiplies + a ~50-matrix fnp.stack per layer (measured 4 ms + 3 ms of
residual per layer), and the range-finder sketches are one contiguous copy of the
weight slice per layer instead of strided views (strided qr/matmul input costs ~0.8 ms
each). Kill-switch: none needed (values identical); V19 stays the reference.

V19: exact cost cut, zero accuracy change. The final layer emits only the mean, whose
(1,) nonlin terms need var = diag(C_pre), D3 and the K4 diagonal -- never C_off, the
(2,1) dslice D21, nor the (1,1)/(2,1)/(2,2) term program. V18 computed all of them at
the last layer and discarded them: the two dense dslice contractions (2k n^3 units,
k = 15 live sources) plus the full C_pre sandwich (2 units, only its diagonal is used).
V19 transports only diag(C_pre) = colsum(w32 * (C @ w32)) (1 unit) and runs the source
machinery in D3-only mode at the last layer (transports + n^2 contractions, no D21).
Expected C: 0.509xB -> ~0.48xB. Kill-switch: V19_FULL_LAST=1 -> exact V18 op stream.
Below this line the V18 docstring follows unchanged.


V18: the V1.6 "K3-dslice feedback into the new-block factors" (dropped since V1.6 as
DS-shadowed) is worth -10% raw on the regen base (F69) and is rank <= 16 in D21.
Birth factors of the B1 hub pair become
  X1 = 3 a_b + Xt,   Xt = 1.5 d(w2) D21            ~ F1 R1,  F1 = d(w2) Q,   R1 = 1.5 Bm
  Y1 = a_b d(w2) + Yt, Yt = 0.5 d(w1) D21^T d(w3)  ~ F2 R2,  F2 = d(w1) Bm^T, R2 = 0.5 Q^T d(w3)
with D21 ~ Q Bm from a rank-R_FB randomized range finder (same recipe as Rres). The
identity-born thin legs F1/F2 travel in their own transported stack Zf (like Z, but never
entering the M-leg einsums); in the dslice contraction the new terms either add to the fused left factors
(LA += Xt*P*w2/3 + P*Yt; LP += A*Yt + Xt*A*w2/3 + Xt*Yt/3; D3 += rowsum(P*3*LPadd)) or
contract against the thin right factors ((AP + Xt*P/3) R2^T F2^T and
(AP*w2 + P*Yt)/3 R1^T F1^T). Extra n^3: ZERO; extra ~16 n^2 r per source-layer.
Lean reference: g_regen_probe.py "regen:c+no31+feedc+usec+fitfull+rres:16+sb2+nowk22
+d21r:16": 2.195e-8 / 2.091e-8 on P2 dumps 0/1 (V17 port: 2.51 / 2.25e-8).
Kill-switch: V18_NO_FB=1 -> exact V17 behaviour (zero thin legs).
Below this line the V17 docstring follows unchanged.


V17 (2026-09-02, F68): the augmented-K3 kappa4 channel (F51/F64) without any
per-source kappa4 memory. The kappa4 matrix core G of the aug chain is, to
R^2 ~0.9-0.97, diag(g) + lambda_l * C_off with ONE frozen scalar per layer
(LAM table, fitted offline on public MLPs). Everything the channel needs then
rides on objects V16b already carries:
  - transported diagonal dG_pre = (W*W) g + lambda (diag(C_pre) - (W*W) var)  [n^2]
  - use side: wk4_row = m dG, wk4_mat = (m/6)(dG_i + dG_j) (== V16b's vec+ww form),
    plus the (3,1) slice wk431 = (m/2) lambda C_pre_off feeding 4 extra n^2 terms
  - K4->K3 feed: hub pair X3 = d(w1) G_pre d(w1) = lambda A d(w1) + P d(w1^2 dG)
    (column scalings of the A/P legs), Y3 = (m/4) w2 1^T (rank-1, folds into the
    P-group + one rank-1 outer product), M_t1 = (m/4)(w2*dG)(w1^2)^T (rank-1 M-type
    -> one extra thin Z/L column).  Extra n^3 cost: ZERO.
Lean reference: scratch/lean_k3_aug.py regen chain (g_regen_probe.py
"regen:c+no31+feedc+usec"): 2.196e-8 / 2.161e-8 on P2 dumps 0/1 vs 4.1e-8 for V16b.
Below this line the V16b docstring follows unchanged.


V16b (2026-09-01, F63): exact cost restructuring of the source machinery.
  - The M leg (B3 = Sym(M x P x P)) is never transported: M_b = diag(S3c)
    + 3*S21^T and S21 = S_sep + Rres with S_sep = diag(e) C_off diag(w1) the
    exact leading (2,1) Wick term (a column-scaled copy of a_b) and Rres a
    D21-driven residual that is numerically low-rank (rank-64 keeps final MSE
    at parity, F63). Hence M = P diag(s) + 3 A diag(e) + 3 Z L^T with Z = P Rr
    a thin (n,r) transported leg and L static.
  - V27 (F80): every fresh (n,n)/(k,n,r) result of the layer body goes out= into a
    persistent named buffer (_Pool); thin stacks and shared-basis factors live in
    ping-pong slot buffers (no concatenates); slow einsum forms -> matmul(out=).
  - All dslice contractions collapse into TWO fused einsums per layer (right
    factors A and P) plus O(n^2 r) thin terms: per source-layer 2 dense
    transports + 2 dense contractions (was 3 + 4).
  - Rres range finder: randomized (Omega = a slice of the layer weight), one
    power iteration, flopscope-billed QR; ~4 n^2 r FLOPs per birth.

Port of the reference kprop k_max=3 SIMPLE factored algorithm (companion paper
arXiv:2605.05179) with cost engineering measured in F42-F45, plus two
accuracy riders measured in F47/F49 (combined 1.44x lower final MSE):
  - vec+ww (augmented-lite v0): the radial K4 channel keeps a per-neuron
    diagonal-content vector instead of a single scalar; born from K(4,) and
    the K(2,2) column sums, transported one layer by the true (W*W) row
    action in place of one average-metric cup. ~n^2/layer, cost-neutral.
  - online mean correction: per-layer ridge-fitted delta = feats @ beta
    (13 per-neuron features already computed in the loop; beta fitted
    offline on the public teacher-forced trajectory, embedded below).
    Applied inside the loop so it propagates.
Base cost engineering:
  - shared-P hub blocks per source layer: B1 = Sym(X1 x P x Y1) with X1 = 3*abar,
    B3 = Sym(M x P x P); identity-born legs all evolve by the same map
    leg <- W @ (w1 * leg), so one P per source serves every sub-block, and the
    wick scaling is folded into the next layer's weight (WD).
  - V1.6 ablations (each measured to cost <~1% final MSE): the B2 hub sub-block
    and the K3-dslice feedback into new-block factors are DS-shadowed at birth
    (their diagonal-slice content migrates into the exact M chain automatically);
    the WK22 struct term is dropped from Y1; nonlin terms pruned at relative
    contribution >= 1e-5 (62 of 90 kept).
  - residual-time engineering: per-source matrices held as batched (k,n,n)
    stacks; all wick vectors built at once in the unified form
    w(k,p) = const*sigma^e*(P1(a)phi + P2(a)Phi); nonlin terms evaluated as one
    stacked einsum per arity; dslice hub contractions fused into 3/4-operand
    einsums with no (k,n,n) intermediates; final layer computes only the mean.

Cost: ~1.83e12 flopscope FLOPs per MLP = 0.833x the 2^41 budget (deterministic,
data-independent). Local mini-dump accuracy: mse_final ~3.2e-8 test-set mean
(V1.6 base alone: ~6.2e-8; exact K3-simple reference: 4.8e-8 at an over-budget
2.24x; cov-prop baseline: 4.4e-6).
Prediction: per-layer post-ReLU means, shape (depth, width). No randomness.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

# ---- embedded coefficient tables (generated by dump_k3_tables2.py) ----
# v2 K3-simple tables (dump_k3_tables2.py; pruned at rel>=1e-5)
WICK_PAIRS = [(0, 1),
 (0, 2),
 (0, 3),
 (0, 4),
 (1, 1),
 (1, 2),
 (2, 1),
 (2, 2),
 (3, 1),
 (3, 2),
 (3, 3),
 (3, 4),
 (4, 1),
 (4, 2),
 (4, 3),
 (4, 4),
 (5, 1),
 (5, 2),
 (6, 1),
 (6, 2),
 (7, 1)]

WICK_UNIFIED = [(1.0, 1, [1.0], [0.0, 1.0]),
 (1.0, 2, [0.0, 1.0], [1.0, 0.0, 1.0]),
 (1.0, 3, [2.0, 0.0, 1.0], [0.0, 3.0, 0.0, 1.0]),
 (1.0, 4, [0.0, 5.0, 0.0, 1.0], [3.0, 0.0, 6.0, 0.0, 1.0]),
 (1.0, 0, [], [1.0]),
 (1.0, 1, [2.0], [0.0, 2.0]),
 (1.0, -1, [1.0], []),
 (2.0, 0, [], [1.0]),
 (1.0, -2, [-0.0, -1.0], []),
 (2.0, -1, [1.0], []),
 (6.0, 0, [], [1.0]),
 (1.0, 1, [24.0], [0.0, 24.0]),
 (1.0, -3, [-1.0, 0.0, 1.0], []),
 (2.0, -2, [-0.0, -1.0], []),
 (6.0, -1, [1.0], []),
 (24.0, 0, [], [1.0]),
 (1.0, -4, [-0.0, 3.0, -0.0, -1.0], []),
 (2.0, -3, [-1.0, 0.0, 1.0], []),
 (1.0, -5, [3.0, 0.0, -6.0, 0.0, 1.0], []),
 (2.0, -4, [-0.0, 3.0, -0.0, -1.0], []),
 (1.0, -6, [-0.0, -15.0, -0.0, 10.0, -0.0, -1.0], [])]

TERM_SPECS = {(1,): [('ones1', 'ones1', 0, None, 1.0),
        ('d3', 'ones1', 8, None, 0.16666666666666666),
        ('g4', 'ones1', 12, None, 0.041666666666666664),
        ('d3', 'd3', 18, None, 0.013888888888888888),
        ('d3', 'g4', 20, None, 0.006944444444444444)],
 (1, 1): [('c_off', 'ones2', 4, 4, 1.0),
          ('d21T', 'ones2', 4, 6, 1.0),
          ('c_off', 'c_off', 6, 6, 0.5),
          ('wk4m', 'ones2', 6, 6, 0.25),
          ('d3row', 'c_off', 4, 12, 0.3333333333333333),
          ('c_off', 'd21T', 6, 8, 1.0),
          ('d3row', 'd21T', 4, 16, 0.16666666666666666),
          ('g4row', 'c_off', 4, 16, 0.08333333333333333),
          ('d3row', 'd21', 6, 12, 0.16666666666666666),
          ('d21T', 'd21T', 6, 12, 0.25),
          ('c_off', 'wk4m', 8, 8, 0.25),
          ('d21T', 'd21', 8, 8, 0.25),
          ('g4row', 'd21T', 4, 18, 0.041666666666666664),
          ('d3row', 'wk4m', 6, 16, 0.08333333333333333),
          ('g4row', 'd21', 6, 16, 0.041666666666666664),
          ('d21T', 'wk4m', 8, 12, 0.25),
          ('g4row', 'wk4m', 6, 18, 0.020833333333333332)],
 (2,): [('ones1', 'ones1', 1, None, 1.0),
        ('d3', 'ones1', 9, None, 0.16666666666666666),
        ('g4', 'ones1', 13, None, 0.041666666666666664)],
 (2, 1): [('c_off', 'ones2', 5, 4, 1.0),
          ('d21T', 'ones2', 5, 6, 0.5),
          ('d21', 'ones2', 7, 4, 0.5),
          ('c_off', 'c_off', 7, 6, 0.5),
          ('wk4m', 'ones2', 7, 6, 0.25),
          ('d3row', 'c_off', 5, 12, 0.16666666666666666),
          ('c_off', 'd21T', 7, 8, 0.5),
          ('c_off', 'd21', 9, 6, 0.5),
          ('c_off', 'd3col', 13, 4, 0.16666666666666666),
          ('d3row', 'd21T', 5, 16, 0.08333333333333333),
          ('g4row', 'c_off', 5, 16, 0.041666666666666664),
          ('d3row', 'd21', 7, 12, 0.08333333333333333),
          ('d21T', 'd21T', 7, 12, 0.125),
          ('c_off', 'wk4m', 9, 8, 0.25),
          ('d21T', 'd21', 9, 8, 0.25),
          ('c_off', 'g4col', 17, 4, 0.041666666666666664),
          ('d21', 'd3col', 17, 4, 0.08333333333333333),
          ('g4row', 'd21T', 5, 18, 0.020833333333333332),
          ('g4row', 'd21', 7, 16, 0.020833333333333332),
          ('d21', 'g4col', 19, 4, 0.020833333333333332)],
 (2, 2): [('c_off', 'ones2', 5, 5, 1.0),
          ('d21T', 'ones2', 5, 7, 1.0),
          ('c_off', 'c_off', 7, 7, 0.5),
          ('wk4m', 'ones2', 7, 7, 0.25),
          ('d3row', 'c_off', 5, 13, 0.3333333333333333),
          ('c_off', 'd21T', 7, 9, 1.0),
          ('d3row', 'd21T', 5, 17, 0.16666666666666666),
          ('g4row', 'c_off', 5, 17, 0.08333333333333333),
          ('d21T', 'd21T', 7, 13, 0.25),
          ('c_off', 'wk4m', 9, 9, 0.25),
          ('g4row', 'd21T', 5, 19, 0.041666666666666664)],
 (3,): [('ones1', 'ones1', 2, None, 1.0),
        ('d3', 'ones1', 10, None, 0.16666666666666666),
        ('g4', 'ones1', 14, None, 0.041666666666666664)],
 (4,): [('ones1', 'ones1', 3, None, 1.0),
        ('d3', 'ones1', 11, None, 0.16666666666666666),
        ('g4', 'ones1', 15, None, 0.041666666666666664)]}

PK2K_TABLE = {(1,): [(((1,),), 1.0)],
 (1, 1): [(((1, 1),), 1.0)],
 (2,): [(((1,), (1,)), -1.0), (((2,),), 1.0)],
 (2, 1): [(((1, 0), (1, 1)), -2.0), (((2, 1),), 1.0)],
 (2, 2): [(((0, 1), (1, 0), (1, 1)), 4.0),
          (((0, 1), (2, 1)), -2.0),
          (((1, 0), (1, 2)), -2.0),
          (((1, 1), (1, 1)), -2.0),
          (((2, 2),), 1.0)],
 (3,): [(((1,), (1,), (1,)), 2.0), (((1,), (2,)), -3.0), (((3,),), 1.0)],
 (4,): [(((1,), (1,), (1,), (1,)), -6.0),
        (((1,), (1,), (2,)), 12.0),
        (((1,), (3,)), -4.0),
        (((2,), (2,)), -3.0),
        (((4,),), 1.0)]}

# Online mean-correction coefficients, (depth, n_features) = (16, 13); ridge-fitted
# offline (lam=1e-3) on the teacher-forced trajectory of 5 public v2-phase2 MLPs
# (k3_aug_corr.py, F47/F49). Feature order:
# 1, mu_pre, var_pre, sigma, alpha, |alpha|, phi, Phi, pred, D3, D21n, K4, sigma*phi
CORR_BETA = [
    [0.0006411968414813632, 0.0, 0.00032431282848653057, -0.0003042144758593203, 0.0, 0.0, 0.0, 0.0, -0.0007625526069403323, 0.0, 0.0, 0.0, -0.0007625526055534857],
    [-0.0004483531031709554, -8.08970612775365e-08, 0.00024032689914639557, 0.0003234947612489975, 7.734051179371133e-06, -8.401877300643312e-06, 0.0026976072084027533, -2.9522256040202606e-06, -1.1604426266687203e-05, 9.457618761612473e-06, -3.3186036376400295e-05, -0.008465131129601192, -0.002371947086869632],
    [-0.00030645743691215543, 2.660655267636766e-06, 0.0006192175275623578, -0.0002468827461681828, -2.9900896046038434e-06, 6.841942903175705e-08, 0.0033128783967715193, 5.123025223409135e-06, -1.831086058750464e-06, -8.948552609048747e-06, -3.292520087183994e-05, 0.024826431355466128, -0.003290775627340336],
    [-0.00010464301263958924, 1.0735707741330386e-06, 0.0009154900192795534, -0.0006467314093895341, 3.6870994916648376e-06, 6.046342808060834e-06, 0.00341846204288833, -8.975061063301133e-06, -8.178979563111877e-06, 9.51453794000349e-05, -3.005588379050443e-05, 0.031163100958057806, -0.003816642326501922],
    [-0.0006681684602995246, -7.2789662985629386e-06, 2.591925967081249e-05, 0.0007797247634619618, 8.477664175402185e-06, 1.7025270836169033e-06, 0.003419282618548069, 7.535308544950828e-07, -5.733309150056555e-06, -2.9160257628201262e-05, -1.722583323732972e-05, 0.04076990884979579, -0.004248021529544134],
    [-0.0006734457317912116, 3.1290157938005224e-05, 0.00022889868689835653, 0.0007465462593369972, 6.851356914590185e-07, 2.1755554980020755e-05, 0.004058574189699476, -4.277653393515956e-06, -6.039384411257449e-05, -9.02015920342041e-05, -1.9819140691513135e-05, 0.041867576221848744, -0.005466361831539052],
    [-0.0006750844151648486, 6.180105939061905e-05, 0.0004389649313793868, 0.0006339652749835816, -6.694983507237225e-07, 4.690187390705702e-05, 0.003977857680309406, 4.383031188174939e-06, -0.00012887677523401884, 0.0002505563048388552, -2.5566781991513985e-05, 0.07694913516123503, -0.0057040529266279925],
    [-0.000972879718186106, 5.1156558236482596e-05, -0.0004913239875599442, 0.0017893987001835238, 1.90937415308986e-06, 3.802173926700858e-05, 0.0038578329372384434, 2.8522169318710464e-06, -0.00010738838919747425, -0.00015415811216305824, -2.5539604454967108e-05, 0.08311668913952446, -0.005918296132102073],
    [-0.0007850991486069435, 7.899688664131035e-05, -0.00022144314048615243, 0.001495955739849875, 1.8359978888302925e-07, 5.023393155879106e-05, 0.0034762600760059295, 3.215904175585605e-06, -0.00015314900768603073, -0.0004475649473906088, -3.028530737478287e-05, 0.06152086703424944, -0.00569825101925022],
    [-0.0005795122172445841, 0.00011465424961793733, 0.00039693472854898914, 0.0007970833957439235, 6.825630979361196e-06, 7.966232390898656e-05, 0.0032776762940374586, -1.387178929061937e-06, -0.0002567385150822406, 0.00028972790478315425, -2.7005665153789784e-05, 0.07590370226079919, -0.005546910408445549],
    [-0.0004949253254431177, 9.892837012135289e-05, 0.0005826341044064549, 0.0006128630824666212, 1.8258883411132417e-06, 6.024891049057027e-05, 0.003354817421260185, -6.690225809034046e-06, -0.00020554230085297872, 0.0002695214755196366, -2.5128600253275433e-05, 0.08226262500093764, -0.006092394095370619],
    [-0.0007584153570441029, 0.0001792621247156503, 0.0001931413288044634, 0.0014013628928244146, 5.054405581529621e-06, 0.00010157511263920278, 0.004187883074782575, -5.420295053799967e-06, -0.00037802160373858374, 0.00014440604770373433, -2.2764810929109396e-05, 0.07738752039037704, -0.008027695439553941],
    [-0.0008291520607940194, 0.0002115021197150293, -0.00012740149902796318, 0.0017326132691935436, -2.1624432383474912e-06, 0.0001087192480144056, 0.004533661385130805, 4.837349485145022e-07, -0.00041674041200178005, 0.00018467937018199515, -1.9126957306380207e-05, 0.07896437764069765, -0.008986219483970914],
    [-0.000658293372173689, 0.00018380103458278174, 0.00046374648579651084, 0.001216035726510531, 4.012828172272205e-06, 9.570320459811334e-05, 0.004325867834172417, -1.2894882645758776e-05, -0.00038451298928645256, 0.0005113367132840055, -2.241500653445156e-05, 0.07104629639605271, -0.00897436256704376],
    [-0.0009527557374376014, 0.00019049799075311402, -0.0009213478791293237, 0.0025553941675787975, 3.508060734949607e-07, 9.248186905581894e-05, 0.004641633979578725, -3.1682752945685096e-06, -0.00038201626600816054, 3.0155769164602614e-05, -2.3334690641924517e-05, 0.07182226701958777, -0.01001369215216409],
    [-0.0006908532933348147, 0.00018408588405074964, 0.0004249876754929348, 0.0013835702671030146, 5.170891351788878e-06, 9.144035619901057e-05, 0.0044972055980902715, -1.0115381018603137e-05, -0.00038927641644205714, 0.0002930281376816756, -2.8026123982155874e-05, 0.157561818094021, -0.010070547059124272],
]

# ---- end embedded tables ----

# Regenerated kappa4 off-diagonal coefficient per layer, fitted at the PRE-activation
# (transported G_pre_off = LAM[l] * C_pre_off after layer l's ReLU); LS fits on the dense
# aug chain, mean over public P2 dumps 0-7 (runs/lam_prefit_d0-7.log, V22). The older
# post-ReLU table (runs/g_regen_10.log) differs mainly at layer 0 (1.95e-3 -> 4.99e-3).
# Suite shape only (gated like CORR_BETA).
LAM = [4.9895e-03, 8.0876e-03, 9.8291e-03, 1.0549e-02, 1.0851e-02, 1.0828e-02, 1.0589e-02, 1.0048e-02, 9.6483e-03, 9.1720e-03, 8.7730e-03, 8.3938e-03, 8.0555e-03, 7.6770e-03, 7.2588e-03, 7.2588e-03]
METRIC_C = 2.0
# debug kill-switches (parity_v17.py): set from the environment before import
import os as _os
import gc as _gc
NO_FEED = _os.environ.get("V17_NO_FEED", "0") == "1"
NO_WK431 = _os.environ.get("V17_NO_WK431", "0") == "1"
NO_REGEN = _os.environ.get("V17_NO_REGEN", "0") == "1"
NO_FB = _os.environ.get("V18_NO_FB", "0") == "1"  # V18: D21 feedback thin legs off
FULL_LAST = _os.environ.get("V19_FULL_LAST", "0") == "1"  # V19: 1 -> V18 op stream
NO_CONFINE = _os.environ.get("V21_NO_CONFINE", "0") == "1"  # V21: 1 -> V20 op stream
QPASS = int(_os.environ.get("V21_QPASS", "1"))  # V21: subspace-iteration passes of the basis range finder
NO_SRC_LAST = _os.environ.get("V19_NO_SRC_LAST", "0") == "1"  # V19 probe: D3(last) := 0, skip source transports
# The V16b CORR_BETA mean rider was ridge-fitted on V16b's OWN trajectory and is
# anti-correlated with V16b's base error; on the regen base it HURTS (dump0: 3.57e-8
# with vs 2.51e-8 without). Off by default until refitted on the V17 trajectory.
NO_CORR = _os.environ.get("V17_NO_CORR", "1") == "1"
# V25: table scale 0.95 (8-dump scan 0.9/0.92/0.95/1.0 -> 2.2752/2.2727/2.2704/2.2782e-8)
LAM = [c * float(_os.environ.get("V17_LAM_SCALE", "1.0")) for c in LAM]
# V25: reference ratio mean(dG)/mean(var) at layer l+1 (8-dump mean, scratch/lam_obs_d0-7.npz)
# and the log-log exponent of the adaptive rule (pooled fit 1.08; 1.0 shipped).
REF_R = [6.58815e-03, 8.18414e-03, 8.53136e-03, 8.38859e-03, 8.10153e-03, 7.74287e-03,
         7.35951e-03, 6.91093e-03, 6.55752e-03, 6.17892e-03, 5.83953e-03, 5.56824e-03,
         5.32459e-03, 5.05165e-03, 4.77181e-03]
BETA = float(_os.environ.get("V25_BETA", "1.0"))
DEBUG = []  # parity_v17.py: per-layer dict of diagnostics when V17_DEBUG=1

# pruned V16b table + the 4 (3,1)-slice use-side terms
_I = WICK_PAIRS.index
TERM_SPECS[(1, 1)].append(('wk431', 'ones2', _I((1, 1)), _I((3, 1)), 1.0 / 3.0))
TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 1)), 1.0 / 6.0))
TERM_SPECS[(2, 1)].append(('wk431', 'ones2', _I((3, 2)), _I((1, 1)), 1.0 / 6.0))
TERM_SPECS[(2, 2)].append(('wk431', 'ones2', _I((1, 2)), _I((3, 2)), 1.0 / 3.0))
ALPHA_DEG = 5
SIG_EMIN, SIG_EMAX = -6, 4
D2_IPS = [(1, 1), (2, 1), (2, 2)]
D1_IPS = [(1,), (2,), (3,), (4,)]
D2_OUT = [(1, 1), (2, 1), (2, 2)]
D1_OUT = [(2,), (3,), (4,)]
MODE0_MISSING = {"d21", "d21T", "d3col", "d3row", "d3", "wk4m", "g4col", "g4row", "g4",
                 "wk431"}


def _statics(n: int) -> dict:
    c = 4 - 2 + n / 2.0 - 1.0
    P2 = (c - 2.0) / (16.0 * 2.0 * c * (1.0 - c) * (2.0 - c))
    return dict(P2=P2, k4_c4=24.0, k4_c22=24.0, wk4_c4=1.0, wk4_c22=1.0 / 3.0,
                cA=6.0 / (n + 4.0), cI=-3.0 / ((n + 2.0) * (n + 4.0)))


def _zero_diag(A):
    fnp.fill_diagonal(A, 0.0)
    return A


def _build_wick_consts():
    np_ = len(WICK_PAIRS)
    C1 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    C2 = [[0.0] * np_ for _ in range(ALPHA_DEG + 1)]
    # sigma-power selection with the constant prefactor folded in
    SELC = [[0.0] * np_ for _ in range(SIG_EMAX - SIG_EMIN + 1)]
    for j, (const, e, p1, p2) in enumerate(WICK_UNIFIED):
        SELC[e - SIG_EMIN][j] = const
        for i, c in enumerate(p1):
            C1[i][j] = c
        for i, c in enumerate(p2):
            C2[i][j] = c
    return C1, C2, SELC


def _term_prog(mode: int):
    """Fused-einsum program for the nonlin terms of one availability mode."""
    missing = MODE0_MISSING if mode == 0 else ({"wk431"} if mode == 2 else set())
    np_ = len(WICK_PAIRS)
    d2 = []  # (group, a, b, wl, wr, coef)
    d1 = []
    for g, ip in enumerate(D2_IPS):
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            if a in missing or b in missing:
                continue
            d2.append((g, a, b, wl, wr, coef))
    for g, ip in enumerate(D1_IPS):
        for a, b, wl, wr, coef in TERM_SPECS.get(ip, []):
            if a in missing or b in missing:
                continue
            d1.append((g, a, b, wl, coef))
    SL2 = [[0.0] * np_ for _ in d2]
    SR2 = [[0.0] * np_ for _ in d2]
    IND2 = [[0.0] * len(D2_IPS) for _ in d2]
    AB2 = []
    for t, (g, a, b, wl, wr, coef) in enumerate(d2):
        SL2[t][wl] = coef
        SR2[t][wr] = 1.0
        IND2[t][g] = 1.0
        AB2.append((a, b) if a <= b else (b, a))
    SL1 = [[0.0] * np_ for _ in d1]
    IND1 = [[0.0] * len(D1_IPS) for _ in d1]
    B1 = []
    for t, (g, a, b, wl, coef) in enumerate(d1):
        SL1[t][wl] = coef
        IND1[t][g] = 1.0
        B1.append((a, b) if a <= b else (b, a))
    return dict(SL2=SL2, SR2=SR2, IND2=IND2, AB2=AB2, SL1=SL1, IND1=IND1, B1=B1)

# ---- V26 (F79): Strassen-Winograd block products on pooled buffers ----------------
STRASSEN_LEVELS = int(_os.environ.get("V26_STRASSEN", "5"))   # V28 ladder (suite harness, C/B | residual): L3 0.3107 | 0.170 s, L4 0.2973 | 0.187, L5 0.2872 | 0.205, L5 + shared basis 0.2680 | 0.242   # dump-0 ladder: 1/2/3 -> C/B 0.344/0.327/0.3125; mini residual at L3 = 0.356 s mean with a 0.42 s tail (7/100 failed) -> ship L1 until the base residual is cut
STRASSEN_HUB = int(_os.environ.get("V26_STRASSEN_HUB", str(STRASSEN_LEVELS)))
STRASSEN_NEW = int(_os.environ.get("V26_STRASSEN_NEW", "0"))   # V29: unused (the newborn rides the family)
CPRE_LEV = int(_os.environ.get("V29_CPRE_LEV", _os.environ.get("V26_STRASSEN", "5")))  # V29: C_pre block family levels
STRASSEN_SB = int(_os.environ.get("V28_STRASSEN_SB", str(STRASSEN_LEVELS)))   # V28: shared-basis formings + contractions   # newborn transport levels (0: ~30 ms residual for 0.5% cost is a bad trade)
STRASSEN_MIN = int(_os.environ.get("V26_STRASSEN_MIN", "32"))  # smallest leaf block side (V28: 32; f32 error grows ~1.4x per level, MSE unchanged to 4 digits at L5)
STRASSEN_FIRST = int(_os.environ.get("V28_STRASSEN_FIRST", "4"))
STRASSEN_FUSE_P = int(_os.environ.get("V28_STRASSEN_FUSE_P", "343"))  # V28: fused per-product leaf when the leaf batch has >= this many blocks (deep levels: 7x smaller pools, +4 ops per leaf)  # V28: level cap of the first predict() of a process (see _predict_core)
WARM = _os.environ.get("V26_WARM", "1") == "1"

# ---- V30: dead-ReLU pruning of the K3 leg machinery (suite shape only) ------------------
PRUNE = _os.environ.get("V30_PRUNE", "1") == "1"


def _sched(name, default):
    return [int(x) for x in _os.environ.get(name, default).split(",")]


# rows of the K3 legs kept at layer l (prefix of the alpha-sorted relabeling)
S_OUT = _sched("V30_S_OUT", "1024,1024,1024,1024,1024,1024,1024,992,992,960,960,960,928,928,928,1024")
# rows kept as the inner index of the transport out of layer l (w1-scaled legs)
S_IN = _sched("V30_S_IN", "1024,1024,1024,992,960,928,896,896,864,864,832,832,800,800,768,1024")
PRUNE_EMU = _os.environ.get("V30_EMU", "0") == "1"   # 1: zero-row emulation (no cost change)
# leading (saturated, alpha >> 0) units of layer l with no K3 readout rows
S_ON = _sched("V30_S_ON", "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
# first layer whose units are relabeled (the first one with a pruned row count)
LR0 = min([i for i in range(len(S_OUT)) if S_OUT[i] < 1024 or S_IN[i] < 1024] + [len(S_OUT)])
LR0 = int(_os.environ.get("V30_LR0", str(LR0)))
SMIN_PR = int(_os.environ.get("V30_SMIN", "24"))     # leaf block minimum of the pruned families
# V36: first layer with no more joins to the shared basis (sources born late stay dense:
# a join costs more than the few transports it would save near the output)
NOJOIN_FROM = int(_os.environ.get("V36_NOJOIN", "99"))
# V38: |alpha| bound of the hub columns a newborn source keeps (0 = all)
HUB_A = float(_os.environ.get("V38_HUB_A", "0"))
# V37: the shared-basis families are rectangular (inner side r = 384 / 224): a Strassen level
# still pays with a small inner block as long as the two outer sides are large, so their leaf
# minimum is lower than the square families' (r = 384 -> 12, r = 224 -> 7 at five levels)
SMIN_SB = int(_os.environ.get("V37_SMIN_SB", "7"))
# V37: recursion levels of the join's own products (range finder, projections, rotation of
# the old factors, basis transport W Qc, lift inner Qc^T); 0 = the dense V29 products
STRASSEN_JOIN = int(_os.environ.get("V37_SJOIN", "5"))
# V37: size cap of a batched Strassen operand, in elements at the top level (slots x rows x
# cols). V29's own largest family is 12 slots of (1024, 1024) at five levels (472 MB combo
# buffer); a family that would exceed it runs one level shallower.
MAX_EL = int(_os.environ.get("V37_MAX_EL", str(12 * 1024 * 1024)))
# probes (emulation only): effective basis ranks per layer (trailing basis columns zeroed)
R1S = _sched("V36_R1S", "0")
R2S = _sched("V36_R2S", "0")


class _VH:
    """V37: a scratch view together with its sub-views, made once. A slice of a remote array
    is a client/server round trip (half of all calls of V30 were slices of the pooled
    Strassen buffers); the pools persist across layers and MLPs, so every sub-view of a
    pooled buffer is created on first use and reused afterwards."""
    __slots__ = ("a", "s7", "q4", "h0", "sw", "qs")

    def __init__(self, a):
        self.a = a
        self.s7 = None    # the seven [:, i*P:(i+1)*P] slabs
        self.q4 = None    # quadrants of the last two axes: (00, 01, 10, 11)
        self.h0 = None    # holder of a[0]
        self.sw = None    # swapaxes(a, -1, -2)
        self.qs = None    # swapped views of the quadrants (hub leaf right operands)

    def sev(self):
        s = self.s7
        if s is None:
            a = self.a
            P = a.shape[1] // 7
            s = self.s7 = tuple(a[:, i * P:(i + 1) * P] for i in range(7))
        return s

    def quad(self):
        qd = self.q4
        if qd is None:
            a = self.a
            h, q = a.shape[-2] // 2, a.shape[-1] // 2
            qd = self.q4 = (a[..., :h, :q], a[..., :h, q:], a[..., h:, :q], a[..., h:, q:])
        return qd

    def first(self):
        e = self.h0
        if e is None:
            e = self.h0 = _VH(self.a[0])
        return e

    def swapped(self):
        s = self.sw
        if s is None:
            s = self.sw = fnp.swapaxes(self.a, -1, -2)
        return s

    def quad_swapped(self, i):
        d = self.qs
        if d is None:
            d = self.qs = {}
        s = d.get(i)
        if s is None:
            s = d[i] = fnp.swapaxes(self.quad()[i], -1, -2)
        return s


class _Strassen:
    """Recursive Strassen on batched operands. Layouts (last two axes = the matrix):
      plain: X (bx, P, m, kd) @ Y (by, P, kd, w) -> out (by, P, m, w), bx in {1, by}
      hub:   X (k, P, m, kd), Y (k, P, w, kd)    -> out (P, m, w) = sum_k X_k @ Y_k^T
    Every combo / product is ONE flopscope op over the whole batch, written with out=
    into pooled buffers shared by role (F53: fresh result buffers dominate residual).
    The seven products: M1=(X11+X22)(Y11+Y22) M2=(X21+X22)Y11 M3=X11(Y12-Y22)
    M4=X22(Y21-Y11) M5=(X11+X12)Y22 M6=(X21-X11)(Y11+Y12) M7=(X12-X22)(Y21+Y22);
    C11=M1+M4-M5+M7 C12=M3+M5 C21=M2+M4 C22=M1-M2+M3+M6.  For the hub kernel the
    right operand is Y^T, whose blocks are (Y^T)11=Y[..,:h,:q], (Y^T)12=Y[..,h:,:q],
    (Y^T)21=Y[..,:h,q:], (Y^T)22=Y[..,h:,q:] (no transposes: the kernel contracts j).
    V37: operands travel as _VH holders (cached sub-views); the op stream of every family
    is unchanged (same ops, same order, same buffers)."""

    def __init__(self, dtype, bmax):
        self.dtype = dtype
        self.bmax = int(bmax)
        self.flat = {}    # V30: (role, P) -> flat scratch buffer shared by every block shape
        self.views = {}   # V30: ((role, P), c, h, q) -> (cap, c, h, q) view of its prefix
        self.hold = {}    # V37: ((role, P), c, h, q, b) -> _VH of the (b, c, h, q) prefix
        self.smin = STRASSEN_MIN   # V30: smallest leaf block side (per family, see _predict_core)

    def clear(self):
        self.flat = {}
        self.views = {}
        self.hold = {}

    def _bufh(self, key, shape4):
        """V30: scratch buffer (b, c, h, q) for role key[0] at recursion width key[1].
        The pruned families run at a different block shape (h, q) on every layer, so the
        pools can no longer be keyed by shape (they would pile up) nor rebuilt per layer
        (freeing GBs is residual time). One flat buffer per (role, P) serves every shape
        through a cached reshaped view of its prefix. Buffers are pure scratch inside one
        family call. V37: returns the cached holder of the (b, c, h, q) prefix."""
        b, c, h, q = (int(x) for x in shape4)
        rk = (key[0], key[1])
        hk = (rk, c, h, q, b)
        e = self.hold.get(hk)
        if e is not None:
            return e
        vk = (rk, c, h, q)
        v = self.views.get(vk)
        if v is None or v.shape[0] < b:
            per = c * h * q
            f = self.flat.get(rk)
            if f is None or f.shape[0] < per * b:
                f = fnp.empty((per * b,), dtype=self.dtype)
                self.flat[rk] = f
                self.views = {k_: v_ for k_, v_ in self.views.items() if k_[0] != rk}
                self.hold = {k_: v_ for k_, v_ in self.hold.items() if k_[0] != rk}
            cap = f.shape[0] // per
            v = fnp.reshape(f[:cap * per], (cap, c, h, q))
            self.views[vk] = v
        e = _VH(v if v.shape[0] == b else v[:b])
        self.hold[hk] = e
        return e

    def level(self, m, kd, w, lev):
        """V28: largest level <= lev at which every block side divides and stays
        >= the leaf minimum (a family whose sides do not allow `lev` runs shallower
        instead of falling back to dense)."""
        while lev > 0 and not self._ok(m, kd, w, lev):
            lev -= 1
        return lev

    def _ok(self, m, kd, w, lev):
        d = 2 ** lev
        return (lev > 0 and m % d == 0 and kd % d == 0 and w % d == 0
                and min(m, kd, w) // d >= self.smin)

    @staticmethod
    def _fit(lev, *elems):
        """V37: largest level <= lev whose biggest stored combo buffer stays within MAX_EL
        at five levels (combos are stored at depths 1..lev-1 when the leaf is fused, i.e.
        from level 4 on, and at depths 1..lev below that)."""
        top = max(elems)
        cap = MAX_EL * 1.75 ** 4
        while lev > 1 and top * 1.75 ** (lev - 1 if lev >= 4 else lev) > cap:
            lev -= 1
        return lev

    def _combos(self, Xh, kind, key, tq=False):
        """Seven Strassen combos of the quadrants (X11,X12,X21,X22) of X (b, P, m, kd) into
        a pooled (b, 7P, h, q) buffer (combo i = slab [:, i*P:(i+1)*P]). tq: the quadrants
        are taken in the transposed block order (hub right operand)."""
        b, P, m, kd = Xh.a.shape
        Bh = self._bufh(key, (b, 7 * P, m // 2, kd // 2))
        o = Bh.sev()
        if tq:
            X11, X21, X12, X22 = Xh.quad()
        else:
            X11, X12, X21, X22 = Xh.quad()
        if kind == "L":
            fnp.add(X11, X22, out=o[0]); fnp.add(X21, X22, out=o[1])
            fnp.copyto(o[2], X11);       fnp.copyto(o[3], X22)
            fnp.add(X11, X12, out=o[4]); fnp.subtract(X21, X11, out=o[5])
            fnp.subtract(X12, X22, out=o[6])
        else:
            fnp.add(X11, X22, out=o[0]); fnp.copyto(o[1], X11)
            fnp.subtract(X12, X22, out=o[2]); fnp.subtract(X21, X11, out=o[3])
            fnp.copyto(o[4], X22);       fnp.add(X11, X12, out=o[5])
            fnp.add(X21, X22, out=o[6])
        return Bh

    @staticmethod
    def _assemble(Mh, outh):
        """M (b, 7P, h, w) products -> out (b, P, m, w) quadrants (8 ops)."""
        M = Mh.sev()
        C11, C12, C21, C22 = outh.quad()
        fnp.add(M[0], M[3], out=C11); fnp.subtract(C11, M[4], out=C11)
        fnp.add(C11, M[6], out=C11)
        fnp.add(M[2], M[4], out=C12)
        fnp.add(M[1], M[3], out=C21)
        fnp.subtract(M[0], M[1], out=C22); fnp.add(C22, M[2], out=C22)
        fnp.add(C22, M[5], out=C22)

    def mm(self, X, Y, out, lev):
        """plain: X (bx, P, m, kd), Y (by, P, kd, w) -> out (by, P, m, w). Operands may be
        arrays or _VH holders (persistent operands: pass a holder to reuse its sub-views)."""
        Xh = X if isinstance(X, _VH) else _VH(X)
        Yh = Y if isinstance(Y, _VH) else _VH(Y)
        bx, P, m, kd = Xh.a.shape
        by, w = Yh.a.shape[0], Yh.a.shape[3]
        lev = self._fit(lev, bx * P * m * kd, by * P * kd * w, max(bx, by) * P * m * w)
        self._mm(Xh, Yh, out if isinstance(out, _VH) else _VH(out), lev)

    def _mm(self, Xh, Yh, outh, lev):
        bx, P, m, kd = Xh.a.shape
        by, w = Yh.a.shape[0], Yh.a.shape[3]
        if not self._ok(m, kd, w, lev):
            fnp.matmul(Xh.a, Yh.a, out=outh.a)
            return
        h, q, v = m // 2, kd // 2, w // 2
        if P >= STRASSEN_FUSE_P and not self._ok(h, q, v, lev - 1):
            # V28: fused leaf (deep levels only) -- one product at a time into the out quadrants
            self._leaf_mm(Xh.quad(), Yh.quad(), outh.quad(), bx, by, P, h, q, v)
            return
        Xc = self._combos(Xh, "L", ("L", P))
        Yc = self._combos(Yh, "R", ("R", P))
        # V28: the products reuse this level's left-combo buffer when the callee recurses
        # (its combos consume Xc before anything is written into out); a fused-leaf callee
        # reads X while writing out, so it keeps a separate M. Square blocks only.
        callee_recurses = self._ok(h // 2, q // 2, v // 2, lev - 2)
        mkey = ("L", P) if (callee_recurses and q == v) else ("M", P)
        Mh = self._bufh(mkey, (by, 7 * P, h, v))
        self._mm(Xc, Yc, Mh, lev - 1)
        self._assemble(Mh, outh)

    def _leaf_mm(self, XQ, YQ, OQ, bx, by, P, h, q, v):
        """Fused leaf of mm: M6->C22, M4->C11, M1 (+C11, +C22), M2->C21 (C22-=), M3->C12
        (C22+=), M5 (C11-=, C12+=), M7 (C11+=): 7 matmul writes + 8 adds, one combo
        pair alive at a time (copies for the raw-quadrant operands keep every matmul
        input contiguous)."""
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        Lb = self._bufh(("L1", P), (bx, P, h, q)).a
        Rb = self._bufh(("R1", P), (by, P, q, v)).a
        Mb = self._bufh(("M1", P), (by, P, h, v)).a
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        fnp.subtract(X21, X11, out=Lb); fnp.add(Y11, Y12, out=Rb)
        fnp.matmul(Lb, Rb, out=C22)                                   # M6 -> C22
        fnp.add(X21, X22, out=Lb)
        fnp.matmul(Lb, Y11, out=C21)                                  # M2 -> C21
        fnp.subtract(C22, C21, out=C22)                               # C22 -= M2
        fnp.subtract(Y21, Y11, out=Rb)
        fnp.matmul(X22, Rb, out=C11)                                  # M4 -> C11
        fnp.add(C21, C11, out=C21)                                    # C21 += M4
        fnp.add(X11, X22, out=Lb); fnp.add(Y11, Y22, out=Rb)
        fnp.matmul(Lb, Rb, out=Mb)                                    # M1
        fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb)
        fnp.matmul(X11, Rb, out=C12)                                  # M3 -> C12
        fnp.add(C22, C12, out=C22)                                    # C22 += M3
        fnp.add(X11, X12, out=Lb)
        fnp.matmul(Lb, Y22, out=Mb)                                   # M5
        fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb); fnp.add(Y21, Y22, out=Rb)
        fnp.matmul(Lb, Rb, out=Mb)                                    # M7
        fnp.add(C11, Mb, out=C11)

    def hub(self, X, Y, out, lev):
        """hub: X (k, P, m, kd), Y (k, P, w, kd) -> out (P, m, w) = sum_k X_k Y_k^T.
        `out` is the (P, m, w) array or a _VH holder of its (1, P, m, w) view."""
        if isinstance(out, _VH):
            outh = out
        else:
            outh = _VH(out[None])
            outh.h0 = _VH(out)
        Xh = X if isinstance(X, _VH) else _VH(X)
        Yh = Y if isinstance(Y, _VH) else _VH(Y)
        k, P, m, kd = Xh.a.shape
        w = Yh.a.shape[2]
        lev = self._fit(lev, k * P * m * kd, k * P * w * kd, P * m * w)
        self._hub(Xh, Yh, outh, lev)

    def _hub(self, Xh, Yh, outh, lev):
        k, P, m, kd = Xh.a.shape
        w = Yh.a.shape[2]
        if not self._ok(m, kd, w, lev):
            # dense: batched GEMM over (k, P) then a k-sum (the einsum form
            # 'kpij,kpcj->pic' takes a slow non-BLAS path: 0.5 s per call at 512^2)
            T = self._bufh(("K", P), (k, P, m, w)).a
            fnp.matmul(Xh.a, Yh.swapped(), out=T)
            fnp.sum(T, axis=0, out=outh.first().a)
            return
        h, q, v = m // 2, kd // 2, w // 2
        if P >= STRASSEN_FUSE_P and not self._ok(h, q, v, lev - 1):
            self._leaf_hub(Xh, Yh, outh.first().quad(), k, P, h, q, v)
            return
        Xc = self._combos(Xh, "L", ("L", P))
        # (Y^T) blocks expressed on the un-transposed Y (rows c, cols j)
        Yc = self._combos(Yh, "R", ("R", P), tq=True)
        Mh = self._bufh(("HM", P), (1, 7 * P, h, v))
        self._hub(Xc, Yc, Mh, lev - 1)
        self._assemble(Mh, outh)

    def _leaf_hub(self, Xh, Yh, OQ, k, P, h, q, v):
        """Fused leaf of hub (same product order as _leaf_mm); each product is a batched
        GEMM over (k, P) into T then a k-sum into its quadrant / the (P, h, v) scratch."""
        X11, X12, X21, X22 = Xh.quad()
        Y11, Y21, Y12, Y22 = Yh.quad()       # transposed block order (see class note)
        C11, C12, C21, C22 = OQ
        Lb = self._bufh(("L1", P), (k, P, h, q)).a
        Rh = self._bufh(("R1", P), (k, P, v, q))
        Rb = Rh.a
        RbT = Rh.swapped()
        Y11T = Yh.quad_swapped(0)
        Y22T = Yh.quad_swapped(3)
        T = self._bufh(("K", P), (k, P, h, v)).a
        Mb = self._bufh(("KS", P), (1, P, h, v)).first().a

        def prod(L_, RT_, dst):
            fnp.matmul(L_, RT_, out=T)
            fnp.sum(T, axis=0, out=dst)

        fnp.subtract(X21, X11, out=Lb); fnp.add(Y11, Y12, out=Rb); prod(Lb, RbT, C22)   # M6 -> C22
        fnp.add(X21, X22, out=Lb); prod(Lb, Y11T, C21)                                # M2 -> C21
        fnp.subtract(C22, C21, out=C22)                                               # C22 -= M2
        fnp.subtract(Y21, Y11, out=Rb); prod(X22, RbT, C11)                           # M4 -> C11
        fnp.add(C21, C11, out=C21)                                                    # C21 += M4
        fnp.add(X11, X22, out=Lb); fnp.add(Y11, Y22, out=Rb); prod(Lb, RbT, Mb)       # M1
        fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb); prod(X11, RbT, C12)                           # M3 -> C12
        fnp.add(C22, C12, out=C22)                                                    # C22 += M3
        fnp.add(X11, X12, out=Lb); prod(Lb, Y22T, Mb)                                 # M5
        fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb); fnp.add(Y21, Y22, out=Rb); prod(Lb, RbT, Mb)  # M7
        fnp.add(C11, Mb, out=C11)


class _Pool:
    """V27 (F80): persistent named scratch buffers. A fresh (n,n) result costs ~0.10 ms
    of residual (allocation + wrapper), the same op written with out= into a pooled
    buffer ~0.013 ms; the pool lives on the estimator across predict() calls (per-MLP
    allocation made the residual creep over a suite, F79). Keys carry the shape, so a
    different width simply adds buffers."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.bufs = {}

    def get(self, name, shape):
        key = (name, tuple(int(x) for x in shape))
        b = self.bufs.get(key)
        if b is None:
            b = fnp.empty(key[1], dtype=self.dtype)
            fnp.copyto(b, 0.0)   # metered page touch (see class note)
            self.bufs[key] = b
        return b

    def get_pair(self, name, shape):
        """V28: an (S, 2, a, b) slot buffer together with its one-time (2S, 1, a, b) view
        (fnp.reshape is a logged, per-element billed op; slabs of whole slots stay
        contiguous, so [2i:2j] of the view is slots i..j-1 with the pair axis folded
        into the Strassen batch)."""
        key = ("pair", name, tuple(int(x) for x in shape))
        e = self.bufs.get(key)
        if e is None:
            b5 = fnp.empty(key[2], dtype=self.dtype)
            fnp.copyto(b5, 0.0)
            b4 = fnp.reshape(b5, (key[2][0] * key[2][1], 1, key[2][2], key[2][3]))
            e = (b5, b4)
            self.bufs[key] = e
        return e


class Estimator(BaseEstimator):
    AGE_OLD = int(_os.environ.get("V21_AGE_OLD", "4"))   # V21: sources with age > AGE_OLD are confined
    R_OLD = int(_os.environ.get("V21_R_OLD", "384"))     # V21: shared basis rank (lean ladder F72)
    AGE_OLD2 = int(_os.environ.get("V24_AGE_OLD2", "7"))  # V24: age gate of the nested tier (0 = off)
    R_OLD2 = int(_os.environ.get("V24_R_OLD2", "224"))    # V24: rank of the nested sub-basis U (8-dump frontier runs/v24_7_*_d0-7.log: 224 best, 128 cliff)
    QPASS2 = int(_os.environ.get("V24_QPASS2", "2"))      # V24: passes of the r1-space range finder
    R_FB = int(_os.environ.get("V37_R_FB", "16"))    # V18: rank of the D21 feedback thin legs (F69 lean ladder: 8/16/32 -> 2.17/2.14/2.14e-8)
    R_RES = 16   # rank of the S21 residual leg (V17 ladder on dumps 0/1: 16/32/64 -> 2.35/2.39/2.41e-8 at 0.492/0.502/0.518xB)

    def __init__(self) -> None:
        self._setup_rng = None
        self._wick_consts = _build_wick_consts()
        self._progs = {0: _term_prog(0), 1: _term_prog(1), 2: _term_prog(2)}
        self._i11 = WICK_PAIRS.index((1, 1))
        self._i21 = WICK_PAIRS.index((2, 1))
        self._i12 = WICK_PAIRS.index((1, 2))
        self._i31 = WICK_PAIRS.index((3, 1))

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)
        if WARM:
            # V26: first-call warm-ups (the residual audit showed one-off 5-30 ms gaps
            # before the first qr / norm.pdf / sandwich of every MLP: library init).
            try:
                f32 = fnp.float32
                z = fnp.zeros((64, 16), dtype=f32)
                fnp.linalg.qr(z + 1.0)
                v = fnp.zeros(64, dtype=f32)
                flops.stats.norm.pdf(v)
                flops.stats.norm.cdf(v)
                e = flops.as_symmetric(fnp.eye(64, dtype=f32), symmetry=(0, 1))
                fnp.einsum("ij,ia,jb->ab", e, z[:, :16] + 1.0, z[:, :16] + 1.0)
                # one tiny call per op signature the chain uses (matmul / einsum path
                # and symmetry machinery initialize lazily: 24 ms before the first
                # small matmul, 1-2 ms before each first einsum subscript)
                a = fnp.zeros((4, 21), dtype=f32) + 1.0
                b = fnp.zeros((21, 64), dtype=f32) + 1.0
                m = fnp.zeros((64, 64), dtype=f32) + 1.0
                t3 = fnp.zeros((3, 64, 64), dtype=f32) + 1.0
                g3 = fnp.zeros((3, 3), dtype=f32) + 1.0
                _ = a @ b
                _ = m @ m
                _ = m @ v
                fnp.einsum("tij,ti,tj,tg->gij", t3, t3[:, :, 0], t3[:, 0, :], g3)
                fnp.einsum("ti,ti,tg->gi", t3[:, :, 0], t3[:, :, 1], g3)
                fnp.einsum("iq,kqn->kin", m[:, :16], t3[:, :16, :])
                fnp.einsum("ki,kc->ic", t3[:, :, 0], t3[:, :, 1])
                fnp.einsum("kij,kij->i", t3, t3)
                fnp.einsum("kij,kj->ki", t3, t3[:, 0, :])
                fnp.einsum("ki,ki->i", t3[:, :, 0], t3[:, :, 1])
                fnp.einsum("kij,kqj->iq", t3, t3[:, :16, :])
                fnp.einsum("kiq,kjq->kij", t3, t3)
                fnp.einsum("kij,kjq->kiq", t3, t3)
                fnp.einsum("kiq,kcq->ic", t3, t3)
                fnp.einsum("pq,kqn->kpn", m[:16, :16], t3[:, :16, :])
                fnp.einsum("cij,cjk->cik", t3[:1], t3)
                fnp.sum(t3, axis=(0, 2))
                fnp.sum(t3, axis=0)
                fnp.diag(m)
                fnp.diagflat(v)
                fnp.fill_diagonal(m, 0.0)
                fnp.matmul(t3, fnp.swapaxes(t3, -1, -2))
                fnp.concatenate([m, m], axis=1)
                fnp.stack([v, v], axis=1)
                fnp.maximum(v, 1e-10)
                fnp.sqrt(v + 1.0)
                fnp.power(v + 1.0, 1.0)
                fnp.clip(v, 0.5, 2.0)
                fnp.mean(v)
                fnp.copy(m)
                fnp.abs(v)
            except Exception:  # noqa: BLE001  warm-up must never fail a submission
                pass

    # ------------------------------------------------------------------
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        was = _gc.isenabled()
        _gc.disable()
        try:
            return self._predict_core(mlp, budget)
        finally:
            if was:
                _gc.enable()

    def _predict_core(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        n = mlp.width
        f32 = fnp.float32
        st = _statics(n)
        metric2 = 2.0 ** 2
        L = len(mlp.weights)
        # vec+ww K4 transport and the fitted mean correction are calibrated for the
        # Phase-2 suite shape ONLY; any other shape (e.g. the grader smoke MLP)
        # runs the shape-generic V1.6 base path. The correction destabilizes small
        # widths (non-finite blowup observed at 16x64) -- never apply it off-suite.
        riders = (n == 1024 and L == len(CORR_BETA))

        C1l, C2l, SELCl = self._wick_consts
        C1 = fnp.asarray(C1l, dtype=f32)
        C2 = fnp.asarray(C2l, dtype=f32)
        SELC = fnp.asarray(SELCl, dtype=f32)
        FP = {}
        for mode, p in self._progs.items():
            FP[mode] = {k: fnp.asarray(v, dtype=f32)
                        for k, v in p.items() if k in ("SL2", "SR2", "IND2", "SL1", "IND1")}
        ones_n = fnp.ones(n, dtype=f32)
        zeros_n = fnp.zeros(n, dtype=f32)
        ones_col = (ones_n)[:, None]
        ones_row = ones_col.T
        ones2 = ones_col * ones_row
        beta_rows = [fnp.asarray(b, dtype=f32) for b in CORR_BETA]
        # V20: one persistent buffer for the d=2 term products (max terms over modes)
        t2max = max(len(p["AB2"]) for p in self._progs.values())
        # V27 (F80): persistent pooled buffers (see _Pool)
        pool = getattr(self, "_pool", None)
        if pool is None or pool.dtype != f32:
            pool = _Pool(f32)
            self._pool = pool

        def NN(name):
            return pool.get(name, (n, n))

        abbuf = pool.get("abbuf", (t2max, n, n))
        T1 = NN("t1")   # layer-local (n,n) scratch

        mu = fnp.zeros(n, dtype=f32)
        C = flops.as_symmetric(fnp.eye(n, dtype=f32), symmetry=(0, 1))
        A_st = P_st = Z_st = L_st = None
        newborn = None
        w2b_list = []
        s_list = []
        e_list = []
        c1_list = []   # per source: lambda_b * w1_b   (X3 = A*c1 + P*c2, column scalings)
        c2_list = []   # per source: w1_b^2 * dG_pre_b
        y_list = []    # per source: (m/4) w2_b       (Y3 = y 1^T)
        g_prev = None  # post-ReLU kappa4 diagonal core of the previous layer
        var_prev = None
        lam_prev = 0.0
        regen = riders and not NO_REGEN  # memoryless kappa4 channel: suite shape only
        bufs = None
        legs = None
        # V26: pooled Strassen products (F79); the pools persist across predict() calls
        # (allocating/freeing ~2.5 GB per MLP made the residual clock creep up over a
        # 100-MLP suite). Pool keys carry the block shapes, so a different width simply
        # adds new buffers; a different depth changes the batch capacity -> rebuild.
        smm = getattr(self, "_smm", None)
        if smm is None or smm.bmax != 2 * (L - 1) or smm.dtype != f32:
            smm = _Strassen(f32, 2 * (L - 1))
            self._smm = smm
        ncall = getattr(self, "_ncall", 0)
        self._ncall = ncall + 1
        s_lev = STRASSEN_LEVELS if ncall > 0 else min(STRASSEN_LEVELS, STRASSEN_FIRST)
        s_sb = STRASSEN_SB
        self._s_hub = s_lev
        r = min(int(self.R_RES), n)  # smoke shapes can be narrower than the rank
        rfb = 0 if NO_FB else min(int(self.R_FB), n)  # V18 feedback rank
        # V21: shared-basis state for old sources (suite-width only: r must be < n)
        r_old = int(self.R_OLD)
        confine = (not NO_CONFINE) and r_old < n
        age_old = int(self.AGE_OLD)
        ka = 0           # number of confined (old) sources = leading slots of the stacks
        Qc = None        # (n, r) current basis of the old legs (pre-wick of this layer)
        fa_side = f2_side = qc_side = z_side = 0   # V27: ping-pong buffer sides
        FAP = FAP2 = None    # V28: interleaved factor slabs (m, 2, r, n) [slot, {A, P}]
        fap4 = fap24 = None  # their one-time (2 (L-1), 1, r, n) views (current side)
        fa_off = 0           # slot offset of FAP inside its side buffer (1 after a nest)
        legs4 = None
        FAo = FPo = None   # (ka, r, n) static factors: A_s = Qc FA_s, P_s = Qc FP_s
        Sg = None        # (r, r) weighted Gram core of the TIER-1 legs in the factor basis
        # V24: nested tier 2 (slots [0:kb] of the stacks, the oldest sources)
        r2 = int(self.R_OLD2)
        age_old2 = int(self.AGE_OLD2)
        nest = confine and age_old2 > 0 and r2 < r_old
        kb = 0           # number of tier-2 sources (leading slots, subset of the ka confined)
        U = None         # (r1, r2) shared sub-basis of tier 2 inside Qc's factor space
        FA2 = FP2 = None   # (kb, r2, n) static factors: A_s = Qc U FA2_s
        S2 = None        # (r2, r2) weighted Gram core of tier 2 in the U basis
        QU = None        # (n, r2) = Qc U, formed once per layer
        dA_list = []     # per source: hub-column Gram weights of the A-type legs
        dP_list = []     # per source: hub-column Gram weights of the P-type legs
        R1T_st = R2T_st = None  # V18: static right factors of the thin legs, (k, n, rfb)
        Zf_st = None  # V18: transported thin legs [F1 | F2], (k, n, 2 rfb); separate from Z
                      # so the M-leg einsums (Z L^T, PPL Z^T) never see them
        K4_sigma = None
        K4_vec = None
        rows = []

        w1_prev = None  # wick w(1) of the previous layer, folded into WD
        # V30: dead-ReLU pruning (suite shape only). Neurons of every layer are relabeled by a
        # cheap pre-estimate of alpha = mu/sigma (descending), so the "dead" units (alpha << 0:
        # zero gain, no K3 readout) are the trailing indices and every restriction is a prefix
        # slice. S_OUT[l]: rows of the K3 legs kept at layer l; S_IN[l]: rows kept as the inner
        # index of the transport out of layer l (legs are w1-scaled there).
        prune = PRUNE and riders
        perm_prev = None
        hub_mask = None   # V38: hub-column mask of the source born at the previous layer
        cdiag_prev = None
        inv_perm = None
        nr = n
        ns = nj = n
        shape_prev = None

        for li, w in enumerate(mlp.weights):
            last = li == L - 1
            trim = last and not FULL_LAST  # V19: mean-only final layer
            skip_src = trim and NO_SRC_LAST
            w32 = w if w.dtype == fnp.float32 else w.astype(f32)
            if prune and li >= LR0:
                # relabel this layer's units by a cheap pre-estimate of alpha (descending);
                # layers before the first pruned one keep their order (no extra ops)
                if perm_prev is not None:
                    w32 = fnp.take(w32, perm_prev, axis=0)
                mu_o = w32.T @ mu
                if cdiag_prev is None:
                    var_a = fnp.sum(w32 * w32, axis=0)
                else:
                    var_a = (w32 * w32).T @ cdiag_prev
                a_apx = mu_o / fnp.sqrt(fnp.maximum(var_a, 1e-10))
                perm = fnp.argsort(-a_apx)
                inv_perm = fnp.argsort(perm)
                w32 = fnp.take(w32, perm, axis=1)
                W = w32.T
                mu = fnp.take(mu_o, perm)
                perm_prev = perm
                nr = int(S_OUT[li])
                nin_prev = int(S_IN[li - 1]) if li > 0 else n
                n_on = int(S_ON[li])
                if PRUNE_EMU:
                    ns, nj = n, n          # emulation: zero rows only, full-size products
                    ro = rf0 = 0
                else:
                    ns, nj = nr, min(nin_prev, n)
                    ro = n_on               # first readout row of the formed (old) legs
                    rf0 = n_on if last else 0   # last layer: the young legs are readout-only too
            else:
                inv_perm = None
                n_on = ro = rf0 = 0
                W = w32.T
                # ---- linear ----
                mu = W @ mu
                ns, nj = n, n
            C_pre = None
            if li == 0:
                # V29: C = I at the input, so the sandwich is the plain Gram w32^T w32
                # (aliased 2-operand einsum: exact, 0.5 u, output tagged symmetric; F77)
                if trim:
                    var = fnp.maximum(fnp.sum(w32 * w32, axis=0), 1e-10)
                else:
                    C_pre = fnp.einsum("ia,ib->ab", w32, w32)
            elif skip_src:
                # probe path only (V19_NO_SRC_LAST): no family at this layer
                if trim:
                    var = fnp.maximum(fnp.sum(w32 * (C @ w32), axis=0), 1e-10)
                else:
                    C_pre = fnp.einsum("ij,ia,jb->ab", C, w32, w32)
            # li >= 1: C_pre is formed after the transport family below (W C rides in it)
            # Source stacks evolve by W @ diag(w1_prev); the newborn (added after
            # last layer's wick) evolves by the raw W. Fold the wick into WD so
            # the stacks never need a separate (k,n,n) scaling pass.
            # Dense legs live in ping-pong slot buffers (F53: no fresh (k,n,n)
            # results, no concatenates): transport k slots into the spare buffer,
            # write the newborn into slot k, swap.
            if A_st is not None and not skip_src:
                k = A_st.shape[0]
                WD = fnp.multiply(W, (w1_prev)[None, :], out=NN("wd"))
                WDb = WD[None]
                # V21: one source per layer crosses the age gate; join it to the shared
                # basis (post-wick legs, like the lean chain), skip the join at the last
                # layer (its legs are used once more, densely, for D3 only).
                ka_t = ka
                if confine and not last and li < NOJOIN_FROM:
                    ka_t = max(ka, min(k, li - age_old))
                # V37: the shared-basis families of this layer (join products, basis transport,
                # formings) run with the rectangular leaf minimum
                s_jn = min(STRASSEN_JOIN, s_lev)
                smm.smin = SMIN_SB
                if ka_t > ka:
                    j = ka  # the joiner (exactly one per layer)
                    w1c = (w1_prev)[:nj, None]
                    # V30: the w1-scaled legs have no rows beyond nj (dead units of the
                    # previous layer), so the range finder lives on the first nj rows
                    Aj = A_st[j][:nj]   # V29: dense legs are already w1-scaled (wick stage)
                    Pj = P_st[j][:nj]
                    dAj = (dA_list[j])[:, None]
                    dPj = (dP_list[j])[:, None]
                    Om = pool.get("om", (nj, r_old))         # fixed sketch (nj, r)
                    fnp.copyto(Om, w32[:nj, :r_old])
                    Qp = fnp.multiply(w1c, Qc[:nj], out=pool.get("qp", (nj, r_old))) if ka > 0 else None
                    Yq = pool.get("yq", (nj, r_old))
                    Yq1 = pool.get("yq1", (n, r_old))
                    Yq2 = pool.get("yq2", (nj, r_old))
                    # V37: the joiner's products as Strassen families over its [A, P] slab
                    lev_p = smm.level(r_old, nj, n, s_jn)    # (r, nj) @ (nj, n): G, projections
                    lev_h = smm.level(nj, n, r_old, s_jn)    # hub: sum_k leg_k G_k^T
                    sjoin = lev_p > 0 and lev_h > 0
                    if sjoin:
                        legs_jh = _VH(legs4["AP0"][2 * j:2 * j + 2][..., :nj, :])   # (2, 1, nj, n)
                        Gj = pool.get("jg", (2, 1, r_old, n))
                        DWj = fnp.stack([dA_list[j], dP_list[j]], axis=0)[:, None, None, :]
                        Yq3 = Yq[None]
                    for _pass in range(QPASS):
                        # Yq = G Om with G = Aj dA Aj^T + Pj dP Pj^T + Qp Sg Qp^T
                        if sjoin:
                            # G_k = (Om^T leg_k) d_k (columns), Yq = sum_k leg_k G_k^T
                            smm.mm(Om.T[None, None], legs_jh, Gj, lev_p)
                            fnp.multiply(Gj, DWj, out=Gj)
                            smm.hub(legs_jh, Gj, Yq3, lev_h)
                        else:
                            fnp.matmul(Aj.T, Om, out=Yq1)
                            fnp.multiply(dAj, Yq1, out=Yq1)
                            fnp.matmul(Aj, Yq1, out=Yq)
                            fnp.matmul(Pj.T, Om, out=Yq1)
                            fnp.multiply(dPj, Yq1, out=Yq1)
                            fnp.matmul(Pj, Yq1, out=Yq2)
                            fnp.add(Yq, Yq2, out=Yq)
                        if ka > 0:
                            # V24: full core of the confined legs = tier-1 core + lifted tier-2 core
                            Sfull = Sg if kb == 0 else Sg + U @ (S2 @ U.T)
                            fnp.matmul(Qp, Sfull @ (Qp.T @ Om), out=Yq2)
                            fnp.add(Yq, Yq2, out=Yq)
                        Qn, _ = fnp.linalg.qr(Yq)           # (n, r) orthonormal
                        Om = Qn
                    if len(R1S) > li and 0 < R1S[li] < r_old:
                        fnp.copyto(Qn[:, R1S[li]:], 0.0)     # probe: effective rank R1S[li]
                    # V27: factors live in ping-pong slot buffers (rotation reads one side,
                    # writes the other; the joiner takes slot m1 = number of tier-1 members)
                    fa_side ^= 1
                    fap_new, fap4_new = pool.get_pair(("fap", fa_side), (L - 1, 2, r_old, n))
                    m1 = ka - kb
                    if sjoin:
                        smm.mm(Qn.T[None, None], legs_jh, fap4_new[2 * m1:2 * m1 + 2], lev_p)
                        FAj = fap_new[m1, 0]
                        FPj = fap_new[m1, 1]
                    else:
                        FAj = fnp.matmul(Qn.T, Aj, out=fap_new[m1, 0])
                        FPj = fnp.matmul(Qn.T, Pj, out=fap_new[m1, 1])
                    if ka > 0:
                        Tq = Qn.T @ Qp                      # (r, r) rotation of the old factors
                        if ka > kb:
                            lev_r = smm.level(r_old, r_old, n, s_jn)
                            if lev_r > 0:
                                smm.mm(Tq[None, None], fap4[2 * fa_off:2 * (fa_off + m1)],
                                       fap4_new[:2 * m1], lev_r)
                            else:
                                fnp.matmul(Tq[None, None], FAP, out=fap_new[:m1])
                            Sg = Tq @ (Sg @ Tq.T)
                        else:
                            Sg = None
                        if kb > 0:
                            U = Tq @ U                      # V24: the sub-basis rides along
                    FAP = fap_new[:m1 + 1]
                    fap4 = fap4_new
                    fa_off = 0
                    FAo = FAP[:, 0]
                    FPo = FAP[:, 1]
                    Sj = (FAj @ (dAj * FAj.T)) + (FPj @ (dPj * FPj.T))
                    Sg = Sj if Sg is None else Sg + Sj
                    # V24: nested move of the oldest tier-1 member into tier 2, in the
                    # orthonormal coordinates Qn of this rebuild (lean: youngest gate first,
                    # then the older gate on the already-confined legs).
                    kb_t = kb
                    if nest:
                        kb_t = max(kb, min(ka_t - 1, li - age_old2))
                    if kb_t > kb:
                        assert kb_t == kb + 1, (li, kb, kb_t)
                        jb = kb                             # stack slot of the mover
                        FA1 = FAo[0]
                        FP1 = FPo[0]
                        dAb = (dA_list[jb])[:, None]
                        dPb = (dP_list[jb])[:, None]
                        S_s = (FA1 @ (dAb * FA1.T)) + (FP1 @ (dPb * FP1.T))   # (r1, r1)
                        G2 = S_s if kb == 0 else S_s + U @ (S2 @ U.T)
                        Om2 = fnp.copy(w32[:r_old, :r2])                        # sketch (r1, r2)
                        for _pass in range(int(self.QPASS2)):
                            Un, _ = fnp.linalg.qr(G2 @ Om2)
                            Om2 = Un
                        if len(R2S) > li and 0 < R2S[li] < r2:
                            fnp.copyto(Un[:, R2S[li]:], 0.0)     # probe: effective rank R2S[li]
                        f2_side ^= 1
                        fap2_new, fap24_new = pool.get_pair(("fap2", f2_side), (L - 1, 2, r2, n))
                        fnp.matmul(Un.T, FA1, out=fap2_new[kb, 0])
                        fnp.matmul(Un.T, FP1, out=fap2_new[kb, 1])
                        if kb > 0:
                            T2 = Un.T @ U                                       # (r2, r2)
                            fnp.matmul(T2[None, None], FAP2, out=fap2_new[:kb])
                        FAP2 = fap2_new[:kb + 1]
                        fap24 = fap24_new
                        FA2 = FAP2[:, 0]
                        FP2 = FAP2[:, 1]
                        S2 = Un.T @ (G2 @ Un)
                        U = Un
                        FAP = FAP[1:]
                        fa_off = 1
                        FAo = FAP[:, 0]
                        FPo = FAP[:, 1]
                        Sg = Sg - S_s
                        kb = kb_t
                    qc_side ^= 1
                    qcb = pool.get(("qc", qc_side), (n, r_old))
                    lev_q = smm.level(ns, nj, r_old, s_jn)
                    if lev_q > 0:
                        # V37: basis transport W Qn as a (one-slot) Strassen family
                        smm.mm(W[None, None][..., :ns, :nj], Qn[None, None],
                               qcb[None, None][..., :ns, :], lev_q)
                        if ns < n:
                            fnp.copyto(qcb[ns:], 0.0)
                        Qc = qcb
                    elif ns < n or nj < n:
                        fnp.matmul(W[:ns, :nj], Qn, out=qcb[:ns])  # wick already inside Qn
                        if ns < n:
                            fnp.copyto(qcb[ns:], 0.0)
                        Qc = qcb
                    else:
                        Qc = fnp.matmul(W, Qn, out=qcb)  # wick already inside Qn
                    ka = ka_t
                elif ka > 0:
                    qc_side ^= 1
                    qcb = pool.get(("qc", qc_side), (n, r_old))
                    lev_q = smm.level(ns, nj, r_old, s_jn)
                    if lev_q > 0:
                        smm.mm(WD[None, None][..., :ns, :nj], Qc[None, None][..., :nj, :],
                               qcb[None, None][..., :ns, :], lev_q)
                        if ns < n:
                            fnp.copyto(qcb[ns:], 0.0)
                        Qc = qcb
                    elif ns < n or nj < n:
                        fnp.matmul(WD[:ns, :nj], Qc[:nj], out=qcb[:ns])
                        if ns < n:
                            fnp.copyto(qcb[ns:], 0.0)
                        Qc = qcb
                    else:
                        Qc = fnp.matmul(WD, Qc, out=qcb)
                if ka > kb:
                    # formed dense legs of the tier-1 sources (pre-wick of this layer):
                    # V28 ONE Strassen family over the interleaved (A, P) factor slab
                    # V30: only the nr live rows of this layer are formed
                    m1 = ka - kb
                    smm.mm(Qc[None, None][..., ro:ns, :], fap4[2 * fa_off:2 * (fa_off + m1)],
                           legs4["AP1"][2 * kb:2 * ka][..., ro:ns, :],
                           smm.level(ns - ro, r_old, n, s_sb))
                if kb > 0:
                    # V24: tier-2 legs through the sub-basis
                    qub = pool.get("qu", (n, r2))
                    fnp.matmul(Qc[ro:ns], U, out=qub[ro:ns])
                    QU = qub
                    smm.mm(QU[None, None][..., ro:ns, :], fap24[:2 * kb],
                           legs4["AP1"][:2 * kb][..., ro:ns, :],
                           smm.level(ns - ro, r2, n, s_sb))
                smm.smin = STRASSEN_MIN
                # V29: the dense young legs were pre-scaled by w1_prev at the previous
                # wick, so the family's left operand is the raw W; the newborn's A leg
                # (slot k A position, born w1-scaled) and the covariance C (slot k P
                # position) ride along: out [ka:k] = transported legs, [k,0] = W a_b,
                # [k,1] = W C (consumed by the C_pre block below, then overwritten by W).
                # Level 0 falls back to the dense batched matmul inside mm().
                extra = 2 if newborn is not None else 0
                if 2 * k + extra > 2 * ka:
                    if ns < n or nj < n or rf0 > 0:
                        # V30: ONE family on the live block (ns rows out, nj rows in). The
                        # covariance slot needs the exact product W C, so the two parts the
                        # block leaves out are added by small dense products (3 ops).
                        smm.smin = SMIN_PR
                        smm.mm(W[None, None][..., rf0:ns, :nj],
                               legs4["AP0"][2 * ka:2 * k + extra][..., :nj, :],
                               legs4["AP1"][2 * ka:2 * k + extra][..., rf0:ns, :],
                               smm.level(ns - rf0, nj, n, s_lev))
                        smm.smin = STRASSEN_MIN
                        if newborn is not None:
                            Cs_ = legs["AP0"][k, 1]      # C (post-ReLU covariance), all rows
                            WCo_ = legs["AP1"][k, 1]     # the W C slot being written
                            if nj < n:
                                t_ = fnp.matmul(W[rf0:ns, nj:], Cs_[nj:, :],
                                                out=pool.get("wc_fix", (ns - rf0, n)))
                                fnp.add(WCo_[rf0:ns], t_, out=WCo_[rf0:ns])
                            if ns < n:
                                fnp.matmul(W[ns:, :], Cs_, out=WCo_[ns:])
                            if rf0 > 0:
                                fnp.matmul(W[:rf0, :], Cs_, out=WCo_[:rf0])
                    else:
                        smm.mm(W[None, None], legs4["AP0"][2 * ka:2 * k + extra],
                               legs4["AP1"][2 * ka:2 * k + extra], smm.level(n, n, n, s_lev))
                z_side ^= 1
                if (ns < n or nj < n) and not PRUNE_EMU:
                    # V37: thin legs on the live block too (their rows beyond nj are zero on
                    # the way in; the rows beyond ns of the ping-pong side are zeroed below)
                    zb_ = pool.get(("z", z_side), (L - 1, n, r + 2))
                    WDv = WD[None, :ns, :nj]
                    fnp.matmul(WDv, Z_st[:, :nj, :], out=zb_[:k, :ns, :])
                    Z_st = zb_[:k]
                    if Zf_st is not None:
                        zfb_ = pool.get(("zf", z_side), (L - 1, n, 2 * rfb))
                        fnp.matmul(WDv, Zf_st[:, :nj, :], out=zfb_[:k, :ns, :])
                        Zf_st = zfb_[:k]
                else:
                    Z_st = fnp.matmul(WDb, Z_st, out=pool.get(("z", z_side), (L - 1, n, r + 2))[:k])
                    if Zf_st is not None:
                        Zf_st = fnp.matmul(WDb, Zf_st, out=pool.get(("zf", z_side), (L - 1, n, 2 * rfb))[:k])
                legs["AP0"], legs["AP1"] = legs["AP1"], legs["AP0"]
                legs4["AP0"], legs4["AP1"] = legs4["AP1"], legs4["AP0"]
            else:
                k = 0
            if newborn is not None and not skip_src:
                a_b, Rr, Lr, s_b, e_b, Ff = newborn
                if A_st is None:
                    # V29: layer 1 has no stack yet: the newborn + C family alone (slot 0
                    # written at birth on the AP0 side; output to AP1, then swap)
                    smm.mm(W[None, None], legs4["AP0"][0:2], legs4["AP1"][0:2],
                           smm.level(n, n, n, s_lev))
                    legs["AP0"], legs["AP1"] = legs["AP1"], legs["AP0"]
                    legs4["AP0"], legs4["AP1"] = legs4["AP1"], legs4["AP0"]
                # V29: C_pre from the transported covariance W C in the newborn's P slot
                WC = legs["AP0"][k, 1]
                if trim:
                    # var = diag(W C W^T) = rowsum((W C) * W)
                    var = fnp.maximum(fnp.sum(fnp.multiply(WC, W, out=T1), axis=1), 1e-10)
                else:
                    C_pre = self._sym_product(WC, w32, n, NN("cpre"), min(CPRE_LEV, s_lev))
                fnp.copyto(legs["AP0"][k, 1], W)
                if hub_mask is not None:
                    fnp.multiply(legs["AP0"][k, 1], hub_mask[None, :], out=legs["AP0"][k, 1])
                    hub_mask = None
                A_st = legs["AP0"][:k + 1, 0]
                P_st = legs["AP0"][:k + 1, 1]
                # V27: the newborn's thin columns go into slot k of the current Z side
                # (the transport above wrote slots [:k] of that side); L was written into
                # its slot at birth.
                zb = pool.get(("z", z_side), (L - 1, n, r + 2))
                thin_live = (ns < n or nj < n) and not PRUNE_EMU   # V37: live block only
                if thin_live:
                    Wv = W[:ns, :nj]
                    fnp.matmul(Wv, Rr[:nj], out=zb[k, :ns])
                else:
                    fnp.matmul(W, Rr, out=zb[k])
                Z_st = zb[:k + 1]
                L_st = pool.get("l", (L - 1, n, r + 2))[:k + 1]
                if Ff is not None:
                    zfb = pool.get(("zf", z_side), (L - 1, n, 2 * rfb))
                    if thin_live:
                        fnp.matmul(Wv, Ff[:nj], out=zfb[k, :ns])
                    else:
                        fnp.matmul(W, Ff, out=zfb[k])
                    Zf_st = zfb[:k + 1]
                s_list.append(s_b)
                e_list.append(e_b)
                newborn = None

            if prune and A_st is not None and not skip_src:
                # V30: rows the banded families above never wrote (stale ping-pong data)
                if ro > 0 and ka > 0:
                    fnp.copyto(legs["AP0"][:ka, :, :ro, :], 0.0)
                if rf0 > 0:
                    fnp.copyto(legs["AP0"][ka:A_st.shape[0], :, :rf0, :], 0.0)
            if prune and nr < n and A_st is not None and not skip_src:
                # V30: K3 legs carry no rows for the dead tail of this layer (the sliced
                # families above never wrote them; the thin / full-size ones did)
                kz_ = A_st.shape[0]
                fnp.copyto(legs["AP0"][:kz_, :, nr:, :], 0.0)
                fnp.copyto(Z_st[:, nr:, :], 0.0)
                if Zf_st is not None:
                    fnp.copyto(Zf_st[:, nr:, :], 0.0)
                if Qc is not None and ka > 0:
                    fnp.copyto(Qc[nr:, :], 0.0)

            # ---- WK slices ----
            if trim:
                C_off = None
            else:
                var = fnp.maximum(fnp.diag(C_pre), 1e-10)
                C_off = _zero_diag(C_pre)
            mode = 0 if A_st is None else 1
            if mode == 1 and skip_src:
                D3, D21 = fnp.zeros(n, dtype=f32), None
                if regen:
                    WW = fnp.multiply(W, W, out=NN("ww"))
                    dG = WW @ (g_prev - var_prev * lam_prev) + var * lam_prev
                    g4row = dG * METRIC_C
                    wk4m = wk431 = None
                elif riders:
                    g4row = ((W * W) @ K4_vec) * 0.5 * float(st["wk4_c4"] * metric2)
                    wk4m = None
                else:
                    g4row = ones_n * (K4_sigma * float(st["wk4_c4"] * metric2))
                    wk4m = None
            elif mode == 1:
                if bufs is None:
                    bufs = {nm: pool.get(nm, (L - 1, n, n))
                            for nm in ("ap", "pp", "t", "mp", "xt", "yt", "u")}
                    bufs["lap"], bufs["lap4"] = pool.get_pair("lap", (L - 1, 2, n, n))   # V26: [LA | LP]
                    bufs["hub"] = pool.get("hub", (n, n))             # V26: hub result
                    bufs["ppl"] = pool.get("ppl", (L - 1, n, r + 2))  # V27: PP L stack
                    bufs["d21"] = NN("d21")
                    bufs["t1"] = T1
                    if rfb > 0:
                        bufs["gyr"] = pool.get("gyr", (L - 1, n, rfb))
                        bufs["gxr"] = pool.get("gxr", (L - 1, n, rfb))
                        bufs["d21fb"] = NN("d21fb")
                D3, D21 = self._dslices(A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,
                                        c1_list, c2_list, y_list, n, bufs,
                                        r, rfb, Zf_st, R1T_st, R2T_st,
                                        need_d21=not trim,
                                        ka=ka, Qc=Qc, kb=kb, U2=U,
                                        apb=legs["AP0"], apb4=legs4["AP0"],
                                        sb1=(fap4[2 * fa_off:2 * (fa_off + ka - kb)] if ka > kb else None),
                                        sb2=(fap24[:2 * kb] if kb > 0 else None), s_sb=s_sb,
                                        nrow=(ns if ((ns < n or ro > 0) and not trim) else None),
                                        r0=ro)
                if prune and li >= LR0:
                    # V30: saturated ("fully on", alpha >> 0) units are linear: they pass
                    # the K3 content along (legs) but read nothing out of it -> no (3,) /
                    # (2,1) readout rows for the leading n_on units of the relabeling
                    n_on = int(S_ON[li])
                    if n_on > 0:
                        fnp.copyto(D3[:n_on], 0.0)
                        if D21 is not None:
                            fnp.copyto(D21[:n_on, :], 0.0)
                if regen:
                    # F68: exact transported diagonal of the regenerated core
                    # G = diag(g) + lam C_off:  dG = (W*W) g + lam (diag(C_pre)
                    # - (W*W) var).  NOT an einsum: same-object repeated operands
                    # crash flopscope (F45).
                    WW = fnp.multiply(W, W, out=NN("ww"))
                    if BETA != 0.0:
                        # V25: adaptive lambda. dG is affine in lam: t_g + lam * t_v.
                        t_g = WW @ g_prev
                        t_v = var - WW @ var_prev
                        dG0 = t_g + t_v * lam_prev                # table value first
                        rr = fnp.mean(dG0) / fnp.mean(var)
                        ref = float(REF_R[min(li - 1, len(REF_R) - 1)])
                        # clamp: an odd MLP must never turn the rule into NaN/inf (zero fallback)
                        lam_prev = lam_prev * fnp.power(fnp.clip(rr / ref, 0.5, 2.0), BETA)
                        dG = t_g + t_v * lam_prev                 # var == diag(C_pre)
                    else:
                        dG = WW @ (g_prev - var_prev * lam_prev) + var * lam_prev  # var == diag(C_pre)
                    g4row = dG * METRIC_C
                    g22c = (dG * (METRIC_C / 6.0))[:, None]
                    wk4m = _zero_diag(fnp.add(g22c, g22c.T, out=NN("wk4m")))
                    wk431 = None if trim else fnp.multiply(C_off, (0.5 * METRIC_C * lam_prev), out=NN("wk431"))
                    if _os.environ.get("V17_DEBUG", "0") == "1":
                        DEBUG.append(dict(layer=li, dG=dG, g_prev=g_prev, var_prev=var_prev,
                                          lam=lam_prev, var=var, D3=D3, D21=D21))
                elif riders:
                    # vec+ww: per-neuron K4 diagonal content transported by the
                    # true (W*W) row action; one avg-metric cup replaced
                    # (/metric_c = *0.5). NOT an einsum: same-object repeated
                    # operands crash flopscope (F45).
                    gv = ((W * W) @ K4_vec) * 0.5
                    g4row = gv * float(st["wk4_c4"] * metric2)
                    g22 = gv * float(0.5 * st["wk4_c22"] * metric2)
                    g22c = (g22)[:, None]
                    wk4m = _zero_diag(fnp.add(g22c, g22c.T, out=NN("wk4m")))
                else:
                    g4v = K4_sigma * float(st["wk4_c4"] * metric2)
                    g22v = K4_sigma * float(st["wk4_c22"] * metric2)
                    g4row = ones_n * g4v
                    wk4m = _zero_diag(ones2 * g22v)
            else:
                D3 = D21 = g4row = wk4m = None

            # ---- wick matrix ----
            sigma = fnp.sqrt(var)
            alpha = mu / sigma
            phi = flops.stats.norm.pdf(alpha).astype(f32)
            Phi = flops.stats.norm.cdf(alpha).astype(f32)
            a2 = alpha * alpha
            a3 = a2 * alpha
            a4 = a3 * alpha
            a5 = a4 * alpha
            APOW = fnp.stack([ones_n, alpha, a2, a3, a4, a5], axis=1)
            inv = 1.0 / sigma
            i2 = inv * inv
            i3 = i2 * inv
            i4 = i3 * inv
            i5 = i4 * inv
            i6 = i5 * inv
            s2 = sigma * sigma
            s3 = s2 * sigma
            s4 = s3 * sigma
            SB = fnp.stack([i6, i5, i4, i3, i2, inv, ones_n, sigma, s2, s3, s4], axis=1)
            phic = (phi)[:, None]
            Phic = (Phi)[:, None]
            W_all = ((APOW @ C1) * phic + (APOW @ C2) * Phic) * (SB @ SELC)
            WT = W_all.T

            # ---- nonlin terms (fused) ----
            pmode = mode if (mode == 0 or (regen and not NO_WK431)) else 2
            prog = self._progs[pmode]
            fpm = FP[pmode]
            b2 = {"ones2": ones2, "c_off": C_off}
            b1 = {"ones1": ones_n}
            if mode == 1:
                if not trim:
                    d3col = fnp.multiply((D3)[:, None], ones_row, out=NN("d3col"))
                    g4col = fnp.multiply((g4row)[:, None], ones_row, out=NN("g4col"))
                    b2.update({"d21": D21, "d21T": D21.T, "wk4m": wk4m,
                               "d3col": d3col, "d3row": d3col.T,
                               "g4col": g4col, "g4row": g4col.T})
                    if regen and not NO_WK431:
                        b2["wk431"] = wk431
                b1.update({"d3": D3, "g4": g4row})
            ab_cache = {}

            def ab(pair, tbl):
                if pair in ab_cache:
                    return ab_cache[pair]
                a, b = pair
                if a in ("ones2", "ones1"):
                    val = tbl[b]
                elif b in ("ones2", "ones1"):
                    val = tbl[a]
                else:
                    val = tbl[a] * tbl[b]
                ab_cache[pair] = val
                return val

            if not trim:
                ab2 = prog["AB2"]
                for t_, (a_, b_) in enumerate(ab2):
                    if a_ in ("ones2", "ones1"):
                        fnp.copyto(abbuf[t_], b2[b_])
                    elif b_ in ("ones2", "ones1"):
                        fnp.copyto(abbuf[t_], b2[a_])
                    else:
                        fnp.multiply(b2[a_], b2[b_], out=abbuf[t_])
                ABstack = abbuf[:len(ab2)]
                WL2 = fpm["SL2"] @ WT
                WR2 = fpm["SR2"] @ WT
                PK2 = fnp.einsum("tij,ti,tj,tg->gij", ABstack, WL2, WR2, fpm["IND2"])
            B1stack = fnp.stack([ab(p, b1) for p in prog["B1"]], axis=0)
            WL1 = fpm["SL1"] @ WT
            PK1 = fnp.einsum("ti,ti,tg->gi", B1stack, WL1, fpm["IND1"])

            if not trim:
                pk11 = fnp.add(PK2[0], PK2[0].T, out=NN("pk11"))
                pk11 = _zero_diag(fnp.multiply(pk11, 0.5, out=pk11))
                pk21 = _zero_diag(PK2[1])
                pk22 = fnp.add(PK2[2], PK2[2].T, out=NN("pk22"))
                pk22 = _zero_diag(fnp.multiply(pk22, 0.5, out=pk22))
            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]

            # ---- online mean correction: delta = feats @ beta[l] (13 dots/neuron);
            # mu still holds the pre-nonlin mean here, pk1v == K(1,) is the pred.
            if riders and not NO_CORR:
                if mode == 1:
                    D3f = D3
                    D21n = fnp.abs(D21, out=T1) @ ones_n if D21 is not None else fnp.zeros(n, dtype=f32)
                    k4f = ones_n * K4_sigma
                else:
                    D3f = fnp.zeros(n, dtype=f32)
                    D21n = fnp.zeros(n, dtype=f32)
                    k4f = fnp.zeros(n, dtype=f32)
                feats = fnp.stack([ones_n, mu, var, sigma, alpha, fnp.abs(alpha),
                                   phi, Phi, pk1v, D3f, D21n, k4f, sigma * phi],
                                  axis=1)
                delta = feats @ beta_rows[li]
            else:
                delta = None

            if last:
                row_ = pk1v if delta is None else pk1v + delta
                rows.append(fnp.take(row_, inv_perm) if inv_perm is not None else row_)
                break

            # ---- wick old blocks + dslice scalings ----
            w1 = W_all[:, self._i11]
            w2 = W_all[:, self._i21]
            w1col = (w1)[:, None]
            w1_prev = w1  # thin stacks / basis pick up this wick via WD at the next linear
            if mode == 1 and ka < A_st.shape[0]:
                # V29: dense young legs pre-scaled by w1 (rows) in place, so the next
                # transport family multiplies by the raw W (W (w1 A) == (W w1) A up to
                # rounding); the joiner reads them scaled too.
                kk = A_st.shape[0]
                fnp.multiply(legs4["AP0"][2 * ka:2 * kk], w1col[None, None],
                             out=legs4["AP0"][2 * ka:2 * kk])
            if mode == 1:
                D3_w = D3 * w1 ** 3
                D21_w = fnp.multiply(w1col * w1col, D21, out=NN("d21w"))
                fnp.multiply(D21_w, (w1)[None, :], out=D21_w)
            else:
                D3_w = None
                D21_w = None

            # ---- new source (V1.6, struct-free Y1 = a_b * D(w2)) ----
            if legs is None:
                # V26: A and P legs interleaved per slot: [:, 0] = A, [:, 1] = P
                # V28: with one-time (2 (L-1), 1, n, n) views for the Strassen batches
                pairs = {nm: pool.get_pair(nm, (L - 1, 2, n, n)) for nm in ("AP0", "AP1")}
                legs = {nm: pairs[nm][0] for nm in pairs}
                legs4 = {nm: pairs[nm][1] for nm in pairs}
            # V29: the newborn's A leg is born into its stack slot (slot = number of
            # sources present at this layer); the next layer's family transports it
            k_b = 0 if A_st is None else A_st.shape[0]
            a_b = fnp.multiply(w1col, C_off, out=legs["AP0"][k_b, 0])
            if rfb > 0 and mode == 1:
                # V18 (F69): D21 feedback thin legs. D21 ~ Qf Bf (rank rfb range finder,
                # one power iteration, sketch = a slice of the layer weight).
                w3 = W_all[:, self._i31]
                Omf = pool.get("omf", (n, rfb))  # V20: contiguous sketch
                fnp.copyto(Omf, w32[:, :rfb])
                Yf = D21 @ Omf
                Qf, _ = fnp.linalg.qr(Yf)
                Yf = D21 @ (D21.T @ Qf)
                Qf, _ = fnp.linalg.qr(Yf)
                Bf = Qf.T @ D21                       # D21 ~= Qf @ Bf
                F1_b = (w2)[:, None] * Qf            # Xt = F1 R1, R1 = 1.5 Bf
                R1T_b = Bf.T * 1.5                              # (n, rfb) = R1^T
                F2_b = w1col * Bf.T                             # Yt = F2 R2, R2 = 0.5 Qf^T d(w3)
                R2T_b = Qf * (w3 * 0.5)[:, None]     # (n, rfb) = R2^T
                Xt_b = fnp.matmul(F1_b, R1T_b.T, out=NN("xtb"))
                Yt_b = fnp.matmul(F2_b, R2T_b.T, out=NN("ytb"))
                X1_b = fnp.multiply(a_b, 3.0, out=NN("x1b"))
                fnp.add(X1_b, Xt_b, out=X1_b)
                Y1_b = fnp.multiply(a_b, (w2)[None, :], out=NN("y1b"))
                fnp.add(Y1_b, Yt_b, out=Y1_b)
                xd = fnp.diag(Xt_b)
                yd = fnp.diag(Yt_b)
                # birth dslices of B1 with P = I (general form; diag(a_b) = 0)
                # (same operation order as the V26 expression: bit-identical)
                D21_new = fnp.multiply((xd)[:, None], Y1_b.T, out=NN("d21new"))
                fnp.multiply(X1_b, Y1_b, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply((yd)[:, None], X1_b.T, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply(D21_new, 1.0 / 3.0, out=D21_new)
                D3_new = xd * yd
            else:
                # birth dslices of B1 with P = I: diag(a_b) = 0, so only the a*Y1 term
                # survives: D21_new = a_b^2 * D(w2)-row; D3_new = 0.
                D21_new = fnp.multiply(a_b, a_b, out=NN("d21new"))
                fnp.multiply(D21_new, (w2)[None, :], out=D21_new)
                D3_new = None
                if rfb > 0:
                    F1_b = fnp.zeros((n, rfb), dtype=f32)
                    F2_b = F1_b
                    R1T_b = F1_b
                    R2T_b = F1_b
            if regen and mode == 1 and not NO_FEED:
                # K4->K3 feed (F68): X3 = diag(w1^2 dG) + lam a_b d(w1), Y3 = y 1^T
                # with y = (m/4) w2; M_t1 = u v^T, u = (m/4) w2*dG, v = w1^2.
                # Birth (P = I) dslices: D21 += (1/3)[dgw y^T + 2 y (.) X3_off]
                # + (1/3) v u^T;  D3 += dgw*y + u*v.
                w1sq = w1 * w1
                y_b = w2 * (0.25 * METRIC_C)
                dgw = w1sq * dG
                u_b = y_b * dG
                c1_b = w1 * lam_prev
                # (V26 expression, same operation order, out= into the pooled buffers)
                fnp.multiply((dgw)[:, None], (y_b)[None, :], out=T1)
                fnp.multiply(T1, 1.0 / 3.0, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply(a_b, (c1_b)[None, :], out=T1)
                fnp.multiply((y_b * (2.0 / 3.0))[:, None], T1, out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                fnp.multiply((w1sq)[:, None], (u_b * (1.0 / 3.0))[None, :], out=T1)
                fnp.add(D21_new, T1, out=D21_new)
                D3_new = ((dgw * y_b + u_b * w1sq) if D3_new is None
                          else D3_new + dgw * y_b + u_b * w1sq)
            else:
                y_b = fnp.zeros(n, dtype=f32)
                u_b = y_b
                c1_b = y_b
                dgw = y_b
                w1sq = y_b
            rep21 = D21_new if D21_w is None else fnp.add(D21_w, D21_new, out=NN("rep21"))
            rep21 = _zero_diag(rep21)

            # ---- pK -> K (per-entry; d=1 outputs are vector ops, d=2 few entries) ----
            pk = {(1,): pk1v, (2,): pk2v, (3,): pk3v, (4,): pk4v,
                  (1, 1): pk11, (2, 1): pk21, (2, 2): pk22}

            def pk_slice(vec):
                nz = tuple(sorted((x for x in vec if x > 0), reverse=True))
                base = pk[nz]
                if len(vec) == 1:
                    return base
                if len(nz) == 2:
                    return base if vec == tuple(sorted(vec, reverse=True)) else base.T
                return ((base)[:, None] if vec[0] > 0
                        else (base)[None, :])

            K = {}
            for out_part, entries in PK2K_TABLE.items():
                if len(out_part) == 2:
                    continue    # V27: matrix parts assembled below with out= (same order)
                acc2 = None
                for vpart, coef in entries:
                    prod = None
                    for blk in vpart:
                        fct = pk_slice(blk)
                        prod = fct if prod is None else prod * fct
                    contrib = prod * coef
                    acc2 = contrib if acc2 is None else acc2 + contrib
                K[out_part] = acc2
            # V27 (F80): the three matrix parts of PK2K_TABLE, unrolled in the generic
            # loop's operation order (bit-identical), written into pooled buffers:
            #   (1,1) = pk11 * 1.0 = pk11
            #   (2,1) = (pk1 (.) pk11) * (-2) + pk21
            #   (2,2) = ((pk1^T (.) pk1) (.) pk11) * 4 + (pk1^T (.) pk21) * (-2)
            #           + (pk1 (.) pk21^T) * (-2) + (pk11 * pk11) * (-2) + pk22
            pk1c = (pk1v)[:, None]
            pk1r = (pk1v)[None, :]
            K11 = pk11
            K21 = fnp.multiply(pk1c, pk11, out=NN("k21"))
            fnp.multiply(K21, -2.0, out=K21)
            fnp.add(K21, pk21, out=K21)
            K22 = fnp.multiply(pk1r, pk1c, out=NN("k22"))
            fnp.multiply(K22, pk11, out=K22)
            fnp.multiply(K22, 4.0, out=K22)
            fnp.multiply(pk1r, pk21, out=T1)
            fnp.multiply(T1, -2.0, out=T1)
            fnp.add(K22, T1, out=K22)
            fnp.multiply(pk1c, pk21.T, out=T1)
            fnp.multiply(T1, -2.0, out=T1)
            fnp.add(K22, T1, out=K22)
            fnp.multiply(pk11, pk11, out=T1)
            fnp.multiply(T1, -2.0, out=T1)
            fnp.add(K22, T1, out=K22)
            fnp.add(K22, pk22, out=K22)
            K21 = _zero_diag(K21)
            K22 = _zero_diag(K22)
            K1v = K[(1,)]
            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]

            # ---- assemble ----
            mu = K1v if delta is None else K1v + delta
            # V27: diagflat + add -> pooled buffer, diagonal written in place (bit-identical:
            # K11 has a zero diagonal)
            C = fnp.add(K11, K11.T, out=NN("c"))
            fnp.multiply(C, 0.5, out=C)
            fnp.fill_diagonal(C, K2v)
            C = flops.as_symmetric(C, symmetry=(0, 1))
            # V29: C rides the next layer's transport family in the P position of the
            # newborn's slot (the family writes W C there; the newborn's P leg W is
            # copied over it once C_pre is formed)
            fnp.copyto(legs["AP0"][k_b, 1], C)
            S3c = K3v - D3_w if D3_w is not None else K3v
            if D3_new is not None:
                S3c = S3c - D3_new
            S21 = _zero_diag(fnp.subtract(K21, rep21, out=NN("s21")))
            # S21 = S_sep + Rres: S_sep = exact leading (2,1) Wick term (the (c_off,
            # ones2) term of pk21 minus 2*pk1*[(c_off, ones2) term of pk11]); M_b^T
            # part 3*S_sep^T = 3*a_b*diag(e) rides on the A leg for free. Rres is
            # compressed to rank r (randomized range finder, one power iteration).
            e_b = W_all[:, self._i12] - 2.0 * pk1v * w1
            S_sep = fnp.multiply((e_b)[:, None], C_off, out=NN("ssep"))
            S_sep = _zero_diag(fnp.multiply(S_sep, (w1)[None, :], out=S_sep))
            Rres = fnp.subtract(S21, S_sep, out=S21)
            Om = pool.get("om_r", (n, r))  # V20: contiguous sketch
            fnp.copyto(Om, w32[:, :r])
            Y = fnp.matmul(Rres, Om, out=pool.get("y", (n, r)))
            Q, _ = fnp.linalg.qr(Y)
            Y2 = fnp.matmul(Rres.T, Q, out=pool.get("y2", (n, r)))
            Y = fnp.matmul(Rres, Y2, out=Y)
            Q, _ = fnp.linalg.qr(Y)
            Bm = fnp.matmul(Q.T, Rres, out=pool.get("bm", (r, n)))   # Rres ~= Q @ Bm
            # M_b = diag(S3c) + 3 S21^T = diag(S3c) + 3 a_b diag(e_b) + 3 Rr Lr^T
            # with Rr = Bm^T (n,r) [transported as Z = P Rr] and Lr = Q (static).
            # extra thin columns: M_t1 = u v^T -> Z column P u, L column v; and the Y3
            # row vector y is a LEFT factor of the hub tensor, so it is transported
            # like a leg (y(l) = P(l) y): Z column P y with a ZERO L column (keeps it
            # out of the M leg). Zeros off-suite / at layer 0 so every source keeps
            # r+2 columns.
            # V27: Rr into a persistent (n, r+2) buffer (read by next layer's W @ Rr);
            # Lr written straight into its stack slot k (static, never transported).
            Rr_full = pool.get("rr", (n, r + 2))
            fnp.multiply(Bm.T, 3.0, out=Rr_full[:, :r])
            fnp.copyto(Rr_full[:, r], u_b)
            fnp.copyto(Rr_full[:, r + 1], y_b)
            # slot of the source born here = number of sources present at this layer
            # (the newborn added above included): it joins the stacks at the next layer
            k_b = 0 if A_st is None else A_st.shape[0]
            lb = pool.get("l", (L - 1, n, r + 2))
            fnp.copyto(lb[k_b, :, :r], Q)
            fnp.copyto(lb[k_b, :, r], w1sq)
            fnp.copyto(lb[k_b, :, r + 1], zeros_n)
            Lr_full = None
            # V18: thin feedback legs [F1 | F2] travel in their own stack Zf (transported
            # like Z, never entering the M-leg einsums); right factors R1T/R2T are static.
            Ff_b = fnp.concatenate([F1_b, F2_b], axis=1) if rfb > 0 else None
            newborn = (a_b, Rr_full, Lr_full, S3c, e_b, Ff_b)
            if rfb > 0:
                r1b = pool.get("r1t", (L - 1, n, rfb))
                r2b = pool.get("r2t", (L - 1, n, rfb))
                fnp.copyto(r1b[k_b], R1T_b)
                fnp.copyto(r2b[k_b], R2T_b)
                R1T_st = r1b[:k_b + 1]
                R2T_st = r2b[:k_b + 1]
            if HUB_A > 0.0 and riders:
                # V38: hub columns of units whose ReLU is (almost) linear or (almost) dead create
                # no new third cumulant: drop them from the newborn source (A columns, P columns
                # at the next layer, the hub rows of the static thin factors)
                hub_mask = (fnp.abs(alpha) < HUB_A).astype(f32)
                fnp.multiply(a_b, hub_mask[None, :], out=a_b)
                fnp.multiply(lb[k_b], hub_mask[:, None], out=lb[k_b])
                fnp.multiply(Rr_full, hub_mask[:, None], out=Rr_full)
                if rfb > 0:
                    fnp.multiply(r1b[k_b], hub_mask[:, None], out=r1b[k_b])
                    fnp.multiply(r2b[k_b], hub_mask[:, None], out=r2b[k_b])
            w2b_list.append(w2)
            # V21: hub-column Gram weights of this source's legs (X1 = 3A, Y1 ~ A d(w2),
            # M ~ P d(s) + 3 A d(e)): A-type 9 + w2^2 + 9 e^2, P-type 1 + s^2
            dA_list.append(9.0 + w2 * w2 + 9.0 * e_b * e_b)
            dP_list.append(1.0 + S3c * S3c)
            c1_list.append(c1_b)
            c2_list.append(dgw)
            y_list.append(y_b)
            if regen:
                # post-ReLU kappa4 diagonal core (r=1 matrix-core harmonic projection)
                k22row = K22 @ ones_n
                g_prev = ((K4v + k22row) * float(st["cA"])
                          + (fnp.sum(K4v) + fnp.sum(k22row)) * float(st["cI"]))
                var_prev = K2v
                lam_prev = float(LAM[min(li, len(LAM) - 1)])
            K4_sigma = (fnp.sum(K4v) * float(st["k4_c4"])
                        + fnp.sum(K22) * float(st["k4_c22"])) * float(st["P2"])
            # per-neuron K4 diagonal content (mean equals K4_sigma by construction;
            # K22 is symmetric so K22 @ ones == its column sums)
            if riders:
                K4_vec = (K4v * float(st["k4_c4"])
                          + (K22 @ ones_n) * float(st["k4_c22"])) * float(n * st["P2"])
            if prune:
                cdiag_prev = K2v
                nin_ = int(S_IN[li])
                if nin_ < n:
                    # V30: rows beyond S_IN are dropped as the inner index of the
                    # next transport (young legs are already w1-scaled; thin legs / basis
                    # pick up w1 through WD at the next layer)
                    kz_ = 0 if A_st is None else A_st.shape[0]
                    if kz_ > 0:
                        fnp.copyto(legs["AP0"][:kz_, :, nin_:, :], 0.0)
                        fnp.copyto(Z_st[:, nin_:, :], 0.0)
                        if Zf_st is not None:
                            fnp.copyto(Zf_st[:, nin_:, :], 0.0)
                        if Qc is not None and ka > 0:
                            fnp.copyto(Qc[nin_:, :], 0.0)
                    fnp.copyto(legs["AP0"][k_b, 0, nin_:, :], 0.0)     # newborn a_b
                    fnp.copyto(Rr_full[nin_:, :], 0.0)
                    if Ff_b is not None:
                        fnp.copyto(Ff_b[nin_:, :], 0.0)
            rows.append(fnp.take(mu, inv_perm) if inv_perm is not None else mu)

        return fnp.stack(rows, axis=0)

    # ------------------------------------------------------------------
    def _sym_product(self, X, Y, n, out, lev):
        """V29: out = X Y for a SYMMETRIC result (X = W C, Y = w32 = W^T): the three block
        products (11, 12, 22) of the 2x2 partition as ONE Strassen family (0.75 of the
        dense count before the recursion); the 21 block is the copy of 12^T. Odd n (smoke
        shapes) falls back to the plain product."""
        h = n // 2
        if n % 2 or n < 4:
            return fnp.matmul(X, Y, out=out)
        pool = self._pool
        X3 = pool.get("sym_x", (3, 1, h, n))
        Y3 = pool.get("sym_y", (3, 1, n, h))
        O3 = pool.get("sym_o", (3, 1, h, h))
        fnp.copyto(X3[0, 0], X[:h]); fnp.copyto(X3[1, 0], X[:h]); fnp.copyto(X3[2, 0], X[h:])
        fnp.copyto(Y3[0, 0], Y[:, :h]); fnp.copyto(Y3[1, 0], Y[:, h:]); fnp.copyto(Y3[2, 0], Y[:, h:])
        self._smm.mm(X3, Y3, O3, self._smm.level(h, n, h, lev))
        fnp.copyto(out[:h, :h], O3[0, 0]); fnp.copyto(out[:h, h:], O3[1, 0])
        fnp.copyto(out[h:, h:], O3[2, 0]); fnp.copyto(out[h:, :h], O3[1, 0].T)
        return out

    def _hub2(self, bufs, apb4, k0, k, n, nrow=None, r0=0):
        """V26: sum over slots k0..k-1 of LA_k A_k^T + LP_k P_k^T as ONE Strassen family
        (LA/LP live interleaved in bufs["lap"], A/P in the legs, so the pair axis folds
        into the contraction batch; V28: on the one-time 4-D views)."""
        out = bufs["hub"]
        smm = self._smm
        if nrow is not None and (nrow < n or r0 > 0):
            # V30: readout rows r0:nrow (no dead tail, no saturated head) x live columns
            smm.smin = SMIN_PR
            smm.hub(bufs["lap4"][2 * k0:2 * k][..., r0:nrow, :], apb4[2 * k0:2 * k][..., :nrow, :],
                    out[None][..., r0:nrow, :nrow],
                    smm.level(nrow - r0, n, nrow, min(STRASSEN_HUB, self._s_hub)))
            smm.smin = STRASSEN_MIN
            if nrow < n:
                fnp.copyto(out[nrow:, :], 0.0)
                fnp.copyto(out[r0:nrow, nrow:], 0.0)
            if r0 > 0:
                fnp.copyto(out[:r0, :], 0.0)
            return out
        smm.hub(bufs["lap4"][2 * k0:2 * k], apb4[2 * k0:2 * k], out[None],
                smm.level(n, n, n, min(STRASSEN_HUB, self._s_hub)))
        return out

    # ------------------------------------------------------------------
    def _dslices(self, A_st, P_st, Z_st, L_st, w2b_list, s_list, e_list,
                 c1_list, c2_list, y_list, n, bufs, rres, rfb, Zf_st, R1T_st, R2T_st,
                 need_d21=True, ka=0, FAo=None, FPo=None, Qc=None,
                 kb=0, FA2=None, FP2=None, U2=None, apb=None, apb4=None,
                 sb1=None, sb2=None, s_sb=0, nrow=None, r0=0):
        """(3,) and (2,1) dslices with the M leg expressed through A, P and the thin
        residual leg: M = P*diag(s) + 3*A*diag(e) + Z L^T (the factor 3 of the
        residual is folded into Z at birth). Two dense contractions (right factors
        A and P) + O(n^2 r) thin terms.
          D21 = 2H + T2 + T3/3 + 2T4/3,  H=(A*P*w2)A^T, T2=(A*A*w2)P^T,
                T3=(P*P)M^T, T4=(M*P)P^T
          D3  = 3 sum_j w2_j A_ij^2 P_ij + sum_j M_ij P_ij^2
        Every (k,n,n) intermediate is written with out= into pooled buffers
        (F53: fresh result buffers dominate residual time).
        need_d21=False (V19, final layer): D3 only -- no left factors, no dense
        contractions, no thin right-factor terms; returns D21 = None.
        """
        k = len(w2b_list)
        W2B = fnp.stack(w2b_list, axis=0)[:, None, :]
        Sb = fnp.stack(s_list, axis=0)[:, None, :]
        Eb = fnp.stack(e_list, axis=0)[:, None, :]
        AP = fnp.multiply(A_st, P_st, out=bufs["ap"][:k])
        PP = fnp.multiply(P_st, P_st, out=bufs["pp"][:k])
        T = bufs["t"][:k]
        # M*P = PP*s + 3 AP*e + (Z L^T)*P
        MP = fnp.matmul(Z_st, fnp.swapaxes(L_st, -1, -2), out=bufs["mp"][:k])   # V27: BLAS path
        fnp.multiply(MP, P_st, out=MP)
        fnp.multiply(PP, Sb, out=T)
        fnp.add(MP, T, out=MP)
        fnp.multiply(AP, Eb * 3.0, out=T)
        fnp.add(MP, T, out=MP)
        if need_d21:
            # leftA = 2 AP*w2 + PP*e   (right factor A)
            LA = fnp.multiply(AP, W2B * 2.0, out=bufs["lap"][:k, 0])
            fnp.multiply(PP, Eb, out=T)
            fnp.add(LA, T, out=LA)
            # leftP = A*A*w2 + PP*(s/3) + (2/3) M*P   (right factor P)
            LP = fnp.multiply(A_st, A_st, out=bufs["lap"][:k, 1])
            fnp.multiply(LP, W2B, out=LP)
            fnp.multiply(PP, Sb * (1.0 / 3.0), out=T)
            fnp.add(LP, T, out=LP)
            fnp.multiply(MP, 2.0 / 3.0, out=T)
            fnp.add(LP, T, out=LP)
        fnp.multiply(P_st, W2B, out=T)
        D3 = (fnp.einsum("kij,kij,kij->i", A_st, A_st, T) * 3.0
              + fnp.einsum("kij,kij->i", MP, P_st))
        if rfb > 0:
            # V18 (F69): B1 pair with X1 = 3A + Xt, Y1 = A*w2 + Yt; Xt = F1 R1, Yt = F2 R2
            # (F1/F2 = transported thin Z columns, R1T/R2T static (k, n, rfb)).
            F1 = Zf_st[:, :, :rfb]
            F2 = Zf_st[:, :, rfb:]
            Xt = fnp.einsum("kiq,kjq->kij", F1, R1T_st, out=bufs["xt"][:k])
            Yt = fnp.einsum("kiq,kjq->kij", F2, R2T_st, out=bufs["yt"][:k])
            U = bufs["u"][:k]
            # LPadd = A*Yt + (1/3) Xt*A*w2 + (1/3) Xt*Yt  (in T);  D3 += 3 rowsum(P*LPadd)
            fnp.multiply(A_st, Yt, out=T)
            fnp.multiply(Xt, A_st, out=U)
            fnp.multiply(U, W2B * (1.0 / 3.0), out=U)
            fnp.add(T, U, out=T)
            fnp.multiply(Xt, Yt, out=U)
            fnp.multiply(U, 1.0 / 3.0, out=U)
            fnp.add(T, U, out=T)
            D3 = D3 + fnp.einsum("kij,kij->i", P_st, T) * 3.0
            if need_d21:
                fnp.add(LP, T, out=LP)
                # LA += (1/3) Xt*P*w2 + P*Yt
                fnp.multiply(Xt, P_st, out=T)
                fnp.multiply(T, W2B * (1.0 / 3.0), out=T)
                fnp.add(LA, T, out=LA)
                fnp.multiply(P_st, Yt, out=U)
                fnp.add(LA, U, out=LA)
                # thin right factors:  GY = AP + (1/3) Xt*P      -> D21 += (GY R2^T) F2^T
                #                      GX = (1/3)(AP*w2 + P*Yt)  -> D21 += (GX R1^T) F1^T
                fnp.multiply(Xt, P_st, out=T)
                fnp.multiply(T, 1.0 / 3.0, out=T)
                fnp.add(T, AP, out=T)                 # GY
                # V27: thin contractions as batched matmul into the free (k,n,n) scratch
                # (Xt is consumed above) + a k-sum written out= (no fresh (n,n) results)
                GYR = fnp.matmul(T, R2T_st, out=bufs["gyr"][:k])
                fnp.matmul(GYR, fnp.swapaxes(F2, -1, -2), out=Xt)
                D21_fb = fnp.sum(Xt, axis=0, out=bufs["d21fb"])
                fnp.multiply(AP, W2B, out=T)
                fnp.add(T, U, out=T)                  # AP*w2 + P*Yt   (U holds P*Yt)
                fnp.multiply(T, 1.0 / 3.0, out=T)     # GX
                GXR = fnp.matmul(T, R1T_st, out=bufs["gxr"][:k])
                fnp.matmul(GXR, fnp.swapaxes(F1, -1, -2), out=Xt)
                fnp.sum(Xt, axis=0, out=bufs["t1"])
                fnp.add(D21_fb, bufs["t1"], out=D21_fb)
            else:
                D21_fb = None
        else:
            D21_fb = None
        # F68 feed pair (X3 = A*c1 + P*c2 column scalings, Y3 = y 1^T):
        #   D21 += (1/3) R y^T + (2/3) (y (.) X3) P^T,  D3 += y * R,
        #   R = rowsum(X3*P) = (A*P) c1 + (P*P) c2; y is the TRANSPORTED Y3 row vector
        #   (last Z column).   MP/T are free scratch now.
        C1 = fnp.stack(c1_list, axis=0)                 # (k, n) static column scalings
        C2 = fnp.stack(c2_list, axis=0)
        Yk = Z_st[:, :, rres + 1]                        # (k, n) transported y = P(l) y
        R = fnp.einsum("kij,kj->ki", AP, C1) + fnp.einsum("kij,kj->ki", PP, C2)
        D3 = D3 + fnp.einsum("ki,ki->i", R, Yk)
        if not need_d21:
            return D3, None
        fnp.multiply(P_st, C2[:, None, :], out=MP)
        fnp.multiply(A_st, C1[:, None, :], out=T)
        fnp.add(T, MP, out=T)
        fnp.multiply(T, (Yk * (2.0 / 3.0))[:, :, None], out=T)
        fnp.add(LP, T, out=LP)
        if ka > 0:
            # V21: old sources through the shared basis: [sum LA FAo^T + LP FPo^T] Qc^T
            # V24: tier-2 sources through U: [sum LA FA2^T + LP FP2^T] U2^T lifted first (U2 = sub-basis; U is the scratch buffer)
            # V28: each tier's contraction sum_k LA_k FA_k^T + LP_k FP_k^T is ONE hub
            # family over the interleaved slabs (rectangular right operand (2m, 1, r, n));
            # level 0 = the dense batched-matmul leaf + k-sum (same billing as the einsum)
            inner = None
            smm = self._smm
            m_ = n if nrow is None else int(nrow)   # V30: live rows of this layer
            r0 = int(r0)                            # V30: first readout row (saturated units skipped)
            mb_ = m_ - r0
            smm.smin = SMIN_SB                      # V37: rectangular leaf minimum (shared basis)
            if ka > kb:
                r1 = sb1.shape[2]
                inner = self._pool.get("inner", (1, mb_, r1))
                smm.hub(bufs["lap4"][2 * kb:2 * ka][..., r0:m_, :], sb1, inner, smm.level(mb_, n, r1, s_sb))
                inner = inner[0]
            if kb > 0:
                r2_ = sb2.shape[2]
                inner2 = self._pool.get("inner2", (1, mb_, r2_))
                smm.hub(bufs["lap4"][:2 * kb][..., r0:m_, :], sb2, inner2, smm.level(mb_, n, r2_, s_sb))
                lift = fnp.matmul(inner2[0], U2.T, out=self._pool.get("lift", (mb_, U2.shape[0])))
                inner = lift if inner is None else fnp.add(inner, lift, out=inner)
            smm.smin = STRASSEN_MIN

            def lift_qc(dst):
                """dst (mb_, m_) = inner Qc[:m_]^T (V37: a one-slot Strassen family)."""
                smm.smin = SMIN_SB
                lev_l = smm.level(mb_, inner.shape[1], m_, min(STRASSEN_JOIN, s_sb))
                if lev_l > 0:
                    smm.mm(inner[None, None], Qc.T[None, None][..., :m_], dst[None, None], lev_l)
                elif m_ < n:
                    fnp.matmul(inner, Qc[:m_].T, out=dst)
                else:
                    fnp.matmul(inner, Qc.T, out=dst)
                smm.smin = STRASSEN_MIN

            # V27: D21 accumulates in one pooled buffer (a + b == b + a exactly, so the
            # hub-first order is bit-identical to V26's D21 + hub)
            if m_ < n or r0 > 0:
                # V30: the (r0:m_, :m_) live block; the other rows / columns of D21 are zero
                if ka < k:
                    D21 = self._hub2(bufs, apb4, ka, k, n, m_, r0)
                    t1v = bufs["t1"][r0:m_, :m_]
                    lift_qc(t1v)
                    fnp.add(D21[r0:m_, :m_], t1v, out=D21[r0:m_, :m_])
                else:
                    D21 = bufs["hub"]
                    fnp.copyto(D21, 0.0)
                    lift_qc(D21[r0:m_, :m_])
            elif ka < k and STRASSEN_HUB > 0:
                D21 = self._hub2(bufs, apb4, ka, k, n)
                lift_qc(bufs["t1"])
                fnp.add(D21, bufs["t1"], out=D21)
            else:
                D21 = fnp.matmul(inner, Qc.T, out=bufs["d21"])
                if ka < k:
                    D21 = (D21 + fnp.einsum("kij,kcj->ic", LA[ka:], A_st[ka:])
                           + fnp.einsum("kij,kcj->ic", LP[ka:], P_st[ka:]))
            fnp.matmul(R.T, Yk, out=bufs["t1"])
            fnp.multiply(bufs["t1"], 1.0 / 3.0, out=bufs["t1"])
            fnp.add(D21, bufs["t1"], out=D21)
        else:
            if STRASSEN_HUB > 0 or (nrow is not None and (nrow < n or r0 > 0)):
                D21 = self._hub2(bufs, apb4, 0, k, n, nrow, r0)
                fnp.matmul(R.T, Yk, out=bufs["t1"])
                fnp.multiply(bufs["t1"], 1.0 / 3.0, out=bufs["t1"])
                fnp.add(D21, bufs["t1"], out=D21)
            else:
                D21 = (fnp.einsum("kij,kcj->ic", LA, A_st)
                       + fnp.einsum("kij,kcj->ic", LP, P_st)
                       + fnp.einsum("ki,kc->ic", R, Yk) * (1.0 / 3.0))
        PPL = fnp.matmul(PP, L_st, out=bufs["ppl"][:k])   # V27: BLAS path (1.2 s -> 8 ms)
        t1 = fnp.einsum("kiq,kcq->ic", PPL, Z_st)
        fnp.multiply(t1, 1.0 / 3.0, out=t1)
        D21 = fnp.add(D21, t1, out=bufs["d21"])   # in place, or hub/dense result -> d21
        if D21_fb is not None:
            fnp.add(D21, D21_fb, out=D21)
        return D3, _zero_diag(D21)


if __name__ == "__main__":
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=1024, depth=16, seed=0)
    compare_against_monte_carlo(Estimator(), mlp, seed=0)
