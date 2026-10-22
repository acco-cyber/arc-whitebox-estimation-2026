"""V41 CORR_BETA refit: per-layer ridge of the baked-truth residual on the 13 dump features.

Inputs:  runs/v41dump/corr_<name>.npz  (feats (16,1024,13) f32, pred (16,1024) f32)
         data/npz/mlp_<id>.npz         (all_layer_means = truth, weights)
The dump npz files carry no MLP id, so id<->file is matched by the exact final-layer MSE
against runs/v41_dump.jsonl (run_est computes the same quantity).

Target: y_l = truth_l - pred_l  (the correction delta_l should approximate this).
Features: the 13 CORR_BETA columns, scaled by the pooled train std (NO centering:
column 0 is the ones/intercept feature; its penalty is 0). Ridge via SVD.

Variants:
  plain : every layer fitted independently on the free-running (uncorrected) trajectory
          residual -- the direct analogue of the V16b-era CORR_BETA fit.
  seq   : layers fitted ascending; each layer's target is orthogonalized against the
          linearised propagated previous correction,
              res_l <- res_l - Phi_l * (W_l^T delta_{l-1}),
          with Phi_l = Phi_cdf(alpha_l) read from the dump features (feature 5) and
          W_l the per-MLP weights (pre-mean transport). This approximates the residual
          the CORRECTED trajectory will actually present at layer l.

CV: leave-one-MLP-out over the 3 dump ids; lam per layer per fold by internal 2-fold CV
on the train MLPs; the shipped table is the all-3-MLP fit at the per-layer consensus lam
(median of the folds' lam*).

usage: python fit_corr41.py [--dump runs/v41dump] [--jsonl runs/v41_dump.jsonl]
writes runs/v41_corr_table_<variant>.npz (betas (16,13) f32, lams (16,)) and prints the
CV table. Projection rows report the held-out MSE change per layer as a fraction of the
mean final-layer residual MSE.
"""
import argparse
import json
import glob
import os
from math import erf, sqrt

import numpy as np

ROOT = "/home/z/my-project/whest"
NPZ = os.path.join(ROOT, "data/npz/mlp_{:04d}.npz")
LAM_GRID = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
_erf = np.vectorize(erf)
SQ2 = sqrt(2.0)


def phi_cdf(a):
    return 0.5 * (1.0 + _erf(a / SQ2))


def ridge(X, y, lam, pen):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    sd = s / (s * s + lam * pen)
    return Vt.T @ (sd * (U.T @ y))


def load_dumps(dump_dir, jsonl):
    recs = [json.loads(l) for l in open(jsonl) if l.strip()]
    scored = {r["mlp"]: r["mse"] for r in recs if not r.get("warm")}
    name2id = {}
    for p in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(jsonl)), "..", "data", "npz", "mlp_*.npz"))):
        z = np.load(p)
        name2id[str(z["mlp_name"])] = int(os.path.basename(p)[4:8])
    files = sorted(f for f in os.listdir(dump_dir) if f.startswith("corr_"))
    out = {}
    for f in files:
        d = np.load(os.path.join(dump_dir, f))
        name = f[len("corr_"):-len(".npz")]
        mid = name2id.get(name)
        if mid is None or mid not in scored:
            raise SystemExit(f"{f}: cannot resolve mlp id (name={name!r})")
        out[mid] = d
    assert set(out) == set(scored), f"id mismatch {sorted(out)} vs {sorted(scored)}"
    return out


def fit_layer(Xs, ys, lam, pen):
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return ridge(X, y, lam, pen)


def run_variant(variant, F, RES, Wt, ids, alpha_of):
    """Returns per-fold rows and the shipped table (all-ids fit)."""
    L = 16
    folds_out = []          # per layer: list of (r2_is, r2_cv, lam, dMSE_frac)
    ship_betas = np.zeros((L, 13))
    ship_lams = np.zeros(L)
    for l in range(L):
        per_fold = []
        for ho in ids:
            tr = [i for i in ids if i != ho]
            r2is, r2cv, lam, dfrac = _fold_layer(variant, l, tr, ho, F, RES, Wt, alpha_of)
            per_fold.append((r2is, r2cv, lam, dfrac))
        # consensus lam for the shipped table (median of fold lam*; tie -> smaller)
        lams = sorted(f[2] for f in per_fold)
        lam_star = lams[len(lams) // 2] if len(lams) % 2 else lams[len(lams) // 2 - 1]
        # shipped fit on ALL ids with the sequential scheme
        beta, r2is_all = _ship_layer(variant, l, ids, lam_star, F, RES, Wt, alpha_of,
                                     ship_betas, ship_lams)
        ship_betas[l] = beta
        ship_lams[l] = lam_star
        folds_out.append(per_fold)
    return folds_out, ship_betas, ship_lams


def _seq_targets(l, deltas, ids_subset, F, RES, Wt, alpha_of, per_id=True):
    """residual targets at layer l under the sequential scheme.
    deltas: dict layer -> delta vector (train-fitted); may be empty for layer 0."""
    out = {}
    for i in ids_subset:
        r = RES[i][l].copy()
        if l > 0 and (l - 1) in deltas:
            Phi = phi_cdf(alpha_of(F[i], l))
            r = r - Phi * (Wt[i][l].T @ deltas[l - 1])
        out[i] = r
    return out


def _fold_layer(variant, l, tr, ho, F, RES, Wt, alpha_of):
    L = 16
    deltas = {}
    ytr = yte = None
    Xtr_l = Xte_l = None
    sd = None
    for ll in range(l + 1):
        Xtr = [F[i][ll] for i in tr]
        Xte = [F[i][ll] for i in [ho]]
        sd = np.stack([np.concatenate([x[..., j].ravel() for x in Xtr])
                       for j in range(13)], axis=1).std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        pen = np.ones(13)
        pen[0] = 0.0
        Xtr_s = [x / sd for x in Xtr]
        Xte_s = [x / sd for x in Xte]
        res_tr = _seq_targets(ll, deltas, tr, F, RES, Wt, alpha_of) \
            if variant == "seq" else {i: RES[i][ll] for i in tr}
        res_te = _seq_targets(ll, deltas, [ho], F, RES, Wt, alpha_of) \
            if variant == "seq" else {ho: RES[ho][ll]}
        ytr = np.concatenate([res_tr[i] for i in tr])
        yte = res_te[ho]
        Xtr_c = np.concatenate(Xtr_s, axis=0)
        Xte_c = Xte_s[0]
        if ll < l:
            # internal 2-fold over train MLPs -> lam for this intermediate layer
            best, bl = -9e9, LAM_GRID[0]
            for lam in LAM_GRID:
                num = den = 0.0
                for j in range(len(tr)):
                    m = np.ones(len(ytr), bool)
                    m[j * 1024:(j + 1) * 1024] = False
                    b = ridge(Xtr_c[m], ytr[m], lam, pen)
                    num += np.sum((ytr[~m] - Xtr_c[~m] @ b) ** 2)
                    den += np.sum(ytr[~m] ** 2)
                v = 1 - num / den if den > 0 else -9e9
                if v > best:
                    best, bl = v, lam
            b = ridge(Xtr_c, ytr, bl, pen)
            deltas[ll] = b / sd
        else:
            best, bl = -9e9, LAM_GRID[0]
            for lam in LAM_GRID:
                num = den = 0.0
                for j in range(len(tr)):
                    m = np.ones(len(ytr), bool)
                    m[j * 1024:(j + 1) * 1024] = False
                    b = ridge(Xtr_c[m], ytr[m], lam, pen)
                    num += np.sum((ytr[~m] - Xtr_c[~m] @ b) ** 2)
                    den += np.sum(ytr[~m] ** 2)
                v = 1 - num / den if den > 0 else -9e9
                if v > best:
                    best, bl = v, lam
            b = ridge(Xtr_c, ytr, bl, pen)
            r2is = 1 - np.mean((ytr - Xtr_c @ b) ** 2) / np.mean(ytr ** 2)
            base = float(np.mean(yte ** 2))
            dms = float(np.mean((yte - Xte_c @ b) ** 2)) - base
            # dMSE as fraction of the mean final-layer base residual MSE (per fold)
            m15 = float(np.mean(RES[ho][15] ** 2))
            return r2is, (1 - (base + dms) / base if base > 0 else float("nan")), bl, dms / m15
    raise RuntimeError


def _ship_layer(variant, l, ids, lam, F, RES, Wt, alpha_of, ship_betas, ship_lams):
    deltas = {}
    for ll in range(l + 1):
        Xtr = [F[i][ll] for i in ids]
        sd = np.stack([np.concatenate([x[..., j].ravel() for x in Xtr])
                       for j in range(13)], axis=1).std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        pen = np.ones(13)
        pen[0] = 0.0
        Xtr_c = np.concatenate([x / sd for x in Xtr], axis=0)
        res = _seq_targets(ll, deltas, ids, F, RES, Wt, alpha_of) \
            if variant == "seq" else {i: RES[i][ll] for i in ids}
        ytr = np.concatenate([res[i] for i in ids])
        if ll < l:
            # lam for this intermediate layer: already shipped by the ascending loop
            b = ridge(Xtr_c, ytr, ship_lams[ll], pen)
            deltas[ll] = b / sd
        else:
            b = ridge(Xtr_c, ytr, lam, pen)
            return b / sd, None
    raise RuntimeError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(ROOT, "runs/v41dump"))
    ap.add_argument("--jsonl", default=os.path.join(ROOT, "runs/v41_dump.jsonl"))
    ap.add_argument("--variant", default="both", choices=["plain", "seq", "both"])
    args = ap.parse_args()

    dumps = load_dumps(args.dump, args.jsonl)
    ids = sorted(dumps)
    L = 16
    F = {i: dumps[i]["feats"].astype(np.float64) for i in ids}
    P = {i: dumps[i]["pred"].astype(np.float64) for i in ids}
    T = {i: np.load(NPZ.format(i))["all_layer_means"].astype(np.float64) for i in ids}
    Wt = {i: np.load(NPZ.format(i))["weights"].astype(np.float64) for i in ids}
    RES = {i: T[i] - P[i] for i in ids}

    def alpha_of(Fi, l):
        return Fi[l][:, 4]

    m15 = np.mean([np.mean(RES[i][15] ** 2) for i in ids])
    print(f"ids={ids}  mean final-layer base residual MSE (== final MSE) {m15:.4e}")
    print(f"per-layer base residual MSE: " +
          " ".join(f"{np.mean([np.mean(RES[i][l]**2) for i in ids]):.2e}" for l in range(L)))

    for variant in (["plain", "seq"] if args.variant == "both" else [args.variant]):
        print(f"\n================ variant: {variant} ================")
        print("layer  res_MSE    in-sample R2   CV-R2(loo-mlp)   lam*      "
              "held-out dMSE / final")
        folds, ship_betas, ship_lams = run_variant(variant, F, RES, Wt, ids, alpha_of)
        tot = 0.0
        for l in range(L):
            r2is = np.mean([f[0] for f in folds[l]])
            r2cv = np.mean([f[1] for f in folds[l]])
            dfr = np.mean([f[3] for f in folds[l]])
            tot += dfr
            lams = sorted({f[2] for f in folds[l]})
            print(f"{l:3d}  {np.mean([np.mean(RES[i][l]**2) for i in ids]):.2e}  "
                  f"{r2is:+.4f}       {r2cv:+.4f}        {lams}  {dfr:+.4%}")
        print(f"projection: sum of held-out per-layer dMSE fractions {tot:+.4%} "
              f"(optimistic upper bound; last-layer-only is the safe lower component "
              f"{folds[15][0][3]:+.4%})")
        out = os.path.join(ROOT, f"runs/v41_corr_table_{variant}.npz")
        np.savez(out, betas=ship_betas.astype(np.float32), lams=ship_lams)
        print("shipped table ->", out)


if __name__ == "__main__":
    main()
