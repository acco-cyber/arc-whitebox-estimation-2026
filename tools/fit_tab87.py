"""R-2a: per-chain ridge refit of the 87-feature CORR_BETA_NEW table (e30a_lp).

Inputs: dump npz files from tools/dump_feats.py (raw per-layer correction inputs
        along the SHIPPED free-running trajectory of the estimator given by --est).
Target: y_l = truth_l - base_l  (base = the exact quantity delta_l is added to;
        at l<15 that is the post-nonlin mean K1v, at l=15 the final-mean pred).
Features: the 87 CORR_FEATS columns reconstructed in float64 from the raw dump
        quantities, exactly as the shipped code consumes them (verified against
        the dumped delta: F @ CORR_BETA_NEW[li] ~= delta to float32 roundoff).
Fit:    per layer, standardized columns (pooled train std, NO centering),
        ridge penalty 1 except the intercept feature '1' (penalty 0), lambda by
        5-block CV over nets (blocks of ~3 nets) on logspace(-4,4,9).

usage: kitvenv/bin/python tools/fit_tab87.py --est est/e30a_lp.py \
           --dump /home/z/my-project/scratch_refit/dumps/iter0 \
           --ids 0,1,...,13 --out /home/z/my-project/scratch_refit/tab_T1.npz
"""
import argparse
import importlib.util
import json
import math
import os
import re

import numpy as np

SQRT2PI = math.sqrt(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
_erf = np.vectorize(math.erf)
_MOD_E = {"": 0.0, "s3": 1.5, "s4": 2.0, "s33": 3.0, "dn": 1.0, "v": 0.0,
          "ds": 1.0, "cr": 1.0, "c2": 2.0, "k4": 0.0}
_HRE = re.compile(r"^h(\d+)(?:\*(\w+))?$")
LAM_GRID = list(np.logspace(-4, 4, 9))


def load_mod(path):
    spec = importlib.util.spec_from_file_location("est_" + os.path.basename(path).replace(".", "_"), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def he_monomials(kmax):
    """He_k probabilists' coefficients in the monomial basis (module's recurrence)."""
    He = [[1.0], [0.0, 1.0]]
    for k in range(2, kmax + 1):
        a = [0.0] + He[k - 1]
        b = He[k - 2] + [0.0, 0.0]
        He.append([a[j] - (k - 1) * b[j] for j in range(k + 1)])
    return He


def feats87(rec, names):
    """rec: dict of (16, n) float arrays with the dump keys -> (16, n, 87) float64."""
    mu = rec["mu"].astype(np.float64)
    var = rec["var"].astype(np.float64)
    n = mu.shape[1]
    sigma = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = mu / sigma
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)
    phi = np.exp(-0.5 * alpha * alpha) / SQRT2PI
    Phi = 0.5 * (1.0 + _erf(alpha / _SQRT2))
    pk1 = rec["pk1"].astype(np.float64)
    pk2 = rec["pk2"].astype(np.float64)
    pk3 = rec["pk3"].astype(np.float64)
    pk4 = rec["pk4"].astype(np.float64)
    d3 = rec["d3"].astype(np.float64)
    d21n = rec["d21n"].astype(np.float64)
    d21s = rec["d21s"].astype(np.float64)
    g4 = rec["g4"].astype(np.float64)
    k4s = rec["k4s"].astype(np.float64)
    crs = rec["crs"].astype(np.float64)
    cr2 = rec["cr2"].astype(np.float64)
    sp = sigma * phi
    base = {"1": np.ones_like(mu), "mu": mu, "var": var, "sigma": sigma, "alpha": alpha,
            "phi": phi, "Phi": Phi, "pred": pk1, "D3": d3, "D21n": d21n, "K4": k4s,
            "g4": g4, "pk2": pk2, "pk3": pk3, "pk4": pk4, "sigma*phi": sp}
    base["|alpha|"] = np.abs(alpha)
    raw = {"": np.ones_like(mu), "s3": d3, "s4": g4, "s33": d3 * d3, "dn": d21n,
           "v": var, "ds": d21s, "cr": crs, "c2": cr2, "k4": k4s}
    kmax = max(int(_HRE.match(nm).group(1)) for nm in names if _HRE.match(nm))
    He = he_monomials(kmax)
    L = mu.shape[0]
    F = np.zeros((L, n, len(names)), dtype=np.float64)
    hcols = {}
    for c, nm in enumerate(names):
        m = _HRE.match(nm)
        if m:
            hcols[c] = (int(m.group(1)), m.group(2) or "")
        else:
            F[:, :, c] = base[nm]
    for c, (k, md) in hcols.items():
        hk = sp * np.polyval(He[k][::-1], alpha)   # He_k(alpha), coefficients low->high
        F[:, :, c] = hk * raw[md] * np.power(var, -_MOD_E[md])
    return F


def ridge_svd(Xs, y, lam, pen, svd=None):
    """Ridge on standardized columns: min ||y - Xs b||^2 + lam ||pen*b||^2."""
    if svd is None:
        svd = np.linalg.svd(Xs, full_matrices=False)
    U, s, Vt = svd
    return Vt.T @ ((s / (s * s + lam * pen)) * (U.T @ y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True, help="estimator file the dumps came from (ship table + CORR_FEATS)")
    ap.add_argument("--dump", required=True, help="dump dir from dump_feats.py")
    ap.add_argument("--ids", default="", help="comma list of dump ids to use (default: all in dir)")
    ap.add_argument("--out", required=True, help="output npz with betas (16,87) f64 + lams")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    mod = load_mod(a.est)
    names = list(mod.CORR_FEATS)
    ship = np.asarray(mod.CORR_BETA_NEW, dtype=np.float64)
    assert ship.shape == (16, len(names)), ship.shape

    ids = ([int(x) for x in a.ids.split(",") if x != ""] if a.ids else
           sorted(int(f[5:9]) for f in os.listdir(a.dump) if f.startswith("dump_")))
    recs = {}
    for mid in ids:
        p = os.path.join(a.dump, f"dump_{mid:04d}.npz")
        recs[mid] = dict(np.load(p))
    print(f"fit ids ({len(ids)}): {ids}", flush=True)

    F = {}
    Y = {}
    dmax = 0.0
    for mid in ids:
        r = recs[mid]
        Ft = feats87(r, names)
        base = r["base"].astype(np.float64)
        delta = r["delta"].astype(np.float64)
        truth = r["truth"].astype(np.float64)
        # verification: reconstructed features reproduce the shipped correction
        for li in range(16):
            d_mod = np.abs(Ft[li] @ ship[li] - delta[li]).max()
            d_rel = d_mod / max(np.abs(delta[li]).max(), 1e-30)
            dmax = max(dmax, d_rel)
        F[mid] = Ft
        base = r["base"].astype(np.float64)
        if "base_asm" in r:
            ba = r["base_asm"].astype(np.float64)
            nz = np.any(ba != 0.0, axis=1)   # base_asm exists only at li<15 (assemble site)
            base = np.where(nz[:, None], ba, base)
        Y[mid] = truth - base
    print(f"feature reconstruction check: max rel deviation vs dumped shipped delta = {dmax:.2e}", flush=True)

    L, nf = 16, len(names)
    ids_sorted = sorted(ids)
    nfold = min(a.folds, len(ids_sorted))
    folds = [ids_sorted[i::nfold] for i in range(nfold)]   # blocks of nets

    ship_beta = np.asarray(mod.CORR_BETA_NEW, dtype=np.float64)
    betas = np.zeros((L, nf))
    lams = np.zeros(L)
    diag = {"ids": ids, "lam_grid": [float(x) for x in LAM_GRID], "layers": []}
    print("layer   base_MSE    ship_1step   cv_MSE     cv_R2    lam*     in_R2", flush=True)
    for li in range(L):
        # per-fold CV over the lambda grid
        sse = {lam: 0.0 for lam in LAM_GRID}
        cnt = 0
        sse_base = 0.0
        sse_ship = 0.0
        for fold in folds:
            tr = [i for i in ids_sorted if i not in fold]
            Xtr = np.concatenate([F[i][li] for i in tr], axis=0)
            ytr = np.concatenate([Y[i][li] for i in tr], axis=0)
            Xte = np.concatenate([F[i][li] for i in fold], axis=0)
            yte = np.concatenate([Y[i][li] for i in fold], axis=0)
            sd = Xtr.std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            pen = np.ones(nf)
            pen[0] = 0.0   # intercept feature '1' unpenalized
            svd = np.linalg.svd(Xtr / sd, full_matrices=False)
            for lam in LAM_GRID:
                b = ridge_svd(None, ytr, lam, pen, svd=svd) / sd
                sse[lam] += float(np.sum((Xte @ b - yte) ** 2))
            cnt += yte.size
            sse_base += float(np.sum(yte ** 2))
            sse_ship += float(np.sum((Xte @ ship_beta[li] - yte) ** 2))
        lam_star = min(LAM_GRID, key=lambda lam: sse[lam])
        # final fit on all ids at lam*
        Xall = np.concatenate([F[i][li] for i in ids_sorted], axis=0)
        yall = np.concatenate([Y[i][li] for i in ids_sorted], axis=0)
        sd = Xall.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        pen = np.ones(nf)
        pen[0] = 0.0
        svd = np.linalg.svd(Xall / sd, full_matrices=False)
        b_all = ridge_svd(None, yall, lam_star, pen, svd=svd) / sd
        betas[li] = b_all
        lams[li] = lam_star
        base_mse = sse_base / cnt
        ship_mse = sse_ship / cnt
        cv_mse = sse[lam_star] / cnt
        cv_r2 = 1.0 - cv_mse / base_mse if base_mse > 0 else float("nan")
        in_r2 = 1.0 - float(np.mean((Xall @ b_all - yall) ** 2)) / float(np.mean(yall ** 2)) \
            if float(np.mean(yall ** 2)) > 0 else float("nan")
        diag["layers"].append({"layer": li, "lam": float(lam_star), "base_mse": base_mse,
                               "ship_1step_mse": ship_mse, "cv_mse": cv_mse, "cv_r2": cv_r2,
                               "in_r2": float(in_r2)})
        print(f"{li:3d}  {base_mse:.4e}  {ship_mse:.4e}  {cv_mse:.4e}  {cv_r2:+.4f}  "
              f"{lam_star:8.3g}  {in_r2:+.4f}", flush=True)

    np.savez(a.out, betas=betas, lams=lams, names=np.array(names, dtype=object),
             ids=np.array(ids))
    with open(a.out.replace(".npz", ".diag.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=1)
    print("table ->", a.out, flush=True)


if __name__ == "__main__":
    main()
