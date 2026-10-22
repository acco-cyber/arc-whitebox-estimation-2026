"""Is the K3 chain's remaining error predictable from the K3-K2 difference (Aitken-style
extrapolation across cumulant orders)?  usage: python aitken.py <k3tag[,k3tag2]> <k2tag> <ids>
For every layer: e3 = truth - m3, d = m3 - m2; through-origin and feature regressions of e3 on d,
pooled over MLPs, leave-one-MLP-out R^2 (= fraction of MSE a correction would remove).
"""
import sys
from math import erf, sqrt, pi

import numpy as np

ROOT = r"E:\Claude code\whest"


def Phi(a):
    return 0.5 * (1.0 + np.vectorize(erf)(a / sqrt(2.0)))


def phi(a):
    return np.exp(-0.5 * a * a) / sqrt(2 * pi)


def g(a):
    return phi(a) + a * Phi(a)


def solve_sigma(mu, m):
    lo = np.full_like(mu, 1e-4)
    hi = np.full_like(mu, 10.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        up = mid * g(mu / mid) < m
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def load_pred(tags, mid):
    for t in tags:
        try:
            return np.load(rf"{ROOT}\runs\pred\pred_{t}_{mid:04d}.npy").astype(np.float64)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(mid)


def loo(F, y, groups):
    num = den = 0.0
    for gval in np.unique(groups):
        te = groups == gval
        b, *_ = np.linalg.lstsq(F[~te], y[~te], rcond=None)
        num += np.sum((y[te] - F[te] @ b) ** 2)
        den += np.sum(y[te] ** 2)
    b, *_ = np.linalg.lstsq(F, y, rcond=None)
    return 1 - num / den, b


def main():
    k3 = sys.argv[1].split(",")
    k2 = sys.argv[2]
    ids = [int(x) for x in sys.argv[3].split(",")]
    data = []
    for mid in ids:
        d = np.load(rf"{ROOT}\data\npz\mlp_{mid:04d}.npz")
        alm = d["all_layer_means"].astype(np.float64)
        W = d["weights"].astype(np.float64)
        m3 = load_pred(k3, mid)
        m2 = np.load(rf"{ROOT}\runs\pred\pred_{k2}_{mid:04d}.npy").astype(np.float64)
        data.append((alm, W, m3, m2))
    print("layer  mse3      mse2      R2(e3~d)  beta     R2(e3~feat)  R2 final-only?")
    for l in range(16):
        e3, dd, grp, al_all, sg_all = [], [], [], [], []
        for gi, (alm, W, m3, m2) in enumerate(data):
            e3.append(alm[l] - m3[l])
            dd.append(m3[l] - m2[l])
            grp.append(np.full(alm.shape[1], gi))
            if l > 0:
                mu = m3[l - 1] @ W[l]
                sg = solve_sigma(mu, np.maximum(m3[l], 1e-12))
            else:
                mu = np.zeros(alm.shape[1])
                sg = np.sqrt((W[0] ** 2).sum(0))
            al_all.append(mu / sg)
            sg_all.append(sg)
        e3 = np.concatenate(e3)
        dd = np.concatenate(dd)
        grp = np.concatenate(grp)
        al = np.concatenate(al_all)
        sg = np.concatenate(sg_all)
        r2a, ba = loo(dd[:, None], e3, grp)
        p = phi(al)
        F = np.stack([dd, dd * Phi(al), dd * p, dd * al * p, p * sg, al * p * sg], axis=1)
        r2b, bb = loo(F, e3, grp)
        mse2 = np.mean((e3 + dd) ** 2)
        print(f"{l:3d}  {np.mean(e3 ** 2):.3e}  {mse2:.3e}  {r2a:+.4f}  {ba[0]:+.4f}   {r2b:+.4f}")


if __name__ == "__main__":
    main()
