"""Is the per-layer error of the chain a function of local per-neuron quantities?
Reconstruct (mu, sigma, alpha) per neuron from the predictions alone, split each layer's
error into the linearly inherited part and the new part, and regress the new part on
Hermite-function features of alpha.
usage: python resid_alpha.py <tag> <ids>
"""
import sys
import numpy as np
from math import erf, sqrt, pi

ROOT = r"E:\Claude code\whest"
SQ2 = sqrt(2.0)


def phi(a):
    return np.exp(-0.5 * a * a) / sqrt(2 * pi)


def Phi(a):
    return 0.5 * (1.0 + np.vectorize(erf)(a / SQ2))


def g(a):
    return phi(a) + a * Phi(a)


def solve_sigma(mu, m):
    lo = np.full_like(mu, 1e-4)
    hi = np.full_like(mu, 10.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = mid * g(mu / mid)
        up = val < m
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def feats(al, sg):
    p = phi(al)
    P = Phi(al)
    cols = [p, al * p, (al * al - 1) * p, (al ** 3 - 3 * al) * p,
            sg * p, sg * al * p, sg * (al * al - 1) * p, sg * (al ** 3 - 3 * al) * p,
            p / sg, al * p / sg, (al * al - 1) * p / sg, P, sg * P]
    return np.stack(cols, axis=1)


def main():
    tag = sys.argv[1]
    ids = [int(x) for x in sys.argv[2].split(",")]
    L = 16
    acc = {l: [] for l in range(L)}
    for mid in ids:
        d = np.load(rf"{ROOT}\data\npz\mlp_{mid:04d}.npz")
        alm = d["all_layer_means"].astype(np.float64)
        W = d["weights"].astype(np.float64)
        p = np.load(rf"{ROOT}\runs\pred\pred_{tag}_{mid:04d}.npy").astype(np.float64)
        e = p - alm
        for l in range(1, L):
            mu = p[l - 1] @ W[l]
            sg = solve_sigma(mu, np.maximum(p[l], 1e-12))
            al = mu / sg
            inh = Phi(al) * (e[l - 1] @ W[l])
            acc[l].append((al, sg, e[l], inh, e[l] - inh))
    print("layer  mse     inh_frac  new_mse   R2(new~feats, pooled over MLPs, in-sample)   R2 cross-MLP   alpha_std  sigma_mean")
    for l in range(1, L):
        al = np.concatenate([a[0] for a in acc[l]])
        sg = np.concatenate([a[1] for a in acc[l]])
        el = np.concatenate([a[2] for a in acc[l]])
        inh = np.concatenate([a[3] for a in acc[l]])
        new = np.concatenate([a[4] for a in acc[l]])
        F = feats(al, sg)
        beta, *_ = np.linalg.lstsq(F, new, rcond=None)
        r2 = 1 - np.mean((new - F @ beta) ** 2) / np.mean(new ** 2)
        # leave-one-MLP-out
        n = 1024
        k = len(acc[l])
        num = den = 0.0
        for j in range(k):
            tr = np.ones(k * n, bool)
            tr[j * n:(j + 1) * n] = False
            b2, *_ = np.linalg.lstsq(F[tr], new[tr], rcond=None)
            num += np.sum((new[~tr] - F[~tr] @ b2) ** 2)
            den += np.sum(new[~tr] ** 2)
        r2cv = 1 - num / den
        # same for the TOTAL error (what a correction would actually remove)
        bt, *_ = np.linalg.lstsq(F, el, rcond=None)
        r2t = 1 - np.mean((el - F @ bt) ** 2) / np.mean(el ** 2)
        print(f"{l:3d}  {np.mean(el**2):.2e}  {1 - np.mean(new**2)/np.mean(el**2):+.3f}   {np.mean(new**2):.2e}   "
              f"{r2:+.4f}   {r2cv:+.4f}   total-err R2 {r2t:+.4f}   {al.std():.2f}   {sg.mean():.3f}")
    # binned bias of the final-layer error vs alpha
    al = np.concatenate([a[0] for a in acc[L - 1]])
    el = np.concatenate([a[2] for a in acc[L - 1]])
    edges = [-99, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 99]
    print("final layer: alpha-bin  count  mean_err  rms_err")
    for a0, a1 in zip(edges[:-1], edges[1:]):
        s = (al >= a0) & (al < a1)
        if s.sum():
            print(f"  [{a0:5.1f},{a1:5.1f})  {s.sum():5d}  {el[s].mean():+.2e}  {np.sqrt((el[s]**2).mean()):.2e}")


if __name__ == "__main__":
    main()
