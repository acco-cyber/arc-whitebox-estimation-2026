"""Per-layer alpha = mu/sigma statistics: cheap diagonal pre-pass vs covariance propagation."""
import sys
import numpy as np
from scipy.stats import norm
from scipy.special import owens_t  # noqa: F401

NPZ = r"E:\Claude code\whest\data\npz\mlp_{:04d}.npz"


def relu_moments(mu, var):
    s = np.sqrt(var)
    a = mu / s
    m = s * (norm.pdf(a) + a * norm.cdf(a))
    e2 = var * ((1 + a * a) * norm.cdf(a) + a * norm.pdf(a))
    return m, np.maximum(e2 - m * m, 0.0), a


def covprop(ws):
    """K=2 style propagation with a 2nd-order Mehler series for the off-diagonal (good enough for alpha)."""
    n = ws[0].shape[0]
    C = np.eye(n)
    mu = np.zeros(n)
    out = []
    for w in ws:
        W = w.T.astype(np.float64)
        mu_z = W @ mu
        Cz = W @ C @ W.T
        var = np.diag(Cz).copy()
        m, v, a = relu_moments(mu_z, var)
        s = np.sqrt(var)
        c1 = norm.cdf(a)
        c2 = norm.pdf(a) / s
        c3 = -a * norm.pdf(a) / var
        Coff = Cz - np.diag(var)
        C = (c1[:, None] * c1[None, :]) * Coff + 0.5 * (c2[:, None] * c2[None, :]) * Coff ** 2 \
            + (1.0 / 6.0) * (c3[:, None] * c3[None, :]) * Coff ** 3
        np.fill_diagonal(C, v)
        mu = m
        out.append((a.copy(), m.copy(), var.copy()))
    return out


def diagprop(ws):
    n = ws[0].shape[0]
    v = np.ones(n)
    mu = np.zeros(n)
    out = []
    for w in ws:
        W = w.T.astype(np.float64)
        mu_z = W @ mu
        var = (W * W) @ v
        m, v, a = relu_moments(mu_z, var)
        mu = m
        out.append((a.copy(), m.copy(), var.copy()))
    return out


def main():
    ids = [int(x) for x in sys.argv[1].split(",")]
    for mid in ids:
        d = np.load(NPZ.format(mid))
        ws = d["weights"]
        gt = d["all_layer_means"].astype(np.float64)
        cp = covprop(ws)
        dp = diagprop(ws)
        print(f"== mlp {mid}: covprop final mse {np.mean((cp[-1][1]-gt[-1])**2):.3e}  diag final mse {np.mean((dp[-1][1]-gt[-1])**2):.3e}")
        print(" L  std(a)  var_mean | frac a<-2  a<-2.3 a<-2.5  a<-3 | a>2.5  a>3 | |a|<2.5 | prepass: rms(a_d-a_c)  misrank@-2.5")
        for l in range(16):
            a = cp[l][0]
            ad = dp[l][0]
            fr = [np.mean(a < -t) for t in (2, 2.3, 2.5, 3)]
            # misrank: neurons in the top-(alive) set by covprop but outside top set by prepass when keeping same count
            k = int(np.sum(a > -2.5))
            top_c = set(np.argsort(-a)[:k].tolist())
            top_d = set(np.argsort(-ad)[:k].tolist())
            mis = len(top_c - top_d)
            print(f"{l+1:2d}  {a.std():5.2f}  {cp[l][2].mean():7.4f} | {fr[0]:6.3f} {fr[1]:6.3f} {fr[2]:6.3f} {fr[3]:6.3f} | "
                  f"{np.mean(a>2.5):5.3f} {np.mean(a>3):5.3f} | {np.mean(np.abs(a)<2.5):5.3f} | {np.sqrt(np.mean((ad-a)**2)):6.3f}  {mis:4d}")


if __name__ == "__main__":
    main()
