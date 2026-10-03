"""Fraction of units per layer with |alpha| >= thr (candidate hub columns to drop), from saved
predictions (alpha reconstructed from consecutive layer means).  usage: python hubfrac.py <tags> <ids>"""
import sys

import numpy as np

sys.path.insert(0, r"E:\Claude code\whest\tools")
from aitken import solve_sigma, load_pred  # noqa: E402

ROOT = r"E:\Claude code\whest"


def main():
    tags = sys.argv[1].split(",")
    ids = [int(x) for x in sys.argv[2].split(",")]
    thrs = [2.0, 2.5, 3.0, 3.5]
    acc = np.zeros((16, len(thrs) * 2 + 1))
    for mid in ids:
        d = np.load(rf"{ROOT}\data\npz\mlp_{mid:04d}.npz")
        W = d["weights"].astype(np.float64)
        m = load_pred(tags, mid)
        for l in range(16):
            if l == 0:
                al = np.zeros(W.shape[1])
            else:
                mu = m[l - 1] @ W[l]
                sg = solve_sigma(mu, np.maximum(m[l], 1e-12))
                al = mu / sg
            row = []
            for t in thrs:
                row.append(np.mean(al >= t))
                row.append(np.mean(al <= -t))
            row.append(np.std(np.clip(al, -8, 8)))
            acc[l] += np.array(row)
    acc /= len(ids)
    print("layer " + " ".join(f"  sat>{t:.1f} dead<-{t:.1f}" for t in thrs) + "  std")
    for l in range(16):
        r = acc[l]
        print(f"{l:4d} " + " ".join(f"   {r[2*i]:.3f}     {r[2*i+1]:.3f}  " for i in range(len(thrs))) + f" {r[-1]:.2f}"
              + "   keep(|a|<2.5)=%.3f keep(|a|<3)=%.3f" % (1 - r[2] - r[3], 1 - r[4] - r[5]))


if __name__ == "__main__":
    main()
