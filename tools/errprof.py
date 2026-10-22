"""Per-layer error profile of saved predictions vs ground truth.
usage: python errprof.py <tag> <ids>      (reads runs/pred/pred_<tag>_<id:04d>.npy)
"""
import sys
import numpy as np

ROOT = r"E:\Claude code\whest"


def main():
    tag = sys.argv[1]
    ids = [int(x) for x in sys.argv[2].split(",")]
    prof = []
    for mid in ids:
        d = np.load(rf"{ROOT}\data\npz\mlp_{mid:04d}.npz")
        alm = d["all_layer_means"].astype(np.float64)
        fm = d["final_means"].astype(np.float64)
        W = d["weights"].astype(np.float64)
        p = np.load(rf"{ROOT}\runs\pred\pred_{tag}_{mid:04d}.npy").astype(np.float64)
        e = p - alm
        mse = (e ** 2).mean(axis=1)
        # propagated error: linearised push of the previous layer's mean error
        # through W (pre-activation mean error), gain ~ fraction active (unknown) -> report raw
        prop = [0.0]
        for l in range(1, 16):
            dmu = W[l].T @ e[l - 1]
            prop.append(float((dmu ** 2).mean()))
        if mid == ids[0]:
            print("final_means vs all_layer_means[15] max abs diff:", np.abs(fm - alm[15]).max())
            print("truth mean of means per layer:", np.round(alm.mean(axis=1), 4))
            print("truth mean of sq means per layer:", np.round((alm ** 2).mean(axis=1), 4))
        prof.append(mse)
        print(f"mlp{mid}: " + " ".join(f"{m:.2e}" for m in mse))
        print(f"   bias  " + " ".join(f"{b:+.1e}" for b in e.mean(axis=1)))
        print(f"   prop  " + " ".join(f"{b:.2e}" for b in prop))
    prof = np.array(prof)
    print("MEAN:  " + " ".join(f"{m:.2e}" for m in prof.mean(axis=0)))


if __name__ == "__main__":
    main()
