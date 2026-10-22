"""Dump per-layer teacher-forced correction records from the e30_a base estimator.

Sets X_TAB_DUMP=1 + X_TAB_FORCE=1 (the estimator's own diagnostics, LP's table-fitting
plumbing), drives est.predict with _TAB_TRUTH set to the npz all_layer_means (1e9 truth),
and saves per net: the pre-correction base, the features' raw ingredients and the truth.

Usage:
  kitvenv/bin/python tools/dump_feats.py --est est/e30a_lp.py --ids 0,1 [--out scratch_refit]
Writes <out>/tab_<id>.npz per net. Each predict takes minutes; results flush per net.
"""
import argparse
import importlib.util
import os
import pathlib
import sys
import time

import numpy as np

B = float(2 ** 41)
NPZ = "/home/z/my-project/whest/data/npz/mlp_{:04d}.npz"


def load_est(path):
    spec = importlib.util.spec_from_file_location("est_" + os.path.basename(path).replace(".", "_"), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", default="/home/z/my-project/scratch_refit")
    a = ap.parse_args()

    os.environ["X_TAB_DUMP"] = "1"
    os.environ["X_TAB_FORCE"] = "1"

    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import SetupContext
    from whestbench.domain import MLP

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ev = load_est(a.est)
    if not getattr(ev, "TAB_DUMP", False) or not getattr(ev, "TAB_FORCE", False):
        raise SystemExit("estimator did not pick up X_TAB_DUMP/X_TAB_FORCE")
    est = ev.Estimator()
    est.setup(SetupContext(width=1024, depth=16, flop_budget=int(B), api_version="1", seed=0))

    for mid in [int(x) for x in a.ids.split(",") if x != ""]:
        f = out / f"tab_{mid:04d}.npz"
        if f.exists():
            print(f"mlp{mid}: dump exists, skip", flush=True)
            continue
        d = np.load(NPZ.format(mid))
        truth = d["all_layer_means"].astype(np.float32)
        gt = d["final_means"].astype(np.float64)
        ws = [fnp.array(w, dtype=fnp.float32) for w in d["weights"]]
        mlp = MLP(width=1024, depth=16, weights=ws, seed=0, name=str(d["mlp_name"]))
        ev._TAB_REC = {}
        ev._TAB_TRUTH = [fnp.asarray(truth[li], dtype=fnp.float32) for li in range(truth.shape[0])]
        t0 = time.perf_counter()
        with flops.BudgetContext(flop_budget=int(B), wall_time_limit_s=1200.0, quiet=True) as ctx:
            pred = np.asarray(est.predict(mlp, int(B)), dtype=np.float64)
        wall = time.perf_counter() - t0
        rec = ev._TAB_REC
        layers = sorted(rec.keys())
        if len(layers) != 16:
            raise SystemExit(f"mlp{mid}: expected 16 layers in _TAB_REC, got {len(layers)}")
        arrays = {}
        for k in ("mu", "var", "pk1", "pk2", "pk3", "pk4", "d3", "d21n", "d21s", "g4",
                  "k4s", "crs", "cr2", "base", "delta_old", "delta", "base_asm"):
            arrays[k] = np.stack([np.asarray(rec[li][k]) for li in layers]).astype(np.float32)
        arrays["lam"] = np.asarray([float(rec[li]["lam"]) for li in layers], dtype=np.float64)
        arrays["truth"] = truth
        arrays["pred_forced_final"] = pred[-1].astype(np.float32)
        mse_forced = float(np.mean((pred[-1] - gt) ** 2))
        np.savez_compressed(f, **arrays)
        print(f"mlp{mid}: dumped {f.name} wall={wall:.0f}s mse_forced_final={mse_forced:.4e} "
              f"lam[8]={arrays['lam'][8]:.3g}", flush=True)


if __name__ == "__main__":
    main()
