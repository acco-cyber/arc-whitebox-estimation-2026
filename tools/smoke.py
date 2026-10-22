"""Smoke-shape parity: run two estimators on random MLPs of non-suite shapes; outputs must be finite
and (for shapes where the new code path is gated off) identical.
Usage: python smoke.py <est_new.py> <est_ref.py>
"""
import importlib.util
import os
import sys
import time

import numpy as np

SHAPES = [(4, 2), (64, 6), (64, 16), (128, 20), (256, 32), (256, 16), (512, 16), (1024, 3)]


def load_est(path):
    spec = importlib.util.spec_from_file_location("est_" + os.path.basename(path).replace(".", "_"), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import SetupContext
    from whestbench.domain import MLP

    mods = [load_est(p) for p in sys.argv[1:3]]
    ests = []
    for m in mods:
        e = m.Estimator()
        e.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", seed=0))
        ests.append(e)
    rng = np.random.default_rng(123)
    ok = True
    for (n, L) in SHAPES:
        ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]
        outs = []
        for e in ests:
            mlp = MLP(width=n, depth=L, weights=[fnp.array(w, dtype=fnp.float32) for w in ws], seed=0)
            t0 = time.perf_counter()
            with flops.BudgetContext(flop_budget=int(1e15), quiet=True) as ctx:
                p = np.asarray(e.predict(mlp, 2 ** 41), dtype=np.float64)
            outs.append((p, ctx.flops_used, time.perf_counter() - t0))
        p0, p1 = outs[0][0], outs[1][0]
        fin = bool(np.all(np.isfinite(p0)))
        diff = float(np.max(np.abs(p0 - p1)))
        shape_ok = p0.shape == (L, n)
        print(f"shape {n}x{L}: finite={fin} shape_ok={shape_ok} maxdiff_vs_ref={diff:.3e} "
              f"flops new/ref={outs[0][1]:.4e}/{outs[1][1]:.4e} t={outs[0][2]:.2f}s", flush=True)
        ok = ok and fin and shape_ok
    print("SMOKE", "OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
