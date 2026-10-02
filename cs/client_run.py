"""Grader-like run: the estimator talks to a flopscope SERVER through the flopscope CLIENT
(no numpy in this process). Usage:
  python client_run.py <estimator.py> <ids> [warm]
Requires FLOPSCOPE_SERVER_URL and a running `flopscope-server`.
"""
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stubs"))

import flopscope as flops  # noqa: E402  (the CLIENT package)
import flopscope.numpy as fnp  # noqa: E402
from whestbench import SetupContext  # noqa: E402
from whestbench.domain import MLP  # noqa: E402

B = 2 ** 41
NPY = r"E:\Claude code\whest\data\npy\mlp_{:04d}"


def load_est(path):
    spec = importlib.util.spec_from_file_location("estimator", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    assert "numpy" not in sys.modules or True
    est_path, ids = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
    warm = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    ev = load_est(est_path)
    est = ev.Estimator()
    t0 = time.perf_counter()
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
        est.setup(SetupContext(width=1024, depth=16, flop_budget=B, seed=0))
    print(f"setup {time.perf_counter() - t0:.2f}s numpy_loaded={'numpy' in sys.modules}", flush=True)
    seq = ([ids[0]] if warm else []) + ids
    for pos, mid in enumerate(seq):
        d = NPY.format(mid)
        with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
            ws = [fnp.load(os.path.join(d, f"w_{l:02d}.npy")) for l in range(16)]
        gt = json.load(open(os.path.join(d, "truth.json")))["final_means"]
        mlp = MLP(width=1024, depth=16, weights=ws, seed=0, name=str(mid))
        t0 = time.perf_counter()
        err = ""
        try:
            with flops.BudgetContext(flop_budget=B, quiet=True) as ctx:
                pred = est.predict(mlp, B)
            sd = ctx.summary_dict()
            C = ctx.flops_used
            resid = sd.get("residual_wall_time_s")
            back = sd.get("flopscope_backend_time_s")
            over = sd.get("flopscope_overhead_time_s")
            with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
                last = pred[15].tolist()
                shape = getattr(pred, "shape", None)
            mse = sum((a - b) ** 2 for a, b in zip(last, gt)) / len(gt)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            err = f"{type(e).__name__}: {e}"
            C, resid, back, over, mse, shape = B, -1, -1, -1, -1, None
        wall = time.perf_counter() - t0
        print(f"mlp{mid}{' (warm)' if warm and pos == 0 else ''}: C/B={C / B:.4f} mse={mse:.4e} "
              f"resid={resid} backend={back} overhead={over} wall={wall:.1f}s shape={shape} {err}", flush=True)


if __name__ == "__main__":
    main()
