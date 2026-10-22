"""Profile the participant-side Python time of one steady predict() in client/server mode.
Usage: python client_prof.py <estimator.py> <mlp_id> [topN]
Requires FLOPSCOPE_SERVER_URL and a running flopscope-server.
"""
import cProfile
import importlib.util
import io
import json
import os
import pstats
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stubs"))

import flopscope as flops  # noqa: E402
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
    est_path, mid = sys.argv[1], int(sys.argv[2])
    topn = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    ev = load_est(est_path)
    est = ev.Estimator()
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
        est.setup(SetupContext(width=1024, depth=16, flop_budget=B, seed=0))
    d = NPY.format(mid)
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
        ws = [fnp.load(os.path.join(d, f"w_{l:02d}.npy")) for l in range(16)]
    mlp = MLP(width=1024, depth=16, weights=ws, seed=0, name=str(mid))
    for rep in range(2):
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        with flops.BudgetContext(flop_budget=B, quiet=True) as ctx:
            if rep == 1:
                pr.enable()
            pred = est.predict(mlp, B)
            if rep == 1:
                pr.disable()
        sd = ctx.summary_dict()
        print(f"rep{rep}: C/B={ctx.flops_used / B:.4f} resid={sd.get('residual_wall_time_s')} "
              f"backend={sd.get('flopscope_backend_time_s')} overhead={sd.get('flopscope_overhead_time_s')} "
              f"wall={time.perf_counter() - t0:.1f}s", flush=True)
        if rep == 1:
            for key in ("tottime", "cumulative"):
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats(key)
                ps.print_stats(topn)
                print(f"===== sorted by {key} =====")
                print(s.getvalue())
            print(json.dumps({k: v for k, v in sd.items() if isinstance(v, (int, float, str))}, indent=1)[:3000])


if __name__ == "__main__":
    main()
