"""Count client->server round trips of one steady predict(), by estimator source line.
Usage: python client_calls.py <estimator.py> <mlp_id> [topN]
Requires FLOPSCOPE_SERVER_URL and a running flopscope-server.
"""
import collections
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stubs"))

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402
import flopscope._connection as _conn  # noqa: E402
from whestbench import SetupContext  # noqa: E402
from whestbench.domain import MLP  # noqa: E402

B = 2 ** 41
NPY = r"E:\Claude code\whest\data\npy\mlp_{:04d}"
COUNTS = collections.Counter()
KIND = collections.Counter()
ON = [False]
EST = [""]


def load_est(path):
    spec = importlib.util.spec_from_file_location("estimator", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def patch():
    cls = None
    for name in dir(_conn):
        obj = getattr(_conn, name)
        if isinstance(obj, type) and hasattr(obj, "send_recv"):
            cls = obj
            break
    assert cls is not None, "no connection class with send_recv"
    orig = cls.send_recv

    def send_recv(self, *a, **k):
        if ON[0]:
            f = sys._getframe(1)
            site = None
            kind = None
            depth = 0
            while f is not None and depth < 40:
                fn = f.f_code.co_filename
                if kind is None and fn.endswith("_remote_array.py") and f.f_code.co_name == "__getitem__":
                    kind = "slice"
                if fn == EST[0]:
                    site = f"{f.f_lineno}:{f.f_code.co_name}"
                    break
                f = f.f_back
                depth += 1
            COUNTS[(site, kind or "op")] += 1
            KIND[kind or "op"] += 1
        return orig(self, *a, **k)

    cls.send_recv = send_recv


def main():
    est_path, mid = sys.argv[1], int(sys.argv[2])
    topn = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    EST[0] = os.path.abspath(est_path)
    patch()
    spec_path = EST[0]
    ev = load_est(spec_path)
    est = ev.Estimator()
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
        est.setup(SetupContext(width=1024, depth=16, flop_budget=B, seed=0))
    d = NPY.format(mid)
    with flops.BudgetContext(flop_budget=10 ** 15, quiet=True):
        ws = [fnp.load(os.path.join(d, f"w_{l:02d}.npy")) for l in range(16)]
    mlp = MLP(width=1024, depth=16, weights=ws, seed=0, name=str(mid))
    for rep in range(3):
        COUNTS.clear()
        KIND.clear()
        ON[0] = True
        t0 = time.perf_counter()
        with flops.BudgetContext(flop_budget=B, quiet=True) as ctx:
            est.predict(mlp, B)
        ON[0] = False
        sd = ctx.summary_dict()
        print(f"rep{rep}: C/B={ctx.flops_used / B:.4f} resid={sd.get('residual_wall_time_s')} "
              f"calls={sum(KIND.values())} {dict(KIND)} wall={time.perf_counter() - t0:.1f}s", flush=True)
    print("top call sites of the last predict (line:function, kind, count):")
    for (site, kind), c in COUNTS.most_common(topn):
        print(f"  {site!s:40s} {kind:6s} {c}")


if __name__ == "__main__":
    main()
