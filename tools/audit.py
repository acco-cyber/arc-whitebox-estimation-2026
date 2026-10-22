"""Per-family FLOP ledger of one steady predict (audit copy made by mk_audit.py).
usage: python audit.py <audit_estimator.py> <mlp_id>
Run with V28_STRASSEN_FIRST=5 so the first predict is the steady op stream (plus one-time
allocations/reshapes).
"""
import collections
import importlib.util
import os
import sys

import numpy as np

B = float(2 ** 41)
U = float(2 ** 31)
NPZ = r"E:\Claude code\whest\data\npz\mlp_{:04d}.npz"


def main():
    path, mid = sys.argv[1], int(sys.argv[2])
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import SetupContext
    from whestbench.domain import MLP

    spec = importlib.util.spec_from_file_location("est_audit", path)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    est = ev.Estimator()
    est.setup(SetupContext(width=1024, depth=16, flop_budget=int(B), api_version="1", seed=0))
    d = np.load(NPZ.format(mid))
    gt = d["final_means"].astype(np.float64)
    ws = [fnp.array(w, dtype=fnp.float32) for w in d["weights"]]
    mlp = MLP(width=1024, depth=16, weights=ws, seed=0, name=str(d["mlp_name"]))
    with flops.BudgetContext(flop_budget=int(B), wall_time_limit_s=3600.0, quiet=True) as ctx:
        ev._AUD[0] = ctx._namespace_stack
        pred = np.asarray(est.predict(mlp, int(B)), dtype=np.float64)
        ev._AUD[0] = None
    log = list(ctx.op_log)
    tot = float(ctx.flops_used)
    print(f"C/B={tot / B:.4f}  units={tot / U:.1f}  ops={len(log)}  mse={np.mean((pred[-1] - gt) ** 2):.4e}")
    fam = collections.defaultdict(lambda: [0.0, 0])
    famop = collections.defaultdict(lambda: [0.0, 0])
    for r in log:
        ns = r.namespace or "-"
        fam[ns][0] += r.flop_cost
        fam[ns][1] += 1
        key = (ns, r.op_name)
        famop[key][0] += r.flop_cost
        famop[key][1] += 1
    print("family            units    share   ops")
    for ns, (f, c) in sorted(fam.items(), key=lambda kv: -kv[1][0]):
        print(f"  {ns:14s} {f / U:8.2f}  {100 * f / tot:5.1f}%  {c:6d}")
    print("family/op (top 40)")
    for (ns, op), (f, c) in sorted(famop.items(), key=lambda kv: -kv[1][0])[:40]:
        print(f"  {ns:14s} {op:18s} {f / U:8.2f}  {100 * f / tot:5.1f}%  {c:6d}")


if __name__ == "__main__":
    main()
