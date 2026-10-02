"""Grader-style local runner: one process, one Estimator instance, MLPs in sequence.

Usage:
  python run_est.py --est <estimator.py> --ids 96,97,98 [--tag name] [--out file.jsonl]
                    [--warm 1] [--save-pred dir] [--threads N]
Environment knobs of the estimator (V21_R_OLD=..., etc.) are read by the estimator at import.
Reports per MLP: C/B (metered FLOPs / 2**41), final-layer MSE vs the baked N=1e9 truth,
residual wall time, wall time. The first predict() of a process is the estimator's warm-up
call (V29 runs it at a shallower Strassen level), so pass --warm 1 to run MLP ids[0] once
unscored first (as the grader's suite position > 0 sees it).
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

B = float(2 ** 41)
NPZ = r"E:\Claude code\whest\data\npz\mlp_{:04d}.npz"


def _peak_mb():
    """Peak working set of this process in MB (Windows), -1 if unavailable."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / 2 ** 20
    except Exception:  # noqa: BLE001
        pass
    return -1.0


def load_est(path):
    spec = importlib.util.spec_from_file_location("est_" + os.path.basename(path).replace(".", "_"), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--save-pred", default="")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--summary", type=int, default=0)
    a = ap.parse_args()

    if a.threads > 0:
        from threadpoolctl import threadpool_limits
        threadpool_limits(a.threads)

    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import SetupContext
    from whestbench.domain import MLP

    ev = load_est(a.est)
    est = ev.Estimator()
    t0 = time.perf_counter()
    est.setup(SetupContext(width=1024, depth=16, flop_budget=int(B), api_version="1", seed=0))
    t_setup = time.perf_counter() - t0
    ids = [int(x) for x in a.ids.split(",") if x != ""]
    seq = ([ids[0]] if a.warm else []) + ids
    res = []
    for pos, mid in enumerate(seq):
        d = np.load(NPZ.format(mid))
        gt = d["final_means"].astype(np.float64)
        ws = [fnp.array(w, dtype=fnp.float32) for w in d["weights"]]
        mlp = MLP(width=1024, depth=16, weights=ws, seed=0, name=str(d["mlp_name"]))
        t0 = time.perf_counter()
        err = ""
        try:
            with flops.BudgetContext(flop_budget=int(B), wall_time_limit_s=1200.0, quiet=True) as ctx:
                pred = est.predict(mlp, int(B))
                pred = np.asarray(pred, dtype=np.float64)
            C = float(ctx.flops_used)
            resid = float(ctx.residual_wall_time_s)
            try:
                nops = len(ctx.op_log)
            except Exception:  # noqa: BLE001
                nops = -1
            if a.summary and pos == len(seq) - 1:
                print(ctx.summary())
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            pred = np.zeros((16, 1024))
            C = B
            resid = -1.0
            nops = -1
        wall = time.perf_counter() - t0
        mse = float(np.mean((pred[-1] - gt) ** 2))
        ok = bool(np.all(np.isfinite(pred))) and not err
        warm = bool(a.warm and pos == 0)
        rec = dict(tag=a.tag, mlp=mid, warm=warm, cb=C / B, mse=mse, score=mse * max(0.1, C / B),
                   resid=resid, wall=wall, ok=ok, err=err, setup=t_setup, nops=nops)
        res.append(rec)
        print(f"[{a.tag}] mlp{mid:3d}{' (warm-up)' if warm else ''}: C/B={C / B:.4f} mse={mse:.4e} "
              f"score={rec['score']:.4e} resid={resid:.3f}s wall={wall:.1f}s ops={nops} "
              f"peak={_peak_mb():.0f}MB {err}", flush=True)
        if a.save_pred and not warm:
            os.makedirs(a.save_pred, exist_ok=True)
            np.save(os.path.join(a.save_pred, f"pred_{a.tag}_{mid:04d}.npy"), pred)
        if a.out:
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    sc = [r for r in res if not r["warm"]]
    if sc:
        print(f"[{a.tag}] MEAN over {len(sc)}: C/B={np.mean([r['cb'] for r in sc]):.4f} "
              f"raw={np.mean([r['mse'] for r in sc]):.4e} score={np.mean([r['score'] for r in sc]):.4e} "
              f"resid max={max(r['resid'] for r in sc):.3f}s mean={np.mean([r['resid'] for r in sc]):.3f}s "
              f"wall mean={np.mean([r['wall'] for r in sc]):.1f}s", flush=True)


if __name__ == "__main__":
    main()
