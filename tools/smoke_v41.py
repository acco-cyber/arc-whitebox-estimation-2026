"""V41 smoke test: exercise the new relabel-sketch (v41alpha) and CORR_BETA dump
(v41corr) code paths at suite WIDTH but depth 2 (riders needs L == len(CORR_BETA),
so the module's CORR_BETA is trimmed to the test depth). Off-suite shape-generic
paths are untouched; this only proves the new branches run and save their dumps.
"""
import importlib.util
import os
import shutil
import sys

import numpy as np

os.environ["V30_LR0"] = "0"
os.environ["V41_DBG"] = "1"
os.environ["V41_DBG_DIR"] = "/tmp/v41dbg"
os.environ["V41_DUMP_DIR"] = "/tmp/v41dump"
for d in ("/tmp/v41dbg", "/tmp/v41dump"):
    shutil.rmtree(d, ignore_errors=True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


import flopscope as flops
import flopscope.numpy as fnp
from whestbench import SetupContext
from whestbench.domain import MLP

L = 2
rng = np.random.default_rng(0)
ws = [(rng.standard_normal((1024, 1024)) / 32.0).astype(np.float32) for _ in range(L)]

# ---- v41alpha: sketch ON ----
ev = load("est/v41alpha.py", "v41a_smoke")
ev.CORR_BETA = ev.CORR_BETA[:L]
est = ev.Estimator()
est.setup(SetupContext(width=1024, depth=L, flop_budget=int(2 ** 41), api_version="1", seed=0))
mlp = MLP(width=1024, depth=L, weights=[fnp.array(w) for w in ws], seed=0, name="smoke")
with flops.BudgetContext(flop_budget=int(2 ** 41), wall_time_limit_s=1200.0, quiet=True) as ctx:
    pred = np.asarray(est.predict(mlp, int(2 ** 41)), dtype=np.float64)
print("v41alpha ON: pred", pred.shape, "finite", bool(np.all(np.isfinite(pred))),
      "flops", f"{ctx.flops_used:.3e}")
dbg = sorted(os.listdir("/tmp/v41dbg"))
print("v41alpha dbg files:", dbg)
a = np.load(f"/tmp/v41dbg/aapx_smoke.npz")["aapx"]
print("  aapx shape", a.shape, "layer0 finite", bool(np.all(np.isfinite(a[0]))))

# ---- v41alpha: sketch OFF must not touch the dbg list either, but runs identical math
os.environ["V41_ASKETCH"] = "0"
ev2 = load("est/v41alpha.py", "v41a_smoke_off")
ev2.CORR_BETA = ev2.CORR_BETA[:L]
est2 = ev2.Estimator()
est2.setup(SetupContext(width=1024, depth=L, flop_budget=int(2 ** 41), api_version="1", seed=0))
with flops.BudgetContext(flop_budget=int(2 ** 41), wall_time_limit_s=1200.0, quiet=True) as ctx2:
    pred2 = np.asarray(est2.predict(mlp, int(2 ** 41)), dtype=np.float64)
print("v41alpha OFF: pred finite", bool(np.all(np.isfinite(pred2))),
      "flops", f"{ctx2.flops_used:.3e}")
a2 = np.load("/tmp/v41dbg/aapx_smoke.npz")["aapx"]
r = np.abs(a[1]).argsort().argsort()
r2 = np.abs(a2[1]).argsort().argsort()
print("  layer1 rank moves >100 (ON vs OFF):", int((np.abs(r - r2) > 100).sum()), "/1024")

# ---- v41corr: dump path ----
os.environ["V41_ASKETCH"] = "1"
ev3 = load("est/v41corr.py", "v41c_smoke")
ev3.CORR_BETA = ev3.CORR_BETA[:L]
est3 = ev3.Estimator()
est3.setup(SetupContext(width=1024, depth=L, flop_budget=int(2 ** 41), api_version="1", seed=0))
with flops.BudgetContext(flop_budget=int(2 ** 41), wall_time_limit_s=1200.0, quiet=True) as ctx3:
    pred3 = np.asarray(est3.predict(mlp, int(2 ** 41)), dtype=np.float64)
print("v41corr dump: pred finite", bool(np.all(np.isfinite(pred3))),
      "flops", f"{ctx3.flops_used:.3e}")
f3 = sorted(os.listdir("/tmp/v41dump"))
print("v41corr dump files:", f3)
d = np.load("/tmp/v41dump/corr_smoke.npz")
print("  feats", d["feats"].shape, "pred", d["pred"].shape, "mu", d["mu"].shape,
      "layers", d["layers"].tolist())
print("  feats row0 layer0:", np.round(d["feats"][0, 0], 4))
print("SMOKE OK")
