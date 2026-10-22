"""Tiny connectivity + API-semantics probe against the flopscope server (client package)."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stubs"))

import flopscope as flops  # noqa: E402
import flopscope.numpy as fnp  # noqa: E402

print("client flopscope", flops.__version__, "numpy loaded:", "numpy" in sys.modules)
f32 = fnp.float32
with flops.BudgetContext(flop_budget=10 ** 12, quiet=True) as ctx:
    a = fnp.asarray([[3.0, 1.0, 2.0], [0.5, -1.0, 4.0]], dtype=f32)
    # take / argsort / negation
    v = fnp.asarray([0.3, -1.0, 2.0, 0.1], dtype=f32)
    p = fnp.argsort(fnp.multiply(v, -1.0))
    print("argsort desc:", p.tolist(), "inv:", fnp.argsort(p).tolist())
    print("take:", fnp.take(v, p).tolist())
    m = fnp.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=f32)
    print("take axis=1:", fnp.take(m, p, axis=1).tolist())
    print("take axis=0:", fnp.take(m, fnp.asarray([1, 0]), axis=0).tolist())
    # flat buffer + reshaped prefix view + out= into slices (aliasing check)
    flat = fnp.empty((24,), dtype=f32)
    view = fnp.reshape(flat[:12], (2, 3, 2))
    fnp.copyto(view, 1.5)
    fnp.add(view[:, 0:1], 2.0, out=view[:, 1:2])
    print("flat prefix after writes through view:", flat[:12].tolist())
    # strided out= for matmul and add
    big = fnp.zeros((4, 4), dtype=f32)
    x = fnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=f32)
    fnp.matmul(x, x, out=big[:2, :2])
    fnp.add(big[:2, :2], x, out=big[:2, :2])
    fnp.copyto(big[2:, :], 7.0)
    print("big:", big.tolist())
    # 4-D ellipsis slicing + out
    t = fnp.zeros((2, 1, 4, 4), dtype=f32)
    fnp.matmul(x[None, None], x[None, None], out=t[0:1][..., :2, :2])
    print("t00:", t[0, 0].tolist())
    w = fnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=f32)
    print("W.T slice:", w.T[:2, :1].tolist(), "diag:", fnp.diag(fnp.zeros((2, 2), dtype=f32)).tolist())
    q, r = fnp.linalg.qr(fnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=f32))
    print("qr ok", q.shape, "flops", ctx.flops_used)
print(ctx.summary_dict())
