import time, sys
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

print([ (d.get('internal_api'), d.get('num_threads'), d.get('version')) for d in threadpool_info()])
n = 1024
rng = np.random.default_rng(0)
A = rng.standard_normal((n, n), dtype=np.float32)
Bm = rng.standard_normal((n, n), dtype=np.float32)
S = rng.standard_normal((8, n, n), dtype=np.float32)
small = rng.standard_normal((2401, 64, 64), dtype=np.float32)
for th in (1, 2, 4, 6, 8, 12):
    with threadpool_limits(th):
        t = time.perf_counter();
        for _ in range(5): A @ Bm
        t1 = (time.perf_counter() - t) / 5
        t = time.perf_counter(); np.matmul(A[None], S); t2 = (time.perf_counter() - t) / 8
        t = time.perf_counter(); np.matmul(small, small); t3 = time.perf_counter() - t
        print(f"threads={th:2d}: 1024^3 {t1*1e3:7.1f} ms ({2*n**3/t1/1e9:6.1f} GFLOPS) | stacked8 per-mat {t2*1e3:7.1f} ms | 2401x64^3 {t3*1e3:7.1f} ms ({2401*2*64**3/t3/1e9:6.1f} GFLOPS)")
