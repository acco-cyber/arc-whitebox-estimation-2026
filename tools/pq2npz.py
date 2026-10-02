"""Convert dataset parquet shards into per-MLP .npz files (float32 weights + baked truth).

Streams one row at a time (iter_batches) so peak memory stays ~1 MLP.
Usage: python pq2npz.py <parquet> [<parquet> ...]
"""
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = r"E:\Claude code\whest\data\npz"


def _flat(arr, shape):
    a = arr
    if isinstance(a, pa.ExtensionArray):
        a = a.storage
    while pa.types.is_list(a.type) or pa.types.is_large_list(a.type) or pa.types.is_fixed_size_list(a.type):
        a = a.flatten()
    return a.to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(shape)


def main():
    os.makedirs(OUT, exist_ok=True)
    for p in sys.argv[1:]:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=1):
            mid = int(batch.column("mlp_id")[0].as_py())
            dst = os.path.join(OUT, f"mlp_{mid:04d}.npz")
            if os.path.exists(dst):
                print("skip", mid, flush=True)
                continue
            w = _flat(batch.column("weights"), (16, 1024, 1024))
            alm = _flat(batch.column("all_layer_means"), (16, 1024))
            fm = _flat(batch.column("final_means"), (1024,))
            np.savez(dst, weights=w, all_layer_means=alm, final_means=fm,
                     mlp_seed=np.int64(batch.column("mlp_seed")[0].as_py()),
                     mlp_name=str(batch.column("mlp_name")[0].as_py()),
                     avg_variance=float(batch.column("avg_variance")[0].as_py()))
            print("wrote", mid, batch.column("mlp_name")[0].as_py(), float(np.abs(fm - alm[-1]).max()), flush=True)


if __name__ == "__main__":
    main()
