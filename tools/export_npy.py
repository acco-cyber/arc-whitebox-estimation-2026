"""Export per-layer .npy weights + truth json for the client/server harness."""
import json
import os
import sys

import numpy as np

SRC = r"E:\Claude code\whest\data\npz\mlp_{:04d}.npz"
DST = r"E:\Claude code\whest\data\npy"
for mid in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(SRC.format(mid))
    out = os.path.join(DST, f"mlp_{mid:04d}")
    os.makedirs(out, exist_ok=True)
    for l in range(16):
        np.save(os.path.join(out, f"w_{l:02d}.npy"), np.ascontiguousarray(d["weights"][l], dtype=np.float32))
    json.dump({"final_means": [float(x) for x in d["final_means"]]}, open(os.path.join(out, "truth.json"), "w"))
    print("exported", mid)
