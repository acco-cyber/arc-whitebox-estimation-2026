"""Build a small local dataset dir (schema 3.0 layout) from one downloaded shard so that
`whest run --dataset <dir>` works offline with the official harness."""
import json
import os
import shutil
import sys

SRC = r"E:\Claude code\whest\data"
DST = r"E:\Claude code\whest\ds_local"
shard = sys.argv[1] if len(sys.argv) > 1 else "mini-00006-of-00007.parquet"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4

os.makedirs(os.path.join(DST, "data"), exist_ok=True)
md = json.load(open(os.path.join(SRC, "metadata.json"), encoding="utf-8"))
md.pop("prepared_splits", None)
md["splits"] = {"mini": md["splits"]["mini"]}
md["splits"]["mini"]["n_mlps"] = n
md["default_split"] = "mini"
json.dump(md, open(os.path.join(DST, "metadata.json"), "w", encoding="utf-8"), indent=1)
dst = os.path.join(DST, "data", "mini-00000-of-00001.parquet")
if not os.path.exists(dst):
    shutil.copyfile(os.path.join(SRC, shard), dst)
print("ok", DST, os.listdir(os.path.join(DST, "data")))
