import json, collections
rows = [json.loads(l) for l in open(r"E:\Claude code\whest\runs\sweep.jsonl", encoding="utf-8") if l.strip()]
print(len(rows), "rows; keys:", sorted(rows[0].keys()))
by = collections.OrderedDict()
for r in rows:
    by.setdefault(r.get("tag"), []).append(r)
for tag, rs in by.items():
    ids = [r.get("id", r.get("mlp")) for r in rs]
    mse = [r.get("mse") for r in rs]
    cb = [r.get("cb", r.get("C/B", r.get("c_over_b"))) for r in rs]
    try:
        m = sum(mse) / len(mse)
        c = sum(cb) / len(cb)
    except Exception:
        m = c = float("nan")
    print(f"{tag:28s} n={len(rs):2d} mse={m:.4e} cb={c:.4f} ids={ids}")
