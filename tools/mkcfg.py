"""Generate a sweep config: python mkcfg.py out.json est ids base_env_json tag1:K=V,K=V tag2:... """
import json, sys
out, est, ids = sys.argv[1], sys.argv[2], sys.argv[3]
base = json.loads(sys.argv[4])
cfgs = []
for spec in sys.argv[5:]:
    tag, _, kv = spec.partition(":")
    env = dict(base)
    if kv:
        for item in kv.split(";"):
            k, _, v = item.partition("=")
            env[k] = v
    cfgs.append({"tag": tag, "est": est, "ids": ids, "env": env})
json.dump(cfgs, open(out, "w"), indent=1)
print(len(cfgs), "configs ->", out)
