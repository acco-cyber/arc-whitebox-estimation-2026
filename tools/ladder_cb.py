#!/usr/bin/env python3
"""C/B-cut ladder driver: run env-knob configs through run_est.py on small id sets,
aggregate final-layer MSE + C/B, and rank by predicted grader score.

Usage: python tools/ladder_cb.py --batch singles        # one-knob cuts
       python tools/ladder_cb.py --batch combos         # multi-knob cuts
       python tools/ladder_cb.py --batch validate --ids 96,97   # re-run given tags
       python tools/ladder_cb.py --summary              # print table only

Local speed: V37_MAX_EL=2500000, --warm 1, ids 0,1 (matches prior session harness).
Predicted grader score = mean_mse * (mean_cb / 1.35)  [local->grader C/B mapping].
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "kitvenv", "bin", "python")
RUNEST = os.path.join(ROOT, "tools", "run_est.py")
EST = os.path.join(ROOT, "submissions", "335854_s10a", "estimator.py")
OUT = os.path.join(ROOT, "runs", "cb_ladder.jsonl")

MAXEL = "2500000"

# name -> extra env (V37_MAX_EL is always set by the driver)
SINGLES = {
    "base":      {},
    "rold320":   {"V21_R_OLD": "320"},
    "rold256":   {"V21_R_OLD": "256"},
    "rold2_192": {"V24_R_OLD2": "192"},
    "rold2_160": {"V24_R_OLD2": "160"},
    "qp2_1":     {"V24_QPASS2": "1"},
    "qp1_0":     {"V21_QPASS": "0"},
    "age2_5":    {"V24_AGE_OLD2": "5"},
    "sout_mid":  {"V30_S_OUT": "1024,1024,1024,1024,1024,1024,1024,960,960,928,928,896,896,864,864,832"},
    "sout_deep": {"V30_S_OUT": "1024,1024,1024,1024,1024,1024,1024,928,928,896,896,864,864,832,832,832"},
    "sin_late":  {"V30_S_IN": "1024,1024,1024,992,960,928,896,896,864,864,832,832,800,768,736,1024"},
    "sminpr16":  {"V30_SMIN": "16"},
    "sjoin6":    {"V37_SJOIN": "6"},
    "sin_deep":  {"V30_S_IN": "1024,1024,1024,960,928,896,864,864,832,800,800,768,736,704,672,1024"},
}

COMBOS = {
    "c1_mod":   {"V21_R_OLD": "320", "V24_R_OLD2": "160", "V24_QPASS2": "1"},
    "c2_aggr":  {"V21_R_OLD": "256", "V24_R_OLD2": "160", "V24_QPASS2": "1",
                 "V30_S_OUT": "1024,1024,1024,1024,1024,1024,1024,928,928,896,896,864,864,832,832,832"},
    "c3_max":   {"V21_R_OLD": "224", "V24_R_OLD2": "128", "V24_QPASS2": "1", "V24_AGE_OLD2": "5",
                 "V30_S_OUT": "1024,1024,1024,1024,1024,1024,1024,896,896,864,864,832,832,832,832,832",
                 "V30_S_IN": "1024,1024,1024,960,928,896,864,864,832,832,800,800,768,736,704,1024"},
}


def load_done():
    done = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[(r["tag"], tuple(r["ids"]))] = r
                except Exception:
                    pass
    return done


def run_cfg(tag, env, ids):
    e = dict(os.environ)
    e["V37_MAX_EL"] = MAXEL
    e.update(env)
    outj = os.path.join(ROOT, "runs", f"cb_{tag}.jsonl")
    cmd = [PY, RUNEST, "--est", EST, "--ids", ids, "--tag", tag, "--out", outj, "--warm", "1"]
    print(f"=== {tag} env={env}", flush=True)
    p = subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True)
    tail = "\n".join(p.stdout.strip().splitlines()[-3:])
    print(tail, flush=True)
    if p.returncode != 0:
        print(f"!!! {tag} FAILED rc={p.returncode}\n{p.stderr[-800:]}", flush=True)
        return None
    # parse per-MLP records from the out jsonl (skip warm-up rows)
    recs = []
    with open(outj) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("warm"):
                continue
            recs.append(r)
    if not recs:
        return None
    mse = sum(r["mse"] for r in recs) / len(recs)
    cb = sum(r["cb"] for r in recs) / len(recs)
    rec = dict(tag=tag, ids=ids.split(","), env=env, mse=mse, cb=cb,
               pred_grader=mse * (cb / 1.35), n=len(recs))
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def summary():
    rows = []
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    if not rows:
        print("no ladder results yet")
        return
    base = next((r for r in rows if r["tag"] == "base" and r["ids"] == ["0", "1"]), None)
    print(f"{'tag':<12} {'ids':<8} {'MSE':>10} {'C/B':>7} {'dMSE%':>7} {'dCB%':>7} {'pred-grad':>10}")
    for r in rows:
        dmse = dcb = float("nan")
        if base:
            dmse = (r["mse"] / base["mse"] - 1) * 100
            dcb = (r["cb"] / base["cb"] - 1) * 100
        print(f"{r['tag']:<12} {','.join(r['ids']):<8} {r['mse']:>10.4e} {r['cb']:>7.4f} "
              f"{dmse:>7.2f} {dcb:>7.2f} {r['pred_grader']:>10.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None, choices=["singles", "combos"])
    ap.add_argument("--ids", default="0,1")
    ap.add_argument("--only", default=None, help="comma list of tags to run")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
        return
    batches = {"singles": SINGLES, "combos": COMBOS}
    cfgs = batches.get(a.batch, {})
    if a.only:
        keep = set(a.only.split(","))
        pool = {**SINGLES, **COMBOS}
        cfgs = {k: v for k, v in pool.items() if k in keep}
    done = load_done()
    for tag, env in cfgs.items():
        if (tag, tuple(a.ids.split(","))) in done:
            print(f"--- {tag} done, skip", flush=True)
            continue
        run_cfg(tag, env, a.ids)
    summary()


if __name__ == "__main__":
    main()
