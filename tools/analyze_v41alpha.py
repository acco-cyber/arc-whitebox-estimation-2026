"""V41 alpha-sketch analysis: MSE/C/B table (ON vs OFF) + relabel-ordering movement.

usage: python analyze_v41alpha.py
reads runs/v41alpha_off.jsonl, runs/v41alpha_on.jsonl and the aapx dumps in
runs/v41dbg/{off,on}/aapx_<name>.npz (rows = relabeled layers 3..15, LR0=3).
"""
import json
import os

import numpy as np

ROOT = "/home/z/my-project/whest"
LR0 = 3


def load(tag):
    p = os.path.join(ROOT, f"runs/v41alpha_{tag}.jsonl")
    recs = [json.loads(l) for l in open(p) if l.strip()]
    return {r["mlp"]: r for r in recs if not r.get("warm")}


def main():
    off, on = load("off"), load("on")
    ids = sorted(set(off) & set(on))
    print("== MSE / C/B (ids", ids, ") ==")
    print(f"{'mlp':>5} {'C/B off':>9} {'C/B on':>9} {'dC/B':>8} | "
          f"{'mse off':>10} {'mse on':>10} {'dmse':>8}")
    for i in ids:
        cb_d = on[i]["cb"] - off[i]["cb"]
        m_d = (on[i]["mse"] - off[i]["mse"]) / off[i]["mse"]
        print(f"{i:5d} {off[i]['cb']:9.4f} {on[i]['cb']:9.4f} {cb_d:+8.4f} | "
              f"{off[i]['mse']:10.4e} {on[i]['mse']:10.4e} {m_d:+8.2%}")
    cb_off = np.mean([off[i]["cb"] for i in ids])
    cb_on = np.mean([on[i]["cb"] for i in ids])
    m_off = np.mean([off[i]["mse"] for i in ids])
    m_on = np.mean([on[i]["mse"] for i in ids])
    print(f"MEAN  {cb_off:9.4f} {cb_on:9.4f} {cb_on-cb_off:+8.4f} | "
          f"{m_off:10.4e} {m_on:10.4e} {(m_on-m_off)/m_off:+8.2%}   "
          f"(score off {m_off*max(0.1,cb_off):.4e} on {m_on*max(0.1,cb_on):.4e})")

    # ordering movement
    dbg = os.path.join(ROOT, "runs/v41dbg")
    names = sorted(os.listdir(os.path.join(dbg, "off")))
    print("\n== relabel ordering movement (a_apx ON vs OFF; units ranked descending) ==")
    print(f"{'layer':>5} " + " ".join(f"{'mlp'+str(i):>12}" for i in ids)
          + "   (>100 = material reordering)")
    agg = []
    for li in range(LR0, 16):
        fracs = []
        for i in ids:
            nm = [f for f in names if f.endswith(f"{i}.npz") or True]  # names are slugs
        # match by identical filename between on/off
        row = []
        for f in sorted(os.listdir(os.path.join(dbg, "off"))):
            po = os.path.join(dbg, "off", f)
            pn = os.path.join(dbg, "on", f)
            if not os.path.exists(pn):
                continue
            a_off = np.load(po)["aapx"]
            a_on = np.load(pn)["aapx"]
            if a_off.shape[0] != 13 or a_on.shape[0] != 13:
                continue
            ro = (-a_off[li - LR0]).argsort().argsort()
            rn = (-a_on[li - LR0]).argsort().argsort()
            row.append(float((np.abs(ro - rn) > 100).mean()))
        fracs = np.mean(row) if row else [float('nan')]
        agg.append(fracs)
        print(f"{li:5d} " + " ".join(f"{x:12.2%}" for x in np.atleast_1d(fracs)))
    agg = np.array(agg)
    print(f"MEAN over relabeled layers: {agg.mean():.2%}   "
          f"max layer: {agg.max():.2%} (layer {LR0 + int(agg.argmax())})")


if __name__ == "__main__":
    main()
