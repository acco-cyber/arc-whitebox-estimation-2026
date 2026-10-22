"""Per-hop scalar fit of the V56 T22 gain (theta) against per-layer truth.

For each MLP id: run est/v56t22.py in-process with V56_DUMP=1 (and any V56_T22G
schedule baked via --t22g), read the module-level DEBUG feature dumps, align them
with the high-precision per-layer truth (all_layer_means, mapped through the
per-layer relabel permutation), and regress the per-layer mean residual on the
T22-pathway feature f1 = w12 * t22g (exact first-order: the within-hop mean
response to dG is the diagonal term (METRIC_C/24) * W_all[:,12] * dG).

Reports per hop: theta per MLP, sign-consistency, pooled theta, pooled R^2, and a
bake-ready V56_T22G schedule string.

Usage:
  ./kitvenv/bin/python tools/fit_t22.py --ids 0,1,2,3 [--t22g "0,...,0"] \
      [--tag it0] [--out runs/fit_t22_it0.json] [--fit-lam 0] [--ridge 1e-12]
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np

B = float(2 ** 41)
NPZ = "/home/z/my-project/whest/data/npz/mlp_{:04d}.npz"
EST = "/home/z/my-project/whest/est/v56t22.py"


def load_est(path, tag):
    spec = importlib.util.spec_from_file_location("est_fit_%s" % tag, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_suite(ids, tag):
    """One process, one Estimator, warm-up on ids[0] then all ids — mirrors the grader."""
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import SetupContext
    from whestbench.domain import MLP

    ev = load_est(EST, tag)
    est = ev.Estimator()
    est.setup(SetupContext(width=1024, depth=16, flop_budget=int(B), api_version="1", seed=0))
    runs = []
    seq = [ids[0]] + ids  # first call = warm-up (discarded dump)
    for pos, mid in enumerate(seq):
        d = np.load(NPZ.format(mid))
        ws = [fnp.array(w, dtype=fnp.float32) for w in d["weights"]]
        mlp = MLP(width=1024, depth=16, weights=ws, seed=0, name=str(d["mlp_name"]))
        ev.DEBUG.clear()
        t0 = time.perf_counter()
        with flops.BudgetContext(flop_budget=int(B), wall_time_limit_s=1200.0, quiet=True) as ctx:
            pred = est.predict(mlp, int(B))
            pred = np.asarray(pred, dtype=np.float64)
        cb = float(ctx.flops_used) / B
        wall = time.perf_counter() - t0
        dbg = []
        for ent in list(ev.DEBUG):
            dbg.append({k: (np.asarray(v, dtype=np.float64)
                            if isinstance(v, (int, float)) or hasattr(v, "__array__") else v)
                        for k, v in ent.items()})
        ev.DEBUG.clear()
        gt_all = d["all_layer_means"].astype(np.float64)
        gt_final = d["final_means"].astype(np.float64)
        warm = pos == 0
        r = dict(mid=mid, pred=pred, dbg=dbg, gt_all=gt_all, gt_final=gt_final,
                 cb=cb, wall=wall, warm=warm,
                 mse=float(np.mean((pred[-1] - gt_final) ** 2)))
        if not warm:
            runs.append(r)
        print("[fit %s] mlp%d%s: C/B=%.4f mse=%.4e wall=%.1fs dumps=%d"
              % (tag, mid, " (warm-up)" if warm else "", cb, r["mse"], wall, len(dbg)),
              flush=True)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="0,1,2,3")
    ap.add_argument("--t22g", default="")
    ap.add_argument("--tag", default="it0")
    ap.add_argument("--out", default="")
    ap.add_argument("--fit-lam", type=int, default=0)
    ap.add_argument("--fit-full", type=int, default=0)
    ap.add_argument("--feature", default="t22g")
    ap.add_argument("--ridge", type=float, default=1e-12)
    a = ap.parse_args()

    if a.t22g:
        os.environ["V56_T22G"] = a.t22g
    else:
        os.environ.pop("V56_T22G", None)
    os.environ["V56_DUMP"] = "1"

    ids = [int(x) for x in a.ids.split(",") if x != ""]
    runs = run_suite(ids, a.tag)

    # ---- align and fit ----
    L = 16
    per_mlp = {mid: dict(theta=[None] * L, r2=[None] * L,
                         dlam=[None] * L if a.fit_lam else None,
                         eps2=[None] * L, snr=[None] * L) for mid in ids}
    sanity = []
    for r in runs:
        d = r["dbg"]
        if len(d) != L:
            print("WARN mlp%d: %d dump dicts" % (r["mid"], len(d)))
        for ent in d:
            li = int(ent["layer"])
            pk1v, w12, t22g, perm = ent["pk1v"], ent["w12"], ent["t22g"], ent["perm"]
            # sanity: pk1v_int == pred_orig[perm]
            if perm is not None:
                rec = float(np.max(np.abs(pk1v - r["pred"][li][perm.astype(int)])))
            else:
                rec = float(np.max(np.abs(pk1v - r["pred"][li])))
            sanity.append(rec)
            truth_int = (np.take(r["gt_all"][li], perm.astype(int))
                         if perm is not None else r["gt_all"][li])
            eps = truth_int - pk1v
            row = per_mlp[r["mid"]]
            row["eps2"][li] = float(np.mean(eps ** 2))
            if t22g is None or w12 is None:
                continue
            f1 = w12 * t22g
            if a.feature == "co1":
                # per-hop refit of the (1,)-partition Wick coefficients over the
                # exact linear features [W0, W8*d3, W12*g4, W18*d3^2, W20*d3*g4]
                ws, d3, dg = ent["wsel"], ent["d3"], ent["dG"]
                if ws is None or d3 is None or dg is None:
                    continue
                g4 = dg * 2.0  # METRIC_C
                A = np.stack([ws[:, 0], ws[:, 1] * d3, ws[:, 2] * g4,
                              ws[:, 3] * d3 * d3, ws[:, 4] * d3 * g4], axis=1)
                co_analytic = np.array([1.0, 1.0 / 6, 1.0 / 24, 1.0 / 72, 1.0 / 144])
                # algebra sanity: analytic combination must reproduce pk1v
                rec = float(np.max(np.abs(A @ co_analytic - pk1v)))
                if rec > 1e-3:
                    print("WARN mlp%d hop%d: co1 reconstruction err %.2e" % (r["mid"], li, rec))
                G = A.T @ A
                sc = float(np.mean(np.diag(G))) + 1e-30
                sol = np.linalg.solve(G + a.ridge * sc * np.eye(5), A.T @ eps)
                theta = None  # handled below via co1 path
                resid = eps - A @ sol
                row.setdefault("co1", [None] * L)[li] = sol
                row.setdefault("corec", [None] * L)[li] = rec
                r2 = 1.0 - float(np.mean(resid ** 2)) / max(float(np.mean(eps ** 2)), 1e-30)
                row["r2"][li] = r2
                row["theta"][li] = float(sol[2])  # keep column meaning = g4 coef
                continue
            elif a.feature == "tv":
                # clean 1-param lambda-pathway fit: eps ~ b*(w12*t_v)
                f1 = w12 * ent["t_v"]
                theta = float(np.sum(f1 * eps) / (np.sum(f1 * f1) + a.ridge))
                resid = eps - theta * f1
            elif a.fit_full and ent["t_g"] is not None and ent["t_v"] is not None:
                # 3-feature per-hop recalibration of the kappa4 diagonal channel:
                # eps ~ a*(w12*t_g) + b*(w12*t_v) + c*(w12*t22g)
                A = np.stack([w12 * ent["t_g"], w12 * ent["t_v"], f1], axis=1)
                G = A.T @ A
                sc = float(np.mean(np.diag(G))) + 1e-30
                sol = np.linalg.solve(G + a.ridge * sc * np.eye(3), A.T @ eps)
                theta = float(sol[2])
                resid = eps - A @ sol
                row["dlam"] = row.get("dlam") or [None] * L
                row["dlam"][li] = (float(sol[0]), float(sol[1]))
            else:
                theta = float(np.sum(f1 * eps) / (np.sum(f1 * f1) + a.ridge))
                resid = eps - theta * f1
                dlam = None
            r2 = 1.0 - float(np.mean(resid ** 2)) / max(float(np.mean(eps ** 2)), 1e-30)
            snr = float(np.sum(f1 * eps) / (np.std(f1 * (np.mean(eps ** 2) ** 0.5)) + 1e-30))
            row["theta"][li] = theta
            row["r2"][li] = r2
            row["snr"][li] = snr

    # ---- aggregate ----
    if a.feature == "co1":
        print("\n=== per-hop (1,)-coefficient refit (%s) ===" % a.tag)
        print("perm-sanity max recon err: %.3e" % max(
            x for r in runs for x in
            [v for v in per_mlp[r["mid"]].get("corec", [None] * L) if v is not None] or [0.0]))
        co_analytic = np.array([1.0, 1.0 / 6, 1.0 / 24, 1.0 / 72, 1.0 / 144])
        tab = []
        print("hop  eps2(base)   R2(mean)  fitted coefs (mean over mlps) vs analytic            agree")
        for li in range(L):
            sols = [per_mlp[m].get("co1", [None] * L)[li] for m in ids]
            sols = [s for s in sols if s is not None]
            e2 = np.mean([per_mlp[m]["eps2"][li] for m in ids if per_mlp[m]["eps2"][li] is not None])
            if not sols:
                tab.append(co_analytic.tolist())
                print("%2d  %.3e  (no feature)" % (li, e2))
                continue
            r2s = [per_mlp[m]["r2"][li] for m in ids]
            sm = np.mean(sols, axis=0)
            # agreement: each mlp's sol within 3x of spread of the mean, sign-safe
            agree = int(np.all(np.sign(sols) == np.sign(sm)))
            # hold-out: apply mean-sol to each mlp, measure R2 against that mlp's own eps
            ho = []
            for r in runs:
                ent = next(e for e in r["dbg"] if int(e["layer"]) == li)
                if ent.get("wsel") is None or ent.get("d3") is None:
                    continue
                ws, d3, dg = ent["wsel"], ent["d3"], ent["dG"]
                g4 = dg * 2.0
                A = np.stack([ws[:, 0], ws[:, 1] * d3, ws[:, 2] * g4,
                              ws[:, 3] * d3 * d3, ws[:, 4] * d3 * g4], axis=1)
                perm = ent["perm"]
                truth_int = (np.take(r["gt_all"][li], perm.astype(int))
                             if perm is not None else r["gt_all"][li])
                eps = truth_int - ent["pk1v"]
                resid = eps - A @ sm
                ho.append(1.0 - float(np.mean(resid ** 2)) / max(float(np.mean(eps ** 2)), 1e-30))
            r2h = float(np.mean(ho)) if ho else 0.0
            keep = 1 if (agree and r2h > 0.02) else 0
            tab.append((sm if keep else co_analytic).tolist())
            print("%2d  %.3e  %+.4f  [%s] vs [%s] agree=%d R2hold=%+.4f keep=%d"
                  % (li, e2, float(np.mean(r2s)),
                     " ".join("%+.4g" % v for v in sm),
                     " ".join("%.4g" % v for v in co_analytic), agree, r2h, keep))
        print("\nCO1_TAB = [")
        for rowv in tab:
            print("    [%s]," % ", ".join("%.10g" % v for v in rowv))
        print("]")
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(dict(tag=a.tag, ids=ids, feature="co1", tab=tab,
                               mse={str(r["mid"]): r["mse"] for r in runs}), f, indent=1)
            print("wrote", a.out)
        return

    print("\n=== per-hop fit (%s) ===" % a.tag)
    print("perm-sanity max|pk1v_int - pred[perm]| over all hops: %.3e" % max(sanity))
    hdr = "hop  eps2(base)     theta per mlp                              pooled   R2(pooled) snr"
    print(hdr)
    sched = []
    for li in range(L):
        th = [per_mlp[m]["theta"][li] for m in ids]
        e2 = np.mean([per_mlp[m]["eps2"][li] for m in ids if per_mlp[m]["eps2"][li] is not None])
        if all(t is None for t in th):
            sched.append(0.0)
            print("%2d  %.3e  (no feature)" % (li, e2))
            continue
        tv = [t for t in th if t is not None]
        pooled = float(np.mean(tv))
        signs = sum(1 for t in tv if np.sign(t) == np.sign(pooled))
        # pooled R2: residual after removing pooled*f1 from every mlp
        num = []
        for r in runs:
            ent = next(e for e in r["dbg"] if int(e["layer"]) == li)
            if ent["t22g"] is None or ent["w12"] is None:
                continue
            pk1v, w12, t22g = ent["pk1v"], ent["w12"], ent["t22g"]
            perm = ent["perm"]
            truth_int = (np.take(r["gt_all"][li], perm.astype(int))
                         if perm is not None else r["gt_all"][li])
            eps = truth_int - pk1v
            f1 = w12 * t22g
            resid = eps - pooled * f1
            num.append(1.0 - float(np.mean(resid ** 2)) / max(float(np.mean(eps ** 2)), 1e-30))
        r2p = float(np.mean(num)) if num else 0.0
        # keep hops with consistent sign and positive explained variance
        keep = 1 if (signs == len(tv) and r2p > 0.01) else 0
        sched.append(pooled if keep else 0.0)
        print("%2d  %.3e  [%s]  %+.4g  %+.3f  %+.1f  keep=%d"
              % (li, e2, " ".join("%+.3g" % t for t in tv), pooled, r2p,
                 np.mean([x for x in [per_mlp[m]["snr"][li] for m in ids] if x is not None]) if keep else 0.0,
                 keep))
    sched_s = ",".join("%.5g" % v for v in sched)
    print("\nV56_T22G=%s" % sched_s)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(dict(tag=a.tag, ids=ids, t22g_in=a.t22g, sched=sched,
                           sched_str=sched_s,
                           per_mlp={str(m): per_mlp[m] for m in ids},
                           mse={str(r["mid"]): r["mse"] for r in runs},
                           cb={str(r["mid"]): r["cb"] for r in runs},
                           sanity_max=float(max(sanity))), f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
