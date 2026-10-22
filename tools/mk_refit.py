"""R-2a: write est/e30a_refit.py = byte-copy of e30a_lp.py with ONLY the
CORR_BETA_NEW rows replaced (+ one header comment line above the table).

usage: kitvenv/bin/python tools/mk_refit.py --base est/e30a_lp.py \
           --table /home/z/my-project/scratch_refit/tab_T2.npz \
           --out est/e30a_refit.py --fits "0-13" [--comment extra text]
"""
import argparse
import os

import numpy as np


def fmt_rows(betas):
    lines = []
    for row in betas:
        vals = ", ".join(repr(float(v)) for v in row)
        lines.append("    [" + vals + "],")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fits", required=True, help="description of the fit nets for the header")
    ap.add_argument("--comment", default="")
    a = ap.parse_args()

    t = np.load(a.table)
    betas = t["betas"].astype(np.float64)
    assert betas.shape == (16, 87), betas.shape

    src = open(a.base, encoding="utf-8").read().splitlines(keepends=True)
    start = next(i for i, ln in enumerate(src) if ln.startswith("CORR_BETA_NEW = ["))
    end = start
    while src[end].rstrip("\n") != "]":
        end += 1
    nrows = end - start - 1
    assert nrows == 16, f"expected 16 rows, found {nrows} (lines {start + 2}..{end})"

    hdr = ("# [R-2a refit 2026-10-18] per-chain ridge refit on local 1e9-truth nets "
           "(koushik_rudra); base e30_a by LordProcciuto (MIT); fits: " + a.fits)
    if a.comment:
        hdr += "; " + a.comment
    out = src[:start] + [hdr + "\n", "CORR_BETA_NEW = [\n"] + \
        [ln + "\n" for ln in fmt_rows(betas)] + src[end + 1:]

    open(a.out, "w", encoding="utf-8").write("".join(out))
    compile(open(a.out, encoding="utf-8").read(), a.out, "exec")

    # verify: the only diff vs base is the replaced block + header line
    import subprocess
    r = subprocess.run(["diff", a.base, a.out], capture_output=True, text=True)
    changed = [ln for ln in r.stdout.splitlines() if ln.startswith(("<", ">"))]
    n_base_rows = sum(1 for ln in changed if ln.startswith("<") and ln[2:].startswith("    ["))
    n_new_rows = sum(1 for ln in changed if ln.startswith(">") and ln[2:].startswith("    ["))
    print(f"wrote {a.out}: diff lines={len(changed)} base_rows={n_base_rows} new_rows={n_new_rows}")
    assert n_new_rows == 16 and n_base_rows == 16, "unexpected diff shape"
    print("diff OK (only table rows + header differ)")


if __name__ == "__main__":
    main()
