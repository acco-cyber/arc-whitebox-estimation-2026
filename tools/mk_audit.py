"""Make an audit copy of an estimator: insert _tag("family") calls before anchor lines.
usage: python mk_audit.py <src.py> <dst.py>
The tags only set the flopscope namespace of the ops that follow (tools/audit.py installs
the hook); the op stream is unchanged.
"""
import sys

HOOK = '''
_AUD = [None]   # audit hook: the active budget's namespace stack (tools/audit.py)


def _tag(name):
    s = _AUD[0]
    if s is not None:
        s[:] = [name]

'''

# (anchor substring, tag, occurrence index or None for all)
ANCHORS = [
    ("if prune and li >= LR0:\n                # relabel", "relabel"),
    ("            C_pre = None\n", "linear"),
    ("                WD = fnp.multiply(W, (w1_prev)[None, :], out=NN(\"wd\"))", "join"),
    ("                if ka > kb:\n                    # formed dense legs", "form_old"),
    ("                extra = 2 if newborn is not None else 0", "young_tr"),
    ("                z_side ^= 1", "thin_tr"),
    ("            if newborn is not None and not skip_src:\n                a_b, Rr, Lr, s_b, e_b, Ff = newborn", "cpre"),
    ("            # ---- WK slices ----", "dsl"),
    ("                if regen:\n                    # F68: exact transported diagonal", "regen"),
    ("            # ---- wick matrix ----", "wick"),
    ("            # ---- nonlin terms (fused) ----", "nonlin"),
    ("            # ---- wick old blocks + dslice scalings ----", "scal"),
    ("            # ---- new source (V1.6, struct-free Y1 = a_b * D(w2)) ----", "birth"),
    ("            # ---- pK -> K (per-entry", "pk2k"),
    ("            # ---- assemble ----", "resleg"),
    ("        k = len(w2b_list)\n        W2B = fnp.stack(w2b_list, axis=0)[:, None, :]", "dsl_elem"),
    ("        if rfb > 0:\n            # V18 (F69): B1 pair with X1 = 3A + Xt", "fb"),
    ("        C1 = fnp.stack(c1_list, axis=0)                 # (k, n) static column scalings", "feed"),
    ("        if ka > 0:\n            # V21: old sources through the shared basis", "hub_old"),
    ("        PPL = fnp.matmul(PP, L_st, out=bufs[\"ppl\"][:k])", "thin_m"),
    ("        out = bufs[\"hub\"]\n        smm = self._smm", "hub_young"),
    ("            def lift_qc(dst):", "lift"),
]


def main():
    src = open(sys.argv[1], encoding="utf-8").read()
    i = src.index("class _VH:") if "class _VH:" in src else src.index("class _Strassen:")
    out = src[:i] + HOOK.lstrip("\n") + "\n\n" + src[i:]
    for anchor, tag in ANCHORS:
        n = out.count(anchor)
        if n != 1:
            print(f"WARNING: anchor for {tag!r} found {n} times")
            if n == 0:
                continue
        j = out.index(anchor)
        # indentation of the anchor's first line
        line_start = out.rfind("\n", 0, j) + 1
        first = out[line_start:out.index("\n", j)]
        indent = first[:len(first) - len(first.lstrip())]
        if j != line_start:
            indent = out[line_start:j] if out[line_start:j].strip() == "" else indent
        out = out[:line_start] + f"{indent}_tag(\"{tag}\")\n" + out[line_start:]
    # lift tag: the def line would be tagged at definition time; retag at call time instead
    out = out.replace('                smm.smin = SMIN_SB\n                lev_l = ', '                _tag("lift")\n                smm.smin = SMIN_SB\n                lev_l = ')
    open(sys.argv[2], "w", encoding="utf-8", newline="\n").write(out)
    import ast
    ast.parse(out)
    print("written", sys.argv[2], "tags:", out.count("_tag(\""))


if __name__ == "__main__":
    main()
