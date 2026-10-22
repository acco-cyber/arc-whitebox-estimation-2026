"""Build a clean submission file from an estimator version: bake schedule defaults, prepend the
docstring paragraphs, check for anything a grader could reject.
usage: python mk_sub.py <src.py> <dst.py> [KEY=VALUE ...]   (KEY = env knob whose default is baked)
"""
import ast
import re
import sys

S1 = r"E:\Claude code\whest\sub\s1\estimator.py"

V37_DOC = """V37 (2026-10-02): (i) the join of a source into the shared basis (range finder, projections,
rotation of the old factors, basis transport W Q, lift inner Q^T) runs as Strassen families with
a rectangular leaf minimum (the shared-basis forming / contraction families too), (ii) the
sub-views of the pooled Strassen scratch buffers are made once and cached (a slice is a call),
(iii) the final layer, which only needs D3, keeps a K3 row band (the 128 most saturated and the
128 most dead units of the relabeling carry no read-out there), (iv) the thin legs are
transported on the live block only. Values match V30 within the range-finder noise;
cost 0.232 -> ~0.207 x B.
"""

BANNED = ["mmap", "tempfile", "atexit", "subprocess", "socket", "ctypes", "threading",
          "multiprocessing", "import numpy", "from numpy", "pickle", "open(", "__import__", "exec(", "eval("]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    kv = dict(a.split("=", 1) for a in sys.argv[3:])
    s = open(src, encoding="utf-8").read()
    for k, v in kv.items():
        pat = re.compile(r'(_os\.environ\.get\("' + re.escape(k) + r'",\s*)"[^"]*"')
        s, n = pat.subn(lambda m: m.group(1) + '"' + v + '"', s)
        if n == 0:
            pat2 = re.compile(r'(_sched\("' + re.escape(k) + r'",\s*)"[^"]*"')
            s, n = pat2.subn(lambda m: m.group(1) + '"' + v + '"', s)
        assert n == 1, (k, n)
    # docstring: take s1's V30 paragraph block and add V37
    d1 = open(S1, encoding="utf-8").read()
    head1 = d1[:d1.index("Below this line the V29 docstring follows unchanged.")]
    assert s.startswith('"""')
    i = s.index("\n\n")  # end of the version-title block of the source docstring
    rest = s[i + 2:]
    head = head1.replace("+ DEAD-RELU PRUNING OF THE K3 LEG MACHINERY (V30, 2026-10-02).",
                         "+ DEAD-RELU PRUNING OF THE K3 LEG MACHINERY (V30, 2026-10-02)\n"
                         "+ STRASSEN JOINS, CACHED VIEWS, FINAL-LAYER BAND (V37, 2026-10-02).")
    s = head + V37_DOC + "Below this line the V29 docstring follows unchanged.\n\n" + rest
    ast.parse(s)
    code_only = re.sub(r'"""[\s\S]*?"""', "", s)
    code_only = re.sub(r"#.*", "", code_only)
    hits = [b for b in BANNED if b in code_only]
    assert not hits, hits
    imports = sorted(set(re.findall(r"^\s*(?:import|from)\s+([\w\.]+)", s, flags=re.M)))
    open(dst, "w", encoding="utf-8", newline="\n").write(s)
    print("written", dst, len(s), "chars; imports:", imports)


if __name__ == "__main__":
    main()
