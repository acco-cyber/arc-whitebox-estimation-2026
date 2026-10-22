import pypdf, sys
src, dst = sys.argv[1], sys.argv[2]
r = pypdf.PdfReader(src)
out = []
for i, p in enumerate(r.pages):
    out.append(f"\n\n===== PAGE {i+1} =====\n" + (p.extract_text() or ""))
open(dst, "w", encoding="utf-8").write("".join(out))
print("pages", len(r.pages), "chars", sum(len(x) for x in out))
