import re, html, sys
src = open(r"E:\Claude code\whest\research\leaderboard.html", encoding="utf-8").read()
rows = re.findall(r"<tr[^>]*>(.*?)</tr>", src, flags=re.S)
out = []
for r in rows:
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S)
    cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip() for c in cells]
    if cells:
        out.append(cells)
print(len(out), "rows")
for c in out[:130]:
    print(" | ".join(x[:40] for x in c).encode("ascii", "replace").decode())
