import io
src = open(r"E:\Claude code\whest\est\v36.py", encoding="utf-8").read()
new = open(r"E:\Claude code\whest\tools\strassen_v37.txt", encoding="utf-8").read()
i = src.index("class _Strassen:")
j = src.index("class _Pool:")
out = src[:i] + new + src[j:]
open(r"E:\Claude code\whest\est\v37.py", "w", encoding="utf-8", newline="\n").write(out)
print("v36 chars", len(src), "v37 chars", len(out))
import re
for pat in ("_buf2\\(", "_bufv\\(", "\\._buf\\(", "_assemble\\(", "smm\\.clear", "\\.views", "smm\\.hub\\(", "smm\\.mm\\("):
    print(pat, [m.start() for m in re.finditer(pat, out)][:20].__len__())
