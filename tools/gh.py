"""Minimal GitHub client (stdlib only; no git binary on this machine).

The token is read from %APPDATA%\\whest-github\\token (never from the repo, never printed).
Pushes are made through the Git Data API: one blob per changed file, one tree, one commit,
then the branch ref is moved (a normal fast-forward commit on the branch).

usage:
  python gh.py whoami
  python gh.py create <repo> [--public]          (private by default)
  python gh.py push <owner/repo> <message> [--branch main]
      pushes the files listed by manifest() below (paths relative to E:\\Claude code\\whest)
"""
import base64
import fnmatch
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = r"E:\Claude code\whest"
API = "https://api.github.com"
TOKEN_FILE = os.path.join(os.environ.get("APPDATA", ""), "whest-github", "token")

# what goes into the repo (relative to ROOT); everything else stays local
INCLUDE = [
    "README.md", "LICENSE", "NOTICE.md", ".gitignore",
    "est/*.py",
    "sub/*/estimator.py",
    "submissions/*", "submissions/*/*",
    "tools/*.py", "tools/*.txt",
    "cs/*.py", "cs/stubs/whestbench/*.py",
    "runs/cfg_*.json", "runs/sweep.jsonl",
    "runs/audit_*.txt", "runs/errprof_*.txt", "runs/resid_alpha_*.txt",
    "docs/*.md",
]
EXCLUDE = ["est/*_audit.py", "tools/q1.py", "tools/wts.py"]
SECRET_RE = re.compile("gh" + r"[pousr]_[A-Za-z0-9]{30,}")   # GitHub token shapes


def _secrets():
    """The live secrets themselves (read at run time, never stored here): refuse to push any file
    that contains the GitHub token or the AIcrowd API key."""
    out = [_token()]
    try:
        import tomllib
        p = os.path.join(os.environ.get("APPDATA", ""), "aicrowd-cli", "config.toml")
        cfg = tomllib.loads(open(p, encoding="utf-8").read())

        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, str) and len(o) >= 32:
                out.append(o)
        walk(cfg)
    except Exception:  # noqa: BLE001
        pass
    return [s for s in out if s]


def _token():
    with open(TOKEN_FILE, encoding="ascii") as f:
        return f.read().strip()


def req(method, path, body=None, ok=(200, 201)):
    url = path if path.startswith("http") else API + path
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", "Bearer " + _token())
    r.add_header("Accept", "application/vnd.github+json")
    r.add_header("X-GitHub-Api-Version", "2022-11-28")
    r.add_header("User-Agent", "whest-tools")
    if data is not None:
        r.add_header("Content-Type", "application/json")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
                txt = resp.read().decode()
                return resp.status, (json.loads(txt) if txt else None)
        except urllib.error.HTTPError as e:
            txt = e.read().decode(errors="replace")
            if e.code in (502, 503, 504) and attempt < 3:
                time.sleep(2 + 3 * attempt)
                continue
            if e.code in ok:
                return e.code, (json.loads(txt) if txt else None)
            return e.code, {"error": txt[:2000]}
        except urllib.error.URLError:
            if attempt < 3:
                time.sleep(2 + 3 * attempt)
                continue
            raise


def whoami():
    st, j = req("GET", "/user")
    if st != 200:
        raise SystemExit(f"auth failed: {st} {j}")
    return j


def create(name, private=True, description=""):
    st, j = req("POST", "/user/repos", {"name": name, "private": private, "description": description,
                                        "auto_init": True})
    if st == 201:
        return j
    if st == 422 and "already exists" in json.dumps(j):
        me = whoami()["login"]
        st2, j2 = req("GET", f"/repos/{me}/{name}")
        return j2
    raise SystemExit(f"create failed: {st} {j}")


def manifest():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = ""
        # prune heavy / third-party trees early
        dirnames[:] = [d for d in dirnames if (rel_dir + "/" + d).lstrip("/") not in
                       ("kit", "dl", "data", "ds_local", "runs/pred", "cs/srv", "cs/cli", "research")
                       and not d.startswith(".") and d != "__pycache__"]
        for fn in filenames:
            rel = (rel_dir + "/" + fn).lstrip("/")
            if any(fnmatch.fnmatch(rel, p) for p in INCLUDE) and not any(fnmatch.fnmatch(rel, p) for p in EXCLUDE):
                out.append(rel)
    return sorted(out)


def _git_blob_sha(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def push(full, message, branch="main"):
    files = manifest()
    st, ref = req("GET", f"/repos/{full}/git/ref/heads/{branch}")
    if st != 200:
        raise SystemExit(f"no branch {branch}: {st} {ref}")
    head = ref["object"]["sha"]
    st, commit = req("GET", f"/repos/{full}/git/commits/{head}")
    base_tree = commit["tree"]["sha"]
    st, tree = req("GET", f"/repos/{full}/git/trees/{base_tree}?recursive=1")
    remote = {e["path"]: e["sha"] for e in tree.get("tree", []) if e["type"] == "blob"}
    entries = []
    secrets = _secrets()
    for rel in files:
        data = open(os.path.join(ROOT, rel.replace("/", os.sep)), "rb").read()
        low = data.decode("utf-8", errors="ignore")
        if SECRET_RE.search(low) or any(s in low for s in secrets):
            raise SystemExit(f"refusing to push {rel}: contains a secret")
        if remote.get(rel) == _git_blob_sha(data):
            continue
        st, b = req("POST", f"/repos/{full}/git/blobs",
                    {"content": base64.b64encode(data).decode(), "encoding": "base64"})
        if st != 201:
            raise SystemExit(f"blob failed for {rel}: {st} {b}")
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": b["sha"]})
    if not entries:
        print("nothing changed")
        return head
    st, t = req("POST", f"/repos/{full}/git/trees", {"base_tree": base_tree, "tree": entries})
    if st != 201:
        raise SystemExit(f"tree failed: {st} {t}")
    st, c = req("POST", f"/repos/{full}/git/commits", {"message": message, "tree": t["sha"], "parents": [head]})
    if st != 201:
        raise SystemExit(f"commit failed: {st} {c}")
    st, r = req("PATCH", f"/repos/{full}/git/refs/heads/{branch}", {"sha": c["sha"], "force": False})
    if st != 200:
        raise SystemExit(f"ref update failed: {st} {r}")
    print(f"pushed {len(entries)} file(s) -> {full}@{branch} {c['sha'][:10]}")
    return c["sha"]


def main():
    cmd = sys.argv[1]
    if cmd == "whoami":
        j = whoami()
        print(j["login"], j.get("name"), "public_repos", j.get("public_repos"), "private", j.get("total_private_repos"))
    elif cmd == "create":
        j = create(sys.argv[2], private="--public" not in sys.argv,
                   description="ARC White-Box Estimation Challenge 2026 (AIcrowd) - estimators, tools, submissions")
        print(j["full_name"], "private" if j["private"] else "PUBLIC", j["html_url"])
    elif cmd == "push":
        br = sys.argv[sys.argv.index("--branch") + 1] if "--branch" in sys.argv else "main"
        push(sys.argv[2], sys.argv[3], br)
    elif cmd == "manifest":
        for f in manifest():
            print(f)


if __name__ == "__main__":
    main()
