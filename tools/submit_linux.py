"""Linux port of tools/submit.py — package -> validate -> submit -> watch -> record -> push.

usage (from anywhere):
  python3 submit_linux.py <estimator.py> <tag> "<description>" [--dry-run] [--no-push]
  python3 submit_linux.py --record <id> <tag> <estimator.py|-> "<description>"
  python3 submit_linux.py --push "<commit message>"

Chain (same as the Windows submit.py): whest validate -> package -> validate-package ->
submit (--format plain) -> parse "submission id N" -> poll grading via the whestbench
AIcrowd client (90 min, 60 s) -> record into submissions/<id>_<tag>/ (estimator.py copied
byte-for-byte + meta.json), append submissions/log.jsonl, regenerate submissions/README.md
(record() / regen_readme() are byte-identical in logic to tools/submit.py) -> push the repo
snapshot to GitHub.

Push note: this tree (/home/z/my-project/whest) is NOT its own git repository (it sits
untracked inside the outer workspace repo), so a local `git add/commit/push` would operate
on the wrong repo. Therefore the push uses the Git-Data-API client ported from tools/gh.py:
one blob per changed file, one tree, one commit, then the branch ref is moved (a normal
fast-forward on main). The GitHub token is read at runtime from $GITHUB_TOKEN (fallback:
~/.config/whest-github/token) and is never printed or stored; the AIcrowd key comes from
$AICROWD_API_KEY or whest's own config. No token literal appears in this file.
"""
import base64
import fnmatch
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = ROOT
WHEST = os.path.join(KIT, "kitvenv", "bin", "whest")
PY = os.path.join(KIT, "kitvenv", "bin", "python")
SUBS = os.path.join(ROOT, "submissions")
REPO = "acco-cyber/arc-whitebox-estimation-2026"
API = "https://api.github.com"
BRANCH = "main"
POLL_TIMEOUT = 90 * 60      # 90 min grading watch
POLL_EVERY = 60             # seconds

# ---------------------------------------------------------------------------
# whest CLI chain
# ---------------------------------------------------------------------------

def run(cmd, timeout=None, cwd=KIT):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def _aick():
    """AICROWD key at runtime: env first, then whest's own config (Linux / Windows paths)."""
    k = os.environ.get("AICROWD_API_KEY")
    if k:
        return k
    for p in (pathlib.Path.home() / ".config" / "aicrowd-cli" / "config.toml",
              pathlib.Path(os.environ.get("APPDATA", "")) / "aicrowd-cli" / "config.toml"):
        try:
            import tomllib
            cfg = tomllib.loads(p.read_text(encoding="utf-8"))

            def walk(o):
                vals = o.values() if isinstance(o, dict) else []
                for v in vals:
                    if isinstance(v, str) and len(v) >= 40:
                        return v
                    if isinstance(v, dict):
                        f = walk(v)
                        if f:
                            return f
            f = walk(cfg)
            if f:
                return f
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit("no AIcrowd key: set AICROWD_API_KEY or run `whest login`")


def _client():
    """whestbench client; re-exec under the kit venv if the current python lacks it."""
    try:
        from whestbench.aicrowd_client import AIcrowdClient  # noqa: F401
    except ImportError:
        os.execv(PY, [PY, os.path.abspath(__file__)] + sys.argv[1:])
    from whestbench.aicrowd_client import AIcrowdClient
    return AIcrowdClient(api_key=_aick())


def status(sid):
    return _client().get_submission_status(int(sid))


def build_chain(est, out_tar):
    """validate -> package -> validate-package. Returns nothing; exits on failure."""
    rc, out = run([WHEST, "validate", "--estimator", est], timeout=600)
    print(out[-1500:].strip())
    if rc != 0:
        raise SystemExit("whest validate failed")
    rc, out = run([WHEST, "package", "--estimator", est, "--output", out_tar], timeout=600)
    if rc != 0:
        print(out)
        raise SystemExit("package failed")
    rc, out = run([WHEST, "validate-package", out_tar], timeout=600)
    print(out.strip()[-500:])
    if rc != 0:
        raise SystemExit("validate-package failed")


def submit_dry_run(est, out_tar, desc):
    """No-upload preview. Tries `submit <tar> --dry-run`; whestbench 0.16.1 wants
    --estimator for dry-run (artifact+--dry-run just warns, exit 2), so fall back.
    Verified in whestbench 0.16.1 cli.py: dry-run packages to a temp dir and never
    contacts AIcrowd."""
    rc, out = run([WHEST, "submit", out_tar, "--dry-run", "--description", desc], timeout=600)
    if rc != 0:
        rc, out = run([WHEST, "submit", "--estimator", est, "--dry-run",
                       "--description", desc], timeout=600)
    print(out[-3000:].strip())
    return rc


def submit_real(out_tar, desc):
    rc, out = run([WHEST, "submit", out_tar, "--description", desc, "--format", "plain"],
                  timeout=900)
    print(out[-1500:].strip())
    m = re.search(r"submission id (\d+)", out)
    if rc != 0 or not m:
        raise SystemExit("submit failed / could not parse the submission id")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# record + README (logic/format identical to tools/submit.py)
# ---------------------------------------------------------------------------

def regen_readme():
    rows = [json.loads(l) for l in open(os.path.join(SUBS, "log.jsonl"), encoding="utf-8") if l.strip()]
    lines = ["# Submissions", "",
             "Every AIcrowd submission of this project, newest first. Score = final-layer MSE x "
             "max(0.1, FLOPs / 2^41), averaged over the hidden suite (lower is better).", "",
             "| id | date (UTC) | tag | score | final-layer MSE | multiplier | status | file | description |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: str(r.get("created_at", "")), reverse=True):
        sc, ms = r.get("score"), r.get("score_secondary")
        mult = (sc / ms) if (sc and ms) else None
        f = f"[{r['dir']}/estimator.py]({r['dir']}/estimator.py)" if r.get("dir") else "-"
        lines.append(f"| {r['id']} | {str(r.get('created_at', ''))[:16].replace('T', ' ')} | {r.get('tag', '')} | "
                     f"{sc:.4e} | {ms:.4e} | {mult:.4f} | {r.get('status', '')} | {f} | {r.get('description', '')} |"
                     if sc else
                     f"| {r['id']} | {str(r.get('created_at', ''))[:16].replace('T', ' ')} | {r.get('tag', '')} | - | - | - | "
                     f"{r.get('status', '')} | {f} | {r.get('description', '')} |")
    open(os.path.join(SUBS, "README.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def record(sid, tag, est_path, description, st):
    d = f"{sid}_{tag}"
    os.makedirs(os.path.join(SUBS, d), exist_ok=True)
    if est_path:
        shutil.copyfile(est_path, os.path.join(SUBS, d, "estimator.py"))
    meta = {"id": sid, "tag": tag, "description": description, "dir": d if est_path else None,
            "status": st.get("grading_status_cd"), "score": st.get("score"),
            "score_secondary": st.get("score_secondary"), "created_at": st.get("created_at"),
            "grading_message": re.sub(r"<[^>]+>", "", st.get("grading_message") or "").strip()}
    json.dump(meta, open(os.path.join(SUBS, d, "meta.json"), "w", encoding="utf-8"), indent=1)
    logp = os.path.join(SUBS, "log.jsonl")
    rows = [json.loads(l) for l in open(logp, encoding="utf-8") if l.strip()] if os.path.exists(logp) else []
    rows = [r for r in rows if r["id"] != sid] + [meta]
    with open(logp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    regen_readme()
    return meta


# ---------------------------------------------------------------------------
# GitHub push — Git-Data-API client ported from tools/gh.py (stdlib only)
# ---------------------------------------------------------------------------

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
PRUNE_DIRS = ("kit", "kitvenv", "dl", "data", "ds_local", "runs/pred", "cs/srv", "cs/cli",
              "research", "tool-results")


def gh_token():
    """GitHub token at runtime: $GITHUB_TOKEN, then a token file outside the repo."""
    t = os.environ.get("GITHUB_TOKEN")
    if t:
        return t.strip()
    for p in (pathlib.Path.home() / ".config" / "whest-github" / "token",
              pathlib.Path(os.environ.get("APPDATA", "")) / "whest-github" / "token"):
        try:
            return p.read_text(encoding="ascii").strip()
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit("no GitHub token: set GITHUB_TOKEN")


def _secrets():
    """Live secrets read at run time; refuse to push any file containing them."""
    out = [gh_token()]
    try:
        out.append(_aick())
    except SystemExit:
        pass
    return [s for s in out if s]


def req(method, path, body=None, ok=(200, 201)):
    url = path if path.startswith("http") else API + path
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", "Bearer " + gh_token())
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


def manifest():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT).replace(os.sep, "/")
        if rel_dir == ".":
            rel_dir = ""
        dirnames[:] = [d for d in dirnames if (rel_dir + "/" + d).lstrip("/") not in PRUNE_DIRS
                       and not d.startswith(".") and d != "__pycache__"]
        for fn in filenames:
            rel = (rel_dir + "/" + fn).lstrip("/")
            if any(fnmatch.fnmatch(rel, p) for p in INCLUDE) and \
               not any(fnmatch.fnmatch(rel, p) for p in EXCLUDE):
                out.append(rel)
    return sorted(out)


def _git_blob_sha(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def push(full, message, branch=BRANCH):
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


# ---------------------------------------------------------------------------
# main flow (mirrors tools/submit.py)
# ---------------------------------------------------------------------------

def main():
    est, tag, desc = sys.argv[1], sys.argv[2], sys.argv[3]
    est = os.path.abspath(est)
    dry = "--dry-run" in sys.argv
    out_tar = os.path.join(os.path.dirname(est), "submission.tar.gz")
    build_chain(est, out_tar)
    if dry:
        rc = submit_dry_run(est, out_tar, desc)
        raise SystemExit(0 if rc == 0 else 1)
    sid = submit_real(out_tar, desc)
    st = status(sid)
    record(sid, tag, est, desc, st)
    if "--no-push" not in sys.argv:
        push(REPO, f"submission {sid} ({tag}): submitted\n\n{desc}")
    t0 = time.time()
    while time.time() - t0 < POLL_TIMEOUT:
        st = status(sid)
        if st.get("grading_status_cd") in ("graded", "failed"):
            break
        time.sleep(POLL_EVERY)
    meta = record(sid, tag, est, desc, st)
    print(json.dumps(meta, indent=1))
    if "--no-push" not in sys.argv:
        push(REPO, f"submission {sid} ({tag}): {meta['status']} {meta.get('score')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--record":
        sid, tag, est, desc = int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
        meta = record(sid, tag, None if est == "-" else os.path.abspath(est), desc, status(sid))
        print(json.dumps(meta, indent=1))
    elif len(sys.argv) > 1 and sys.argv[1] == "--push":
        push(REPO, sys.argv[2])
    else:
        if len(sys.argv) < 4:
            raise SystemExit(__doc__)
        main()
