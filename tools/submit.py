"""Package -> validate -> submit -> watch -> record -> push to GitHub, in one step.

usage (from anywhere):
  python submit.py <estimator.py> <tag> "<description>" [--dry-run] [--no-push]

Records every submission under submissions/<id>_<tag>/ (estimator.py + meta.json), appends
submissions/log.jsonl, regenerates submissions/README.md, then pushes the repo snapshot
(tools/gh.py). The estimator is copied byte-for-byte; the AIcrowd key never leaves
whest's own config.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = r"E:\Claude code\whest"
KIT = os.path.join(ROOT, "kit")
WHEST = os.path.join(KIT, ".venv", "Scripts", "whest.exe")
PY = os.path.join(KIT, ".venv", "Scripts", "python.exe")
SUBS = os.path.join(ROOT, "submissions")
REPO = "acco-cyber/arc-whitebox-estimation-2026"
B = 2 ** 41


def run(cmd, timeout=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, cwd=KIT, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def status(sid):
    rc, out = run([PY, os.path.join(ROOT, "tools", "acct.py"), "status", str(sid)])
    m = re.search(r"^\{.*^\}", out, flags=re.S | re.M)
    return json.loads(m.group(0)) if m else {"raw": out}


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


def push(msg):
    rc, out = run([PY, os.path.join(ROOT, "tools", "gh.py"), "push", REPO, msg])
    print(out.strip())
    return rc


def main():
    est, tag, desc = sys.argv[1], sys.argv[2], sys.argv[3]
    est = os.path.abspath(est)
    dry = "--dry-run" in sys.argv
    out_tar = os.path.join(os.path.dirname(est), "submission.tar.gz")
    rc, out = run([WHEST, "validate", "--estimator", est], timeout=600)
    print(out[-1500:])
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
    if dry:
        rc, out = run([WHEST, "submit", out_tar, "--dry-run", "--description", desc], timeout=600)
        print(out[-3000:])
        return
    rc, out = run([WHEST, "submit", out_tar, "--description", desc, "--format", "plain"], timeout=900)
    print(out[-1500:])
    m = re.search(r"submission id (\d+)", out)
    if not m:
        raise SystemExit("could not parse the submission id")
    sid = int(m.group(1))
    st = status(sid)
    record(sid, tag, est, desc, st)
    if "--no-push" not in sys.argv:
        push(f"submission {sid} ({tag}): submitted\n\n{desc}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    t0 = time.time()
    while time.time() - t0 < 5400:
        st = status(sid)
        if st.get("grading_status_cd") in ("graded", "failed"):
            break
        time.sleep(60)
    meta = record(sid, tag, est, desc, st)
    print(json.dumps(meta, indent=1))
    if "--no-push" not in sys.argv:
        push(f"submission {sid} ({tag}): {meta['status']} {meta['score']}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--record":
        # python submit.py --record <id> <tag> <estimator.py|-> "<description>"
        sid, tag, est, desc = int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
        meta = record(sid, tag, None if est == "-" else os.path.abspath(est), desc, status(sid))
        print(json.dumps(meta, indent=1))
    elif len(sys.argv) > 1 and sys.argv[1] == "--push":
        push(sys.argv[2])
    else:
        main()
