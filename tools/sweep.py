"""Run several estimator configs in parallel subprocesses and tabulate results.

Usage: python sweep.py <configs.json> [max_parallel]
configs.json: [{"tag": "...", "est": "path", "env": {"K": "V"}, "ids": "96,97", "warm": 0}, ...]
Results are appended to runs/sweep.jsonl and printed as a table (paired vs the first config).
"""
import json
import os
import subprocess
import sys
import time

ROOT = r"E:\Claude code\whest"
PY = os.path.join(ROOT, "kit", ".venv", "Scripts", "python.exe")
RUN = os.path.join(ROOT, "tools", "run_est.py")
OUT = os.path.join(ROOT, "runs", "sweep.jsonl")


def main():
    cfgs = json.load(open(sys.argv[1], encoding="utf-8"))
    maxp = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    pending = list(cfgs)
    running = []
    t0 = time.time()
    stamp = time.strftime("%H%M%S")
    while pending or running:
        while pending and len(running) < maxp:
            c = pending.pop(0)
            env = dict(os.environ)
            env.update({k: str(v) for k, v in c.get("env", {}).items()})
            env["PYTHONWARNINGS"] = "ignore"
            env["PYTHONIOENCODING"] = "utf-8"
            th = str(c.get("threads", 4))
            env["OPENBLAS_NUM_THREADS"] = th
            env["OMP_NUM_THREADS"] = th
            log = os.path.join(ROOT, "runs", f"sw_{stamp}_{c['tag']}.log")
            cmd = [PY, RUN, "--est", c["est"], "--ids", c["ids"], "--tag", c["tag"],
                   "--warm", str(c.get("warm", 0)), "--out", OUT,
                   "--save-pred", os.path.join(ROOT, "runs", "pred")]
            f = open(log, "w", encoding="utf-8")
            p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=os.path.join(ROOT, "kit"))
            running.append((c, p, f, log))
            print(f"[{time.time()-t0:6.0f}s] started {c['tag']}", flush=True)
        time.sleep(3)
        for item in list(running):
            c, p, f, log = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                tail = [ln.rstrip() for ln in open(log, encoding="utf-8", errors="replace") if ln.startswith("[")]
                print(f"[{time.time()-t0:6.0f}s] done {c['tag']} rc={p.returncode}", flush=True)
                for ln in tail:
                    print("   ", ln, flush=True)
                if p.returncode != 0:
                    err = open(log, encoding="utf-8", errors="replace").read()[-3000:]
                    print("    --- log tail ---\n" + err, flush=True)


if __name__ == "__main__":
    main()
