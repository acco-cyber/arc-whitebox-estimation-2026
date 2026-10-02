"""Account / submission status helper (reads the key whest login stored)."""
import json, os, sys, pathlib

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

from whestbench.aicrowd_client import AIcrowdClient, describe_error

SLUG = "arc-white-box-estimation-challenge-2026"


def _key():
    k = os.environ.get("AICROWD_API_KEY")
    if k:
        return k
    p = pathlib.Path(os.environ["APPDATA"]) / "aicrowd-cli" / "config.toml"
    cfg = tomllib.loads(p.read_text())
    for v in cfg.values():
        if isinstance(v, str) and len(v) >= 40:
            return v
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, str) and len(vv) >= 40:
                    return vv
    raise SystemExit("no key")


def main():
    c = AIcrowdClient(api_key=_key())
    who = c.whoami()
    print("user:", {k: who.get(k) for k in ("id", "username", "name") if k in who})
    cmd = sys.argv[1] if len(sys.argv) > 1 else "elig"
    if cmd == "elig":
        try:
            e = c.check_eligibility(challenge_slug=SLUG)
            print(json.dumps(e, indent=1)[:4000])
        except Exception as ex:  # noqa
            print("eligibility error:", describe_error(ex))
    elif cmd == "status":
        for sid in sys.argv[2:]:
            try:
                s = c.get_submission_status(int(sid))
                keep = {k: s.get(k) for k in ("id", "grading_status_cd", "score", "score_secondary",
                                               "grading_message", "created_at", "participant_id", "meta")}
                print(json.dumps(keep, indent=1, default=str)[:6000])
            except Exception as ex:  # noqa
                print(sid, "error:", describe_error(ex))


if __name__ == "__main__":
    main()
