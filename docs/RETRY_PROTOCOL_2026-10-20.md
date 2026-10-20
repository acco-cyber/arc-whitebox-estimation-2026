# Retry protocol — grader smoke outage (Oct 20, ~18:00 UTC onset)

## State at writing (22:47 UTC Oct 20)
- Last board-wide grade: oliver_chen 17:55 UTC (now #1, 1.30e-9 / MSE 1.10e-8).
- ALL submissions since fail "Smoke test failed" — INCLUDING the byte-identical raw
  control of graded 337079 (337390) => content-independent grader-side outage.
- Best entry 337079 = 2.9663e-9 intact; nomination auto-fallback = 337079 + 337066 (safe,
  no action needed; manual UI nomination open until Oct 22 23:59 UTC if desired).
- Quota: Oct 20 6/10 used; Oct 21 = fresh 10. Deadline Oct 21 23:59 UTC.
- Board watch: `~/.venv/bin/python /home/z/my-project/scripts/board_check.py`
  (prints "dark" while the outage lasts; a "<N> m ago" string = grading resumed).

## Fire order (validated packages, byte-ready in /home/z/my-project/whest/scratch/)
1. `s33taul26.tar.gz` — s27 + X_DEAD_TAU_L "0-4:2.6" (expected ~2.94-2.96e-9; knob string
   grader-proven by s29). Submit + poll status (acct.py status <id>).
2. If s33 GRADES: `s32fuse1.tar.gz` — s27 + FUSE_P 343->1 (expected ~2.84-2.88e-9 if the
   fused path is clean at full size; bit-exact + cheaper + faster at mini 128/256).
3. If BOTH grade: `s35combo.tar.gz` — s27 + tau_L + FUSE_P (both deltas independently
   graded; expected mult ~0.1427-0.1445 x MSE 1.9861e-8 => ~2.84-2.87e-9).
4. Record every outcome in submissions/<id>_<tag>/ + log.jsonl + README.md, update the
   research doc, push. If s33 fails smoke AGAIN post-recovery: the tau_L-alone delta is
   genuinely smoke-fatal (the s29 bundle's QPASS2=1 was masking); skip to s32, and if s32
   also fails, hold the remaining quota for the Oct-21 window / organizer fix.

## Watch notes
- The Oct-16 outage lasted overnight and ended with the flopscope migration + deadline
  extension (organizer topic 18251). Check the challenge Discussion tab for a new
  organizer post; escalate there if the outage exceeds ~12 h (forum login is web-only).
