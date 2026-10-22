# Final Leaderboard Nomination — 2026-10-22 (Nanoloom / koushik_rudra)

## Executive summary

Phase 2 submissions closed 2026-10-21 23:59 UTC ("challenge_completed" — API-verified).
2026-10-22 is **nomination day**: select up to 2 valid Phase 2 entries for the
**Phase 2 Private Re-evaluation** (fresh MLPs, private seeds, flopscope v0.12.3).
Deadline: **2026-10-22 23:59 UTC**. Prize rankings come exclusively from this re-eval.

**NOMINATION EXECUTED AND CONFIRMED (GraphQL `selectSubmission`, verified persisted):**

| # | Submission | Public score | MSE (score_secondary) | Multiplier | Role |
|---|-----------|--------------|----------------------|------------|------|
| 1 | **337551** | **2.9575029e-9** | 1.9854289e-8 | 0.148960 | Best score; cheapest build (lowest cost multiplier in pool) |
| 2 | **337066** | **2.9711234e-9** | 1.9446764e-8 | 0.152782 | Vanilla ALG058 verbatim; lowest MSE in pool; graded 2.9711 twice (our 337066 + TristanMiano's 336809, byte-identical artifact) |

Platform state after execution: `selectedCount: 2, remaining: 0,
selectedSubmissionIds: ["337551", "337066"]` (round 1429, window open,
deadline 2026-10-22T23:59Z). Re-verified with an independent query.

## Why this pair

- **Prize metric = best of the two nominated files on fresh seeds.** Max-of-two is
  maximized by diversifying across orthogonal risk profiles, not by doubling up
  near-identical builds.
- 337551 = cheapest engine build (mult 0.14896, 0.27% cheaper than s27's 0.14936)
  with suite MSE identical to the s27 family (1.9854e-8). Its score gain is purely
  the billing cut. Content (recovered from Oct-21 session records on GitHub,
  submissions/337551_s38hub7): ALG058-V001 + tau_b 2.2 (graded 337079 base) +
  **Strassen depth 7 at the single main young-hub site** (leaf 8, smin=8 wrapped at
  that call site only; global STRASSEN_MIN stays 32; old-level site stays depth 5;
  first-predict cap preserved; level() clamps to 6 where shapes disallow).
- 337066 = unmodified ALG058-V001 (TristanMiano, MIT; e30_a engine by LordProcciuto,
  V29 by 504aldo). It has the **lowest MSE of all valid entries** (1.9447e-8, 2.0%
  below the 337551 family) and is the only artifact **proven byte-stable across two
  independent gradings** (identical 2.9711e-9 both times). Zero custom-knob risk.
- Hedge logic: if the cost-cut knob in 337551 interacts badly with fresh-seed
  distributions, 337066 is untouched by that failure mode; conversely if fresh-seed
  MSE preserves family ordering (expected), 337551 still carries the top score.
- The near-twin 337545 (2.9577e-9, 0.008% behind 337551; = same base with hub
  Strassen depth 6 instead of 7) was **rejected** in favor of 337066: nominating both
  twins doubles exposure to one knob family (hub-depth + tau_b) — including the
  documented grader runtime-margin risk (deep builds run 10% slower on some grader
  days → zero-prediction fallbacks; receipts in forum topic 18252) — while 337066's
  fresh-seed behavior is the most evidenced in the pool.

## "All submissions getting error" — diagnosis

9 Phase 2 submissions failed with `Error : Smoke test failed`:

```
336420 336421 336427 336442 336463        (Oct 16 outage window)
337369 337386 337388 337390              (Oct 20, 16:23–19:24 UTC)
337561 337569 337581 337589
337605 337613 337642                     (Oct 21, 16:32–22:39 UTC)
```

**Root cause: grader-side smoke-test outage, NOT submission content.** Proof:
337390 was a byte-identical re-upload of the already-graded 337079 and ALSO
smoke-failed; the leaderboard showed zero grades board-wide during the windows
(mirrors the Oct-16 outage). Outcome: failed entries cannot be resubmitted
(submissions are closed) and do not affect the final ranking — only the 2
nominated files count in the private re-eval, and both graded clean already.

## All valid (graded) Phase 2 entries — final board

| id | tag (where known) | score | MSE | mult |
|----|-------------------|-------|-----|------|
| 337551 | Oct-21 session | **2.9575029e-9** | 1.98543e-8 | 0.148960 |
| 337545 | Oct-21 session | 2.9577463e-9 | 1.98535e-8 | 0.148979 |
| 337079 | s27 taub22 | 2.9663011e-9 | 1.98607e-8 | 0.149356 |
| 337066 | s25 ALG058 | 2.9711234e-9 | 1.94468e-8 | 0.152782 |
| 337069 | s26 ALG055 | 2.9830025e-9 | 1.95309e-8 | 0.152733 |
| 337367 | s29 micro | 2.9941694e-9 | 2.00904e-8 | 0.149035 |
| 337370 | s31 dose20 | 3.0485965e-9 | 2.06360e-8 | 0.147732 |
| 336882 | e30_a-class | 3.1162826e-9 | 2.05540e-8 | 0.151615 |

(Catastrophic grades excluded: 336109, 336264, 336272, 336788, 336899, 337088, 337532 —
multiplier-forced-1.0 signatures; documented in submissions/log.jsonl.)

## Timeline & obligations

- 2026-10-21 23:59 UTC — Phase 2 submissions closed.
- **2026-10-22 23:59 UTC — nomination deadline (DONE, above).**
- 2026-10-28 23:59 UTC — algorithmic-contribution write-ups due (forum topic 18251).
- Prize ranking = private re-eval on fresh seeds, flopscope v0.12.3 (same version
  that graded both nominated entries).

## Reproduction

```
# selection state (read)
POST https://www.aicrowd.com/graphql
{"query":"query($slug: ID!){ challenge(slug: $slug){ rounds{ id name
 selectionWindowOpen selectionDeadline viewerSelectionContext{ selectedCount
 remaining selectedSubmissionIds } } } }","variables":{"slug":"arc-white-box-estimation-challenge-2026"}}
# Authorization: Token <AICROWD_API_KEY>

# nominate
mutation SelectSubmission($submissionId: ID!) {
  selectSubmission(input: { submissionId: $submissionId }) {
    submission { id }
    selectionContext { selectedCount remaining selectedSubmissionIds }
    errors { code message }
  }
}
```
