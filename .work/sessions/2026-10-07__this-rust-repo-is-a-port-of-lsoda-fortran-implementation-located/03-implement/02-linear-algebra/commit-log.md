# Ticket 02: linear-algebra

**Status:** ✅ Done

## Local Test Result

`cargo test linalg`: 8 passed, 0 failed, no warnings.

## Review Notes

Implemented by a sonnet subagent (high effort); reviewed in this session. Re-ran the tests and compared `dgefa` against LINPACK DGEFA by reading: loop bounds, pivot selection and the -1/pivot multiplier match. `dgbfa`, `dgesl` and `dgbsl` were not re-read line by line; they are covered by the dense-vs-banded comparison tests (9 shapes) and singular-detection tests. Conventions: column-major with `lda`, 0-based `ipvt`, `info` as in LINPACK. No changes after review.

## Blockers / Challenges

None.

## Commits

- c5afbb6 feat: dense and banded LU factor and solve (ticket 02)

## Time Spent

2m (ticket-start.js to ticket-complete.js)
