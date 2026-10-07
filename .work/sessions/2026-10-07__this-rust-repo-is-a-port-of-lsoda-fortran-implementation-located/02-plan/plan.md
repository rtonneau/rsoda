# Implementation Plan

**Session:** this Rust repo is a port of LSODA fortran implementation located in DEV/FORTRAN/odepack
**Date:** 2026-10-07T09:57:41.042Z
**Estimated effort:** 4-6 days

## Strategy

Port LSODA bottom-up from the Reference implementation: public types and errors first, then the leaf numerical routines (linear algebra, norms, coefficients, interpolation), then the Jacobian preparation, the core stepper (DSTODA), and finally the driver (DLSODA) with its stepping API. Analytic tests land once the driver exists; the Fortran driver and Golden-file harness come last because they need gfortran, which is not installed on this machine.

## Tickets Overview

- **Ticket 1:** Crate skeleton: module layout, `LsodaError`, `Tolerance`, `JacobianKind`, `OdeSystem` trait, closure adapter, builder with validation, machine constants.
- **Ticket 2:** Dense and banded LU factorization and solve (DGEFA/DGESL/DGBFA/DGBSL equivalents).
- **Ticket 3:** Leaf routines: weighted norms, error-weight vector, method coefficients (CFODE), interpolation (INTDY), and the corrector linear solve (SOLSY).
- **Ticket 4:** Jacobian preparation (PRJA) for all four Jacobian kinds.
- **Ticket 5:** Core stepper (STODA) including order selection and Adams/BDF method switching.
- **Ticket 6:** Driver (LSODA): first-step size, `integrate`, `step`, tcrit variants, re-entry, error returns, warning and method-switch hooks.
- **Ticket 7:** Analytic-solution test suite across all Jacobian kinds and itask modes.
- **Ticket 8:** Fortran driver, Golden-file generator and comparison harness, README, NOTICE and rustdoc example.

## Sequencing Rationale

Each ticket depends only on earlier ones: the stepper needs the leaf routines and Jacobian preparation, the driver needs the stepper, and the tests need the driver. Golden files come last because they need the user to install gfortran.

## Risks & Mitigation

- **Risk:** The stepper is long and subtle (order selection, stiffness detection), so small porting slips change step sequences. → **Mitigation:** port constants and control flow verbatim from `odepack_sub1.f`, use the highest effort level for tickets 5 and 6, and add the Golden-file comparison in ticket 8.
- **Risk:** gfortran is not installed here, so Golden files cannot be generated in an unattended run. → **Mitigation:** ticket 8 commits the driver, the generator script and a harness whose Golden tests skip with a clear message when files are absent; the user generates them after installing gfortran.
- **Risk:** LU rounding differs from the fork's LAPACK path. → **Mitigation:** compare counters exactly and values within about 1e-12 relative.

## Assumptions

- Rust edition 2024 toolchain is available (`cargo` is on PATH).
- `f64` only, `std`, zero dependencies.
- The Reference implementation in `C:\DEV\FORTRAN\odepack\src` is the source of truth, read directly for each routine.
