# Session: this Rust repo is a port of LSODA fortran implementation located in DEV/FORTRAN/odepack

**Date:** 2026-10-07T09:47:48.980Z
**Status:** Grill phase complete

## Problem Statement

Rust has no maintained port of the LSODA ODE solver (automatic switching between a nonstiff Adams method and a stiff BDF method). rsoda ports LSODA from the thread-safe Modern Fortran fork of ODEPACK in `DEV/FORTRAN/odepack`, with the same numerics and an idiomatic Rust API. LSODAR (root-finding) is out of scope.

## Context & Constraints

- **Current behavior:** `rsoda` is an empty `cargo new --lib` skeleton (edition 2024, no dependencies, a placeholder `add` function). The Reference implementation is about 7,400 lines of Fortran (`odepack.f`, `odepack_sub1.f`, `odepack_sub2.f`, `odepack_mod.f90`) with tests `test_lsoda.f90` and `test_threadsafe.f90` and a benchmark. It replaces common blocks with a derived type, adds an `ierr` argument to callbacks, stores error messages in a string instead of printing, and uses LAPACK/OpenBLAS instead of LINPACK.
- **Pain point:** no Rust LSODA with step-for-step fidelity to the Fortran.
- **Dependencies:** none at runtime. `gfortran` is not installed on this machine (only `cmake` and `cargo`), so the user must install it once to generate golden files. ODEPACK is declared public domain (see the fork's LICENSE).
- **Tech stack:** Rust 2024, `f64` only, `std`, zero dependencies; a Fortran driver (built with gfortran) for golden-file generation only.

## Success Metrics

- Step, order, counter (nst, nfe, nje, nqu) and Method-switch sequences match the Golden files exactly; state values match within about 1e-12 relative.
- All four Jacobian kinds work (dense and banded, user-supplied and finite-difference), and all itask modes (1 to 5, including tcrit variants) work.
- Analytic-solution tests (e.g. Robertson, Van der Pol) pass without the Fortran, and the library never prints to stdout.

## Architecture & Approach

Single crate with modules per Fortran subsystem (stoda, prja, solsy, intdy, LU routines, and so on). Numerics are a faithful port (same algorithm, constants, step/order/switching logic); the API is idiomatic.

- **System:** an `OdeSystem` trait with `&mut self` methods for the right-hand side and Jacobian, returning `Result`; a negative `ierr` becomes an `Err`. A closure adapter wraps plain functions.
- **Solver:** a stateful `Lsoda` solver built through a builder with a typed `Tolerance` (scalar or per-component), `h0`, `hmax`, `hmin`, `mxstep`, `mxordn`, `mxords`, `mxhnil`, and a Jacobian kind. Defaults match the Fortran. `build()` validates and returns a `Result` mirroring the fork's istate=-3 checks.
- **Stepping:** `integrate(tout)` (itask 1), `step()` (itask 2), and tcrit-bounded variants (itask 4/5); re-entry is ordinary method calls with setters for changing tolerances and options.
- **Errors:** `Result<_, LsodaError>` with one variant per negative istate, carrying its data. Warnings and Method-switch events are opt-in hooks.
- **Linear algebra:** hand-ported dense and banded LU factor/solve routines in pure Rust.
- **Verification:** a Fortran driver dumps Golden files (trajectory plus counters and Method-switch sequence) that are committed; Rust tests compare against them, plus analytic tests.

## Assumptions & Trade-offs

- Not doing LSODAR, root-finding, the rest of ODEPACK, generic float types, no_std, or publishing to crates.io.
- Bitwise equality with the Fortran is not required: the fork uses LAPACK for LU while rsoda hand-ports it, so last-bit rounding can differ.
- Fortran golden generation is a one-off step needing the user to install gfortran.
- Done means: complete LSODA port, tests, rustdoc with an example, README and a NOTICE crediting ODEPACK (public domain) and the fork.

## Open Questions

- Exact location and format of the Fortran driver and Golden files in the repo (decide while planning).
- Which gfortran distribution the user installs (MSYS2 or conda).

## Notes

Glossary terms added in `.work/GLOSSARY.md`: LSODA, Reference implementation, Method, Method switch, Jacobian kind, Golden file. No ADR recorded.
