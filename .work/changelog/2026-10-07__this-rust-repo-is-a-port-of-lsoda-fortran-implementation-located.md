---
bump: minor
floor: minor
---
- Add a Rust port of the LSODA ODE solver, which switches automatically between a nonstiff (Adams) and a stiff (BDF) method.
- Support dense and banded Jacobians, either supplied by the user or computed by finite differences.
- Offer all five ITASK modes through `integrate`, `step`, `advance_past`, `integrate_tcrit` and `step_tcrit`, with tolerances and options set through a builder and setters.
- Report failures as a `LsodaError` value instead of printing, with opt-in hooks for warnings and method switches.
- Add a Fortran driver and a test harness to compare results against the Fortran reference; it needs gfortran and no reference data is included yet.
