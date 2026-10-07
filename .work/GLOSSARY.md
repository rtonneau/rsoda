# rsoda glossary

rsoda is a Rust port of the LSODA solver for ordinary differential equation initial value problems, translated from the thread-safe Modern Fortran fork of ODEPACK. It exists to give Rust users the same automatic stiff/nonstiff integration with matching numerics.

<!-- Format adapted from mattpocock-skills' domain-modeling (GLOSSARY-FORMAT.md): https://github.com/mattpocock/skills.git -->

<!--
How to add a term (the moment it is resolved, not in a batch at the end):

**Term**:
One or two sentences: what it IS, not what it does.
_Avoid_: the other words people use for it

- Be opinionated: one word per concept; list the others under _Avoid_.
- Only terms specific to this project. General programming concepts and
  implementation details don't belong here: this is a glossary, not a spec.
- Group terms under "### <area>" subheadings once natural clusters emerge.
-->

## Language

**LSODA**:
The solver that integrates an ODE system, automatically switching between a nonstiff and a stiff method as the problem demands. rsoda ports LSODA only, not LSODAR.
_Avoid_: ODEPACK (the whole package), LSODAR (adds root-finding, out of scope)

**Reference implementation**:
The thread-safe Modern Fortran fork of ODEPACK in `DEV/FORTRAN/odepack`, the source of truth for behavior and golden data.
_Avoid_: original ODEPACK, the Fortran

**Method**:
Either the nonstiff (Adams) or the stiff (BDF) integration scheme LSODA is currently using.
_Avoid_: mode, scheme

**Method switch**:
The moment LSODA changes Method because it detected stiffness or its absence.
_Avoid_: stiffness flip

**Jacobian kind**:
How the Jacobian is obtained and stored: dense or banded, user-supplied or finite-difference (Fortran `jt` 1, 2, 4, 5).
_Avoid_: jt, Jacobian mode

**Golden file**:
A committed record of a Reference implementation run (trajectory plus step, order and Method-switch counters) that rsoda's tests compare against.
_Avoid_: fixture, baseline
