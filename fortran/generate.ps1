<#
.SYNOPSIS
  Builds fortran/golden_driver.f90 against the Reference implementation
  (Fortran ODEPACK) and writes the Golden files into tests/golden/.

.DESCRIPTION
  STATUS: UNBUILT AND UNTESTED. This script and the driver were written on a
  machine without gfortran. Expect to fix small problems on the first run.

  Requires gfortran on PATH. The Reference implementation's source directory
  (the one holding odepack_mod.f90, odepack.f, lapack/) is given with
  -OdepackSrc or the ODEPACK_SRC environment variable; no path is hardcoded.

.PARAMETER OdepackSrc
  Path to the Reference implementation's `src` directory.

.PARAMETER OutDir
  Where to write the Golden files. Default: tests/golden in this repository.

.PARAMETER BuildDir
  Scratch directory for object files and the executable.
  Default: .scratch/golden-build in this repository (git-ignored).

.EXAMPLE
  $env:ODEPACK_SRC = "<path to odepack>/src"; ./fortran/generate.ps1
#>
param(
    [string]$OdepackSrc = $env:ODEPACK_SRC,
    [string]$OutDir,
    [string]$BuildDir
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $OutDir)   { $OutDir   = Join-Path $repo 'tests/golden' }
if (-not $BuildDir) { $BuildDir = Join-Path $repo '.scratch/golden-build' }

if (-not (Get-Command gfortran -ErrorAction SilentlyContinue)) {
    throw 'gfortran not found on PATH. Install it (see README) to regenerate the Golden files.'
}
if (-not $OdepackSrc) {
    throw 'Give the ODEPACK source directory with -OdepackSrc or the ODEPACK_SRC environment variable.'
}
if (-not (Test-Path (Join-Path $OdepackSrc 'odepack_mod.f90'))) {
    throw "odepack_mod.f90 not found in '$OdepackSrc'."
}

New-Item -ItemType Directory -Force $OutDir, $BuildDir | Out-Null

# -ffp-contract=off keeps a*b+c from being fused, so the floating-point
# operation order matches the Rust port. No -ffast-math.
$common = @('-O2', '-ffp-contract=off', '-J', $BuildDir)
$legacy = $common + @('-std=legacy')

function Invoke-Gfortran([string[]]$GfArgs) {
    & gfortran @GfArgs
    if ($LASTEXITCODE -ne 0) { throw "gfortran failed: gfortran $($GfArgs -join ' ')" }
}

# Compile order matters for the Fortran modules.
$objs = @()
foreach ($f in 'odepack_common.f90', 'odepack_interface.f90') {
    $o = Join-Path $BuildDir ($f + '.o')
    Invoke-Gfortran ($common + @('-c', (Join-Path $OdepackSrc $f), '-o', $o)); $objs += $o
}
$sources = @('odepack_sub1.f', 'odepack_sub2.f', 'odepack.f') +
    (Get-ChildItem (Join-Path $OdepackSrc 'lapack') -Filter *.f | ForEach-Object { 'lapack/' + $_.Name })
foreach ($f in $sources) {
    $o = Join-Path $BuildDir (($f -replace '[\\/]', '_') + '.o')
    Invoke-Gfortran ($legacy + @('-c', (Join-Path $OdepackSrc $f), '-o', $o)); $objs += $o
}
$o = Join-Path $BuildDir 'odepack_mod.f90.o'
Invoke-Gfortran ($common + @('-c', (Join-Path $OdepackSrc 'odepack_mod.f90'), '-o', $o)); $objs += $o

$exe = Join-Path $BuildDir 'golden_driver'
Invoke-Gfortran ($common + @((Join-Path $PSScriptRoot 'golden_driver.f90')) + $objs + @('-o', $exe))

& $exe $OutDir
if ($LASTEXITCODE -ne 0) { throw "golden_driver exited with code $LASTEXITCODE" }
Write-Host "Golden files written to $OutDir"
