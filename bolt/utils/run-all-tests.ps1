#!/usr/bin/env pwsh
# run-all-tests.ps1 -- One command to run every BOLT PE/COFF validation script
#
# This is the master script. Run it before pushing to make sure nothing is broken.
# It calls each test script in order and reports a final summary.
#
# Usage:
#   .\run-all-tests.ps1
#   .\run-all-tests.ps1 -BuildDir D:\llvm\build -Z3Path D:\z3\build\release\z3.exe
#   .\run-all-tests.ps1 -SkipZ3   # if you do not have z3.exe

param(
    [string]$BuildDir = "D:\llvm-upstream\llvm-project\build",
    [string]$Z3Path   = "D:\z3\build\release\z3.exe",
    [switch]$SkipZ3
)

$ErrorActionPreference = "Continue"
$ScriptDir = $PSScriptRoot

Write-Host ""
Write-Host "BOLT PE/COFF -- Full Test Suite" -ForegroundColor Cyan
Write-Host ("=" * 60)
Write-Host "  Build dir: $BuildDir"
Write-Host "  z3 path:   $(if ($SkipZ3) { '(skipped)' } else { $Z3Path })"
Write-Host ""

$results = @()

function Run-Script {
    param([string]$Name, [string]$Script, [string[]]$ExtraArgs = @())

    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor DarkGray
    Write-Host "Running: $Name" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor DarkGray

    $fullPath = Join-Path $ScriptDir $Script
    if (-not (Test-Path $fullPath)) {
        Write-Host "  [SKIP] Script not found: $fullPath" -ForegroundColor Yellow
        $script:results += [PSCustomObject]@{ Name = $Name; Status = "SKIP"; Detail = "not found" }
        return
    }

    & $fullPath -BuildDir $BuildDir @ExtraArgs
    $code = $LASTEXITCODE

    $status = if ($code -eq 0) { "PASS" } else { "FAIL" }
    $script:results += [PSCustomObject]@{ Name = $Name; Status = $status; Detail = "exit $code" }
}

# 1. Lit tests (the fastest and most important check)
Run-Script "Regression tests" "test-no-regression.ps1"

# 2. Emit + link (profile-guided rewrite of small binary)
Run-Script "Emit + link" "test-emit-link.ps1"

# 3. z3 profile test (identity rewrite + small binary profile)
if (-not $SkipZ3 -and (Test-Path $Z3Path)) {
    Run-Script "z3 profile test" "test-z3-profile.ps1" @("-Z3Path", $Z3Path, "-SkipLargeTest")
} else {
    Write-Host ""
    Write-Host "  [SKIP] z3 profile test (no z3.exe or -SkipZ3)" -ForegroundColor Yellow
    $results += [PSCustomObject]@{ Name = "z3 profile test"; Status = "SKIP"; Detail = "no z3" }
}

# 4. PE structural validation (if z3 is available)
if (-not $SkipZ3 -and (Test-Path $Z3Path)) {
    Run-Script "PE validation (z3)" "validate-pecoff.ps1" @("-Original", $Z3Path)
} else {
    $results += [PSCustomObject]@{ Name = "PE validation"; Status = "SKIP"; Detail = "no z3" }
}

# 5. Cross-validation (if z3 is available)
if (-not $SkipZ3 -and (Test-Path $Z3Path)) {
    Run-Script "Cross-validation (z3)" "cross-validate-pecoff.ps1" @("-Binary", $Z3Path)
} else {
    $results += [PSCustomObject]@{ Name = "Cross-validation"; Status = "SKIP"; Detail = "no z3" }
}

# 6. Instrumentation runtime (will likely skip, not yet built)
Run-Script "Instrumentation runtime" "test-instr-runtime.ps1"

# 7. PE instrumentation (will likely skip, not yet implemented)
Run-Script "PE instrumentation" "test-instrument-pecoff.ps1"

# Final summary
Write-Host ""
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""

$pass = ($results | Where-Object { $_.Status -eq "PASS" }).Count
$fail = ($results | Where-Object { $_.Status -eq "FAIL" }).Count
$skip = ($results | Where-Object { $_.Status -eq "SKIP" }).Count

foreach ($r in $results) {
    $color = switch ($r.Status) { "PASS" { "Green" } "FAIL" { "Red" } "SKIP" { "Yellow" } }
    $mark  = switch ($r.Status) { "PASS" { "[OK]  " } "FAIL" { "[FAIL]" } "SKIP" { "[SKIP]" } }
    Write-Host "  $mark $($r.Name)" -ForegroundColor $color
}

Write-Host ""
Write-Host "  Passed: $pass  Failed: $fail  Skipped: $skip" -ForegroundColor $(if ($fail -gt 0) { "Red" } else { "Green" })
Write-Host ""

exit $fail
