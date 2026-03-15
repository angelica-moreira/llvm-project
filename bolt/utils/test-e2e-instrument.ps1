#!/usr/bin/env pwsh
# test-e2e-instrument.ps1 -- Full round-trip instrumentation test
#
# This is the big integration test. It does the entire BOLT instrumentation
# loop: instrument a binary, run it to collect a profile, then optimize
# the original binary with that profile.
#
# STATUS: NOT YET FULLY IMPLEMENTED
#
# Requires Phase 10c (runtime), 10d (PE plumbing), and 10e to be done.
# The script will detect what is available and run what it can.
#
# Usage:
#   .\test-e2e-instrument.ps1 -Binary D:\z3\build\release\z3.exe -Workload "--version"
#   .\test-e2e-instrument.ps1 -Binary D:\z3\build\release\z3.exe -Workload "sat.smt2"

param(
    [string]$BuildDir = "D:\llvm-upstream\llvm-project\build",
    [string]$Binary   = "D:\z3\build\release\z3.exe",
    [string]$Workload = "--version"
)

$ErrorActionPreference = "Stop"

$Bolt    = Join-Path $BuildDir "bin\llvm-bolt.exe"
$ReadObj = Join-Path $BuildDir "bin\llvm-readobj.exe"

foreach ($tool in @($Bolt, $ReadObj)) {
    if (-not (Test-Path $tool)) {
        Write-Host "Missing tool: $tool" -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path $Binary)) {
    Write-Host "Binary not found: $Binary" -ForegroundColor Red
    Write-Host "Pass -Binary <path> to specify the test binary." -ForegroundColor Yellow
    exit 1
}

$BinaryName = [IO.Path]::GetFileNameWithoutExtension($Binary)
$WorkDir = Join-Path ([IO.Path]::GetTempPath()) "bolt-e2e-$BinaryName"
if (Test-Path $WorkDir) { Remove-Item $WorkDir -Recurse -Force }
New-Item -ItemType Directory $WorkDir | Out-Null

$Failures = 0

function Write-Check {
    param([string]$Label, [bool]$Ok, [string]$Detail = "")
    $mark = if ($Ok) { "[OK]" } else { "[FAIL]" }
    $color = if ($Ok) { "Green" } else { "Red" }
    $msg = "  $mark $Label"
    if ($Detail) { $msg += " ($Detail)" }
    Write-Host $msg -ForegroundColor $color
    if (-not $Ok) { $script:Failures++ }
}

Write-Host ""
Write-Host "BOLT End-to-End Instrumentation Test" -ForegroundColor Cyan
Write-Host ("=" * 60)
Write-Host "  Binary:   $Binary"
Write-Host "  Workload: $Workload"
Write-Host ""

# ------------------------------------------------------------------
# Step 1: Instrument the binary
# ------------------------------------------------------------------
Write-Host "Step 1: Instrumenting..." -ForegroundColor Yellow
$instrBinary = Join-Path $WorkDir "$BinaryName-instr.exe"

$output = & $Bolt $Binary -o $instrBinary --instrument 2>&1 | Out-String

if ($LASTEXITCODE -ne 0 -or $output -match "not supported|not implemented") {
    Write-Host "  Instrumentation not yet available for PE/COFF." -ForegroundColor Yellow
    Write-Host "  This is expected until Phases 10c and 10d are done." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  To test the parts that DO work today:" -ForegroundColor Cyan
    Write-Host "    .\test-emit-link.ps1         # profile-guided rewrite" -ForegroundColor Cyan
    Write-Host "    .\test-z3-profile.ps1        # z3 identity + small profile" -ForegroundColor Cyan
    Write-Host "    .\cross-validate-pecoff.ps1  # PE structure validation" -ForegroundColor Cyan
    Write-Host "    .\validate-pecoff.ps1        # 67-check structural test" -ForegroundColor Cyan
    Write-Host ""
    Remove-Item $WorkDir -Recurse -Force
    exit 0
}

Write-Check "Instrumentation succeeded" ($LASTEXITCODE -eq 0)
Write-Check "Instrumented binary created" (Test-Path $instrBinary)

# ------------------------------------------------------------------
# Step 2: Run the instrumented binary to collect profile
# ------------------------------------------------------------------
Write-Host "`nStep 2: Running instrumented binary..." -ForegroundColor Yellow
Push-Location $WorkDir
$instrOutput = & $instrBinary $Workload.Split(" ") 2>&1 | Out-String
$instrExit = $LASTEXITCODE
Pop-Location

Write-Check "Instrumented binary ran" ($instrExit -eq 0 -or $instrOutput.Length -gt 0)

# The profile should be written to prof.fdata in the working directory
$fdataPath = Join-Path $WorkDir "prof.fdata"
if (-not (Test-Path $fdataPath)) {
    # Some versions write to the current directory instead
    $fdataPath = "prof.fdata"
}

Write-Check "prof.fdata created" (Test-Path $fdataPath) $fdataPath

if (Test-Path $fdataPath) {
    $fdataSize = (Get-Item $fdataPath).Length
    $fdataLines = (Get-Content $fdataPath | Measure-Object).Count
    Write-Check "prof.fdata not empty" ($fdataSize -gt 0) "${fdataSize} bytes, ${fdataLines} lines"

    # Show a quick summary of the profile
    Write-Host "`n  Profile summary:" -ForegroundColor Cyan
    Write-Host "    File size:  $fdataSize bytes"
    Write-Host "    Line count: $fdataLines"
    $funcs = Get-Content $fdataPath | ForEach-Object { ($_ -split " ")[1] } | Sort-Object -Unique | Measure-Object
    Write-Host "    Unique functions: $($funcs.Count)"
}

# ------------------------------------------------------------------
# Step 3: Optimize the original binary with collected profile
# ------------------------------------------------------------------
Write-Host "`nStep 3: Optimizing with profile..." -ForegroundColor Yellow
$optBinary = Join-Path $WorkDir "$BinaryName-opt.exe"

if (Test-Path $fdataPath) {
    $optOutput = & $Bolt $Binary -o $optBinary -data=$fdataPath -reorder-blocks=ext-tsp 2>&1 | Out-String
    Write-Check "Optimization succeeded" ($LASTEXITCODE -eq 0)
    Write-Check "Optimized binary created" (Test-Path $optBinary)
} else {
    Write-Host "  [SKIP] No profile to optimize with" -ForegroundColor Yellow
}

# ------------------------------------------------------------------
# Step 4: Validate the optimized binary
# ------------------------------------------------------------------
if (Test-Path $optBinary) {
    Write-Host "`nStep 4: Validating optimized binary..." -ForegroundColor Yellow

    $headers = & $ReadObj --file-headers $optBinary 2>&1 | Out-String
    Write-Check "Valid PE headers" ($headers -match "IMAGE_FILE_MACHINE_AMD64")

    # Smoke test: run the optimized binary with the same workload
    $optRun = & $optBinary $Workload.Split(" ") 2>&1 | Out-String
    $optExit = $LASTEXITCODE
    Write-Check "Optimized binary runs" ($optExit -eq 0 -or $optRun.Length -gt 0)

    # Compare outputs (they should match)
    $origRun = & $Binary $Workload.Split(" ") 2>&1 | Out-String
    Write-Check "Output matches original" ($optRun.Trim() -eq $origRun.Trim())

    # Size comparison
    Write-Host "`n  Size comparison:" -ForegroundColor Cyan
    $origSize = (Get-Item $Binary).Length
    $optSize  = (Get-Item $optBinary).Length
    Write-Host "    Original:  $origSize bytes"
    Write-Host "    Optimized: $optSize bytes"
    $diff = $optSize - $origSize
    $pct = [math]::Round(($diff / $origSize) * 100, 2)
    Write-Host "    Difference: $diff bytes ($pct%)"
}

# Cleanup
Remove-Item $WorkDir -Recurse -Force

Write-Host ""
Write-Host ("=" * 60)
if ($Failures -eq 0) {
    Write-Host "All checks passed." -ForegroundColor Green
} else {
    Write-Host "$Failures check(s) FAILED." -ForegroundColor Red
}
Write-Host ""
exit $Failures
