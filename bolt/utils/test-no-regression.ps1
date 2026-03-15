#!/usr/bin/env pwsh
# test-no-regression.ps1 -- Make sure our PE/COFF changes did not break anything
#
# Runs the BOLT lit tests and JITLink tests to catch regressions introduced
# by the COFF edge lowering code in JITLinkLinker.cpp.
#
# This is the script you want to run before every commit.
#
# Usage:
#   .\test-no-regression.ps1
#   .\test-no-regression.ps1 -BuildDir D:\llvm\build

param(
    [string]$BuildDir = "D:\llvm-upstream\llvm-project\build"
)

$ErrorActionPreference = "Stop"

$LLVMLit = Join-Path $BuildDir "bin\llvm-lit.py"
$BoltTestDir = Join-Path $BuildDir "tools\bolt\test"
$Python = "python"

# We need the process-debug-line shim on Windows because the original
# is a Unix shell script. If someone already created it, great. If not,
# we make a minimal .bat so lit does not fail during tool resolution.
$shimPath = Join-Path $BuildDir "bin\process-debug-line.bat"
if (-not (Test-Path $shimPath)) {
    Write-Host "Creating process-debug-line.bat shim (lit needs it on Windows)..." -ForegroundColor Yellow
    "@echo off`r`nrem Stub for Windows - the real script is a Unix shell script" | Set-Content $shimPath
}

# Check that the lit site config exists. If LLVM_INCLUDE_TESTS was OFF
# during the cmake run, this file might not exist and we cannot run lit.
$siteConfig = Join-Path $BoltTestDir "lit.site.cfg.py"
if (-not (Test-Path $siteConfig)) {
    Write-Host "ERROR: $siteConfig not found." -ForegroundColor Red
    Write-Host "This usually means LLVM_INCLUDE_TESTS was OFF when you ran cmake." -ForegroundColor Red
    Write-Host "Run: cmake . -DLLVM_INCLUDE_TESTS=ON  (from your build dir)" -ForegroundColor Red
    Write-Host ""
    Write-Host "If that fails with dependency errors, you can still run the" -ForegroundColor Yellow
    Write-Host "PECOFF tests directly with:" -ForegroundColor Yellow
    Write-Host "  python $LLVMLit -v $BoltTestDir\X86\PECOFF\" -ForegroundColor Cyan
    exit 1
}

$Failures = 0

function Run-TestSuite {
    param([string]$Name, [string]$TestPath)

    Write-Host ""
    Write-Host "Running $Name..." -ForegroundColor Cyan
    Write-Host ("-" * 60)

    if (-not (Test-Path $TestPath)) {
        Write-Host "  [SKIP] Test path not found: $TestPath" -ForegroundColor Yellow
        return
    }

    $output = & $Python $LLVMLit -v $TestPath 2>&1 | Out-String
    $code = $LASTEXITCODE

    # Show the summary line
    $output -split "`n" | Where-Object { $_ -match "Passed|Failed|Total|FAIL:|PASS:" } | ForEach-Object {
        $color = if ($_ -match "FAIL") { "Red" } else { "Green" }
        Write-Host "  $_" -ForegroundColor $color
    }

    if ($code -ne 0) {
        Write-Host "  [FAIL] $Name returned exit code $code" -ForegroundColor Red
        $script:Failures++
    } else {
        Write-Host "  [OK] $Name passed" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "BOLT Regression Test Suite" -ForegroundColor Cyan
Write-Host ("=" * 60)

# The PE/COFF tests are the ones we care about most.
Run-TestSuite "PE/COFF lit tests" (Join-Path $BoltTestDir "X86\PECOFF")

# If the full BOLT test suite exists, run it too. This catches any ELF
# regressions from our JITLink changes. On Windows many of these tests
# will be unsupported (they need ELF binaries) but the ones that do run
# should still pass.
$allBoltTests = Join-Path $BoltTestDir "X86"
if (Test-Path $allBoltTests) {
    Run-TestSuite "All BOLT X86 lit tests" $allBoltTests
}

Write-Host ""
Write-Host ("=" * 60)
if ($Failures -eq 0) {
    Write-Host "No regressions detected." -ForegroundColor Green
} else {
    Write-Host "$Failures test suite(s) FAILED." -ForegroundColor Red
}
Write-Host ""
exit $Failures
