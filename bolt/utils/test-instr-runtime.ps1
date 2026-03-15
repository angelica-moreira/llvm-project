#!/usr/bin/env pwsh
# test-instr-runtime.ps1 -- Test the Windows instrumentation runtime library
#
# STATUS: NOT YET IMPLEMENTED
#
# This script will be usable once Phase 10c (Windows instrumentation runtime)
# is complete. It needs the bolt_rt_instr_windows CMake target to exist.
#
# What it will test:
#   1. Build the runtime library (bolt_rt_instr_windows target)
#   2. Verify the .lib/.obj was produced
#   3. Link a tiny test harness that calls __bolt_instr_setup / __bolt_instr_fini
#   4. Run the harness and check that prof.fdata was created
#   5. Verify the fdata file is not empty and has valid content
#
# The runtime lives in bolt/runtime/sys_windows_x86_64.h and provides
# freestanding wrappers around Windows API calls (WriteFile, VirtualAlloc,
# GetCurrentProcessId, etc). It must not depend on the CRT.
#
# Usage (once implemented):
#   .\test-instr-runtime.ps1
#   .\test-instr-runtime.ps1 -BuildDir D:\llvm\build

param(
    [string]$BuildDir = "D:\llvm-upstream\llvm-project\build"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "Windows Instrumentation Runtime Test" -ForegroundColor Cyan
Write-Host ("=" * 60)

# Check if the runtime target exists
$runtimeLib = Join-Path $BuildDir "lib\bolt_rt_instr_windows.lib"
if (-not (Test-Path $runtimeLib)) {
    Write-Host ""
    Write-Host "  The Windows instrumentation runtime has not been built yet." -ForegroundColor Yellow
    Write-Host "  This is expected if Phase 10c is not complete." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  To implement it:" -ForegroundColor Cyan
    Write-Host "    1. Create bolt/runtime/sys_windows_x86_64.h" -ForegroundColor Cyan
    Write-Host "    2. Add bolt_rt_instr_windows target to bolt/runtime/CMakeLists.txt" -ForegroundColor Cyan
    Write-Host "    3. Build with: ninja bolt_rt_instr_windows" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

# TODO: Once the runtime exists, add these checks:
#
# Step 1: Verify the .lib was produced
# Write-Check "Runtime library exists" (Test-Path $runtimeLib)
#
# Step 2: Create a minimal test harness
# $harness = @"
# extern void __bolt_instr_setup();
# extern void __bolt_instr_fini();
# int main() {
#     __bolt_instr_setup();
#     // do some work
#     __bolt_instr_fini();
#     return 0;
# }
# "@
#
# Step 3: Compile and link
# clang-cl /c test_harness.c
# lld-link test_harness.obj bolt_rt_instr_windows.lib /OUT:test_harness.exe
#
# Step 4: Run and check for prof.fdata
# & .\test_harness.exe
# Write-Check "prof.fdata created" (Test-Path "prof.fdata")
# Write-Check "prof.fdata not empty" ((Get-Item "prof.fdata").Length -gt 0)

Write-Host "  Runtime library found. Full test not yet implemented." -ForegroundColor Yellow
exit 0
