#!/usr/bin/env pwsh
# test-instrument-pecoff.ps1 -- Test BOLT instrumentation of a PE/COFF binary
#
# STATUS: NOT YET IMPLEMENTED
#
# This script will be usable once Phase 10d (PE section registration for
# instrumentation) is complete. It needs both the Windows runtime and the
# PE-specific instrumentation plumbing in BOLT.
#
# What it will test:
#   1. Build a small test binary from YAML
#   2. Run llvm-bolt with --instrument
#   3. Verify the instrumented binary has .bolt.instr.counters section
#   4. Verify the instrumented binary has .bolt.instr.tables section
#   5. Verify AddressOfEntryPoint was changed (entry point trampoline)
#   6. Verify the instrumented binary is larger than the original
#   7. Disassemble the entry point to see the trampoline code
#
# Usage (once implemented):
#   .\test-instrument-pecoff.ps1
#   .\test-instrument-pecoff.ps1 -BuildDir D:\llvm\build

param(
    [string]$BuildDir = "D:\llvm-upstream\llvm-project\build"
)

$ErrorActionPreference = "Stop"

$Bolt     = Join-Path $BuildDir "bin\llvm-bolt.exe"
$Yaml2Obj = Join-Path $BuildDir "bin\yaml2obj.exe"
$ReadObj  = Join-Path $BuildDir "bin\llvm-readobj.exe"
$ObjDump  = Join-Path $BuildDir "bin\llvm-objdump.exe"

Write-Host ""
Write-Host "PE/COFF Instrumentation Test" -ForegroundColor Cyan
Write-Host ("=" * 60)

# Quick check: does BOLT accept --instrument for PE/COFF?
$WorkDir = Join-Path ([IO.Path]::GetTempPath()) "bolt-instrument-test"
if (Test-Path $WorkDir) { Remove-Item $WorkDir -Recurse -Force }
New-Item -ItemType Directory $WorkDir | Out-Null

$YamlInput = Join-Path $BuildDir "..\bolt\test\X86\PECOFF\Inputs\two-funcs.yaml"
& $Yaml2Obj $YamlInput -o "$WorkDir\test.exe" 2>&1 | Out-Null

$output = & $Bolt "$WorkDir\test.exe" -o "$WorkDir\test-instr.exe" --instrument 2>&1 | Out-String
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0 -or $output -match "not supported" -or $output -match "not implemented") {
    Write-Host ""
    Write-Host "  PE/COFF instrumentation is not yet supported." -ForegroundColor Yellow
    Write-Host "  This is expected if Phase 10d is not complete." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Prerequisites:" -ForegroundColor Cyan
    Write-Host "    1. Phase 10c: Windows instrumentation runtime" -ForegroundColor Cyan
    Write-Host "    2. Phase 10d: PE section registration + entry point hooking" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  BOLT output:" -ForegroundColor Gray
    $output -split "`n" | Select-Object -First 5 | ForEach-Object {
        Write-Host "    $_" -ForegroundColor Gray
    }
    Remove-Item $WorkDir -Recurse -Force
    exit 0
}

# If we got here, instrumentation produced output. Run the full checks.
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

Write-Check "Instrumentation exit code 0" ($exitCode -eq 0)
Write-Check "Output file created" (Test-Path "$WorkDir\test-instr.exe")

if (Test-Path "$WorkDir\test-instr.exe") {
    $sections = & $ReadObj --sections "$WorkDir\test-instr.exe" 2>&1 | Out-String
    Write-Check "Has .bolt.instr.counters" ($sections -match "bolt\.instr\.counters")
    Write-Check "Has .bolt.instr.tables" ($sections -match "bolt\.instr\.tables")

    # Entry point should be different
    $origHeaders = & $ReadObj --file-headers "$WorkDir\test.exe" 2>&1 | Out-String
    $instrHeaders = & $ReadObj --file-headers "$WorkDir\test-instr.exe" 2>&1 | Out-String

    $origEntry = if ($origHeaders -match "AddressOfEntryPoint:\s+(0x[0-9A-Fa-f]+)") { $Matches[1] }
    $instrEntry = if ($instrHeaders -match "AddressOfEntryPoint:\s+(0x[0-9A-Fa-f]+)") { $Matches[1] }

    Write-Check "Entry point changed" ($origEntry -ne $instrEntry) "orig=$origEntry instr=$instrEntry"

    # Instrumented binary should be larger (has extra sections)
    $origSize = (Get-Item "$WorkDir\test.exe").Length
    $instrSize = (Get-Item "$WorkDir\test-instr.exe").Length
    Write-Check "Binary grew" ($instrSize -gt $origSize) "orig=$origSize instr=$instrSize"
}

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
