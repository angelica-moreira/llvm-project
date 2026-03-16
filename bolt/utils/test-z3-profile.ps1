#!/usr/bin/env pwsh
# test-z3-profile.ps1 -- Test BOLT profile-guided optimization on z3.exe
#
# This script tests three things:
#   1. Identity rewrite (no profile) -- should be byte-identical
#   2. Profile-guided optimization on a small binary (two-funcs)
#   3. Profile-guided optimization on z3.exe -- verifies the direct
#      relocation resolver works on a real binary with 13K+ functions
#
# Usage:
#   .\test-z3-profile.ps1 -Z3Path D:\z3\build\release\z3.exe
#   .\test-z3-profile.ps1 -Z3Path D:\z3\build\release\z3.exe -SkipLargeTest

param(
    [string]$BuildDir = "D:\llvm-upstream\llvm-project\build",
    [string]$Z3Path = "D:\z3\build\release\z3.exe",
    [int]$TimeoutSeconds = 300,
    [switch]$SkipLargeTest
)

$ErrorActionPreference = "Stop"

$Bolt     = Join-Path $BuildDir "bin\llvm-bolt.exe"
$Yaml2Obj = Join-Path $BuildDir "bin\yaml2obj.exe"
$ReadObj  = Join-Path $BuildDir "bin\llvm-readobj.exe"
$YamlInput = Join-Path $BuildDir "..\bolt\test\X86\PECOFF\Inputs\two-funcs.yaml"

foreach ($tool in @($Bolt, $Yaml2Obj, $ReadObj)) {
    if (-not (Test-Path $tool)) {
        Write-Host "Missing tool: $tool" -ForegroundColor Red
        exit 1
    }
}

$WorkDir = Join-Path ([IO.Path]::GetTempPath()) "bolt-z3-profile-test"
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
Write-Host "BOLT Profile-Guided Optimization Test" -ForegroundColor Cyan
Write-Host ("=" * 60)

# ------------------------------------------------------------------
# Part 1: Identity rewrite of z3 (baseline, no profile)
# ------------------------------------------------------------------
Write-Host "`nPart 1: Identity rewrite of z3.exe" -ForegroundColor Yellow

if (Test-Path $Z3Path) {
    & $Bolt $Z3Path -o "$WorkDir\z3-id.exe" 2>&1 | Out-Null
    Write-Check "z3 identity rewrite exits cleanly" ($LASTEXITCODE -eq 0)

    if (Test-Path "$WorkDir\z3-id.exe") {
        $origHash = (Get-FileHash $Z3Path -Algorithm SHA256).Hash
        $idHash   = (Get-FileHash "$WorkDir\z3-id.exe" -Algorithm SHA256).Hash
        Write-Check "Identity is byte-identical" ($origHash -eq $idHash)
    }
} else {
    Write-Host "  [SKIP] z3.exe not found at $Z3Path" -ForegroundColor Yellow
}

# ------------------------------------------------------------------
# Part 2: Profile-guided optimization on small binary (two-funcs)
# ------------------------------------------------------------------
Write-Host "`nPart 2: Profile optimization on two-funcs.exe" -ForegroundColor Yellow

& $Yaml2Obj $YamlInput -o "$WorkDir\two-funcs.exe" 2>&1 | Out-Null

$profile = "1 func_0x140001030 1b 1 func_0x140001000 0 0 100`n"
[IO.File]::WriteAllText("$WorkDir\two-funcs.fdata", $profile, [Text.UTF8Encoding]::new($false))

$output = & $Bolt "$WorkDir\two-funcs.exe" -o "$WorkDir\two-funcs-opt.exe" `
    "-data=$WorkDir\two-funcs.fdata" -reorder-blocks=ext-tsp 2>&1 | Out-String
Write-Check "two-funcs profile optimization succeeds" ($LASTEXITCODE -eq 0)
Write-Check "Relocations resolved" ($output -match "resolved relocations")
Write-Check "Functions were rewritten" ($output -match "functions rewritten")

# Validate the output PE
if (Test-Path "$WorkDir\two-funcs-opt.exe") {
    $headers = & $ReadObj --file-headers "$WorkDir\two-funcs-opt.exe" 2>&1 | Out-String
    Write-Check "Output has valid PE headers" ($headers -match "IMAGE_FILE_MACHINE_AMD64")
}

# ------------------------------------------------------------------
# Part 3: Profile-guided optimization on z3 with no layout changes
# ------------------------------------------------------------------
Write-Host "`nPart 3: z3 profile optimization (no layout changes)" -ForegroundColor Yellow

if ($SkipLargeTest) {
    Write-Host "  [SKIP] Large binary test skipped (-SkipLargeTest)" -ForegroundColor Yellow
} elseif (-not (Test-Path $Z3Path)) {
    Write-Host "  [SKIP] z3.exe not found at $Z3Path" -ForegroundColor Yellow
} else {
    # This profile targets two functions but won't cause layout changes
    # because the edge goes to the very next instruction. The output
    # binary should be byte-identical to the input.
    $z3profile = "1 func_0x1400010c0 0 1 func_0x140001220 0 0 500`n"
    [IO.File]::WriteAllText("$WorkDir\z3.fdata", $z3profile, [Text.UTF8Encoding]::new($false))

    $outLog = "$WorkDir\z3-output.txt"
    $outExe = "$WorkDir\z3-opt.exe"

    # Use cmd redirect to avoid PowerShell pipe deadlock on large output
    $proc = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c `"$Bolt `"$Z3Path`" -o `"$outExe`" -data=`"$WorkDir\z3.fdata`" -reorder-blocks=ext-tsp > `"$outLog`" 2>&1`"" `
        -PassThru -NoNewWindow

    $done = $proc.WaitForExit($TimeoutSeconds * 1000)
    if (-not $done) {
        Write-Check "z3 completes within timeout" $false "${TimeoutSeconds}s"
        try { $proc.Kill() } catch {}
    } else {
        Write-Check "z3 profile optimization succeeds" ($proc.ExitCode -eq 0)

        $logContent = Get-Content $outLog -Raw
        Write-Check "Relocations resolved" ($logContent -match "resolved relocations for \d+ functions")
        Write-Check "No layout changes means 0 functions rewritten" ($logContent -match "0 functions rewritten")

        # Output should be byte-identical when nothing changed
        $origHash = (Get-FileHash $Z3Path -Algorithm SHA256).Hash
        $optHash  = (Get-FileHash $outExe -Algorithm SHA256).Hash
        Write-Check "Output byte-identical (no layout changes)" ($origHash -eq $optHash)

        # Make sure the output actually works
        $smt = "(set-logic QF_LIA)`n(declare-const x Int)`n(assert (> x 5))`n(check-sat)`n(exit)"
        [IO.File]::WriteAllText("$WorkDir\test.smt2", $smt, [Text.UTF8Encoding]::new($false))
        $result = & $outExe "$WorkDir\test.smt2" 2>&1 | Out-String
        Write-Check "Optimized z3 produces correct output" ($result.Trim() -eq "sat")
    }
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
