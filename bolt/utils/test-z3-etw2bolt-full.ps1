#!/usr/bin/env pwsh
# test-z3-etw2bolt-full.ps1 -- Full etw2bolt pipeline test on z3.exe
#
# This script runs the complete pipeline on z3.exe:
#   1. Run z3 through BOLT to discover functions
#   2. Build a realistic profile from the discovered functions
#   3. Optimize z3 with the profile
#   4. Verify the optimized binary produces correct results
#   5. Run z3 through the etw2bolt symlink (aggregate-only mode)
#
# Usage:
#   .\test-z3-etw2bolt-full.ps1
#   .\test-z3-etw2bolt-full.ps1 -Z3Path D:\z3\build\release\z3.exe

param(
    [string]$BuildDir = "D:\llvm-upstream\llvm-project\build",
    [string]$Z3Path = "D:\z3\build\release\z3.exe"
)

$ErrorActionPreference = "Stop"

$Bolt     = Join-Path $BuildDir "bin\llvm-bolt.exe"
$Etw2Bolt = Join-Path $BuildDir "bin\etw2bolt.exe"

if (-not (Test-Path $Bolt)) { Write-Host "Missing: $Bolt" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Z3Path)) { Write-Host "Missing: $Z3Path" -ForegroundColor Red; exit 1 }

$WorkDir = Join-Path ([IO.Path]::GetTempPath()) "bolt-z3-etw2bolt-full"
if (Test-Path $WorkDir) { Remove-Item $WorkDir -Recurse -Force }
New-Item -ItemType Directory $WorkDir | Out-Null

$Pass = 0; $Fail = 0

function Check {
    param([string]$Label, [bool]$Ok)
    if ($Ok) { Write-Host "  PASS: $Label" -ForegroundColor Green; $script:Pass++ }
    else     { Write-Host "  FAIL: $Label" -ForegroundColor Red; $script:Fail++ }
}

Write-Host ""
Write-Host "Z3 etw2bolt Full Pipeline Test" -ForegroundColor Cyan
Write-Host ("=" * 60)

# ------------------------------------------------------------------
# Step 1: Identity rewrite to verify z3 goes through BOLT cleanly
# ------------------------------------------------------------------
Write-Host "`n--- Step 1: Identity rewrite ---" -ForegroundColor Yellow

$outLog = "$WorkDir\z3-identity.txt"
$idExe = "$WorkDir\z3-id.exe"
$proc = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c `"$Bolt `"$Z3Path`" -o `"$idExe`" > `"$outLog`" 2>&1`"" `
    -PassThru -NoNewWindow
$proc.WaitForExit(120000) | Out-Null
Check "identity rewrite succeeds" ($proc.ExitCode -eq 0)

$origHash = (Get-FileHash $Z3Path -Algorithm SHA256).Hash
$idHash   = (Get-FileHash $idExe -Algorithm SHA256).Hash
Check "identity is byte-identical" ($origHash -eq $idHash)

# Quick SMT test
$smt = "(set-logic QF_LIA)`n(declare-const x Int)`n(assert (> x 5))`n(check-sat)`n(exit)"
[IO.File]::WriteAllText("$WorkDir\test.smt2", $smt, [Text.UTF8Encoding]::new($false))
$result = & $idExe "$WorkDir\test.smt2" 2>&1 | Out-String
Check "identity z3 produces sat" ($result.Trim() -eq "sat")

# ------------------------------------------------------------------
# Step 2: Build realistic profile from z3 function addresses
# ------------------------------------------------------------------
Write-Host "`n--- Step 2: Build profile data ---" -ForegroundColor Yellow

# Grab function names from the BOLT output
$boltLog = Get-Content $outLog
$funcNames = [regex]::Matches(($boltLog -join "`n"), "func_0x([0-9a-f]+)") |
    ForEach-Object { $_.Value } | Sort-Object -Unique

Write-Host "  Found $($funcNames.Count) unique functions"
Check "discovered functions" ($funcNames.Count -gt 100)

# Build a profile with edges between consecutive function pairs.
# This simulates what etw2bolt would produce from real ETW data.
$fdataLines = @()
for ($i = 0; $i -lt [Math]::Min(200, $funcNames.Count - 1); $i++) {
    $from = $funcNames[$i]
    $to = $funcNames[$i + 1]
    $count = 1000 - ($i * 4)
    if ($count -lt 10) { $count = 10 }
    $fdataLines += "1 $from 0 1 $to 0 0 $count"
}

$fdataPath = "$WorkDir\z3-profile.fdata"
[IO.File]::WriteAllText($fdataPath, ($fdataLines -join "`n") + "`n", [Text.UTF8Encoding]::new($false))
Write-Host "  Created profile with $($fdataLines.Count) branch edges"
Check "profile fdata created" ($fdataLines.Count -gt 50)

# ------------------------------------------------------------------
# Step 3: Optimize z3 with profile
# ------------------------------------------------------------------
Write-Host "`n--- Step 3: Profile-guided optimization ---" -ForegroundColor Yellow

$optLog = "$WorkDir\z3-opt.txt"
$optExe = "$WorkDir\z3-opt.exe"
$proc = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c `"$Bolt `"$Z3Path`" -o `"$optExe`" -data=`"$fdataPath`" -reorder-blocks=ext-tsp > `"$optLog`" 2>&1`"" `
    -PassThru -NoNewWindow
$proc.WaitForExit(300000) | Out-Null
Check "bolt optimization succeeds" ($proc.ExitCode -eq 0)

$optOutput = Get-Content $optLog
$resolvedLine = $optOutput | Where-Object { $_ -match "resolved relocations" }
$rewrittenLine = $optOutput | Where-Object { $_ -match "functions rewritten" }
$modifiedLine = $optOutput | Where-Object { $_ -match "functions had layout modified" }
Write-Host "  $resolvedLine"
Write-Host "  $modifiedLine"
Write-Host "  $rewrittenLine"

Check "relocations resolved" ($resolvedLine -match "resolved relocations for \d+ functions")

# ------------------------------------------------------------------
# Step 4: Verify optimized binary works
# ------------------------------------------------------------------
Write-Host "`n--- Step 4: Verify optimized binary ---" -ForegroundColor Yellow

if (Test-Path $optExe) {
    $optResult = & $optExe "$WorkDir\test.smt2" 2>&1 | Out-String
    Check "optimized z3 produces sat" ($optResult.Trim() -eq "sat")

    $optSize = (Get-Item $optExe).Length
    $origSize = (Get-Item $Z3Path).Length
    Check "output size matches original" ($optSize -eq $origSize)
} else {
    Check "optimized binary exists" $false
}

# ------------------------------------------------------------------
# Step 5: Test etw2bolt symlink (aggregate-only mode)
# ------------------------------------------------------------------
Write-Host "`n--- Step 5: Test etw2bolt help and validation ---" -ForegroundColor Yellow

$helpOut = & $Etw2Bolt --help 2>&1 | Out-String
Check "etw2bolt shows help" ($helpOut -match "BOLT ETW data aggregator")
Check "etw2bolt has -etwdata option" ($helpOut -match "etwdata")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 60)
Write-Host "Results: $Pass passed, $Fail failed" -ForegroundColor $(if ($Fail -eq 0) { "Green" } else { "Red" })

Remove-Item $WorkDir -Recurse -Force
exit $Fail
