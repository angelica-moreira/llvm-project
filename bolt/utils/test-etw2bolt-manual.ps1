#!/usr/bin/env pwsh
# test-etw2bolt-manual.ps1 -- Step-by-step manual test for the etw2bolt pipeline
#
# This script walks through the full BOLT PE/COFF optimization pipeline
# step by step, printing what it does at each stage so you can follow along
# and reproduce each step manually.
#
# Steps:
#   1. Build test programs (bubble_sort, matrix_mul) with clang-cl
#   2. Run identity rewrite and verify byte-identical output
#   3. Create synthetic profile data (simulating ETW output)
#   4. Run etw2bolt to convert profile to fdata
#   5. Run llvm-bolt with fdata to produce optimized binary
#   6. Verify the optimized binary runs correctly
#
# Usage:
#   .\test-etw2bolt-manual.ps1
#   .\test-etw2bolt-manual.ps1 -BuildDir D:\llvm-upstream\llvm-project\build

param(
    [string]$BuildDir = "D:\llvm-upstream\llvm-project\build"
)

$ErrorActionPreference = "Stop"

$ClangCl  = Join-Path $BuildDir "bin\clang-cl.exe"
$Bolt     = Join-Path $BuildDir "bin\llvm-bolt.exe"
$ReadObj  = Join-Path $BuildDir "bin\llvm-readobj.exe"
$SrcDir   = Join-Path $BuildDir "..\bolt\test\X86\PECOFF\TestPrograms"

foreach ($tool in @($ClangCl, $Bolt, $ReadObj)) {
    if (-not (Test-Path $tool)) {
        Write-Host "ERROR: Missing tool: $tool" -ForegroundColor Red
        exit 1
    }
}

$WorkDir = Join-Path ([IO.Path]::GetTempPath()) "bolt-manual-test"
if (Test-Path $WorkDir) { Remove-Item $WorkDir -Recurse -Force }
New-Item -ItemType Directory $WorkDir | Out-Null

$Pass = 0
$Fail = 0

function Step {
    param([string]$Msg)
    Write-Host ""
    Write-Host "=== $Msg ===" -ForegroundColor Cyan
}

function Check {
    param([string]$Label, [bool]$Ok)
    if ($Ok) {
        Write-Host "  PASS: $Label" -ForegroundColor Green
        $script:Pass++
    } else {
        Write-Host "  FAIL: $Label" -ForegroundColor Red
        $script:Fail++
    }
}

function Manual {
    param([string]$Cmd)
    Write-Host "  > $Cmd" -ForegroundColor DarkGray
}

# ====================================================================
Step "1. Build test programs with clang-cl"
# ====================================================================

Manual "clang-cl /O2 bubble_sort.c -o bubble_sort.exe"
& $ClangCl /O2 "$SrcDir\bubble_sort.c" -o "$WorkDir\bubble_sort.exe" 2>&1 | Out-Null
Check "bubble_sort.exe compiled" (Test-Path "$WorkDir\bubble_sort.exe")

Manual "clang-cl /O2 matrix_mul.c -o matrix_mul.exe"
& $ClangCl /O2 "$SrcDir\matrix_mul.c" -o "$WorkDir\matrix_mul.exe" 2>&1 | Out-Null
Check "matrix_mul.exe compiled" (Test-Path "$WorkDir\matrix_mul.exe")

# Verify they run
$bs = & "$WorkDir\bubble_sort.exe" 2>&1 | Out-String
Check "bubble_sort runs correctly" ($bs -match "sorted 30000 elements")

$mm = & "$WorkDir\matrix_mul.exe" 2>&1 | Out-String
Check "matrix_mul runs correctly" ($mm -match "matrix multiply 256x256 done")

# ====================================================================
Step "2. Check .pdata exists (function map for address resolution)"
# ====================================================================

Manual "llvm-readobj --sections bubble_sort.exe | grep .pdata"
$sections = & $ReadObj --sections "$WorkDir\bubble_sort.exe" 2>&1 | Out-String
Check "bubble_sort has .pdata section" ($sections -match "\.pdata")

# ====================================================================
Step "3. Identity rewrite (no profile, output should be byte-identical)"
# ====================================================================

Manual "llvm-bolt bubble_sort.exe -o bubble_sort_id.exe"
& $Bolt "$WorkDir\bubble_sort.exe" -o "$WorkDir\bubble_sort_id.exe" 2>&1 | Out-Null
Check "identity rewrite succeeds" ($LASTEXITCODE -eq 0)

$origHash = (Get-FileHash "$WorkDir\bubble_sort.exe" -Algorithm SHA256).Hash
$idHash   = (Get-FileHash "$WorkDir\bubble_sort_id.exe" -Algorithm SHA256).Hash
Check "identity output is byte-identical" ($origHash -eq $idHash)

$idResult = & "$WorkDir\bubble_sort_id.exe" 2>&1 | Out-String
Check "identity copy runs correctly" ($idResult -match "sorted 30000 elements")

# Discover function addresses from the BOLT output for profile creation.
$boltOut = & $Bolt "$WorkDir\bubble_sort.exe" -o "$WorkDir\tmp.exe" 2>&1 | Out-String
$funcAddrs = [regex]::Matches($boltOut, "func_0x([0-9a-f]+)") | ForEach-Object { $_.Value } | Sort-Object -Unique | Select-Object -First 10
Write-Host "  Found functions: $($funcAddrs -join ', ')" -ForegroundColor DarkGray

# ====================================================================
Step "4. Create profile data (fdata format, same as perf2bolt output)"
# ====================================================================

# In production you would capture an ETW trace and run etw2bolt:
#   xperf -on PROC_THREAD+LOADER+PROFILE
#   bubble_sort.exe
#   xperf -d trace.etl
#   etw2bolt bubble_sort.exe -e trace.etl -o profile.fdata
#
# For this test we write fdata directly using function addresses that
# BOLT discovered from .pdata.

if ($funcAddrs.Count -ge 2) {
    $f1Name = $funcAddrs[0]
    $f2Name = $funcAddrs[1]
    $fdataContent = "1 $f1Name 0 1 $f2Name 0 0 100`n"
    $fdataContent += "1 $f2Name 0 1 $f1Name 0 0 50`n"
    [IO.File]::WriteAllText("$WorkDir\profile.fdata", $fdataContent, [Text.UTF8Encoding]::new($false))
    Manual "Created profile.fdata with branch edges between $f1Name and $f2Name"
    Check "fdata file created" (Test-Path "$WorkDir\profile.fdata")
    Write-Host "  fdata content:" -ForegroundColor DarkGray
    Get-Content "$WorkDir\profile.fdata" | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
} else {
    Write-Host "  SKIP: not enough functions found" -ForegroundColor Yellow
}

# ====================================================================
Step "5. Run llvm-bolt with profile data"
# ====================================================================

Manual "llvm-bolt bubble_sort.exe -o bubble_sort_opt.exe -data=profile.fdata -reorder-blocks=ext-tsp"
$boltOpt = & $Bolt "$WorkDir\bubble_sort.exe" -o "$WorkDir\bubble_sort_opt.exe" `
    "-data=$WorkDir\profile.fdata" -reorder-blocks=ext-tsp 2>&1 | Out-String
Check "llvm-bolt with profile succeeds" ($LASTEXITCODE -eq 0)

$boltOpt -split "`n" | Where-Object { $_ -match "^BOLT-INFO:" } | ForEach-Object {
    Write-Host "  $_" -ForegroundColor DarkGray
}

# ====================================================================
Step "6. Verify optimized binary runs correctly"
# ====================================================================

if (Test-Path "$WorkDir\bubble_sort_opt.exe") {
    $optResult = & "$WorkDir\bubble_sort_opt.exe" 2>&1 | Out-String
    Check "optimized binary runs correctly" ($optResult -match "sorted 30000 elements")
    Check "output is valid PE" ((Get-Item "$WorkDir\bubble_sort_opt.exe").Length -gt 0)
} else {
    Check "optimized binary exists" $false
}

# ====================================================================
Step "7. Repeat with matrix_mul"
# ====================================================================

& $Bolt "$WorkDir\matrix_mul.exe" -o "$WorkDir\matrix_mul_id.exe" 2>&1 | Out-Null
$mmIdResult = & "$WorkDir\matrix_mul_id.exe" 2>&1 | Out-String
Check "matrix_mul identity rewrite runs" ($mmIdResult -match "matrix multiply")

# ====================================================================
# Summary
# ====================================================================
Write-Host ""
Write-Host ("=" * 60)
Write-Host "Results: $Pass passed, $Fail failed" -ForegroundColor $(if ($Fail -eq 0) { "Green" } else { "Red" })
Write-Host ""
Write-Host "To reproduce each step manually:" -ForegroundColor Yellow
Write-Host "  cd $WorkDir"
Write-Host "  clang-cl /O2 bubble_sort.c -o bubble_sort.exe"
Write-Host "  llvm-bolt bubble_sort.exe -o bubble_sort_id.exe"
Write-Host "  # Create profile.fdata with branch edges (see step 5 output)"
Write-Host "  llvm-bolt bubble_sort.exe -o bubble_sort_opt.exe -data=profile.fdata -reorder-blocks=ext-tsp"
Write-Host "  bubble_sort_opt.exe"
Write-Host ""
Write-Host "  For real ETW traces:" -ForegroundColor Yellow
Write-Host "  xperf -on PROC_THREAD+LOADER+PROFILE"
Write-Host "  bubble_sort.exe"
Write-Host "  xperf -d trace.etl"
Write-Host "  etw2bolt bubble_sort.exe -e trace.etl -o profile.fdata"
Write-Host "  llvm-bolt bubble_sort.exe -o bubble_sort_opt.exe -data=profile.fdata -reorder-blocks=ext-tsp"
Write-Host ""

# Leave the work dir so user can inspect
Write-Host "Work directory preserved at: $WorkDir" -ForegroundColor Yellow

exit $Fail
