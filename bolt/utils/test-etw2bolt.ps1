#!/usr/bin/env pwsh
# test-etw2bolt.ps1 -- Test the etw2bolt tool end-to-end
#
# Tests three things:
#   1. etw2bolt converts synthetic branch CSV to valid fdata
#   2. llvm-bolt accepts the fdata and produces a valid binary
#   3. The optimized binary runs correctly
#
# Usage:
#   .\test-etw2bolt.ps1
#   .\test-etw2bolt.ps1 -Z3Path D:\z3\build\release\z3.exe

param(
    [string]$BuildDir = "D:\llvm-upstream\llvm-project\build",
    [string]$Z3Path = "D:\z3\build\release\z3.exe"
)

$ErrorActionPreference = "Stop"

$Etw2Bolt = Join-Path $BuildDir "bin\etw2bolt.exe"
$Bolt     = Join-Path $BuildDir "bin\llvm-bolt.exe"
$Yaml2Obj = Join-Path $BuildDir "bin\yaml2obj.exe"
$YamlInput = Join-Path $BuildDir "..\bolt\test\X86\PECOFF\Inputs\two-funcs.yaml"

foreach ($tool in @($Etw2Bolt, $Bolt, $Yaml2Obj)) {
    if (-not (Test-Path $tool)) {
        Write-Host "Missing tool: $tool" -ForegroundColor Red
        exit 1
    }
}

$WorkDir = Join-Path ([IO.Path]::GetTempPath()) "bolt-etw2bolt-test"
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
Write-Host "etw2bolt End-to-End Test" -ForegroundColor Cyan
Write-Host ("=" * 60)

# ------------------------------------------------------------------
# Part 1: Convert synthetic branches on two-funcs.exe
# ------------------------------------------------------------------
Write-Host "`nPart 1: etw2bolt on two-funcs.exe" -ForegroundColor Yellow

& $Yaml2Obj $YamlInput -o "$WorkDir\two-funcs.exe" 2>&1 | Out-Null

# two-funcs has functions at 0x140001000 and 0x140001030
# Create synthetic branch data between them
$csv = @"
# Synthetic ETW LBR branch records for two-funcs.exe
# from_address,to_address,mispredicted
0x140001038,0x140001000,0
0x140001038,0x140001000,0
0x140001038,0x140001000,0
0x140001038,0x140001000,1
0x140001000,0x140001030,0
0x140001000,0x140001030,0
"@
[IO.File]::WriteAllText("$WorkDir\branches.csv", $csv, [Text.UTF8Encoding]::new($false))

$output = & $Etw2Bolt "-exe=$WorkDir\two-funcs.exe" "-csv=$WorkDir\branches.csv" "-o=$WorkDir\two-funcs.fdata" -v 2>&1 | Out-String
Write-Check "etw2bolt exits cleanly" ($LASTEXITCODE -eq 0)
Write-Check "Functions loaded from .pdata" ($output -match "loaded \d+ functions")
Write-Check "Branch records matched" ($output -match "\d+ matched to functions")

if (Test-Path "$WorkDir\two-funcs.fdata") {
    $fdata = Get-Content "$WorkDir\two-funcs.fdata"
    Write-Check "fdata file is not empty" ($fdata.Count -gt 0)
    Write-Check "fdata has correct format" ($fdata[0] -match "^1 func_0x\w+ \w+ 1 func_0x\w+ \w+ \d+ \d+$")

    # Show the fdata for manual inspection
    Write-Host "  fdata content:" -ForegroundColor DarkGray
    $fdata | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
} else {
    Write-Check "fdata file created" $false
}

# ------------------------------------------------------------------
# Part 2: Feed fdata into llvm-bolt
# ------------------------------------------------------------------
Write-Host "`nPart 2: llvm-bolt with etw2bolt output" -ForegroundColor Yellow

if (Test-Path "$WorkDir\two-funcs.fdata") {
    $boltOut = & $Bolt "$WorkDir\two-funcs.exe" -o "$WorkDir\two-funcs-opt.exe" `
        "-data=$WorkDir\two-funcs.fdata" -reorder-blocks=ext-tsp 2>&1 | Out-String
    Write-Check "llvm-bolt accepts fdata" ($LASTEXITCODE -eq 0)
    Write-Check "No errors in output" (-not ($boltOut -match "ERROR"))
    Write-Check "Relocations resolved" ($boltOut -match "resolved relocations")
}

# ------------------------------------------------------------------
# Part 3: etw2bolt on z3.exe (real binary)
# ------------------------------------------------------------------
Write-Host "`nPart 3: etw2bolt on z3.exe" -ForegroundColor Yellow

if (-not (Test-Path $Z3Path)) {
    Write-Host "  [SKIP] z3.exe not found at $Z3Path" -ForegroundColor Yellow
} else {
    # Synthetic branches hitting real z3 functions
    $z3csv = @"
# Synthetic branch records for z3.exe
0x1400010c0,0x140001220,0
0x140001220,0x1400010c0,0
0x1400010c0,0x140001220,1
0x140001400,0x1400010c0,0
0x140001400,0x1400010c0,0
0x140001400,0x1400010c0,0
0x1400010c0,0x140001400,0
0x140002df0,0x140001400,0
0x140002df0,0x140001400,0
0x140004d60,0x140004e50,0
0x140004e50,0x140004d60,0
0x140004e50,0x140004d60,0
0x140004e50,0x140004d60,1
"@
    [IO.File]::WriteAllText("$WorkDir\z3-branches.csv", $z3csv, [Text.UTF8Encoding]::new($false))

    $z3out = & $Etw2Bolt "-exe=$Z3Path" "-csv=$WorkDir\z3-branches.csv" "-o=$WorkDir\z3.fdata" -v 2>&1 | Out-String
    Write-Check "etw2bolt on z3 succeeds" ($LASTEXITCODE -eq 0)
    Write-Check "All 13 records matched" ($z3out -match "13 matched to functions")
    Write-Check "34K+ functions loaded" ($z3out -match "loaded 3\d{4} functions")

    if (Test-Path "$WorkDir\z3.fdata") {
        $z3fdata = Get-Content "$WorkDir\z3.fdata"
        Write-Check "z3 fdata has edges" ($z3fdata.Count -gt 0)

        # Run through llvm-bolt
        $outLog = "$WorkDir\z3-bolt.txt"
        $outExe = "$WorkDir\z3-opt.exe"
        $proc = Start-Process -FilePath "cmd.exe" `
            -ArgumentList "/c `"$Bolt `"$Z3Path`" -o `"$outExe`" -data=`"$WorkDir\z3.fdata`" -reorder-blocks=ext-tsp > `"$outLog`" 2>&1`"" `
            -PassThru -NoNewWindow
        $proc.WaitForExit(120000) | Out-Null
        Write-Check "llvm-bolt on z3 succeeds" ($proc.ExitCode -eq 0)

        if (Test-Path $outExe) {
            # Verify the optimized binary works
            $smt = "(set-logic QF_LIA)`n(declare-const x Int)`n(assert (> x 5))`n(check-sat)`n(exit)"
            [IO.File]::WriteAllText("$WorkDir\test.smt2", $smt, [Text.UTF8Encoding]::new($false))
            $result = & $outExe "$WorkDir\test.smt2" 2>&1 | Out-String
            Write-Check "Optimized z3 runs correctly" ($result.Trim() -eq "sat")
        }
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
