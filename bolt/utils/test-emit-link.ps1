#!/usr/bin/env pwsh
# test-emit-link.ps1 -- Verify that BOLT can emit and link optimized COFF code
#
# This is the gate test for Phase 10a (JITLink COFF fix).  It builds a small
# PE binary from YAML, creates a fake profile, and runs BOLT with block
# reordering enabled.  If JITLink can handle the emitted relocations and the
# output binary has valid PE headers, we are good.
#
# Usage:
#   .\test-emit-link.ps1
#   .\test-emit-link.ps1 -BuildDir D:\llvm\build

param(
    [string]$BuildDir = "D:\llvm-upstream\llvm-project\build"
)

$ErrorActionPreference = "Stop"

$Bolt     = Join-Path $BuildDir "bin\llvm-bolt.exe"
$Yaml2Obj = Join-Path $BuildDir "bin\yaml2obj.exe"
$ReadObj  = Join-Path $BuildDir "bin\llvm-readobj.exe"
$ObjDump  = Join-Path $BuildDir "bin\llvm-objdump.exe"

foreach ($tool in @($Bolt, $Yaml2Obj, $ReadObj, $ObjDump)) {
    if (-not (Test-Path $tool)) {
        Write-Host "Missing tool: $tool" -ForegroundColor Red
        exit 1
    }
}

$YamlInput = Join-Path $BuildDir "..\bolt\test\X86\PECOFF\Inputs\two-funcs.yaml"
if (-not (Test-Path $YamlInput)) {
    Write-Host "Missing test input: $YamlInput" -ForegroundColor Red
    exit 1
}

$WorkDir = Join-Path ([IO.Path]::GetTempPath()) "bolt-emit-link-test"
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
Write-Host "BOLT Emit+Link Test (Phase 10a)" -ForegroundColor Cyan
Write-Host ("=" * 50)

# Step 1: Build the test binary from YAML.
Write-Host "`nBuilding test binary..." -ForegroundColor Yellow
& $Yaml2Obj $YamlInput -o "$WorkDir\test.exe" 2>&1 | Out-Null
Write-Check "yaml2obj produced test.exe" (Test-Path "$WorkDir\test.exe")

# Step 2: Create a minimal fdata profile.
# The profile says "the call from func_0x140001030 offset 0x1b to func_0x140001000
# was taken 100 times".  The exact numbers do not matter much -- we just need the
# profile reader to accept it and mark the function as having profile data.
$profile = "1 func_0x140001030 1b 1 func_0x140001000 0 0 100`n"
[IO.File]::WriteAllText("$WorkDir\test.fdata", $profile, [Text.UTF8Encoding]::new($false))
Write-Check "Profile created" (Test-Path "$WorkDir\test.fdata")

# Step 3: Run BOLT with profile and block reordering.
Write-Host "`nRunning BOLT with profile..." -ForegroundColor Yellow
$boltOutput = & $Bolt "$WorkDir\test.exe" -o "$WorkDir\test-opt.exe" `
    -data="$WorkDir\test.fdata" -reorder-blocks=ext-tsp -v=1 2>&1

$exitCode = $LASTEXITCODE
$boltText = $boltOutput | Out-String

Write-Check "BOLT exit code is 0" ($exitCode -eq 0) "got $exitCode"
Write-Check "No JITLink errors" (-not ($boltText -match "JITLink failed")) ""
Write-Check "Functions rewritten" ($boltText -match "functions rewritten") ""
Write-Check "Output file created" (Test-Path "$WorkDir\test-opt.exe") ""

# Step 4: Verify the output is a valid PE.
if (Test-Path "$WorkDir\test-opt.exe") {
    Write-Host "`nValidating output PE..." -ForegroundColor Yellow

    $headers = & $ReadObj --file-headers "$WorkDir\test-opt.exe" 2>&1 | Out-String
    Write-Check "readobj can parse output" ($LASTEXITCODE -eq 0) ""
    Write-Check "Has AMD64 machine type" ($headers -match "IMAGE_FILE_MACHINE_AMD64") ""
    Write-Check "Has PE32+ magic" ($headers -match "0x20B") ""

    $sections = & $ReadObj --sections "$WorkDir\test-opt.exe" 2>&1 | Out-String
    Write-Check "Has .text section" ($sections -match "\.text") ""

    # Step 5: Make sure disassembly works.
    $disasm = & $ObjDump -d "$WorkDir\test-opt.exe" 2>&1 | Out-String
    Write-Check "Disassembly succeeds" ($disasm -match "pushq") ""

    # Step 6: Same file size as original.
    $origSize = (Get-Item "$WorkDir\test.exe").Length
    $optSize = (Get-Item "$WorkDir\test-opt.exe").Length
    Write-Check "File size preserved" ($origSize -eq $optSize) "orig=$origSize opt=$optSize"
}

# Step 7: Verify identity rewrite still works (no profile case).
Write-Host "`nRegression: identity rewrite..." -ForegroundColor Yellow
& $Bolt "$WorkDir\test.exe" -o "$WorkDir\test-id.exe" 2>&1 | Out-Null
if (Test-Path "$WorkDir\test-id.exe") {
    $origHash = (Get-FileHash "$WorkDir\test.exe" -Algorithm SHA256).Hash
    $idHash = (Get-FileHash "$WorkDir\test-id.exe" -Algorithm SHA256).Hash
    Write-Check "Identity copy byte-identical" ($origHash -eq $idHash) ""
}

# Cleanup.
Remove-Item $WorkDir -Recurse -Force

Write-Host ""
Write-Host ("=" * 50)
if ($Failures -eq 0) {
    Write-Host "All checks passed." -ForegroundColor Green
} else {
    Write-Host "$Failures check(s) FAILED." -ForegroundColor Red
}
Write-Host ""
exit $Failures
