#!/usr/bin/env pwsh
# cross-validate-pecoff.ps1 -- Compare what BOLT sees with raw binary parsing
#
# This script reads a PE/COFF binary three different ways and makes sure
# the numbers agree.  Useful after making changes to the BOLT PE reader
# to catch parsing bugs early.
#
#   Method 1: llvm-readobj --unwind (the reference implementation)
#   Method 2: llvm-bolt (our PE/COFF rewrite path)
#   Method 3: direct byte reading from the file (ground truth)
#
# Usage:
#   .\cross-validate-pecoff.ps1 -Binary C:\path\to\app.exe
#   .\cross-validate-pecoff.ps1 -Binary C:\path\to\app.exe -BuildDir D:\llvm\build

param(
    [Parameter(Mandatory)]
    [string]$Binary,

    [string]$BuildDir = "D:\llvm-upstream\llvm-project\build"
)

$ErrorActionPreference = "Stop"

# ---- tool paths ----

$ReadObj = Join-Path $BuildDir "bin\llvm-readobj.exe"
$Bolt    = Join-Path $BuildDir "bin\llvm-bolt.exe"

foreach ($tool in @($ReadObj, $Bolt)) {
    if (-not (Test-Path $tool)) {
        Write-Host "Cannot find $tool -- check your BuildDir" -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path $Binary)) {
    Write-Host "Binary not found: $Binary" -ForegroundColor Red
    exit 1
}

$BinaryName = Split-Path $Binary -Leaf

# ---- helpers ----

function Write-Check {
    param([string]$Label, [bool]$Ok, [string]$Detail = "")
    $mark = if ($Ok) { "[OK]" } else { "[FAIL]" }
    $color = if ($Ok) { "Green" } else { "Red" }
    $msg = "  $mark $Label"
    if ($Detail) { $msg += " ($Detail)" }
    Write-Host $msg -ForegroundColor $color
    if (-not $Ok) { $script:Failures++ }
}

$Failures = 0

Write-Host ""
Write-Host "Cross-validating $BinaryName" -ForegroundColor Cyan
Write-Host ("=" * 60)

# ======================================================================
# Step 1: Get section layout from llvm-readobj
# ======================================================================

Write-Host ""
Write-Host "Reading PE sections..." -ForegroundColor Yellow

$secOutput = & $ReadObj --sections $Binary 2>&1

# We need the .pdata and .rdata details to do raw parsing later.
# Parse each section into a hashtable keyed by name.
$sections = @{}
$currentSec = $null

foreach ($line in $secOutput) {
    $s = "$line".Trim()
    if ($s -match "^Name:\s+(\.\w+)") {
        $currentSec = $Matches[1]
        $sections[$currentSec] = @{}
    }
    elseif ($currentSec -and $s -match "^(VirtualSize|VirtualAddress|PointerToRawData):\s+(0x[0-9A-Fa-f]+)") {
        $sections[$currentSec][$Matches[1]] = [Convert]::ToUInt32($Matches[2], 16)
    }
    elseif ($currentSec -and $s -match "^RawDataSize:\s+(\d+)") {
        # llvm-readobj prints this as a plain decimal number.
        $sections[$currentSec]["RawDataSize"] = [uint32]$Matches[1]
    }
}

$secNames = ($sections.Keys | Sort-Object) -join ", "
Write-Host "  Sections found: $secNames"

if (-not $sections.ContainsKey(".pdata")) {
    Write-Host "  No .pdata section -- nothing to validate for exception handling." -ForegroundColor Yellow
    Write-Host "  (The binary might still be valid, just no SEH to cross-check.)"
    exit 0
}

$pdataVA   = $sections[".pdata"]["VirtualAddress"]
$pdataRaw  = $sections[".pdata"]["PointerToRawData"]
$pdataSize = $sections[".pdata"]["VirtualSize"]

Write-Host ("  .pdata: VA=0x{0:X}  FileOff=0x{1:X}  Size={2}" -f $pdataVA, $pdataRaw, $pdataSize)

# Figure out which section holds UNWIND_INFO.  Usually .xdata, but many
# binaries (like z3) put it in .rdata instead.
$unwindSec = $null
$unwindSecName = ""

if ($sections.ContainsKey(".xdata")) {
    $unwindSec = $sections[".xdata"]
    $unwindSecName = ".xdata"
}

# ======================================================================
# Step 2: Raw byte parsing -- read .pdata and UNWIND_INFO directly
# ======================================================================

Write-Host ""
Write-Host "Parsing raw bytes from $BinaryName..." -ForegroundColor Yellow

$fileBytes = [System.IO.File]::ReadAllBytes($Binary)
$totalEntries = [Math]::Floor($pdataSize / 12)

Write-Host "  File size: $($fileBytes.Length) bytes"
Write-Host "  Expected .pdata entries: $totalEntries (= $pdataSize / 12)"

# We might need to discover the unwind section from the first entry.
if (-not $unwindSec) {
    $firstUnwindRVA = [BitConverter]::ToUInt32($fileBytes, $pdataRaw + 8)
    # Walk sections to find which one contains this RVA.
    foreach ($name in $sections.Keys) {
        $sec = $sections[$name]
        $start = $sec["VirtualAddress"]
        $end   = $start + $sec["VirtualSize"]
        if ($firstUnwindRVA -ge $start -and $firstUnwindRVA -lt $end) {
            $unwindSec = $sec
            $unwindSecName = $name
            break
        }
    }
}

if (-not $unwindSec) {
    Write-Host "  Could not locate unwind info section -- skipping raw validation" -ForegroundColor Red
    $Failures++
} else {
    Write-Host "  Unwind info lives in $unwindSecName"

    $uwVA  = $unwindSec["VirtualAddress"]
    $uwRaw = $unwindSec["PointerToRawData"]
    $uwSize = $unwindSec["RawDataSize"]

    # Walk every .pdata entry and classify by unwind flags.
    $rawNone = 0
    $rawHandler = 0
    $rawChain = 0
    $rawSkipped = 0

    for ($i = 0; $i -lt $totalEntries; $i++) {
        $off = $pdataRaw + ($i * 12)
        $beginRVA  = [BitConverter]::ToUInt32($fileBytes, $off)
        $endRVA    = [BitConverter]::ToUInt32($fileBytes, $off + 4)
        $unwindRVA = [BitConverter]::ToUInt32($fileBytes, $off + 8)

        # Null entry means end of table (or padding).
        if ($beginRVA -eq 0 -and $endRVA -eq 0) {
            $rawSkipped++
            continue
        }

        # Make sure the unwind RVA falls inside the unwind section.
        $relOff = $unwindRVA - $uwVA
        if ($unwindRVA -lt $uwVA -or ($relOff + 4) -gt $uwSize) {
            # Shouldn't happen if the binary is well-formed.
            $rawSkipped++
            continue
        }

        $fileOff = $uwRaw + $relOff
        $byte0 = $fileBytes[$fileOff]
        $flags = ($byte0 -shr 3) -band 0x1F

        # Bit 2 = UNW_FLAG_CHAININFO, bits 0/1 = EHANDLER/UHANDLER
        if ($flags -band 4) {
            $rawChain++
        } elseif ($flags -band 3) {
            $rawHandler++
        } else {
            $rawNone++
        }
    }

    Write-Host "  Raw results: none=$rawNone  handler=$rawHandler  chain=$rawChain  skipped=$rawSkipped"
}

# ======================================================================
# Step 3: llvm-readobj --unwind flag counts
# ======================================================================

Write-Host ""
Write-Host "Running llvm-readobj --unwind..." -ForegroundColor Yellow

$uwOutput = & $ReadObj --unwind $Binary 2>&1

# Count the Flags lines.  llvm-readobj prints them like:
#   Flags [ (0x0)
#   Flags [ (0x3)
# We just need the hex value inside the parens.

$roNone = 0; $roHandler = 0; $roChain = 0

foreach ($line in $uwOutput) {
    $s = "$line"
    if ($s -match "Flags \[ \(0x([0-9A-Fa-f]+)\)") {
        $val = [Convert]::ToInt32($Matches[1], 16)
        if ($val -band 4) { $roChain++ }
        elseif ($val -band 3) { $roHandler++ }
        else { $roNone++ }
    }
}

$roTotal = $roNone + $roHandler + $roChain
Write-Host "  readobj results: none=$roNone  handler=$roHandler  chain=$roChain  total=$roTotal"

# ======================================================================
# Step 4: Run BOLT and capture its summary lines
# ======================================================================

Write-Host ""
Write-Host "Running llvm-bolt (identity pass)..." -ForegroundColor Yellow

# Output to NUL since we only care about the log messages.
$boltOut = & $Bolt $Binary -o NUL 2>&1

$boltParsed = 0; $boltChained = 0; $boltDiscovered = 0
$boltHandlers = 0; $boltDisasm = 0; $boltDisasmFail = 0
$boltCFG = 0; $boltCFGFail = 0

foreach ($line in $boltOut) {
    $s = "$line"
    if ($s -match "parsed (\d+) \.pdata entries, (\d+) chained") {
        $boltParsed  = [int]$Matches[1]
        $boltChained = [int]$Matches[2]
    }
    elseif ($s -match "(\d+) functions discovered") {
        $boltDiscovered = [int]$Matches[1]
    }
    elseif ($s -match "(\d+) functions with exception handlers") {
        $boltHandlers = [int]$Matches[1]
    }
    elseif ($s -match "disassembled (\d+) functions \((\d+) failed\)") {
        $boltDisasm     = [int]$Matches[1]
        $boltDisasmFail = [int]$Matches[2]
    }
    elseif ($s -match "built CFG for (\d+) functions \((\d+) failed\)") {
        $boltCFG     = [int]$Matches[1]
        $boltCFGFail = [int]$Matches[2]
    }
}

Write-Host "  BOLT parsed:     $boltParsed entries, $boltChained chained"
Write-Host "  BOLT discovered: $boltDiscovered functions"
Write-Host "  BOLT handlers:   $boltHandlers skipped"
Write-Host "  BOLT disasm:     $boltDisasm ok, $boltDisasmFail failed"
Write-Host "  BOLT CFG:        $boltCFG ok, $boltCFGFail failed"

# ======================================================================
# Step 5: Cross-check everything
# ======================================================================

Write-Host ""
Write-Host "Cross-validation results" -ForegroundColor Cyan
Write-Host ("-" * 60)

# Total entry counts should all agree.
Write-Check "Total .pdata entries" `
    ($totalEntries -eq $roTotal -and $totalEntries -eq $boltParsed) `
    "raw=$totalEntries readobj=$roTotal bolt=$boltParsed"

# Chain counts.
if ($unwindSec) {
    Write-Check "Chained entry count" `
        ($rawChain -eq $roChain -and $rawChain -eq $boltChained) `
        "raw=$rawChain readobj=$roChain bolt=$boltChained"
}

# Handler counts.  BOLT reports how many functions it skipped due to
# handlers.  The raw and readobj counts include chained entries too,
# but BOLT only counts non-chained functions.  So we compare raw/readobj
# to each other, and verify BOLT's handler count is <= rawHandler.
if ($unwindSec) {
    Write-Check "Handler count (raw vs readobj)" `
        ($rawHandler -eq $roHandler) `
        "raw=$rawHandler readobj=$roHandler"

    Write-Check "BOLT handler skip count matches" `
        ($boltHandlers -eq $rawHandler) `
        "bolt=$boltHandlers raw=$rawHandler"
}

# Discovered = total - chained.
$expectedDiscovered = $totalEntries - $boltChained
Write-Check "Discovered = total - chained" `
    ($boltDiscovered -eq $expectedDiscovered) `
    "bolt=$boltDiscovered expected=$expectedDiscovered"

# No-handler = discovered - handlers.  That should equal disasm attempted.
$expectedAttempted = $boltDiscovered - $boltHandlers
$actualAttempted = $boltDisasm + $boltDisasmFail
Write-Check "Disasm attempted = discovered - handlers" `
    ($actualAttempted -eq $expectedAttempted) `
    "attempted=$actualAttempted expected=$expectedAttempted"

if ($unwindSec) {
    # The raw "none" count should match the readobj "none" count.
    Write-Check "No-handler count (raw vs readobj)" `
        ($rawNone -eq $roNone) `
        "raw=$rawNone readobj=$roNone"
}

# Sanity: disasm success rate should be very high.
if ($actualAttempted -gt 0) {
    $successPct = [Math]::Round(100.0 * $boltDisasm / $actualAttempted, 1)
    Write-Check "Disassembly success rate > 99%" `
        ($successPct -ge 99.0) `
        "$successPct%"
}

# CFG should be close to disasm count.
Write-Check "CFG built for most disassembled functions" `
    ($boltCFG -ge ($boltDisasm - 10)) `
    "cfg=$boltCFG disasm=$boltDisasm"

# Quick identity rewrite check: run BOLT with a real output file and
# make sure the copy matches the original.
Write-Host ""
Write-Host "Running identity copy check..." -ForegroundColor Yellow

$tempOut = Join-Path ([System.IO.Path]::GetTempPath()) "bolt-xval-$([guid]::NewGuid().ToString('N').Substring(0,8)).exe"

try {
    & $Bolt $Binary -o $tempOut 2>$null | Out-Null

    if (Test-Path $tempOut) {
        $origHash = (Get-FileHash $Binary -Algorithm SHA256).Hash
        $copyHash = (Get-FileHash $tempOut -Algorithm SHA256).Hash
        Write-Check "Identity copy is byte-identical" `
            ($origHash -eq $copyHash) `
            ""

        # Quick smoke test: does the copy have the same file size?
        $origSize = (Get-Item $Binary).Length
        $copySize = (Get-Item $tempOut).Length
        Write-Check "Output file size matches" `
            ($origSize -eq $copySize) `
            "orig=$origSize copy=$copySize"
    } else {
        Write-Check "Identity copy produced output" $false "no output file"
    }
} finally {
    if (Test-Path $tempOut) { Remove-Item $tempOut -Force }
}

# ======================================================================
# Summary
# ======================================================================

Write-Host ""
Write-Host ("=" * 60)
if ($Failures -eq 0) {
    Write-Host "All checks passed." -ForegroundColor Green
} else {
    Write-Host "$Failures check(s) FAILED." -ForegroundColor Red
}
Write-Host ""

exit $Failures
