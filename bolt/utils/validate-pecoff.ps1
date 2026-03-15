<#
.SYNOPSIS
    Validates a PE/COFF binary produced by llvm-bolt against the original.

.DESCRIPTION
    Runs a thorough set of checks comparing the original PE binary with the
    bolt-processed one.  Covers:

      - File size and SHA-256 hash
      - PE file headers (machine, entry point, image base, image size, etc.)
      - Section layout (names, virtual sizes, raw sizes, characteristics)
      - Import table (DLL names + imported symbols)
      - Export table
      - Base relocation table (.reloc)
      - Debug directory (CodeView, POGO, Repro)
      - Load configuration (CFG tables, security cookie, guard flags)
      - Unwind information (.pdata RuntimeFunction count)
      - TLS directory
      - PE resources (.rsrc)
      - PE optional header checksum
      - Byte-level comparison of the .text section
      - Full binary byte-for-byte diff (first N differences)
      - SizeBench BinaryBytes deep analysis (when a PDB is available)
      - Optional functional smoke test

    Exit code 0 means every check passed.

.PARAMETER Original
    Path to the original (pre-bolt) PE binary.

.PARAMETER Bolt
    Path to the bolt-processed PE binary.

.PARAMETER Pdb
    (Optional) Path to the PDB matching the original binary.
    Enables the SizeBench BinaryBytes deep analysis step.

.PARAMETER RunCommand
    (Optional) Arguments to pass to the bolt binary as a quick smoke test.
    The binary path is prepended automatically.
    Example: "--version" or "-in"

.PARAMETER MaxByteDiffs
    (Optional) How many byte-level differences to report before stopping.
    Defaults to 10.

.EXAMPLE
    .\validate-pecoff.ps1 -Original z3.exe -Bolt z3-bolt.exe
    .\validate-pecoff.ps1 -Original test.dll -Bolt test-bolt.dll -Pdb test.pdb
    .\validate-pecoff.ps1 -Original z3.exe -Bolt z3-bolt.exe -RunCommand "--version"
#>

param(
    [Parameter(Mandatory)] [string] $Original,
    [Parameter(Mandatory)] [string] $Bolt,
    [string] $Pdb,
    [string] $RunCommand,
    [int]    $MaxByteDiffs = 10
)

$ErrorActionPreference = "Stop"
$script:pass  = 0
$script:fail  = 0
$script:skip  = 0

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

function Write-Step([string]$msg) {
    Write-Host "`n--- $msg ---" -ForegroundColor Cyan
}

function Pass([string]$msg) {
    Write-Host "  [OK]   $msg" -ForegroundColor Green
    $script:pass++
}

function Fail([string]$msg) {
    Write-Host "  [FAIL] $msg" -ForegroundColor Red
    $script:fail++
}

function Warn([string]$msg) {
    Write-Host "  [WARN] $msg" -ForegroundColor Yellow
}

function Skip([string]$msg) {
    Write-Host "  [SKIP] $msg" -ForegroundColor DarkGray
    $script:skip++
}

function Info([string]$msg) {
    Write-Host "  $msg"
}

# Pull a value from llvm-readobj text output.  Returns empty string on miss.
function Get-Field([string]$text, [string]$field) {
    if ($text -match "${field}\s*:\s*(.+)") { return $Matches[1].Trim() }
    return ""
}

# Parse section records out of llvm-readobj --sections output.
# Returns a hashtable:  "sectionName.property" -> value
function Parse-Sections([string[]]$lines) {
    $result  = @{}
    $current = $null
    foreach ($line in $lines) {
        if ($line -match "^\s+Name:\s+(\S+)") { $current = $Matches[1] }
        if ($current) {
            if ($line -match "VirtualSize:\s+(0x\w+)")       { $result["$current.VirtualSize"]       = $Matches[1] }
            if ($line -match "RawDataSize:\s+(\d+)")          { $result["$current.RawDataSize"]       = $Matches[1] }
            if ($line -match "VirtualAddress:\s+(0x\w+)")     { $result["$current.VirtualAddress"]    = $Matches[1] }
            if ($line -match "PointerToRawData:\s+(0x\w+)")   { $result["$current.PointerToRawData"]  = $Matches[1] }
            if ($line -match "Characteristics \[.*\(0x(\w+)\)") { $result["$current.Characteristics"] = $Matches[1] }
        }
    }
    return $result
}

# ---------------------------------------------------------------------------
#  locate tools
# ---------------------------------------------------------------------------

$readobj = $null
foreach ($candidate in @(
    "D:\llvm-upstream\llvm-project\build\bin\llvm-readobj.exe",
    (Join-Path $PSScriptRoot "..\..\build\bin\llvm-readobj.exe")
)) {
    if (Test-Path $candidate) { $readobj = (Resolve-Path $candidate).Path; break }
}
if (-not $readobj) {
    $found = Get-Command "llvm-readobj.exe" -ErrorAction SilentlyContinue
    if ($found) { $readobj = $found.Source }
}
if (-not $readobj) { Write-Host "WARNING: llvm-readobj.exe not found; most checks will be skipped." -ForegroundColor Yellow }

$sizebench = "D:\SizeBench\src\BinaryBytes\bin\x64\Debug\net8.0-windows10.0.17763\win-x64\BinaryBytes.exe"
if (-not (Test-Path $sizebench)) { $sizebench = $null }


# ===========================================================================
#  1.  File size and hash
# ===========================================================================

Write-Step "File size and SHA-256 hash"

if (-not (Test-Path $Original)) { Fail "original not found: $Original"; exit 1 }
if (-not (Test-Path $Bolt))     { Fail "bolt binary not found: $Bolt";  exit 1 }

$origSize = (Get-Item $Original).Length
$boltSize = (Get-Item $Bolt).Length
Info "Original : $origSize bytes"
Info "Bolt     : $boltSize bytes"

if ($boltSize -eq 0)                      { Fail "bolt binary is empty" }
elseif ($boltSize -lt ($origSize * 0.5))   { Fail "bolt binary suspiciously small ($boltSize vs $origSize)" }
elseif ($boltSize -gt ($origSize * 2.0))   { Fail "bolt binary suspiciously large ($boltSize vs $origSize)" }
else                                       { Pass "file size reasonable" }

$origHash = (Get-FileHash $Original -Algorithm SHA256).Hash
$boltHash = (Get-FileHash $Bolt     -Algorithm SHA256).Hash

if ($origHash -eq $boltHash) {
    Pass "SHA-256 identical (identity rewrite confirmed)"
} else {
    Warn "SHA-256 differs (expected when bolt applied optimizations)"
    Info "  original : $origHash"
    Info "  bolt     : $boltHash"
}


# ===========================================================================
#  2.  PE file headers
# ===========================================================================

Write-Step "PE file headers"

if ($readobj) {
    $origHdr = & $readobj --file-headers $Original 2>&1 | Out-String
    $boltHdr = & $readobj --file-headers $Bolt     2>&1 | Out-String

    foreach ($field in @(
        "Machine",
        "SectionCount",
        "OptionalHeaderSize",
        "AddressOfEntryPoint",
        "ImageBase",
        "SectionAlignment",
        "FileAlignment",
        "SizeOfCode",
        "SizeOfInitializedData",
        "SizeOfImage",
        "SizeOfHeaders",
        "Subsystem",
        "DLLCharacteristics"
    )) {
        $a = Get-Field $origHdr $field
        $b = Get-Field $boltHdr $field
        if (-not $a -and -not $b) { continue }   # field not present
        if ($a -eq $b) { Pass "$field = $a" }
        else           { Fail "$field : original=$a  bolt=$b" }
    }

    # PE checksum (separate because it lives on a different line)
    $origChk = Get-Field $origHdr "CheckSum"
    $boltChk = Get-Field $boltHdr "CheckSum"
    if ($origChk -eq $boltChk) { Pass "CheckSum = $origChk" }
    else                       { Fail "CheckSum : original=$origChk  bolt=$boltChk" }
} else {
    Skip "llvm-readobj not available"
}


# ===========================================================================
#  3.  Section layout
# ===========================================================================

Write-Step "Section layout"

if ($readobj) {
    $origSec = & $readobj --sections $Original 2>&1
    $boltSec = & $readobj --sections $Bolt     2>&1

    $origNames = @($origSec | Select-String "^\s+Name:\s+(\S+)" | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $boltNames = @($boltSec | Select-String "^\s+Name:\s+(\S+)" | ForEach-Object { $_.Matches[0].Groups[1].Value })

    Info "Original sections : $($origNames -join ', ')"
    Info "Bolt sections     : $($boltNames -join ', ')"

    $missing = $origNames | Where-Object { $_ -notin $boltNames }
    $added   = $boltNames | Where-Object { $_ -notin $origNames }

    if ($missing) { Fail "sections missing in bolt output: $($missing -join ', ')" }
    else          { Pass "all original sections present" }

    if ($added) { Warn "bolt added sections: $($added -join ', ')" }

    # Compare per-section properties.
    $oMap = Parse-Sections $origSec
    $bMap = Parse-Sections $boltSec

    foreach ($key in ($oMap.Keys | Sort-Object)) {
        $ov = $oMap[$key]; $bv = $bMap[$key]
        if (-not $bv) { Fail "$key missing in bolt"; continue }
        if ($ov -eq $bv) { Pass "$key = $ov" }
        else             { Fail "$key : original=$ov  bolt=$bv" }
    }
} else {
    Skip "llvm-readobj not available"
}


# ===========================================================================
#  4.  Import table
# ===========================================================================

Write-Step "Import table"

if ($readobj) {
    $origImp = & $readobj --coff-imports $Original 2>&1
    $boltImp = & $readobj --coff-imports $Bolt     2>&1

    $origDlls = @($origImp | Select-String "Name:\s+(\S+\.dll)" -AllMatches | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $boltDlls = @($boltImp | Select-String "Name:\s+(\S+\.dll)" -AllMatches | ForEach-Object { $_.Matches[0].Groups[1].Value })

    $origSyms = ($origImp | Select-String "Symbol:\s+").Count
    $boltSyms = ($boltImp | Select-String "Symbol:\s+").Count

    Info "Original : $($origDlls.Count) DLLs, $origSyms symbols"
    Info "Bolt     : $($boltDlls.Count) DLLs, $boltSyms symbols"

    if ($origDlls.Count -eq $boltDlls.Count) { Pass "DLL count matches" }
    else { Fail "DLL count mismatch: $($origDlls.Count) vs $($boltDlls.Count)" }

    if ($origSyms -eq $boltSyms) { Pass "imported symbol count matches ($origSyms)" }
    else { Fail "imported symbol count mismatch: $origSyms vs $boltSyms" }

    # check each DLL name is present
    $missingDll = $origDlls | Where-Object { $_ -notin $boltDlls }
    if ($missingDll) { Fail "DLLs missing in bolt: $($missingDll -join ', ')" }
} else {
    Skip "llvm-readobj not available"
}


# ===========================================================================
#  5.  Export table
# ===========================================================================

Write-Step "Export table"

if ($readobj) {
    $origExp = (& $readobj --coff-exports $Original 2>&1 | Select-String "Name:").Count
    $boltExp = (& $readobj --coff-exports $Bolt     2>&1 | Select-String "Name:").Count

    Info "Original : $origExp exports"
    Info "Bolt     : $boltExp exports"
    if ($origExp -eq $boltExp) { Pass "export count matches" }
    else { Fail "export count mismatch: $origExp vs $boltExp" }
} else {
    Skip "llvm-readobj not available"
}


# ===========================================================================
#  6.  Base relocations (.reloc)
# ===========================================================================

Write-Step "Base relocations (.reloc)"

if ($readobj) {
    $origRel = (& $readobj --coff-basereloc $Original 2>&1 | Select-String "Type:").Count
    $boltRel = (& $readobj --coff-basereloc $Bolt     2>&1 | Select-String "Type:").Count

    Info "Original : $origRel entries"
    Info "Bolt     : $boltRel entries"
    if ($origRel -eq $boltRel) { Pass "base relocation count matches ($origRel)" }
    else { Fail "base relocation count mismatch: $origRel vs $boltRel" }
} else {
    Skip "llvm-readobj not available"
}


# ===========================================================================
#  7.  Debug directory
# ===========================================================================

Write-Step "Debug directory"

if ($readobj) {
    $origDbg = & $readobj --coff-debug-directory $Original 2>&1
    $boltDbg = & $readobj --coff-debug-directory $Bolt     2>&1

    $origTypes = @($origDbg | Select-String "Type:\s+(\S+)" | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $boltTypes = @($boltDbg | Select-String "Type:\s+(\S+)" | ForEach-Object { $_.Matches[0].Groups[1].Value })

    Info "Original debug entries: $($origTypes -join ', ')"
    Info "Bolt debug entries    : $($boltTypes -join ', ')"

    if (($origTypes -join ',') -eq ($boltTypes -join ',')) { Pass "debug directory types match" }
    else { Fail "debug directory types differ" }

    # If there is a PDB reference, compare the GUID
    $origGuid = ($origDbg | Select-String "PDBSignature:\s+\{(.+?)\}" | ForEach-Object { $_.Matches[0].Groups[1].Value })
    $boltGuid = ($boltDbg | Select-String "PDBSignature:\s+\{(.+?)\}" | ForEach-Object { $_.Matches[0].Groups[1].Value })
    if ($origGuid -and $boltGuid) {
        if ($origGuid -eq $boltGuid) { Pass "PDB GUID matches" }
        else { Warn "PDB GUID differs (orig=$origGuid bolt=$boltGuid)" }
    }
} else {
    Skip "llvm-readobj not available"
}


# ===========================================================================
#  8.  Load configuration (CFG, security cookie, guard tables)
# ===========================================================================

Write-Step "Load configuration"

if ($readobj) {
    $origLC = & $readobj --coff-load-config $Original 2>&1 | Out-String
    $boltLC = & $readobj --coff-load-config $Bolt     2>&1 | Out-String

    foreach ($field in @(
        "SecurityCookie",
        "GuardCFCheckFunctionPointer",
        "GuardCFFunctionTable",
        "GuardCFFunctionCount",
        "GuardFlags",
        "GuardAddressTakenIatEntryTable",
        "GuardAddressTakenIatEntryCount",
        "GuardLongJumpTargetTable",
        "GuardLongJumpTargetCount"
    )) {
        $a = Get-Field $origLC $field
        $b = Get-Field $boltLC $field
        if (-not $a -and -not $b) { continue }
        if ($a -eq $b) { Pass "$field = $a" }
        else           { Fail "$field : original=$a  bolt=$b" }
    }
} else {
    Skip "llvm-readobj not available"
}


# ===========================================================================
#  9.  Unwind information (.pdata / .xdata)
# ===========================================================================

Write-Step "Unwind information (.pdata)"

if ($readobj) {
    $origUW = (& $readobj --unwind $Original 2>&1 | Select-String "RuntimeFunction \{").Count
    $boltUW = (& $readobj --unwind $Bolt     2>&1 | Select-String "RuntimeFunction \{").Count

    Info "Original : $origUW RuntimeFunction entries"
    Info "Bolt     : $boltUW RuntimeFunction entries"
    if ($origUW -eq $boltUW) { Pass "RuntimeFunction count matches ($origUW)" }
    else { Fail "RuntimeFunction count mismatch: $origUW vs $boltUW" }
} else {
    Skip "llvm-readobj not available"
}


# ===========================================================================
#  10.  TLS directory
# ===========================================================================

Write-Step "TLS directory"

if ($readobj) {
    $origTLS = & $readobj --coff-tls-directory $Original 2>&1
    $boltTLS = & $readobj --coff-tls-directory $Bolt     2>&1

    foreach ($field in @("StartAddressOfRawData","EndAddressOfRawData",
                         "AddressOfIndex","AddressOfCallBacks","SizeOfZeroFill")) {
        $origLine = $origTLS | Select-String "${field}:" | Select-Object -First 1
        $boltLine = $boltTLS | Select-String "${field}:" | Select-Object -First 1
        if (-not $origLine -and -not $boltLine) { continue }
        $a = ("$origLine" -replace ".*${field}:\s*","").Trim()
        $b = ("$boltLine" -replace ".*${field}:\s*","").Trim()
        if ($a -eq $b) { Pass "TLS $field = $a" }
        else           { Fail "TLS $field : original=$a  bolt=$b" }
    }
} else {
    Skip "llvm-readobj not available"
}


# ===========================================================================
#  11.  Resources (.rsrc)
# ===========================================================================

Write-Step "Resources"

if ($readobj) {
    $origRes = (& $readobj --coff-resources $Original 2>&1 | Select-String "Entry \{").Count
    $boltRes = (& $readobj --coff-resources $Bolt     2>&1 | Select-String "Entry \{").Count

    Info "Original : $origRes resource entries"
    Info "Bolt     : $boltRes resource entries"
    if ($origRes -eq $boltRes) { Pass "resource count matches" }
    else { Fail "resource count mismatch: $origRes vs $boltRes" }
} else {
    Skip "llvm-readobj not available"
}


# ===========================================================================
#  12.  Byte-level comparison
# ===========================================================================

Write-Step "Byte-level binary comparison"

$origBytes = [System.IO.File]::ReadAllBytes((Resolve-Path $Original).Path)
$boltBytes = [System.IO.File]::ReadAllBytes((Resolve-Path $Bolt).Path)

if ($origBytes.Length -ne $boltBytes.Length) {
    Fail "file lengths differ ($($origBytes.Length) vs $($boltBytes.Length))"
} else {
    $diffs = 0
    for ($i = 0; $i -lt $origBytes.Length; $i++) {
        if ($origBytes[$i] -ne $boltBytes[$i]) {
            if ($diffs -lt $MaxByteDiffs) {
                Info ("  offset 0x{0:X8}: orig=0x{1:X2} bolt=0x{2:X2}" -f $i, $origBytes[$i], $boltBytes[$i])
            }
            $diffs++
        }
    }
    if ($diffs -eq 0) {
        Pass "binaries are byte-identical"
    } else {
        Warn "$diffs byte(s) differ"
        if ($diffs -gt $MaxByteDiffs) { Info "  (showing first $MaxByteDiffs)" }
    }
}


# ===========================================================================
#  13.  SizeBench BinaryBytes deep analysis
# ===========================================================================

Write-Step "SizeBench BinaryBytes"

if ($Pdb -and $sizebench) {
    if (-not (Test-Path $Pdb)) {
        Fail "PDB not found: $Pdb"
    } else {
        $dbOut = [IO.Path]::ChangeExtension($Bolt, ".sizebench.db")
        Info "running BinaryBytes (this may take a moment)..."

        $sbOutput = & $sizebench "/pdb-file=$Pdb" "/binary-file=$Bolt" "/out-file=$dbOut" "/include-all-symbols" 2>&1
        $sbExit = $LASTEXITCODE

        # BinaryBytes sometimes appends .db to the path we gave it.
        $dbActual = $dbOut
        if (-not (Test-Path $dbActual) -and (Test-Path "$dbActual.db")) {
            $dbActual = "$dbActual.db"
        }

        if ($sbExit -eq 0 -and (Test-Path $dbActual)) {
            $dbSize = (Get-Item $dbActual).Length
            Pass "BinaryBytes completed (db = $([math]::Round($dbSize/1KB, 1)) KB)"
            Info "output: $dbActual"
        } else {
            Fail "BinaryBytes failed (exit code $sbExit)"
            $sbOutput | Select-Object -Last 5 | ForEach-Object { Info "  $_" }
        }
    }
} elseif (-not $Pdb) {
    Skip "no -Pdb provided (pass -Pdb to enable SizeBench analysis)"
} else {
    Skip "SizeBench BinaryBytes.exe not found at expected path"
}


# ===========================================================================
#  14.  Functional smoke test
# ===========================================================================

Write-Step "Functional smoke test"

if ($RunCommand) {
    $cmdArgs = $RunCommand -split "\s+"
    try {
        $output = & $Bolt @cmdArgs 2>&1 | Out-String
        $rc = $LASTEXITCODE
        if ($output.Length -gt 0) {
            Pass "binary executed (exit code $rc, $($output.Length) chars output)"
            $output.Split("`n") | Select-Object -First 3 | ForEach-Object { Info "  $_" }
        } else {
            Fail "binary returned exit code $rc with no output"
        }
    } catch {
        Fail "binary crashed: $_"
    }
} else {
    Skip "no -RunCommand provided"
}


# ===========================================================================
#  summary
# ===========================================================================

Write-Host "`n==========================================" -ForegroundColor White
Write-Host "  Passed : $($script:pass)" -ForegroundColor Green
Write-Host "  Failed : $($script:fail)" -ForegroundColor $(if ($script:fail -gt 0) { "Red" } else { "Green" })
Write-Host "  Skipped: $($script:skip)" -ForegroundColor DarkGray
Write-Host "==========================================" -ForegroundColor White

exit $script:fail
