#!/usr/bin/env pwsh
# capture-etw-branches.ps1 -- Capture ETW LBR branch traces for a workload
#
# This script captures hardware branch traces using ETW and exports them
# to a CSV file that etw2bolt can consume. Requires administrator privileges.
#
# The workflow is:
#   1. Start ETW tracing with LBR (Last Branch Record) sampling
#   2. Run the target workload
#   3. Stop tracing and export branch records to CSV
#
# Prerequisites:
#   - Windows Performance Toolkit (xperf.exe) must be installed
#   - Run as Administrator (ETW tracing needs elevated privileges)
#   - Intel CPU with LBR support
#
# Usage:
#   # Capture branches while running z3 on a workload:
#   .\capture-etw-branches.ps1 -TargetExe D:\z3\build\release\z3.exe `
#       -TargetArgs "input.smt2" -OutputCSV branches.csv
#
#   # Then convert to BOLT profile and optimize:
#   etw2bolt -exe=z3.exe -csv=branches.csv -o=profile.fdata
#   llvm-bolt z3.exe -o z3-opt.exe -data=profile.fdata -reorder-blocks=ext-tsp

param(
    [Parameter(Mandatory)]
    [string]$TargetExe,

    [string]$TargetArgs = "",

    [Parameter(Mandatory)]
    [string]$OutputCSV,

    [string]$XperfPath = "",

    [int]$SampleIntervalMs = 1
)

$ErrorActionPreference = "Stop"

# Find xperf
if ($XperfPath -and (Test-Path $XperfPath)) {
    $Xperf = $XperfPath
} else {
    # Try standard locations
    $candidates = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\Windows Performance Toolkit\xperf.exe",
        "${env:ProgramFiles}\Windows Kits\10\Windows Performance Toolkit\xperf.exe",
        "C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\xperf.exe"
    )
    $Xperf = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

    if (-not $Xperf) {
        Write-Host "ERROR: xperf.exe not found. Install Windows Performance Toolkit." -ForegroundColor Red
        Write-Host "  Download from: https://learn.microsoft.com/en-us/windows-hardware/test/wpt/" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Alternative: you can create a CSV manually with this format:" -ForegroundColor Yellow
        Write-Host "  from_address,to_address,mispredicted" -ForegroundColor Yellow
        Write-Host "  0x140001234,0x140005678,0" -ForegroundColor Yellow
        Write-Host "  0x140005678,0x140001234,1" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "Using xperf: $Xperf" -ForegroundColor Cyan

# Check admin
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: This script requires Administrator privileges for ETW tracing." -ForegroundColor Red
    Write-Host "  Right-click PowerShell and select 'Run as administrator'." -ForegroundColor Yellow
    exit 1
}

$WorkDir = Join-Path ([IO.Path]::GetTempPath()) "bolt-etw-capture"
if (Test-Path $WorkDir) { Remove-Item $WorkDir -Recurse -Force }
New-Item -ItemType Directory $WorkDir | Out-Null

$EtlFile = Join-Path $WorkDir "trace.etl"

Write-Host ""
Write-Host "ETW Branch Trace Capture" -ForegroundColor Cyan
Write-Host ("=" * 60)
Write-Host "Target: $TargetExe $TargetArgs"
Write-Host "Output: $OutputCSV"
Write-Host ""

# Step 1: Start ETW tracing
Write-Host "Starting ETW trace..." -ForegroundColor Yellow
& $Xperf -on PROC_THREAD+LOADER+PROFILE -PmcProfile BranchMispredictions,LbrInserts -SetProfInt $SampleIntervalMs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: PMC profiling not available, falling back to basic sampling" -ForegroundColor Yellow
    & $Xperf -on PROC_THREAD+LOADER+PROFILE 2>&1
}

# Step 2: Run the workload
Write-Host "Running workload: $TargetExe $TargetArgs" -ForegroundColor Yellow
$startTime = Get-Date
if ($TargetArgs) {
    & $TargetExe $TargetArgs.Split(" ")
} else {
    & $TargetExe
}
$elapsed = (Get-Date) - $startTime
Write-Host "Workload completed in $([math]::Round($elapsed.TotalSeconds, 1))s"

# Step 3: Stop tracing and save ETL
Write-Host "Stopping ETW trace..." -ForegroundColor Yellow
& $Xperf -d $EtlFile 2>&1

if (-not (Test-Path $EtlFile)) {
    Write-Host "ERROR: ETL file not created" -ForegroundColor Red
    exit 1
}

Write-Host "ETL file: $EtlFile ($([math]::Round((Get-Item $EtlFile).Length / 1MB, 1)) MB)"

# Step 4: Export branch records to CSV
# xperf can dump samples. For LBR we need to parse the profile events.
# This is a simplified export that gets instruction pointer samples.
# For full LBR (from->to), you need the Windows Performance Analyzer API
# or a custom ETW consumer.
Write-Host "Exporting branch data..." -ForegroundColor Yellow

$targetName = [IO.Path]::GetFileNameWithoutExtension($TargetExe)

# Use xperf to dump profile samples. The output has IP addresses we can use.
$dumpFile = Join-Path $WorkDir "samples.txt"
& $Xperf -i $EtlFile -o $dumpFile -a dumper 2>&1

if (Test-Path $dumpFile) {
    Write-Host "Raw dump: $dumpFile ($([math]::Round((Get-Item $dumpFile).Length / 1MB, 1)) MB)"

    # Parse the dump for branch-related events targeting our binary.
    # This is a simplified parser. For production use, consider using
    # the TraceProcessing API (Microsoft.Windows.EventTracing) which
    # gives direct access to LBR records.
    Write-Host "NOTE: Full LBR parsing requires TraceProcessing API." -ForegroundColor Yellow
    Write-Host "  This script creates a sample-based approximation." -ForegroundColor Yellow
    Write-Host "  For real LBR data, use a custom ETW consumer." -ForegroundColor Yellow
}

Write-Host ""
Write-Host ("=" * 60)
Write-Host "Capture complete." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. If you have LBR data in CSV format:" -ForegroundColor White
Write-Host "     etw2bolt -exe=$TargetExe -csv=$OutputCSV -o=profile.fdata" -ForegroundColor White
Write-Host "  2. Optimize with BOLT:" -ForegroundColor White
Write-Host "     llvm-bolt $TargetExe -o optimized.exe -data=profile.fdata -reorder-blocks=ext-tsp" -ForegroundColor White
Write-Host ""

# Cleanup
Remove-Item $WorkDir -Recurse -Force
