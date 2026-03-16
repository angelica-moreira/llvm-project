#!/usr/bin/env pwsh
# filter-etw-dump.ps1 -- Filter xperf dump to only events for a target binary
#
# xperf dumps everything on the system (900MB+). This script extracts only
# the SampledProfile and I-Start (image load) events for your binary,
# shrinking the file from hundreds of MB to a few MB.
#
# Usage:
#   .\filter-etw-dump.ps1 -DumpFile D:\z3\trace-dump.txt -BinaryName z3.exe -OutputFile D:\z3\trace-filtered.txt
#
# Then pass the filtered file to etw2bolt:
#   etw2bolt z3.exe -e trace.etl -o profile.fdata -etw-dump=trace-filtered.txt

param(
    [Parameter(Mandatory)]
    [string]$DumpFile,

    [Parameter(Mandatory)]
    [string]$BinaryName,

    [string]$OutputFile = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DumpFile)) {
    Write-Host "ERROR: $DumpFile not found" -ForegroundColor Red
    exit 1
}

if ($OutputFile -eq "") {
    $dir = Split-Path $DumpFile -Parent
    $base = [IO.Path]::GetFileNameWithoutExtension($DumpFile)
    $OutputFile = Join-Path $dir "$base-$($BinaryName -replace '\.exe$','').txt"
}

$inputSize = [math]::Round((Get-Item $DumpFile).Length / 1MB, 1)
Write-Host "Filtering $DumpFile ($inputSize MB) for $BinaryName ..." -ForegroundColor Cyan

[IO.StreamReader]$reader = [IO.File]::OpenText($DumpFile)
[IO.StreamWriter]$writer = [IO.File]::CreateText($OutputFile)

$samples = 0
$images = 0
$total = 0

while ($null -ne ($line = $reader.ReadLine())) {
    $total++

    # Progress every 500K lines
    if ($total % 500000 -eq 0) {
        Write-Host "  $([math]::Round($total / 1000000, 1))M lines scanned, $samples samples found" -ForegroundColor DarkGray
    }

    # Only keep lines that mention our binary
    if ($line.IndexOf($BinaryName, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        continue
    }

    # Keep SampledProfile and I-Start events
    if ($line -match "SampledProfile") {
        $writer.WriteLine($line)
        $samples++
    }
    elseif ($line -match "I-Start") {
        $writer.WriteLine($line)
        $images++
    }
}

$writer.Close()
$reader.Close()

$outputSize = [math]::Round((Get-Item $OutputFile).Length / 1KB, 1)
Write-Host ""
Write-Host "Done:" -ForegroundColor Green
Write-Host "  Scanned $total lines"
Write-Host "  Found $samples SampledProfile events for $BinaryName"
Write-Host "  Found $images I-Start (image load) events"
Write-Host "  Output: $OutputFile ($outputSize KB)"
Write-Host ""
Write-Host "Next step:" -ForegroundColor Yellow
Write-Host "  etw2bolt $BinaryName -e <trace.etl> -o profile.fdata -etw-dump=$OutputFile"
