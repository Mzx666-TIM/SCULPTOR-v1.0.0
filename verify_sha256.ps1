$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$sumPath = Join-Path $root "SHA256SUMS"
if (-not (Test-Path -LiteralPath $sumPath)) {
    Write-Host "SHA256SUMS not found in $root"
    exit 2
}

$checked = 0
$bad = 0
foreach ($line in Get-Content -LiteralPath $sumPath -Encoding UTF8) {
    if (-not $line.Trim()) { continue }
    $m = [regex]::Match($line, '^([0-9a-fA-F]{64})\s{2}(.+)$')
    if (-not $m.Success) {
        Write-Host "MALFORMED $line"
        $bad++
        continue
    }
    $expected = $m.Groups[1].Value.ToLowerInvariant()
    $rel = $m.Groups[2].Value -replace '/', [IO.Path]::DirectorySeparatorChar
    $path = Join-Path $root $rel
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Host "MISSING   $rel"
        $bad++
        continue
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        Write-Host "MISMATCH  $rel"
        $bad++
    }
    $checked++
}

if ($bad -eq 0) {
    Write-Host "Verified $checked files: OK"
    exit 0
}
Write-Host "$bad problem(s) in $checked files"
exit 1
