param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$ResolvedDataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
$OutputRootPath = [System.IO.Path]::GetFullPath($OutputRoot)
$Protocols = @("segment_level", "recording_disjoint", "vessel_name_disjoint")

New-Item -ItemType Directory -Force -Path $OutputRootPath | Out-Null
$LogRoot = Join-Path $OutputRootPath "logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

Push-Location $RepoRoot
try {
    foreach ($Protocol in $Protocols) {
        $RunRoot = Join-Path $OutputRootPath "smoke_$Protocol"
        $Manifest = Join-Path $RepoRoot "protocols\isolation_comparison_v1\$Protocol\split_manifest.json"
        $LogPath = Join-Path $LogRoot "smoke_$Protocol.log"
        $Arguments = @(
            "scripts/train/train_deepship_macnna.py",
            "--data-root", $ResolvedDataRoot,
            "--split-manifest", $Manifest,
            "--protocol-name", $Protocol,
            "--output-root", $RunRoot,
            "--device", "cuda",
            "--epochs", "1",
            "--num-workers", "0",
            "--max-train-batches", "2",
            "--max-eval-batches", "2",
            "--allow-experiment-overrides",
            "--seed", "42"
        )
        Write-Host "Starting smoke run: $Protocol"
        & python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        if ($LASTEXITCODE -ne 0) {
            throw "Smoke run failed: $Protocol. See $LogPath"
        }
    }
    Write-Host "All three smoke runs completed."
}
finally {
    Pop-Location
}
