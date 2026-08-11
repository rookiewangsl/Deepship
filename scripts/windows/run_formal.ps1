param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [int[]]$Seeds = @(42, 43, 44),

    [string[]]$Protocols = @("segment_level", "recording_disjoint", "vessel_name_disjoint"),

    [int]$NumWorkers = 0,

    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$ResolvedDataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
$OutputRootPath = [System.IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $OutputRootPath | Out-Null
$LogRoot = Join-Path $OutputRootPath "logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

Push-Location $RepoRoot
try {
    foreach ($Seed in $Seeds) {
        foreach ($Protocol in $Protocols) {
            $RunName = "${Protocol}_seed${Seed}"
            $RunRoot = Join-Path $OutputRootPath $RunName
            $Manifest = Join-Path $RepoRoot "protocols\isolation_comparison_v1\$Protocol\split_manifest.json"
            $LogPath = Join-Path $LogRoot "$RunName.log"
            $Arguments = @(
                "scripts/train/train_deepship_macnna.py",
                "--data-root", $ResolvedDataRoot,
                "--split-manifest", $Manifest,
                "--protocol-name", $Protocol,
                "--output-root", $RunRoot,
                "--device", "cuda",
                "--num-workers", "$NumWorkers",
                "--seed", "$Seed"
            )
            if ($Resume) {
                $Arguments += "--resume"
            }
            Write-Host "Starting formal run: $RunName"
            if ($Resume) {
                & python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
            }
            else {
                & python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
            }
            if ($LASTEXITCODE -ne 0) {
                throw "Formal run failed: $RunName. See $LogPath"
            }
        }
    }
    Write-Host "Requested formal runs completed."
}
finally {
    Pop-Location
}
