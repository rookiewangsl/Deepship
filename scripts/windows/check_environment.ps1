param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$ResolvedDataRoot = (Resolve-Path -LiteralPath $DataRoot).Path

Push-Location $RepoRoot
try {
    Write-Host "Repository: $RepoRoot"
    Write-Host "DeepShip data: $ResolvedDataRoot"
    & git status --short
    if ($LASTEXITCODE -ne 0) {
        throw "Git status check failed."
    }
    $GitChanges = & git status --porcelain --untracked-files=all
    if ($LASTEXITCODE -ne 0) {
        throw "Git worktree check failed."
    }
    if ($GitChanges) {
        throw "Formal training requires a clean git worktree. Commit or remove local changes first."
    }
    & git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Git commit check failed."
    }
    & python -c "import torch, torchaudio; print('torch:', torch.__version__); print('torchaudio:', torchaudio.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_version:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
    if ($LASTEXITCODE -ne 0) {
        throw "Python/PyTorch environment check failed."
    }
    & python scripts/prepare/validate_deepship_protocols.py --data-root $ResolvedDataRoot --protocol all --no-write-reports
    if ($LASTEXITCODE -ne 0) {
        throw "DeepShip protocol validation failed."
    }
    Write-Host "Environment and all three frozen protocols passed validation."
}
finally {
    Pop-Location
}
