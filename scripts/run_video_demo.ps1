param(
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

try {
    $ActivateScript = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
    if (Test-Path -LiteralPath $ActivateScript) {
        . $ActivateScript
        Write-Host "[OK] Environment demo aktif" -ForegroundColor Green
    }
    else {
        Write-Host "[INFO] .venv tidak ditemukan; menggunakan Python aktif" -ForegroundColor Yellow
    }

    Write-Host "[1/2] Menyiapkan paket demo..." -ForegroundColor Cyan
    python scripts\prepare_video_demo.py
    if ($LASTEXITCODE -ne 0) {
        throw "Persiapan demo gagal."
    }

    Write-Host "[2/2] Menjalankan validasi dan eksekusi notebook..." -ForegroundColor Cyan
    python scripts\validate_video_demo.py
    if ($LASTEXITCODE -ne 0) {
        throw "Demo berstatus NOT_READY. Periksa demo\outputs\demo_validation_report.json."
    }

    Write-Host ""
    Write-Host "READY - paket demo lolos validasi." -ForegroundColor Green
    Write-Host "Notebook: demo\01_podanauli_video_demo.ipynb"

    if (-not $NoOpen) {
        $CodeCommand = Get-Command code -ErrorAction SilentlyContinue
        if ($null -ne $CodeCommand) {
            Start-Process -FilePath $CodeCommand.Source -ArgumentList "demo\01_podanauli_video_demo.ipynb"
        }
        else {
            Write-Host "VS Code CLI tidak ditemukan. Buka notebook secara manual." -ForegroundColor Yellow
        }
    }
}
finally {
    Pop-Location
}
