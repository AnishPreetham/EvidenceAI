# =============================================================
# Evidence AI - start the application (OFFLINE RUNTIME)
# =============================================================
#   powershell -ExecutionPolicy Bypass -File .\start.ps1
#   .\start.ps1 -ApiOnly     FastAPI only
#   .\start.ps1 -UiOnly      Streamlit only
# =============================================================

param(
    [switch]$ApiOnly,
    [switch]$UiOnly,
    [int]$ApiPort = 8000,
    [int]$UiPort  = 8501
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py   = Join-Path $root ".venv\Scripts\python.exe"

function Say($text)  { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Good($text) { Write-Host "  [OK]   $text" -ForegroundColor Green }
function Warn($text) { Write-Host "  [WARN] $text" -ForegroundColor Yellow }
function Bad($text)  { Write-Host "  [FAIL] $text" -ForegroundColor Red }

if (-not (Test-Path $py)) {
    Bad "No virtual environment found. Run .\setup.ps1 first."
    exit 1
}

Say "Preflight"

# Ollama daemon
try {
    $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    Good "Ollama is running"
} catch {
    Warn "Ollama is not responding - starting it..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
        Good "Ollama started"
    } catch {
        Bad "Could not start Ollama. Run 'ollama serve' in another terminal."
        exit 1
    }
}

# Required model
$tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
if ($tags.models.name -contains "qwen2.5vl:3b") {
    Good "qwen2.5vl:3b available"
} else {
    Bad "qwen2.5vl:3b missing. Run: ollama pull qwen2.5vl:3b"
    exit 1
}

# Tesseract
if (Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe") {
    Good "Tesseract OCR available"
} else {
    Warn "Tesseract missing - images will be indexed without OCR text"
}

# Index
$dbPath = Join-Path $root "index\store.db"
if (Test-Path $dbPath) {
    Good "Persistent index found"
} else {
    Warn "No index yet - ingest files from the UI, or run:"
    Warn "  .venv\Scripts\python scripts\ingest_demo.py"
}

Say "Launching"

if (-not $UiOnly) {
    Start-Process -FilePath $py `
        -ArgumentList "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "$ApiPort" `
        -WorkingDirectory $root
    Good "FastAPI   http://127.0.0.1:$ApiPort/docs"
}

if (-not $ApiOnly) {
    Good "Streamlit http://localhost:$UiPort"
    Write-Host "`n  Press Ctrl+C to stop.`n" -ForegroundColor Yellow
    & $py -m streamlit run (Join-Path $root "app.py") --server.port $UiPort
} else {
    Write-Host "`n  API running. Press Ctrl+C to stop.`n" -ForegroundColor Yellow
    while ($true) { Start-Sleep -Seconds 3600 }
}
