# Import the model trained in Colab into Ollama and benchmark it.
# Usage (from the project root):  powershell -ExecutionPolicy Bypass -File colab\import_finetuned.ps1
param(
    [string]$Zip = "$env:USERPROFILE\Downloads\careimpact_ft.zip",
    [int]$EvalLimit = 60
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$run = Join-Path $root "models\runs\colab_1p5b_lora"

if (-not (Test-Path $Zip)) { throw "Not found: $Zip. Download careimpact_ft.zip from Google Drive (folder 'careimpact') first, or pass -Zip <path>." }

$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) { $ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" }
if (-not (Test-Path $ollama)) { throw "Ollama not found. Install it from https://ollama.com/download" }

Write-Host "1/3 Unpacking to $run"
New-Item -ItemType Directory -Force $run | Out-Null
Expand-Archive -Path $Zip -DestinationPath $run -Force

Write-Host "2/3 Registering the model with Ollama as careimpact-ft"
Push-Location $run
try { & $ollama create careimpact-ft -f Modelfile; if ($LASTEXITCODE) { throw "ollama create failed" } }
finally { Pop-Location }

Write-Host "3/3 Benchmarking on $EvalLimit held-out examples (this takes a while on CPU)"
Push-Location $root
try { python -m careimpact.cli evaluate --model careimpact-ft --limit $EvalLimit }
finally { Pop-Location }

Write-Host "`nDone. Compare the scores on the Model Benchmarking page (or data\reports\benchmark_report.md)."
