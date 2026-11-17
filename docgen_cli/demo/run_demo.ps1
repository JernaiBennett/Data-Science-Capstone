# Runs docgen on copies of the demo files so the originals are never changed.
# Usage (from the docgen_cli folder, with the virtual environment active):
#   .\demo\run_demo.ps1
param([string]$Model = "models\codet5p_full_ft")

$ErrorActionPreference = "Stop"
$demo = $PSScriptRoot
$work = Join-Path $demo "_work"
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Path $work | Out-Null
Copy-Item -Recurse (Join-Path $demo "shop") (Join-Path $work "shop")

Write-Host "`n=== 1. Full-style docstrings (--full) proposed for shop\pricing.py (diff, nothing is changed) ===`n"
docgen doc (Join-Path $work "shop\pricing.py") --full --model $Model

Write-Host "`n=== 2. Apply summary-only docstrings to the whole shop package ===`n"
docgen doc (Join-Path $work "shop") --write --model $Model
Write-Host "`nCompare with: git diff --no-index demo\shop demo\_work\shop"

Write-Host "`n=== 3. Snippets in other languages (one model load for all files) ===`n"
docgen snippet (Get-ChildItem (Join-Path $demo "snippets")).FullName --model $Model

Write-Host "`n=== 4. Bug report titles (one model load for all files) ===`n"
docgen bug (Get-ChildItem (Join-Path $demo "bugs")).FullName --model $Model
