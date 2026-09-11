param(
    [string]$Config = "configs/experiments/ari_fsc_adapted_q26_locked_test_v1.json"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "D:\miniconda3\envs\ml\python.exe"
$matlab = "D:\Program Files\MATLAB\R2025a\bin\matlab.exe"
if (-not (Test-Path -LiteralPath $matlab)) { $matlab = "D:\Program Files\MATLAB\R2025b\bin\matlab.exe" }
if (-not (Test-Path -LiteralPath $matlab)) { $matlab = "D:\MATLAB\R2025b\bin\matlab.exe" }
$configPath = (Resolve-Path -LiteralPath (Join-Path $root $Config)).Path
$env:PYTHONPATH = Join-Path $root "src"
Push-Location $root
try {
    & $python scripts\prepare_ari_fsc_locked_test_waveforms.py $configPath
    if ($LASTEXITCODE -ne 0) { throw "locked test waveform preparation failed" }
    $prepared = Join-Path $root "data\processed\ari_fsc_adapted_q26_test_v1"
    $features = Join-Path $root "artifacts\ari_fsc_adapted_q26_v2\test"
    $matlabCommand = "addpath(fullfile('$($root.Replace('\','/'))','matlab')); mcar.prepare_ari_fsc_test_split('$($prepared.Replace('\','/'))','$($features.Replace('\','/'))');"
    & $matlab -batch $matlabCommand
    if ($LASTEXITCODE -ne 0) { throw "locked test MCA feature preparation failed" }
    & $python scripts\predict_evaluate_ari_fsc_test_secondary.py $configPath
    if ($LASTEXITCODE -ne 0) { throw "locked test inference/secondary evaluation failed" }
    $erbCommand = "addpath(fullfile('$($root.Replace('\','/'))','matlab')); mcar.evaluate_ari_fsc_test_erb('$($configPath.Replace('\','/'))');"
    & $matlab -batch $erbCommand
    if ($LASTEXITCODE -ne 0) { throw "locked test ERB evaluation failed" }
    & $python scripts\finalize_ari_fsc_locked_test.py $configPath
    if ($LASTEXITCODE -ne 0) { throw "locked test finalization failed" }
}
finally {
    Pop-Location
}
