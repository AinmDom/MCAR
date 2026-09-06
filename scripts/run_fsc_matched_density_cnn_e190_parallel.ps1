$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = 'D:\miniconda3\envs\ml\python.exe'
Set-Location $repo
$runs = @(
  'sonicom_fsc_q14_film_seed20260821_e130',
  'sonicom_fsc_q14_film_seed20260822_e130_retry1',
  'sonicom_fsc_q14_film_seed20260823_e130',
  'sonicom_fsc_q50_film_seed20260821_e130',
  'sonicom_fsc_q50_film_seed20260822_e130',
  'sonicom_fsc_q50_film_seed20260823_e130'
)
$log = Join-Path $repo 'artifacts/training/fsc_matched_density_cnn_e190_parallel.log'
foreach ($run in $runs) {
  $report = Join-Path $repo "artifacts/training/$run/training_report.json"
  while (-not (Test-Path $report)) {
    Add-Content $log "[$(Get-Date -Format o)] WAIT $run"
    Start-Sleep -Seconds 30
  }
  $status = (Get-Content $report | ConvertFrom-Json).status
  if ($status -ne 'completed') { throw "E130 failed: $run status=$status" }
}
Add-Content $log "[$(Get-Date -Format o)] ALL E130 COMPLETE; finalizing CNN hashes"
& $python scripts/finalize_fsc_cnn_configs.py *>> $log
if ($LASTEXITCODE -ne 0) { throw "CNN config finalization failed" }
$cnn = @(
  'sonicom_fsc_q14_cnn_seed20260821_e190',
  'sonicom_fsc_q14_cnn_seed20260822_e190',
  'sonicom_fsc_q14_cnn_seed20260823_e190',
  'sonicom_fsc_q50_cnn_seed20260821_e190',
  'sonicom_fsc_q50_cnn_seed20260822_e190',
  'sonicom_fsc_q50_cnn_seed20260823_e190'
)
foreach ($run in $cnn) {
  $out = Join-Path $repo "artifacts/training/$run.parallel.out.log"
  $err = Join-Path $repo "artifacts/training/$run.parallel.err.log"
  $p = Start-Process -FilePath $python -WorkingDirectory $repo -WindowStyle Hidden -ArgumentList @('-m','mcar.training.train_film_siren_stage_c',"configs/experiments/$run.json") -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
  Add-Content $log "[$(Get-Date -Format o)] START $run pid=$($p.Id)"
}
Add-Content $log "[$(Get-Date -Format o)] STARTED all six CNN E190 runs in parallel"
