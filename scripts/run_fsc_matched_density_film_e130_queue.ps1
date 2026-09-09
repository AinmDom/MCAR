$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = 'D:\miniconda3\envs\ml\python.exe'
$configs = @(
  'configs/experiments/sonicom_fsc_q14_film_seed20260821_e130.json',
  'configs/experiments/sonicom_fsc_q14_film_seed20260822_e130.json',
  'configs/experiments/sonicom_fsc_q14_film_seed20260823_e130.json',
  'configs/experiments/sonicom_fsc_q50_film_seed20260821_e130.json',
  'configs/experiments/sonicom_fsc_q50_film_seed20260822_e130.json',
  'configs/experiments/sonicom_fsc_q50_film_seed20260823_e130.json'
)
Set-Location $repo
$log = Join-Path $repo 'artifacts/training/fsc_matched_density_film_e130_queue.log'
foreach ($config in $configs) {
  $runName = [IO.Path]::GetFileNameWithoutExtension($config)
  $outDir = Join-Path $repo "artifacts/training/$runName"
  $history = Join-Path $outDir 'history.csv'
  while ($config -eq $configs[0] -and (Test-Path $history)) {
    $last = Get-Content $history | Select-Object -Last 1
    if ($last -match '^130,') { break }
    Add-Content $log "[$(Get-Date -Format o)] WAIT $config current run in progress"
    Start-Sleep -Seconds 30
  }
  if (Test-Path $history) {
    $last = Get-Content $history | Select-Object -Last 1
    if ($last -match '^130,') {
      Add-Content $log "[$(Get-Date -Format o)] SKIP $config already has cycle=130"
      continue
    }
  }
  Add-Content $log "[$(Get-Date -Format o)] START $config"
  & $python -m mcar.training.train_film_siren_stage_c $config *>> $log
  $code = $LASTEXITCODE
  Add-Content $log "[$(Get-Date -Format o)] END $config exit=$code"
  if ($code -ne 0) { exit $code }
}
Add-Content $log "[$(Get-Date -Format o)] COMPLETE all six E130 runs"
