param(
    [int]$PollSeconds = 60,
    [string]$WslDistro = 'Ubuntu-22.04'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$resultRoot = Join-Path $projectRoot 'results\sonicom_validation_v351_ranf_fsp_mca_subject_hrtf'
$figureRoot = Join-Path $resultRoot 'figures'
$logPath = Join-Path $resultRoot 'automatic_postprocess.log'
$statusPath = Join-Path $resultRoot 'automatic_postprocess_status.json'
$lockPath = Join-Path $resultRoot 'automatic_postprocess.lock'

New-Item -ItemType Directory -Force -Path $figureRoot | Out-Null
$lockStream = [System.IO.File]::Open(
    $lockPath,
    [System.IO.FileMode]::OpenOrCreate,
    [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None
)

function Write-RunLog {
    param([string]$Message)
    $line = '[{0}] {1}' -f (Get-Date -Format o), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function Write-Status {
    param(
        [string]$Status,
        [string]$Message
    )
    $payload = [ordered]@{
        schema_version = '1.0'
        updated_on = (Get-Date -Format o)
        status = $Status
        message = $Message
        result_root = 'results/sonicom_validation_v351_ranf_fsp_mca_subject_hrtf'
        figure_png = 'results/sonicom_validation_v351_ranf_fsp_mca_subject_hrtf/figures/validation44_subject_contralateral_hrtf_comparison.png'
        figure_pdf = 'results/sonicom_validation_v351_ranf_fsp_mca_subject_hrtf/figures/validation44_subject_contralateral_hrtf_comparison.pdf'
        selected_direction_table = 'results/sonicom_validation_v351_ranf_fsp_mca_subject_hrtf/selected_direction_per_subject.csv'
        test_subject_count_read = 0
    }
    $json = $payload | ConvertTo-Json -Depth 4
    [System.IO.File]::WriteAllText($statusPath, $json, [System.Text.UTF8Encoding]::new($false))
}

function Invoke-NativeLogged {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )
    Write-RunLog ("Running: {0} {1}" -f $FilePath, ($ArgumentList -join ' '))
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $nativeOutput = & $FilePath @ArgumentList 2>&1
    $nativeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($nativeOutput) {
        Add-Content -LiteralPath $logPath -Value $nativeOutput -Encoding UTF8
    }
    if ($nativeExitCode -ne 0) {
        throw "Native command failed with exit code $nativeExitCode`: $FilePath"
    }
}

function Invoke-WslText {
    param([string]$Command)
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $lines = @(& wsl.exe -d $WslDistro -- bash -lc $Command 2>$null)
    $nativeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($nativeExitCode -ne 0) {
        throw "WSL status command failed with exit code $nativeExitCode."
    }
    return ($lines -join "`n").Trim()
}

try {
    Set-Location -LiteralPath $projectRoot
    Write-Status -Status 'waiting_for_ranf' -Message 'Waiting for the frozen RANF validation pipeline.'
    Write-RunLog 'Automatic post-processing watcher started.'

    while ($true) {
        $phase = Invoke-WslText -Command 'cat /home/ill3/ranf-work/exp/mcar_q26_validation_v351_main/validation_pipeline_phase.txt 2>/dev/null || true'
        $exitStatus = Invoke-WslText -Command 'cat /home/ill3/ranf-work/exp/mcar_q26_validation_v351_main/validation_pipeline_exit_status.txt 2>/dev/null || true'

        if ($phase -eq 'complete' -and $exitStatus -eq '0') {
            Write-RunLog 'Frozen RANF validation pipeline completed successfully.'
            break
        }
        if ($exitStatus -and $exitStatus -ne '0') {
            throw "RANF validation pipeline failed with exit status $exitStatus."
        }

        $processCountText = Invoke-WslText -Command "pgrep -fc 'ranf.2_adapting_neural_field|ranf.3_evaluating_neural_field|launch_mcar_q26_validation' || true"
        $processCount = if ($processCountText) { [int]$processCountText } else { 0 }
        if ($processCount -eq 0) {
            throw 'RANF validation process disappeared before reporting completion.'
        }
        Start-Sleep -Seconds $PollSeconds
    }

    Write-Status -Status 'exporting_ranf' -Message 'Exporting 44 validation SOFA predictions.'
    $exportCommand = @(
        '/home/ill3/.venvs/ranf/bin/python'
        '/mnt/d/cuc/CSMT/MCAR/scripts/export_ranf_q26_predictions.py'
        '--mapping /home/ill3/ranf-work/mcar_q26_formal/subject_mapping.csv'
        '--eval-root /home/ill3/ranf-work/exp/mcar_q26_validation_v351_main/log/eval'
        '--output-root /mnt/d/cuc/CSMT/MCAR/artifacts/reconstruction/sonicom_ranf_q26_validation_frozen'
        '--split val'
    ) -join ' '
    Invoke-NativeLogged -FilePath 'wsl.exe' -ArgumentList @(
        '-d', $WslDistro, '--', 'bash', '-lc', $exportCommand
    )

    $ranfSubjectRoot = Join-Path $projectRoot 'artifacts\reconstruction\sonicom_ranf_q26_validation_frozen\subjects'
    $ranfSofaCount = @(
        Get-ChildItem -LiteralPath $ranfSubjectRoot -Recurse -File -Filter 'prediction.sofa'
    ).Count
    if ($ranfSofaCount -ne 44) {
        throw "Expected 44 exported RANF validation SOFAs, found $ranfSofaCount."
    }

    Write-Status -Status 'plotting' -Message 'Generating the 44-subject MATLAB comparison figure.'
    $matlabCommand = "addpath('matlab'); p=mcar.plot_v351_ranf_fsp_mca_validation_subject_hrtf(); disp(p);"
    Invoke-NativeLogged -FilePath 'matlab.exe' -ArgumentList @(
        '-batch', $matlabCommand
    )

    $pngPath = Join-Path $figureRoot 'validation44_subject_contralateral_hrtf_comparison.png'
    $pdfPath = Join-Path $figureRoot 'validation44_subject_contralateral_hrtf_comparison.pdf'
    $tablePath = Join-Path $resultRoot 'selected_direction_per_subject.csv'
    $qualityPath = Join-Path $resultRoot 'quality_checks.csv'
    foreach ($requiredPath in @($pngPath, $pdfPath, $tablePath, $qualityPath)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Required post-processing output is missing: $requiredPath"
        }
        if ((Get-Item -LiteralPath $requiredPath).Length -eq 0) {
            throw "Required post-processing output is empty: $requiredPath"
        }
    }

    $quality = Import-Csv -LiteralPath $qualityPath
    $subjectCheck = $quality | Where-Object { $_.Check -eq 'SubjectCount' }
    $nonfiniteCheck = $quality | Where-Object { $_.Check -eq 'NonfiniteSelectedDirectionMetrics' }
    if ($subjectCheck.Value -ne '44' -or $nonfiniteCheck.Value -ne '0') {
        throw 'MATLAB quality checks did not confirm 44 finite validation subjects.'
    }

    Write-Status -Status 'completed' -Message 'RANF export and MATLAB HRTF comparison completed successfully.'
    Write-RunLog 'Automatic post-processing completed successfully.'
}
catch {
    Write-Status -Status 'failed' -Message $_.Exception.Message
    Write-RunLog ("FAILED: {0}" -f $_.Exception.Message)
    throw
}
finally {
    if ($lockStream) {
        $lockStream.Dispose()
    }
}
