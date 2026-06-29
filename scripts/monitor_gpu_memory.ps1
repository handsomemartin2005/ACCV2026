param(
    [int]$LimitMiB = 12000,
    [int]$IntervalSeconds = 10,
    [string]$LogPath = "run\gpu_memory_monitor.log"
)

$ErrorActionPreference = "Continue"
$logDir = Split-Path -Parent $LogPath
if ($logDir) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

function Write-MonitorLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    Add-Content -Path $LogPath -Value "$timestamp $Message"
}

$breaches = 0
Write-MonitorLog "started limit_mib=$LimitMiB interval_seconds=$IntervalSeconds"

while ($true) {
    $trainProcs = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match '^python(\.exe)?$' -and
            $_.CommandLine -match 'run_full_experiment_suite\.py|run_ablation\.py'
        }

    if (-not $trainProcs) {
        Write-MonitorLog "finished no_training_process"
        break
    }

    $usedText = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    $usedMiB = 0
    [int]::TryParse(($usedText -replace '[^\d]', ''), [ref]$usedMiB) | Out-Null
    $procIds = ($trainProcs | ForEach-Object { $_.ProcessId }) -join ','
    Write-MonitorLog "memory_used_mib=$usedMiB training_pids=$procIds"

    if ($usedMiB -gt $LimitMiB) {
        $breaches += 1
        Write-MonitorLog "breach count=$breaches limit_mib=$LimitMiB"
    } else {
        $breaches = 0
    }

    if ($breaches -ge 2) {
        Write-MonitorLog "stopping training because memory exceeded limit twice"
        foreach ($proc in $trainProcs) {
            try {
                Stop-Process -Id $proc.ProcessId -Force
                Write-MonitorLog "stopped pid=$($proc.ProcessId)"
            } catch {
                Write-MonitorLog "failed_to_stop pid=$($proc.ProcessId) error=$($_.Exception.Message)"
            }
        }
        break
    }

    Start-Sleep -Seconds $IntervalSeconds
}

