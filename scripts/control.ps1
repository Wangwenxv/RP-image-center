param([ValidateSet('start','stop')][string]$Action = 'start')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskLogs = Join-Path $taskRoot 'logs'
$taskRuntime = Join-Path $taskRoot 'runtime'
$taskRecord = Join-Path $taskRuntime 'gateway-process.json'
$taskPort = 7865
$taskConfig = Join-Path $taskRoot 'config.local.json'
if (Test-Path -LiteralPath $taskConfig) {
    $taskOptions = Get-Content -Raw -LiteralPath $taskConfig | ConvertFrom-Json
    if ($taskOptions.port) { $taskPort = [int]$taskOptions.port }
}
if ($Action -eq 'start') {
    if (!(Test-Path -LiteralPath $taskPython)) { throw 'Run install.cmd first.' }
    try {
        $taskHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$taskPort/health" -TimeoutSec 5
        if ($taskHealth.ok -and $taskHealth.service -eq 'rp-image-center' -and $taskHealth.root -eq $taskRoot) { Write-Output "Already running: http://127.0.0.1:$taskPort/docs"; exit 0 }
    } catch { }
    New-Item -ItemType Directory -Force -Path $taskLogs,$taskRuntime | Out-Null
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-m','uvicorn','image_center.app:app','--host','127.0.0.1','--port',"$taskPort") -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs 'gateway.stdout.log') -RedirectStandardError (Join-Path $taskLogs 'gateway.stderr.log')
    @{pid=$taskProcess.Id;root=$taskRoot;started=$taskProcess.StartTime.ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $taskRecord
    for ($taskTry=0;$taskTry -lt 30;$taskTry++) {
        Start-Sleep -Milliseconds 500
        if ($taskProcess.HasExited) { throw "Gateway exited. See $taskLogs\gateway.stderr.log" }
        try {
            $taskHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$taskPort/health" -TimeoutSec 5
            if ($taskHealth.ok -and $taskHealth.service -eq 'rp-image-center' -and $taskHealth.root -eq $taskRoot) { Write-Output "Started: http://127.0.0.1:$taskPort/docs"; exit 0 }
        } catch { }
    }
    throw "Gateway startup timed out. See $taskLogs\gateway.stderr.log"
}
try {
    $taskHealth = Invoke-RestMethod -Uri "http://127.0.0.1:$taskPort/health" -TimeoutSec 5
    if ($taskHealth.service -eq 'rp-image-center' -and $taskHealth.root -eq $taskRoot) {
        Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$taskPort/admin/stop-native" -TimeoutSec 20 | Out-Null
    }
} catch { }
foreach ($taskItem in @(@{file=$taskRecord;marker='image_center.app:app'},@{file=(Join-Path $taskRuntime 'native-process.json');marker='sd-server.exe'})) {
    if (!(Test-Path -LiteralPath $taskItem.file)) { continue }
    $taskSaved = Get-Content -Raw -LiteralPath $taskItem.file | ConvertFrom-Json
    $taskPid = [int]$taskSaved.pid
    $taskProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$taskPid" -ErrorAction SilentlyContinue
    if (!$taskProcessInfo) { continue }
    $taskCommand = [string]$taskProcessInfo.CommandLine
    if ($taskCommand.Contains($taskRoot) -and $taskCommand.Contains($taskItem.marker)) {
        # Windows venv Python is a launcher; stop its verified uvicorn child as well.
        if ($taskItem.marker -eq 'image_center.app:app') {
            Get-CimInstance Win32_Process -Filter "ParentProcessId=$taskPid" | Where-Object { ([string]$_.CommandLine).Contains('image_center.app:app') } | ForEach-Object { Stop-Process -Id $_.ProcessId }
        }
        Stop-Process -Id $taskPid -ErrorAction SilentlyContinue
    }
}
Write-Output 'Stopped RP Image Center and its owned native engine.'
