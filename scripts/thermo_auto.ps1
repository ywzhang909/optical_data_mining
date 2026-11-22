# 温度脉动仪批量出报告的自动化脚本（Windows / PowerShell 5+）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\thermo_auto.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\thermo_auto.ps1 -Days 20261014 20261015
#   powershell -ExecutionPolicy Bypass -File scripts\thermo_auto.ps1 -SliceRule 30min -Python .venv\Scripts\python.exe
#
# 它做四件事：发现日期目录 → 逐次重试运行（本机偶发内存崩溃，靠逐文件缓存续跑）→
# 写日志 → 输出报告目录。报告内含 slices/ 时间片中间数据。
#
# 注册为计划任务（每天 02:00 跑前一天的完整数据）：
#   schtasks /Create /TN ThermoDaily /SC DAILY /ST 02:00 /TR ^
#     "powershell -ExecutionPolicy Bypass -File D:\ZhangH\code\optical_data_mining\scripts\thermo_auto.ps1"

param(
    [string]$DataRoot = "data/daily_data",
    [string]$Out = "data/reports",
    [string]$RunName = "",                       # 默认 auto_<时间戳>
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$SliceRule = "10min",
    [string[]]$Days = @(),                       # 空 = 自动发现全部日期
    [int]$MaxAttempts = 20,
    [int]$WatchdogSeconds = 2400,
    [string]$LogDir = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo

if (-not (Test-Path $Python)) {
    Write-Error "找不到 Python：$Python（先运行 uv sync 或传入 -Python）"
}

if (-not $RunName) {
    $RunName = "auto_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}
if (-not $LogDir) {
    $LogDir = Join-Path $env:TEMP "thermo_auto"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir "$RunName.log"

if (-not $Days -or $Days.Count -eq 0) {
    $Days = Get-ChildItem -Path $DataRoot -Directory |
        Where-Object { $_.Name -match '^\d{8}$' } |
        Sort-Object Name |
        ForEach-Object { $_.Name }
}
if (-not $Days -or $Days.Count -eq 0) {
    Write-Error "在 $DataRoot 下没有找到 YYYYMMDD 日期目录"
}
Write-Host "待分析日期：$($Days -join ', ')"

$env:THERMO_WATCHDOG = "$WatchdogSeconds"
$env:PYTHONIOENCODING = "utf-8"

$argList = @(
    "-m", "ui.analysis.thermo",
    "--data-root", $DataRoot,
    "--out", $Out,
    "--run-name", $RunName,
    "--slice-rule", $SliceRule,
    "--days"
) + $Days

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $attemptLog = Join-Path $LogDir "$RunName.attempt$attempt.log"
    Write-Host "第 $attempt/$MaxAttempts 次：$Python $($argList -join ' ')"
    & $Python @argList *>&1 | Tee-Object -FilePath $attemptLog
    $code = $LASTEXITCODE
    Add-Content -Path $log -Value ("{0} attempt {1} exit={2}" -f (Get-Date -Format s), $attempt, $code)
    if ($code -eq 0) {
        Write-Host "完成：$Out\thermo_$RunName"
        exit 0
    }
    if ($code -eq 2) {
        Write-Error "配置或数据错误（exit=2），不再重试；详见 $attemptLog"
        exit 2
    }
    Write-Warning "第 $attempt 次异常退出（exit=$code），缓存已保留，重试中…"
    Start-Sleep -Seconds 3
}

Write-Error "连续 $MaxAttempts 次失败，请查看 $LogDir 下的日志"
exit 1
