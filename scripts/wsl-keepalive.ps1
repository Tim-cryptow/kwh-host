# The keep-alive check for docs/windows-wsl2.md step 5 ("Keep Ubuntu running"), on a real Windows PC.
#
# Run it after `bash ~/wsl-check.sh` has finished its stage 2 in Ubuntu. That stage copies this file
# to your Windows user folder (its code folder, if you have one) and prints the command, e.g.:
#
#   powershell -ExecutionPolicy Bypass -File "C:\Users\<you>\code\wsl-keepalive.ps1" -Distro Ubuntu-24.04
#
# Run it as yourself, not as administrator: the guide's commands should work for an ordinary user.
# It takes about 15 minutes and asks you to close the Ubuntu window once. In order:
#   1. The guide's scheduled task, exactly as written. Does it register without administrator
#      rights, does it open a window, and do Ubuntu and the host keep running with no Ubuntu window?
#   2. The logon path: Ubuntu stopped, then started by the task alone. Does the host's service come
#      back by itself, with no window opened?
#   3. One setting instead of a task that runs forever: [general] instanceIdleTimeout=-1 in
#      .wslconfig (WSL 2.5.4 and later). It asks first, and puts the file back afterwards.
#   4. The results: wsl-check.tgz next to this file. The test host's service is removed.
#   5. The default: with no task and no setting, how soon after its last session WSL stops Ubuntu.
# Everything it prints also goes to wsl-keepalive.log next to this file. Its scheduled task is
# removed at the end, also when stopped with Ctrl+C.
param(
  [string]$Distro = "Ubuntu-24.04",
  [int]$HoldSeconds = 180
)
$ErrorActionPreference = "Continue"
$env:WSL_UTF8 = "1"                       # wsl.exe writes UTF-8 instead of UTF-16
$TaskName = "kWh host (WSL)"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Log = Join-Path $Here "wsl-keepalive.log"
$WslConfig = Join-Path $env:USERPROFILE ".wslconfig"
$CheckScript = 'exec bash $HOME/kwh-host-src/scripts/wsl-check.sh'   # $HOME is Ubuntu's, expanded by bash
$Verdicts = New-Object System.Collections.Generic.List[string]
$script:SavedConfig = $null

function Say([string]$m) {
  $line = "[{0}] {1}" -f [DateTime]::UtcNow.ToString("HH:mm:ss"), $m
  Write-Host $line
  Add-Content -Path $Log -Value $line -Encoding UTF8
}

function Verdict([string]$m) { $Verdicts.Add($m); Say $m }

function Clean($lines) {
  # wsl.exe output: drop the NULs of UTF-16 (older WSL ignores WSL_UTF8), trailing space, empty lines
  @($lines | ForEach-Object { ("" + $_) -replace "`0", "" } | ForEach-Object { $_.TrimEnd() } | Where-Object { $_ -ne "" })
}

function Test-Running {
  $names = Clean (& wsl.exe --list --running --quiet)
  return [bool]($names -contains $Distro)
}

function Wait-Running([bool]$want, [int]$max) {
  # seconds until the distro is (or is no longer) running; -1 if not within $max
  $sw = [Diagnostics.Stopwatch]::StartNew()
  while ($sw.Elapsed.TotalSeconds -lt $max) {
    if ((Test-Running) -eq $want) { return [int]$sw.Elapsed.TotalSeconds }
    Start-Sleep -Seconds 2
  }
  return -1
}

function Wait-Seconds([int]$secs, [string]$what) {
  $end = [DateTime]::UtcNow.AddSeconds($secs)
  while ([DateTime]::UtcNow -lt $end) {
    $left = [int]($end - [DateTime]::UtcNow).TotalSeconds
    Write-Host -NoNewline ("`r    {0}: {1}:{2:00} to go   " -f $what, [int][Math]::Floor($left / 60), ($left % 60))
    Start-Sleep -Seconds 1
  }
  Write-Host ("`r    {0}: done{1}" -f $what, (" " * 16))
}

function Ask([string]$q) {
  while ($true) {
    $a = ("" + (Read-Host ($q + " (y/n)"))).Trim().ToLower()
    if ($a -eq "y" -or $a -eq "yes") { Add-Content -Path $Log -Value ("    " + $q + " y") -Encoding UTF8; return $true }
    if ($a -eq "n" -or $a -eq "no") { Add-Content -Path $Log -Value ("    " + $q + " n") -Encoding UTF8; return $false }
  }
}

function Invoke-Probe([string]$label, [int]$window = 0, [switch]$Quiet) {
  # wsl-check.sh probe: the host's state inside Ubuntu, ending with "RESULT key=value ..."
  $out = Clean (& wsl.exe -d $Distro --exec bash -c "$CheckScript probe $label $window 2>&1")
  $r = @{ exit = $LASTEXITCODE; lines = @() }
  foreach ($l in $out) {
    if ($l.StartsWith("RESULT ")) {
      foreach ($kv in $l.Substring(7).Split(" ")) {
        $i = $kv.IndexOf("=")
        if ($i -gt 0) { $r[$kv.Substring(0, $i)] = $kv.Substring($i + 1) }
      }
    } else {
      $r.lines += $l
    }
  }
  if (-not $Quiet) { foreach ($l in $r.lines) { Say ("    " + $l) } }
  return $r
}

function Num($v) { if ($null -eq $v -or "$v" -eq "") { return -1 }; return [int]$v }

function Remove-Task {
  if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  }
}

function Set-WslConfig {
  $had = Test-Path -LiteralPath $WslConfig
  $bytes = $null
  $lines = @()
  if ($had) { $bytes = [IO.File]::ReadAllBytes($WslConfig); $lines = [IO.File]::ReadAllLines($WslConfig) }
  $script:SavedConfig = @{ had = $had; bytes = $bytes }
  $new = New-Object System.Collections.Generic.List[string]
  $done = $false
  foreach ($l in $lines) {
    if ($l -match '^\s*instanceIdleTimeout\s*=') { continue }
    $new.Add($l)
    if (-not $done -and $l -match '^\s*\[general\]\s*$') { $new.Add("instanceIdleTimeout=-1"); $done = $true }
  }
  if (-not $done) {
    if ($new.Count -gt 0) { $new.Add("") }
    $new.Add("[general]")
    $new.Add("instanceIdleTimeout=-1")
  }
  [IO.File]::WriteAllLines($WslConfig, [string[]]$new.ToArray())
  Say ("    {0} for now:" -f $WslConfig)
  foreach ($l in $new) { Say ("      " + $l) }
}

function Restore-WslConfig {
  if ($null -eq $script:SavedConfig) { return }
  if ($script:SavedConfig.had) {
    [IO.File]::WriteAllBytes($WslConfig, [byte[]]$script:SavedConfig.bytes)
  } else {
    Remove-Item -LiteralPath $WslConfig -ErrorAction SilentlyContinue
  }
  $script:SavedConfig = $null
  Say ("    {0} is back as it was" -f $WslConfig)
}

try {
  Say "keep-alive check for $Distro"
  $os = Get-CimInstance Win32_OperatingSystem
  $battery = @(Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue).Count -gt 0
  $admin = $false
  try {
    $me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    $admin = $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  } catch { }
  $info = "Windows: {0}, build {1}; battery: {2}; PowerShell {3}; as administrator: {4}"
  Say ($info -f $os.Caption, $os.BuildNumber, $battery, $PSVersionTable.PSVersion, $admin)
  if ($admin) { Say "  (as administrator, step 1 says nothing about ordinary users; run it in a normal PowerShell if you can)" }
  $wslVersion = $null
  foreach ($l in (Clean (& wsl.exe --version))) {
    Say ("  " + $l)
    if (-not $wslVersion -and $l -match '(\d+\.\d+\.\d+(\.\d+)?)') { $wslVersion = [version]$Matches[1] }
  }
  $distros = Clean (& wsl.exe --list --quiet)
  if (-not ($distros -contains $Distro)) {
    Say ("There is no WSL distro named {0}; yours: {1}. Run this again with -Distro and one of those." -f $Distro, ($distros -join ", "))
    exit 1
  }
  if (-not (Test-Running)) {
    Say "$Distro is not running. In Ubuntu run   bash ~/wsl-check.sh   (stage 2), then this, with that window still open."
    exit 1
  }

  Say "0. The test host, before anything"
  $p = Invoke-Probe "start"
  if ($p.exit -ne 0 -or $p.platform -ne "live") {
    Say "The test host is not live. In Ubuntu run   bash ~/wsl-check.sh   (stage 2) and let it finish first."
    exit 1
  }

  # --- 1 -------------------------------------------------------------------------------------------
  Say "1. The guide's task, exactly as written (docs/windows-wsl2.md, step 5)"
  if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    if (-not (Ask "  A scheduled task named '$TaskName' already exists. Replace it for this check? (It is removed at the end.)")) { exit 1 }
    Remove-Task
  }
  $a = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d $Distro --exec sleep infinity"
  $t = New-ScheduledTaskTrigger -AtLogOn
  $registered = $false
  try {
    Register-ScheduledTask -TaskName $TaskName -Action $a -Trigger $t -ErrorAction Stop | Out-Null
    $registered = $true
    Verdict "1a. The guide's commands register the task$(if (-not $admin) { ' without administrator rights' })."
  } catch {
    Verdict ("1a. The guide's commands do NOT register the task: " + $_.Exception.Message.Trim())
    $who = "$env:USERDOMAIN\$env:USERNAME"
    $t = New-ScheduledTaskTrigger -AtLogOn -User $who
    try {
      Register-ScheduledTask -TaskName $TaskName -Action $a -Trigger $t -ErrorAction Stop | Out-Null
      $registered = $true
      Verdict "    With the trigger limited to this user (New-ScheduledTaskTrigger -AtLogOn -User $who) it registers."
    } catch {
      Verdict ("    With -User it does not register either: " + $_.Exception.Message.Trim())
    }
  }

  if ($registered) {
    $task = Get-ScheduledTask -TaskName $TaskName
    $s = $task.Settings
    $info = "    the task's settings: stops after {0}; starts on battery: {1}; stops on battery: {2}; runs as {3} ({4})"
    Say ($info -f $s.ExecutionTimeLimit, (-not $s.DisallowStartIfOnBatteries), $s.StopIfGoingOnBatteries,
      $task.Principal.UserId, $task.Principal.LogonType)
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 4
    Say ("    started: task {0}, wsl.exe processes now {1}" -f (Get-ScheduledTask -TaskName $TaskName).State, @(Get-Process wsl -ErrorAction SilentlyContinue).Count)
    $window = Ask "  Did a new window (or a new tab in Terminal) open just now?"
    Verdict ("1b. Starting the task " + $(if ($window) { "opens a window." } else { "opens no window." }))
    Write-Host ""
    Write-Host "  Now close the Ubuntu window (or tab) where you ran wsl-check.sh, and any other Ubuntu window."
    if ($window) { Write-Host "  Leave open the window the task just opened." }
    [void](Read-Host "  Press Enter here once they are closed")
    $closed = [DateTime]::UtcNow
    Say "    Ubuntu windows closed"
    Wait-Seconds $HoldSeconds "Ubuntu and the host should keep running"
    $since = [int]([DateTime]::UtcNow - $closed).TotalSeconds
    if (Test-Running) {
      $p = Invoke-Probe "task-hold" $since
      $gap = Num $p.gap
      if ($p.exit -eq 0 -and $p.platform -eq "live" -and $gap -ge 0 -and $gap -le 30 -and (Num $p.up) -ge $since) {
        Verdict ("1c. With the task and no Ubuntu window for {0} s, Ubuntu kept running and the host stayed live (heartbeats at most {1} s apart)." -f $since, $gap)
      } else {
        $info = "1c. With the task and no Ubuntu window for {0} s, Ubuntu kept running but the host did NOT stay live: service {1}, engine {2}, platform {3}, longest heartbeat gap {4} s, Ubuntu up {5} s."
        Verdict ($info -f $since, $p.service, $p.engine, $p.platform, $gap, $p.up)
      }
    } else {
      Verdict "1c. With the task running, Ubuntu STOPPED anyway once its windows were closed."
    }

    # --- 2 -----------------------------------------------------------------------------------------
    Say "2. The logon path: Ubuntu stopped, then started by the task alone"
    Stop-ScheduledTask -TaskName $TaskName
    & wsl.exe --shutdown
    if ((Wait-Running $false 60) -lt 0) { Say "    $Distro did not stop after wsl --shutdown" }
    Say "    Ubuntu stopped (wsl --shutdown); starting the task, as Windows does at logon"
    Start-ScheduledTask -TaskName $TaskName
    $w = Wait-Running $true 90
    if ($w -lt 0) {
      Verdict "2. The task did NOT start Ubuntu within 90 s."
    } else {
      Say "    the task started Ubuntu in $w s; now waiting for the host's service, with no Ubuntu window"
      $p = $null
      $deadline = [DateTime]::UtcNow.AddSeconds(240)
      while ([DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Seconds 15
        $p = Invoke-Probe "logon" -Quiet
        if ($p.exit -eq 0) { break }
      }
      foreach ($l in $p.lines) { Say ("    " + $l) }
      if ($p.exit -eq 0) {
        Verdict ("2. Started by the task alone, Ubuntu brought the host back by itself: its service {0} s and its engine {1} s after Ubuntu started. (Its heartbeats fail now: the test platform does not survive a restart.)" -f $p.svc_boot, $p.eng_boot)
      } else {
        Verdict ("2. Started by the task alone, Ubuntu did NOT bring the host back within 4 minutes: service {0}, engine {1}." -f $p.service, $p.engine)
      }
    }
  }
  Remove-Task
  Say "    the task is removed"

  # --- 3 -------------------------------------------------------------------------------------------
  Say "3. Instead of a task that runs forever: instanceIdleTimeout=-1 in .wslconfig"
  if ($null -eq $wslVersion -or $wslVersion -lt [version]"2.5.4") {
    Verdict "3. Skipped: this WSL ($wslVersion) is older than 2.5.4, which introduced instanceIdleTimeout."
  } elseif (-not (Ask "  This adds [general] instanceIdleTimeout=-1 to $WslConfig for about 5 minutes, then puts the file back as it was. Go ahead?")) {
    Verdict "3. Skipped (you said no)."
  } else {
    Set-WslConfig
    & wsl.exe --shutdown
    [void](Wait-Running $false 60)
    $null = & wsl.exe -d $Distro --exec true
    $started = [DateTime]::UtcNow
    Say "    Ubuntu started by a command that ended at once: no window, no task, nothing holding it"
    Wait-Seconds $HoldSeconds "Ubuntu should keep running"
    $since = [int]([DateTime]::UtcNow - $started).TotalSeconds
    if (Test-Running) {
      $p = Invoke-Probe "setting-hold" $since
      $gap = Num $p.gap
      if ($p.exit -eq 0 -and $gap -ge 0 -and $gap -le 75 -and (Num $p.up) -ge $since) {
        Verdict ("3. With instanceIdleTimeout=-1 and nothing holding it, Ubuntu kept running for {0} s and so did the host (service up {1} s after Ubuntu started; it kept heartbeating, at most {2} s apart, into the platform that is gone)." -f $since, $p.svc_boot, $gap)
      } else {
        Verdict ("3. With instanceIdleTimeout=-1, Ubuntu kept running for {0} s, but the host did not: service {1}, engine {2}, longest heartbeat gap {3} s, Ubuntu up {4} s." -f $since, $p.service, $p.engine, $gap, $p.up)
      }
    } else {
      Verdict "3. With instanceIdleTimeout=-1, Ubuntu still STOPPED with nothing holding it."
    }
    Restore-WslConfig
    & wsl.exe --shutdown
    [void](Wait-Running $false 60)
  }

  # --- 4 -------------------------------------------------------------------------------------------
  Say "4. The results"
  foreach ($l in (Clean (& wsl.exe -d $Distro --exec bash -c "$CheckScript collect 2>&1"))) { Say ("    " + $l) }
  $lastSession = [DateTime]::UtcNow

  # --- 5 -------------------------------------------------------------------------------------------
  Say "5. The default: no task, no setting"
  $open = @(Get-Process wsl -ErrorAction SilentlyContinue).Count
  if ($open -gt 0) {
    Write-Host "  WSL sessions still open: $open. Close every Ubuntu window now."
    [void](Read-Host "  Press Enter once they are closed")
    $lastSession = [DateTime]::UtcNow
  }
  if ((Wait-Running $false 300) -ge 0) {
    Verdict ("5. With nothing holding it, WSL stopped Ubuntu {0} s after its last session ended." -f [int]([DateTime]::UtcNow - $lastSession).TotalSeconds)
  } else {
    Verdict "5. With nothing holding it, Ubuntu was still running 5 minutes after its last session ended."
  }

  Say "Summary"
  foreach ($v in $Verdicts) { Say ("  " + $v) }
  Say ("Results: {0} and {1}" -f (Join-Path $Here "wsl-check.tgz"), $Log)
} finally {
  Remove-Task
  Restore-WslConfig
}
