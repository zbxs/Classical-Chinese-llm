# Run in local PowerShell; keep this terminal open while using the page.
$ErrorActionPreference = "Stop"
Write-Host "Open http://127.0.0.1:17860/ after the SSH connection is established."
& ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -N -L 127.0.0.1:17860:127.0.0.1:17860 seetacloud-4090d
