@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0run_secretflow_docker.ps1" -Target mnist %*
