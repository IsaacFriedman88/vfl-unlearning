@echo off
setlocal EnableDelayedExpansion
set DATASET=%1
shift
set REMAINING_ARGS=
:collect
if "%~1"=="" goto run
set REMAINING_ARGS=!REMAINING_ARGS! "%~1"
shift
goto collect
:run
powershell -ExecutionPolicy Bypass -File "%~dp0run_secretflow_docker.ps1" -Target launcher -Dataset %DATASET% !REMAINING_ARGS!
