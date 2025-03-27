@echo off
setlocal enabledelayedexpansion

set SUCCESS_LOG=successful_installations.log
set FAILURE_LOG=failed_installations.log

echo Successful installations: > %SUCCESS_LOG%
echo Failed installations: > %FAILURE_LOG%

for /f "tokens=*" %%i in (requirements.txt) do (
    echo Installing %%i
    pip install %%i --ignore-installed --no-cache-dir
    if !errorlevel! equ 0 (
        echo %%i >> %SUCCESS_LOG%
    ) else (
        echo %%i >> %FAILURE_LOG%
    )
)

echo Installation complete. Check %SUCCESS_LOG% and %FAILURE_LOG% for details.