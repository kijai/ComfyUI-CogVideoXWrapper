@echo off

set "python_exec=..\..\..\python_embeded\python.exe"

echo Installing node...

if exist "%python_exec%" (
    echo Installing with ComfyUI Portable
    "%python_exec%" -s -m pip install --pre onediff onediffx && pip install nexfort"
) else (
    echo Installing with system Python
    pip install --pre onediff onediffx && pip install nexfort"
)

pause