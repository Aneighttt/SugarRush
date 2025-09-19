@echo off

:: 运行 server.exe
start  server.exe


:: 等待 1 秒
timeout /t 1 >nul

:: 运行 b.exe
start  .\client\SugarBean.exe