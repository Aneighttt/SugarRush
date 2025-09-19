@echo off
echo 正在强制关闭所有相关进程...

:: 强制关闭服务器进程
echo 关闭服务器进程...
taskkill /f /im server.exe >nul 2>&1
if %errorlevel% == 0 (
    echo 服务器进程已关闭
) else (
    echo 未找到服务器进程或已经关闭
)

:: 强制关闭客户端进程
echo 关闭客户端进程...
taskkill /f /im SugarBean.exe >nul 2>&1
if %errorlevel% == 0 (
    echo 客户端进程已关闭
) else (
    echo 未找到客户端进程或已经关闭
)

:: 额外关闭可能的Unity相关进程
echo 关闭Unity相关进程...
taskkill /f /im UnityCrashHandler64.exe >nul 2>&1

echo.
echo 所有进程关闭完成！
:: 自动退出，不等待用户按键
exit