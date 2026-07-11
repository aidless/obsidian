@echo off
REM setup_scheduler.bat — 注册每日定时任务
REM 用管理员权限运行一次即可

set TASK_NAME=PaperKB-DailyGrow
set SCRIPT=E:\peS2o_kb_faiss\daily_grow.bat
set TIME=03:00

echo 注册每日 %TIME% 自动运行: %SCRIPT%
schtasks /create /tn "%TASK_NAME%" /tr "\"%SCRIPT%\"" /sc daily /st %TIME% /f

if %errorlevel% equ 0 (
    echo [OK] 任务已注册
    echo 查看任务: schtasks /query /tn "%TASK_NAME%"
    echo 删除任务: schtasks /delete /tn "%TASK_NAME%" /f
    echo 立即运行: schtasks /run /tn "%TASK_NAME%"
) else (
    echo [FAIL] 需要管理员权限, 请右键此文件 → "以管理员身份运行"
)

pause