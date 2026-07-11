@echo off
REM post_daily_grow_check.bat
REM 在 daily_grow 调度任务之后 5 分钟运行,作为独立健康检查兜底
REM 若 daily_grow 内部健康检查失败,这里再跑一次并把结果写到日志
REM
REM 注册方法 (管理员 cmd):
REM   schtasks /create /tn "KB Daily Health" /tr "E:\peS2o_kb_faiss\post_daily_grow_check.bat" ^
REM     /sc daily /st 03:10
REM
REM 或修改原 daily_grow 任务的 "多个程序" 列表

setlocal
set KB_DIR=E:\peS2o_kb_faiss
set PY=py -3
set LOG=%KB_DIR%\post_health_check.log
set HEALTH_JSON=%KB_DIR%\last_health_check.json

echo ════════════════════════════════════════════════════════════════ > %LOG%
echo  POST DAILY_GROW HEALTH CHECK — %date% %time% >> %LOG%
echo ════════════════════════════════════════════════════════════════ >> %LOG%

cd /d "%KB_DIR%"

REM 跑严格模式,JSON 输出供其他工具消费
%PY% "%KB_DIR%\kb_health.py" --strict --sample 2000 --json >> %LOG% 2>&1
set RC=%ERRORLEVEL%

echo. >> %LOG%
echo Exit code: %RC% >> %LOG%

if %RC% EQU 0 (
    echo ✅ KB healthy — no action needed >> %LOG%
) else if %RC% EQU 1 (
    echo ⚠ KB WARN — see %HEALTH_JSON% for details >> %LOG%
    echo ⚠ 建议下次手动跑: py -3 "%KB_DIR%\finalize_rebuild.py" >> %LOG%
) else (
    echo 🚨 KB DANGER — 主索引严重漂移 >> %LOG%
    echo 🚨 立即跑: py -3 "%KB_DIR%\finalize_rebuild.py" >> %LOG%
)

endlocal