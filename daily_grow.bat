@echo off
REM daily_grow.bat — 每日论文 KB 自生长 (Windows 包装)
REM
REM 用法:
REM   daily_grow.bat                默认配置 (cs.LG + cs.CL + cs.AI, 最近 1 天, 最多 100 篇)
REM   daily_grow.bat 7              最近 7 天
REM   daily_grow.bat 7 cs.LG cs.AI  自定义类别

setlocal

set DAYS=%1
if "%DAYS%"=="" set DAYS=1
shift

set CATS=%1 %2 %3 %4 %5 %6 %7 %8 %9
if "%CATS%"=="" set CATS=cs.LG cs.CL cs.AI

REM cd 到 KB 目录以便 self_grow.py 找到正确的相对路径
cd /d "E:\peS2o_kb_faiss"

REM UTF-8 输出
chcp 65001 >nul

echo ============================================================
echo  Daily Grow — %date% %time%
echo  Days: %DAYS% | Cats: %CATS%
echo ============================================================

py -3 "E:\peS2o_kb_faiss\daily_grow.py" --days %DAYS% --cats %CATS% --max 200

echo ============================================================
echo  Done. KB log: E:\peS2o_kb_faiss\daily_grow.log
echo ============================================================

endlocal