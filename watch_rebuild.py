#!/usr/bin/env python3
"""watch_rebuild.py — 监控重建进度, 完成时通知

用法: py -3 watch_rebuild.py [--interval 60]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

KB_DIR = Path(r'E:/peS2o_kb_faiss')
STATE_FILE = KB_DIR / 'rebuild_state.json'
LOG_FILE = KB_DIR / 'rebuild.log'
PID_FILE = KB_DIR / 'rebuild.pid'


def find_python_pid():
    """Find running rebuild python process."""
    import subprocess
    out = subprocess.run(['powershell', '-Command',
                          "Get-Process python -ErrorAction SilentlyContinue | "
                          "Where-Object { $_.StartTime -gt (Get-Date).AddHours(-6) } | "
                          "Select-Object -ExpandProperty Id"],
                         capture_output=True, text=True)
    pids = [int(line.strip()) for line in out.stdout.split('\n') if line.strip().isdigit()]
    return pids


def main():
    parser = argparse.ArgumentParser(description='watch_rebuild')
    parser.add_argument('--interval', type=int, default=60, help='检查间隔(秒)')
    args = parser.parse_args()

    print(f'Watching rebuild every {args.interval}s. Ctrl+C to stop.')
    last_count = -1

    while True:
        try:
            now = time.strftime('%H:%M:%S')

            # PID status
            pids = find_python_pid()
            pid_status = f'PID={",".join(map(str, pids))}' if pids else 'NO PROCESS'

            # State file
            if STATE_FILE.exists():
                try:
                    with open(STATE_FILE, encoding='utf-8') as f:
                        s = json.load(f)
                    count = s.get('embedded_count', 0)
                    finished = s.get('finished_at')
                    started = s.get('started_at', '?')[:19]
                    delta = count - last_count
                    last_count = count
                    print(f'[{now}] {pid_status} | {started} | {count:,} done | delta {delta:+d} | done={bool(finished)}')

                    if finished:
                        print(f'  REBUILD COMPLETE at {finished}')
                        print(f'  Run: py -3 E:\\peS2o_kb_faiss\\finalize_rebuild.py')
                        return 0
                except Exception as e:
                    print(f'[{now}] state read error: {e}')
            else:
                # No state file means either never started or done+cleaned
                if NEW_FAISS := (KB_DIR / 'papers_clean.index').exists():
                    print(f'[{now}] no state file but papers_clean.index exists — ready to swap')
                    print(f'  Run: py -3 E:\\peS2o_kb_faiss\\finalize_rebuild.py')
                    return 0
                print(f'[{now}] {pid_status} | no state file')

            time.sleep(args.interval)
        except KeyboardInterrupt:
            print('\nstopped')
            return 0


if __name__ == '__main__':
    main()