#!/usr/bin/env python3
"""kb_growth_log.py — 记录 KB 增长曲线, 让"自我生长"看得见

每隔 X 小时记录一次 KB 状态, 输出 CSV 格式增长日志
方便观察 daily_grow 是否真的让 KB 在持续增长
"""
import csv
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import faiss

KB = Path(r'E:/peS2o_kb_faiss')
LOG_FILE = KB / 'kb_growth_log.csv'


def get_stats() -> dict:
    idx = faiss.read_index(str(KB / 'papers.index'))
    faiss_n = idx.ntotal

    conn = sqlite3.connect(str(KB / 'papers.db'))
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM papers')
    sql_n = cur.fetchone()[0]
    conn.close()

    ids_file = KB / 'paper_ids.txt'
    ids_n = sum(1 for _ in open(ids_file, encoding='utf-8')) if ids_file.exists() else 0

    idx_size = os.path.getsize(str(KB / 'papers.index')) / 1e9
    db_size = os.path.getsize(str(KB / 'papers.db')) / 1e9

    state_path = KB / 'daily_grow_state.json'
    daily_state = None
    if state_path.exists():
        try:
            import json
            with open(state_path, encoding='utf-8') as f:
                daily_state = json.load(f)
        except Exception:
            pass

    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'faiss_vectors': faiss_n,
        'sqlite_papers': sql_n,
        'paper_ids': ids_n,
        'index_gb': round(idx_size, 3),
        'db_gb': round(db_size, 3),
        'last_daily_run': (daily_state or {}).get('last_run', ''),
        'last_added': (daily_state or {}).get('last_added', 0),
        'total_runs': (daily_state or {}).get('runs', 0),
    }


def main():
    stats = get_stats()

    # Append to CSV (create with header if needed)
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(stats)

    print(f'[{stats["timestamp"]}]')
    print(f'  FAISS: {stats["faiss_vectors"]:,} vectors ({stats["index_gb"]} GB)')
    print(f'  SQLite: {stats["sqlite_papers"]:,} papers ({stats["db_gb"]} GB)')
    print(f'  paper_ids.txt: {stats["paper_ids"]:,} entries')
    print(f'  daily_grow runs: {stats["total_runs"]}, last added: {stats["last_added"]}')
    print(f'  logged to: {LOG_FILE}')


if __name__ == '__main__':
    main()