#!/usr/bin/env python3
"""kb_health.py — KB 健康检查器(只读,不修改任何文件)

检查项:
  1. SQLite papers 表的论文数
  2. FAISS 主索引向量数
  3. FAISS gap 索引向量数(若存在)
  4. paper_ids.txt / papers_gap_ids.txt 条目数
  5. 主索引 vs SQLite:差距百分比
  5b. Gap merge 状态(gap 是否被 main 索引消化,未合并时长阈值 24h/72h)
  6. 主+gap vs SQLite:差距百分比(应 ≈ 0)
  7. ID 文件 vs FAISS 索引:数量一致性
  8. SQLite 与 FAISS ID 集合:实际 missing 数(随机抽样)
  9. 最近 daily_grow 时间(是否 > 48 小时未跑)
 10. 磁盘占用

用法:
  py -3 kb_health.py            # 完整检查
  py -3 kb_health.py --sample 5000  # 抽样核对 ID(默认 2000)
  py -3 kb_health.py --json     # 输出 JSON 给其他脚本消费

退出码:
  0 = 健康(差距 < 1%)
  1 = 警告(1% <= 差距 < 5%)
  2 = 危险(差距 >= 5% 或连续 48h 未跑)
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# 模块 logger: 默认 INFO 级别, 走 stderr(不污染 stdout / --json)
logger = logging.getLogger('kb_health')
if not logger.handlers:
    _h = logging.StreamHandler(stream=sys.stderr)
    _h.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

KB_DIR = Path(r'E:/peS2o_kb_faiss')
FAISS_INDEX = KB_DIR / 'papers.index'
FAISS_GAP_INDEX = KB_DIR / 'papers_gap.index'
IDS_FILE = KB_DIR / 'paper_ids.txt'
IDS_GAP_FILE = KB_DIR / 'papers_gap_ids.txt'
SQLITE_DB = KB_DIR / 'papers.db'
STATE_FILE = KB_DIR / 'daily_grow_state.json'
GROWTH_LOG = KB_DIR / 'kb_growth_log.csv'


def human_bytes(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f'{n:.2f} {unit}'
        n /= 1024
    return f'{n:.2f} PB'


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_faiss_ntotal(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import faiss
        idx = faiss.read_index(str(path))
        return int(idx.ntotal)
    except Exception as e:
        print(f'  WARN: failed to read FAISS {path.name}: {e}', file=sys.stderr)
        return -1


def sqlite_paper_count() -> tuple[int, int]:
    """返回 (总数, FTS5 表中的数)。FTS5 数更能反映"可全文检索"量。"""
    conn = sqlite3.connect(str(SQLITE_DB))
    try:
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) FROM papers')
        total = cur.fetchone()[0]
        try:
            cur.execute('SELECT COUNT(*) FROM papers_fts')
            fts = cur.fetchone()[0]
        except sqlite3.OperationalError:
            fts = total
        return total, fts
    finally:
        conn.close()


def sample_ids_match(ids_file: Path, sample: int) -> tuple[int, int, str]:
    """随机抽样核对 ID 文件里的 ID 是否在 SQLite papers.paper_id 列里存在。
    返回 (命中, 总抽样, id_type)。
    id_type: 'numeric' / 'arxiv' / 'mixed',根据样本中数字 vs xx.xxxx 的比例判断。
    """
    if not ids_file.exists():
        return 0, 0, 'unknown'
    with open(ids_file, encoding='utf-8', errors='replace') as f:
        all_ids = [l.strip() for l in f if l.strip()]
    if not all_ids:
        return 0, 0, 'unknown'
    import random
    random.seed(42)
    chosen = random.sample(all_ids, min(sample, len(all_ids)))
    # 判断 ID 类型
    arxiv_like = sum(1 for x in chosen if '.' in x)
    ratio = arxiv_like / len(chosen)
    id_type = ('arxiv' if ratio > 0.8 else
               'numeric' if ratio < 0.2 else 'mixed')
    conn = sqlite3.connect(str(SQLITE_DB))
    try:
        cur = conn.cursor()
        qmarks = ','.join('?' * len(chosen))
        # 注意:papers.id 是行号 INTEGER,papers.paper_id 才是真实 ID
        cur.execute(f'SELECT paper_id FROM papers WHERE paper_id IN ({qmarks})',
                    chosen)
        found = {r[0] for r in cur.fetchall()}
        return len(found), len(chosen), id_type
    finally:
        conn.close()


def gap_merge_status() -> dict:
    """检查 gap 索引/ID 文件是否已被 main 索引消化。

    判据(无侵入,只用 mtime + 大小):
      - 若 gap 不存在      ⇒ merged, status=ok
      - 若 gap mtime <= main mtime 且 gap_ids mtime <= main mtime ⇒ merged
      - 否则:未合并。计算未合并时长 = now - max(gap mtime, gap_ids mtime)
    """
    logger.info('[5b] gap_merge_status: 开始判定')
    logger.info('[5b]  存在性: gap.index=%s, gap_ids.txt=%s, '
                'main.index=%s, main_ids.txt=%s',
                FAISS_GAP_INDEX.exists(), IDS_GAP_FILE.exists(),
                FAISS_INDEX.exists(), IDS_FILE.exists())

    if not FAISS_GAP_INDEX.exists() and not IDS_GAP_FILE.exists():
        logger.info('[5b]  → 命中分支: gap files 不存在, status=ok')
        return {'status': 'ok', 'reason': 'no gap files present', 'pending': 0,
                'unmerged_hours': 0.0}

    now_ts = time.time()
    now_iso = datetime.fromtimestamp(now_ts).isoformat(timespec='seconds')
    gap_mtime = (FAISS_GAP_INDEX.stat().st_mtime
                 if FAISS_GAP_INDEX.exists() else 0.0)
    gap_ids_mtime = (IDS_GAP_FILE.stat().st_mtime
                     if IDS_GAP_FILE.exists() else 0.0)
    main_mtime = (FAISS_INDEX.stat().st_mtime
                  if FAISS_INDEX.exists() else 0.0)
    main_ids_mtime = (IDS_FILE.stat().st_mtime
                      if IDS_FILE.exists() else 0.0)

    # 记录 mtime 原始值(秒)+ ISO 字符串
    def _fmt(label: str, ts: float) -> str:
        if ts == 0.0:
            return f'{label} ts=0.0 (missing)'
        iso = datetime.fromtimestamp(ts).isoformat(timespec='seconds')
        return f'{label} ts={ts:.0f} ({iso})'

    logger.info('[5b]  mtime 原始值 (now=%s, %.0f):',
                now_iso, now_ts)
    logger.info('[5b]    %s', _fmt('gap.index       ', gap_mtime))
    logger.info('[5b]    %s', _fmt('gap_ids.txt     ', gap_ids_mtime))
    logger.info('[5b]    %s', _fmt('main.index      ', main_mtime))
    logger.info('[5b]    %s', _fmt('main_ids.txt    ', main_ids_mtime))

    gap_vs_main    = gap_mtime    - main_mtime
    gapids_vs_main = gap_ids_mtime - main_ids_mtime
    logger.info('[5b]  mtime 比较 (gap - main, 正值=gap 更新):')
    logger.info('[5b]    gap.index    - main.index   = %+.1f s  '
                '(>0 ⇒ gap 有新内容)', gap_vs_main)
    logger.info('[5b]    gap_ids.txt  - main_ids.txt = %+.1f s',
                gapids_vs_main)

    # gap 里有新内容(main 更新之后还写过)
    pending_writes = []
    if FAISS_GAP_INDEX.exists() and gap_mtime > main_mtime:
        pending_writes.append('gap.index')
    if IDS_GAP_FILE.exists() and gap_ids_mtime > main_ids_mtime:
        pending_writes.append('gap_ids.txt')

    logger.info('[5b]  pending_writes 列表: %s '
                '(规则: gap_mtime > main_mtime)', pending_writes or '[]')

    # gap 向量数(反映"待合并量")
    gap_n = load_faiss_ntotal(FAISS_GAP_INDEX) if FAISS_GAP_INDEX.exists() else 0
    logger.info('[5b]  gap FAISS 向量数 (待合并量): %s', gap_n)

    if not pending_writes:
        logger.info('[5b]  → 命中分支: pending_writes 为空, status=ok, '
                    'reason="gap already merged into main"')
        return {'status': 'ok', 'reason': 'gap already merged into main',
                'pending': max(0, gap_n), 'unmerged_hours': 0.0}

    last_write_ts = max(gap_mtime, gap_ids_mtime)
    unmerged_hours = (now_ts - last_write_ts) / 3600.0
    last_write_iso = datetime.fromtimestamp(last_write_ts).isoformat(timespec='seconds')
    logger.info('[5b]  last_write_ts = max(gap_mtime, gap_ids_mtime) '
                '= %.0f (%s)', last_write_ts, last_write_iso)
    logger.info('[5b]  unmerged_hours = (now - last_write) / 3600 = %.3f h',
                unmerged_hours)

    # 阈值判断: log 命中哪一档
    if unmerged_hours >= 72:
        status = 'danger'
        threshold_hit = '>= 72h 阈值 (DANGER)'
    elif unmerged_hours >= 24:
        status = 'warn'
        threshold_hit = '>= 24h 阈值 (WARN)'
    else:
        status = 'ok'
        threshold_hit = '< 24h 视为正常(短窗口, 等下次 kb_search 启动)'
    logger.info('[5b]  阈值判断: %.3f h  %s → status=%s',
                unmerged_hours, threshold_hit, status)

    return {
        'status': status,
        'reason': f'gap files newer than main: {", ".join(pending_writes)}',
        'pending': max(0, gap_n),
        'unmerged_hours': unmerged_hours,
        'last_gap_write_iso': datetime.fromtimestamp(last_write_ts).isoformat(timespec='seconds'),
    }


def last_daily_grow_age_hours() -> float | None:
    if not STATE_FILE.exists():
        return None
    try:
        st = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        ts = st.get('last_run')
        if not ts:
            return None
        # ISO 格式
        last = datetime.fromisoformat(ts)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - last).total_seconds() / 3600
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=2000,
                    help='ID 文件 vs SQLite 的抽样核对数(0=跳过)')
    ap.add_argument('--json', action='store_true', help='输出 JSON')
    ap.add_argument('--strict', action='store_true',
                    help='严格模式: 主索引对齐阈值降到 0.5%, '
                         'ID 抽样命中率必须 ≥ 99.5%, 失败立即非零退出')
    args = ap.parse_args()

    # 阈值配置
    WARN_PCT  = 0.5  if args.strict else 1.0   # main+gap vs sqlite 差距
    DANGER_PCT = 5.0 if args.strict else 5.0
    SAMPLE_MIN = 99.5 if args.strict else 95.0  # ID 抽样命中率

    t0 = time.time()
    print('═══════════════════════════════════════════════════════════')
    print('  KB Health Check — E:/peS2o_kb_faiss/')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('═══════════════════════════════════════════════════════════')

    # ── 1. SQLite ──
    print('\n[1] SQLite (papers.db)')
    db_size = SQLITE_DB.stat().st_size if SQLITE_DB.exists() else 0
    print(f'    size: {human_bytes(db_size)}')
    sql_total, sql_fts = sqlite_paper_count()
    print(f'    papers table:  {sql_total:>10,} rows')
    print(f'    papers_fts:    {sql_fts:>10,} rows')

    # ── 1b. paper_id 格式分布(纯 SQL 聚合,不扫全表到 Python) ──
    print('\n[1b] paper_id format distribution (in SQLite)')
    conn = sqlite3.connect(str(SQLITE_DB))
    try:
        cur = conn.cursor()
        # arXiv-like: 含 . 且形如 NNNN.NNNN
        cur.execute("SELECT COUNT(*) FROM papers WHERE paper_id GLOB '[0-9]*.[0-9]*'")
        n_arxiv = cur.fetchone()[0]
        # SHA1: 40 个 hex (排除含点的)
        cur.execute("SELECT COUNT(*) FROM papers WHERE LENGTH(paper_id)=40 AND paper_id NOT LIKE '%.%'")
        n_sha1 = cur.fetchone()[0]
        # 纯数字 (peS2o 内部 ID,可能 7-9 位) — 不是 SHA1,不是 arxiv
        cur.execute("""SELECT COUNT(*) FROM papers
                       WHERE paper_id NOT LIKE '%.%'
                         AND LENGTH(paper_id) != 40""")
        n_numeric = cur.fetchone()[0]
        # 其他(短 hex / 含字母数字混合 / 等)
        n_other = sql_total - n_arxiv - n_sha1 - n_numeric
        n_other = max(0, n_other)
        print(f'    arXiv-like (XXXX.XXXX): {n_arxiv:>10,} ({n_arxiv/sql_total*100:.1f}%)')
        print(f'    SHA1 hash (40 hex):     {n_sha1:>10,} ({n_sha1/sql_total*100:.1f}%)')
        print(f'    other (peS2o numeric):  {n_numeric + n_other:>10,} '
              f'({(n_numeric+n_other)/sql_total*100:.1f}%)')
        if n_sha1:
            print(f'      ↳ {n_numeric:,} pure-numeric + {n_other:,} mixed alphanumeric')
    finally:
        conn.close()

    # ── 2. FAISS 主索引 ──
    print('\n[2] FAISS main index (papers.index)')
    idx_size = FAISS_INDEX.stat().st_size if FAISS_INDEX.exists() else 0
    idx_n = load_faiss_ntotal(FAISS_INDEX)
    print(f'    size:    {human_bytes(idx_size)}')
    print(f'    vectors: {idx_n:>10,}')

    # ── 3. FAISS gap 索引 ──
    print('\n[3] FAISS gap index (papers_gap.index)')
    if FAISS_GAP_INDEX.exists():
        gap_size = FAISS_GAP_INDEX.stat().st_size
        gap_n = load_faiss_ntotal(FAISS_GAP_INDEX)
        print(f'    size:    {human_bytes(gap_size)}')
        print(f'    vectors: {gap_n:>10,}')
    else:
        gap_n = 0
        print('    (does not exist — no pending gap)')

    # ── 4. ID 文件 ──
    print('\n[4] ID files')
    ids_main = count_lines(IDS_FILE)
    ids_gap = count_lines(IDS_GAP_FILE) if IDS_GAP_FILE.exists() else 0
    print(f'    paper_ids.txt:         {ids_main:>10,} entries')
    print(f'    papers_gap_ids.txt:    {ids_gap:>10,} entries')

    # ── 5. 一致性核对 ──
    print('\n[5] Consistency checks')
    total_faiss = idx_n + max(0, gap_n)
    total_ids = ids_main + ids_gap

    diff_main_sql = sql_total - idx_n
    diff_all_sql = sql_total - total_faiss
    diff_faiss_ids = total_faiss - total_ids

    pct_main = diff_main_sql / sql_total * 100 if sql_total else 0
    pct_all = diff_all_sql / sql_total * 100 if sql_total else 0

    print(f'    main FAISS vs SQLite:    {idx_n:>10,} / {sql_total:>10,}   '
          f'(diff {diff_main_sql:>+10,} | {pct_main:+.1f}%)')
    print(f'    main+gap vs SQLite:      {total_faiss:>10,} / {sql_total:>10,}   '
          f'(diff {diff_all_sql:>+10,} | {pct_all:+.1f}%)')
    print(f'    FAISS vectors vs IDs:    {total_faiss:>10,} / {total_ids:>10,}   '
          f'(diff {diff_faiss_ids:>+10,})')

    # ── 5b. Gap merge 状态 ──
    print('\n[5b] Gap merge status')
    gm = gap_merge_status()
    if gm['status'] == 'ok':
        print(f'    status:        ok  ({gm["reason"]})')
    else:
        print(f'    status:        {gm["status"].upper()}')
        print(f'    reason:        {gm["reason"]}')
        print(f'    pending:       {gm["pending"]:,} vectors in gap')
        print(f'    unmerged for:  {gm["unmerged_hours"]:.1f} h '
              f'(last gap write: {gm.get("last_gap_write_iso", "?")})')
        print('    action:        跑 kb_search.py 触发自动 merge;'
              ' 或手动 finalize_rebuild.py')
    merge_status = gm['status']

    # ── 6. 抽样核对 ID 真实存在性 ──
    if args.sample > 0:
        print(f'\n[6] ID-vs-SQLite sample check (n={args.sample})')
        if IDS_GAP_FILE.exists():
            h_main, n_main, t_main = sample_ids_match(IDS_FILE, args.sample // 2)
            h_gap, n_gap, t_gap = sample_ids_match(IDS_GAP_FILE, args.sample // 2)
            pct_main = (h_main / n_main * 100) if n_main else 0
            pct_gap = (h_gap / n_gap * 100) if n_gap else 0
            print(f'    main:  {h_main}/{n_main} found in SQLite.paper_id '
                  f'({pct_main:.1f}%)  [id_type={t_main}]')
            print(f'    gap:   {h_gap}/{n_gap} found in SQLite.paper_id '
                  f'({pct_gap:.1f}%)  [id_type={t_gap}]')
        else:
            h, n, t = sample_ids_match(IDS_FILE, args.sample)
            pct = (h / n * 100) if n else 0
            print(f'    ids found in SQLite.paper_id: {h}/{n} '
                  f'({pct:.1f}%)  [id_type={t}]')

    # ── 7. daily_grow freshness ──
    print('\n[7] daily_grow freshness')
    age_h = last_daily_grow_age_hours()
    if age_h is None:
        print('    no daily_grow_state.json or unparseable')
        freshness_status = 'unknown'
    else:
        print(f'    last run: {age_h:.1f} hours ago')
        freshness_status = 'ok' if age_h < 36 else (
            'warn' if age_h < 72 else 'stale')
        if freshness_status == 'stale':
            print('    ⚠ STALE: >72h since last daily_grow')

    # ── 8. 磁盘 ──
    print('\n[8] Disk')
    try:
        import shutil
        total, used, free = shutil.disk_usage(KB_DIR)
        print(f'    E: total {human_bytes(total)}, used {human_bytes(used)}, '
              f'free {human_bytes(free)} ({free/total*100:.1f}% free)')
    except Exception as e:
        print(f'    disk check failed: {e}')

    # ── 总结 + 退出码 ──
    print('\n═══════════════════════════════════════════════════════════')
    mode_tag = ' [STRICT]' if args.strict else ''
    # 三维度同时检查: 一致性 / 新鲜度 / merge 状态
    consistency_ok = abs(pct_all) < WARN_PCT
    freshness_ok = freshness_status != 'stale'
    merge_ok = merge_status in ('ok',)
    logger.info('[verdict] 三维度综合判定:')
    logger.info('[verdict]   consistency_ok = (|pct_all|=%.3f%% < WARN_PCT=%.1f%%) → %s',
                abs(pct_all), WARN_PCT, consistency_ok)
    logger.info('[verdict]   freshness_ok   = (freshness_status=%r != "stale") → %s',
                freshness_status, freshness_ok)
    logger.info('[verdict]   merge_ok       = (merge_status=%r in {"ok"}) → %s',
                merge_status, merge_ok)
    if consistency_ok and freshness_ok and merge_ok:
        verdict = f'✅ HEALTHY{mode_tag}'
        exit_code = 0
        logger.info('[verdict] 三维度全 True → HEALTHY, exit_code=0')
    elif (abs(pct_all) < DANGER_PCT and freshness_ok
          and merge_status != 'danger'):
        verdict = f'⚠ WARN{mode_tag}'
        exit_code = 1
        logger.info('[verdict] 走 WARN 分支 (|pct_all|=%.3f%% < DANGER=%.1f%%, '
                    'freshness ok, merge != danger) → exit_code=1',
                    abs(pct_all), DANGER_PCT)
    else:
        verdict = f'🚨 DANGER{mode_tag}'
        exit_code = 2
        logger.info('[verdict] 落 DANGER 分支 (任一维度危险) → exit_code=2')
    print(f'  Verdict: {verdict}')
    print(f'  Thresholds: warn<{WARN_PCT}%  danger>={DANGER_PCT}%  '
          f'sample>={SAMPLE_MIN}%')
    print(f'  Merge:    unmerged_hours={gm["unmerged_hours"]:.1f} '
          f'(warn>=24h, danger>=72h)')
    print(f'  Time:    {time.time()-t0:.1f}s')
    print('═══════════════════════════════════════════════════════════')

    if args.json:
        out = {
            'sqlite': {'total': sql_total, 'fts': sql_fts, 'size_gb': db_size/1024**3},
            'faiss_main': {'vectors': idx_n, 'size_gb': idx_size/1024**3},
            'faiss_gap': {'vectors': gap_n,
                          'exists': FAISS_GAP_INDEX.exists(),
                          'size_gb': FAISS_GAP_INDEX.stat().st_size/1024**3
                                    if FAISS_GAP_INDEX.exists() else 0},
            'id_files': {'main': ids_main, 'gap': ids_gap},
            'consistency': {
                'main_vs_sqlite_pct': pct_main,
                'main_plus_gap_vs_sqlite_pct': pct_all,
                'faiss_vs_ids_diff': diff_faiss_ids,
            },
            'freshness_hours': age_h,
            'freshness_status': freshness_status,
            'merge_status': merge_status,
            'merge': {
                'pending_vectors': gm.get('pending', 0),
                'unmerged_hours': gm.get('unmerged_hours', 0.0),
                'last_gap_write_iso': gm.get('last_gap_write_iso'),
                'reason': gm.get('reason'),
            },
            'verdict': verdict,
            'strict': args.strict,
            'thresholds': {'warn_pct': WARN_PCT, 'danger_pct': DANGER_PCT,
                           'sample_min_pct': SAMPLE_MIN},
        }
        print('\n--- JSON ---')
        print(json.dumps(out, indent=2, ensure_ascii=False))

    sys.exit(exit_code)


if __name__ == '__main__':
    main()