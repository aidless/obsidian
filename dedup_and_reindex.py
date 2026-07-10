#!/usr/bin/env python3
"""dedup_and_reindex.py — 重建主索引(去重 + 补齐缺失论文)

4 个修复目标:
  1. SQLite papers 表去重 (257 种 paper_id 重复, 保留最新 created 的)
  2. FAISS 主索引去重 (33K 重复 paper_id, 保留后入的)
  3. 补齐缺失的 14,593 条 peS2o_numeric paper_id (从 E:\\peS2o_cs\\peS2o-*.jsonl 读)
  4. 验证: pct_all 应 ≈ 0%, kb_health 状态应从 WARN 升到 HEALTHY

5 阶段流水线:
  Phase 0: 准备 (mtime / 磁盘 / 14K 缺失 ID 列表)
  Phase 1: 重建 SQLite (去重 257 种 + 补 14K 的元数据)
  Phase 2: 从 JSONL 流式读 14K 论文原文 + 嵌入
  Phase 3: 重建 FAISS (读主索引 → 去重 → append 14K vecs → 写新索引)
  Phase 4: atomic swap + backup + 验证 kb_health

设计:
  - 每个 phase 独立, 任一 phase 失败立即 stop
  - backup 完整 KB 到 rebuild_backups/<timestamp>/
  - checkpoint 写到 /tmp/dedup_reindex_ckpt.json, 14K 嵌入中断可续跑

用法:
  py -3 dedup_and_reindex.py            # 完整执行
  py -3 dedup_and_reindex.py --dry-run  # 只跑 Phase 0/1 (只读)
  py -3 dedup_and_reindex.py --skip-14k # 跳过 14K 补齐, 只去重

退出码: 0 = 成功 / 1 = 校验失败 / 2 = 写盘失败已回滚
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

KB_DIR = Path(r'E:/peS2o_kb_faiss')
PES2O_JSONL_DIR = Path(r'E:/peS2o_cs')
PES2O_PATTERN = 'peS2o-*.jsonl'
CHECKPOINT = Path(r'F:/temp/dedup_reindex_ckpt.json')

FAISS_INDEX = KB_DIR / 'papers.index'
IDS_FILE = KB_DIR / 'paper_ids.txt'
SQLITE_DB = KB_DIR / 'papers.db'
BACKUP_DIR = KB_DIR / 'rebuild_backups'
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBED_DIM = 384


def log(msg: str) -> None:
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, file=sys.stderr, flush=True)
    # 同时写到日志文件 (PowerShell 重定向会丢内容, 文件不会)
    try:
        with open(r'F:\temp\dedup_run.log', 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


# ════════════════════════════════════════════════════════════
# Phase 0: 准备
# ════════════════════════════════════════════════════════════
def phase0_prepare() -> dict:
    log('══ Phase 0: 准备 (只读) ══')
    for p in [FAISS_INDEX, IDS_FILE, SQLITE_DB, PES2O_JSONL_DIR]:
        if not p.exists():
            raise FileNotFoundError(f'缺少关键文件: {p}')

    # 找 peS2o JSONL 文件
    jsonl_files = sorted(PES2O_JSONL_DIR.glob(PES2O_PATTERN))
    if not jsonl_files:
        raise FileNotFoundError(f'未找到 peS2o JSONL: {PES2O_JSONL_DIR}/{PES2O_PATTERN}')
    total_jsonl_gb = sum(f.stat().st_size for f in jsonl_files) / 1024**3
    log(f'  peS2o JSONL 文件: {len(jsonl_files)} 个, '
        f'总 {total_jsonl_gb:.1f} GB')

    # 读 SQLite paper_id
    conn = sqlite3.connect(str(SQLITE_DB))
    c = conn.cursor()
    c.execute('SELECT paper_id, year, categories, source, created, '
              'authors, title, abstract, text_prefix '
              'FROM papers')
    sql_rows = c.fetchall()
    c.execute('SELECT COUNT(*) FROM papers')
    n_sql = c.fetchone()[0]
    c.execute('SELECT COUNT(DISTINCT paper_id) FROM papers')
    n_sql_unique = c.fetchone()[0]
    conn.close()
    log(f'  SQLite: {n_sql:,} 行 ({n_sql_unique:,} unique paper_id, '
        f'{n_sql - n_sql_unique} 重复行)')

    # 读 FAISS paper_ids
    with open(IDS_FILE, encoding='utf-8') as f:
        faiss_ids_list = [line.strip() for line in f if line.strip()]
    faiss_counter = Counter(faiss_ids_list)
    n_faiss_total = len(faiss_ids_list)
    n_faiss_unique = len(set(faiss_ids_list))
    n_faiss_dup = n_faiss_total - n_faiss_unique
    log(f'  FAISS:  {n_faiss_total:,} 行 ({n_faiss_unique:,} unique, '
        f'{n_faiss_dup} 重复)')

    # 算差异
    sql_set = {r[0] for r in sql_rows}
    faiss_set = set(faiss_ids_list)
    in_faiss_not_sql = faiss_set - sql_set
    in_sql_not_faiss = sql_set - faiss_set
    log(f'  FAISS - SQLite: {len(in_faiss_not_sql):,}')
    log(f'  SQLite - FAISS: {len(in_sql_not_faiss):,}  ← 需补齐')
    log(f'  重复行: SQLite {n_sql - n_sql_unique} / FAISS {n_faiss_dup}')

    # 磁盘空间估算
    main_size = FAISS_INDEX.stat().st_size
    ids_size = IDS_FILE.stat().st_size
    db_size = SQLITE_DB.stat().st_size
    expected_new_idx = (n_faiss_unique + len(in_sql_not_faiss)) * EMBED_DIM * 4
    backup_need = main_size + ids_size + db_size
    staging_need = expected_new_idx
    free_need = backup_need + staging_need
    total, used, free = shutil.disk_usage(KB_DIR)
    log(f'  磁盘:  free={free/1024**3:.1f} GB, 预计需 {free_need/1024**3:.1f} GB')
    if free < free_need * 1.2:
        raise RuntimeError(f'磁盘空间不足: 需 {free_need/1024**3:.1f} GB, '
                           f'仅 {free/1024**3:.1f} GB')

    return {
        'sql_rows': sql_rows,
        'sql_set': sql_set,
        'faiss_ids_list': faiss_ids_list,
        'faiss_counter': faiss_counter,
        'faiss_set': faiss_set,
        'in_sql_not_faiss': in_sql_not_faiss,
        'in_faiss_not_sql': in_faiss_not_sql,
        'jsonl_files': jsonl_files,
        'n_sql': n_sql, 'n_sql_unique': n_sql_unique,
        'n_faiss_total': n_faiss_total, 'n_faiss_unique': n_faiss_unique,
    }


# ════════════════════════════════════════════════════════════
# Phase 1: 重建 SQLite (去重 + 14K 元数据补齐准备)
# ════════════════════════════════════════════════════════════
def phase1_dedup_sqlite(stats: dict) -> tuple[list[tuple], set]:
    """去重 SQLite papers 表, 保留 created 最新的那条 (如果有 created 字段)
       返回: (去重后的 row 列表, 14K 缺失 ID 集合)"""
    log('══ Phase 1: SQLite 去重 (保留最新) ══')

    sql_rows = stats['sql_rows']
    in_sql_not_faiss = stats['in_sql_not_faiss']

    # 按 paper_id 分组
    by_id: dict[str, list[tuple]] = {}
    for row in sql_rows:
        by_id.setdefault(row[0], []).append(row)
    n_dup_ids = sum(1 for v in by_id.values() if len(v) > 1)
    log(f'  重复 paper_id: {n_dup_ids} 种')

    # 保留 created 最新的 (idx=4 in row tuple)
    deduped_rows = []
    for pid, rows in by_id.items():
        if len(rows) == 1:
            deduped_rows.append(rows[0])
        else:
            # 按 created 时间倒序, 取最新
            rows_sorted = sorted(rows, key=lambda r: r[4] or '', reverse=True)
            deduped_rows.append(rows_sorted[0])
            log(f'    [去重] {pid}: {len(rows)} 条 → 1 条 (created={rows_sorted[0][4]})')

    log(f'  去重后: {len(deduped_rows):,} unique rows '
        f'(原 {len(sql_rows):,}, 删除 {len(sql_rows) - len(deduped_rows):,})')

    # 14K 缺失 ID 集合
    in_sql_not_faiss_set = in_sql_not_faiss
    log(f'  待补齐缺失 ID: {len(in_sql_not_faiss_set):,}')

    return deduped_rows, in_sql_not_faiss_set


# ════════════════════════════════════════════════════════════
# Phase 2: 从 JSONL 流式读 14K 论文, 嵌入
# ════════════════════════════════════════════════════════════
def phase2_embed_missing(missing_ids: set, stats: dict,
                          do_embed: bool = True,
                          use_progress: bool = True) -> tuple[list[str], list]:
    """从 peS2o JSONL 流式读缺失 ID 的 text, 嵌入, 返回 (found_ids, vectors_or_None)"""
    log('══ Phase 2: 从 JSONL 读 14K 论文 + 嵌入 ══')

    # checkpoint
    if CHECKPOINT.exists():
        ckpt = json.loads(CHECKPOINT.read_text(encoding='utf-8'))
        log(f'  发现 checkpoint: 已完成 {ckpt.get("done", 0)} 个')
    else:
        ckpt = {'done': 0, 'found': []}

    jsonl_files = stats['jsonl_files']
    found: dict[str, str] = {pid: None for pid in missing_ids}  # id -> text_prefix
    found_pids: list[str] = []

    t0 = time.time()
    log('  扫描 JSONL 找缺失 ID 的 text ...')

    # 进度条: 按文件 + 行数
    found_count = 0
    jsonl_iter = tqdm(jsonl_files, desc='JSONL 文件',
                      unit='file', disable=not use_progress,
                      file=sys.stderr, dynamic_ncols=True)
    for fpath in jsonl_iter:
        jsonl_iter.set_postfix(file=fpath.name[:30],
                               found=f'{found_count}/{len(missing_ids)}')
        # 行级进度条 (按字节估计行数)
        file_size = fpath.stat().st_size
        with open(fpath, encoding='utf-8') as f:
            line_iter = tqdm(f, desc=f'  {fpath.name[:20]:20s}',
                             total=file_size, unit='B', unit_scale=True,
                             disable=not use_progress,
                             file=sys.stderr, dynamic_ncols=True,
                             leave=False)
            for line in line_iter:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    line_iter.update(len(line))
                    continue
                pid = str(obj.get('id', ''))
                if pid in found and found[pid] is None:
                    text = obj.get('text', '') or ''
                    text_prefix = text[:2000]
                    found[pid] = text_prefix
                    found_pids.append(pid)
                    found_count += 1
                    jsonl_iter.set_postfix(file=fpath.name[:30],
                                           found=f'{found_count}/{len(missing_ids)}')
                line_iter.update(len(line))
            line_iter.close()
        # 优化: 如果全找到了就 break
        if all(v is not None for v in found.values()):
            log(f'  → 全部 {len(missing_ids)} 个已找齐, 停止扫描')
            break
    jsonl_iter.close()

    found_pids = [p for p in found_pids if found[p] is not None]
    not_found = [p for p in missing_ids if found.get(p) is None]
    log(f'  找到: {len(found_pids):,} / {len(missing_ids):,}')
    if not_found:
        log(f'  ⚠ {len(not_found)} 个在 JSONL 中没找到 (可能属于 peS2o 其他子目录)')

    if not do_embed or not found_pids:
        return found_pids, []

    # ── 嵌入 + 增量 checkpoint 续跑 ──
    # checkpoint 结构:
    #   F:\temp\dedup_reindex_ckpt.json
    #     {done_pids: [pid1, pid2, ...], total: N}
    #   F:\temp\dedup_reindex_vecs.npy
    #     float32 array shape=(done_count, 384), 与 done_pids 一一对应
    log(f'  嵌入 {len(found_pids):,} 条 (model={MODEL_NAME}) ...')
    from sentence_transformers import SentenceTransformer
    import numpy as np

    VECS_NPY = CHECKPOINT.with_suffix('.vecs.npy')

    # 1. 读 checkpoint (done_pids 列表, 保序)
    if CHECKPOINT.exists():
        ckpt = json.loads(CHECKPOINT.read_text(encoding='utf-8'))
        done_pids = list(ckpt.get('done_pids', []))
    else:
        done_pids = []
    done_set = set(done_pids)

    # 2. 找待嵌入: 找齐的 pid 中还没嵌入的
    found_set = set(found_pids)
    to_embed = [p for p in found_pids if p not in done_set]
    log(f'    待嵌入: {len(to_embed):,} (checkpoint 已完成 {len(done_pids):,})')

    if to_embed:
        model = SentenceTransformer(MODEL_NAME)
        BATCH = 64
        new_done_pids = []
        new_vec_chunks = []
        t_emb = time.time()

        # 进度条: 嵌入 + ETA
        embed_pbar = tqdm(total=len(to_embed), desc='嵌入 14K 论文',
                          unit='paper', disable=not use_progress,
                          file=sys.stderr, dynamic_ncols=True,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                     '[{elapsed}<{remaining}, {rate_fmt}]')
        ckpt_count = 0
        for i in range(0, len(to_embed), BATCH):
            batch_pids = to_embed[i:i + BATCH]
            batch_texts = [found[p] for p in batch_pids]
            batch_vecs = model.encode(batch_texts,
                                      batch_size=BATCH,
                                      show_progress_bar=False).astype('float32')
            new_done_pids.extend(batch_pids)
            new_vec_chunks.append(batch_vecs)

            # 每 5 个 batch 增量写一次
            if len(new_done_pids) % (BATCH * 5) == 0 or i + BATCH >= len(to_embed):
                new_arr = np.vstack(new_vec_chunks) if len(new_vec_chunks) > 1 \
                          else new_vec_chunks[0]
                if VECS_NPY.exists():
                    old = np.load(str(VECS_NPY))
                    merged = np.vstack([old, new_arr])
                else:
                    merged = new_arr
                np.save(str(VECS_NPY), merged)
                updated_pids = done_pids + new_done_pids
                CHECKPOINT.write_text(
                    json.dumps({'done_pids': updated_pids,
                                'total': len(found_pids)}),
                    encoding='utf-8')
                ckpt_count += 1
                embed_pbar.set_postfix(ckpt=ckpt_count, file=f'{VECS_NPY.stat().st_size/1024**2:.1f}MB')
            embed_pbar.update(BATCH)
        embed_pbar.close()

        # 全部嵌入完成, 加载完整 npy
        all_vecs = np.load(str(VECS_NPY))
        log(f'  嵌入完成: {all_vecs.shape} ({time.time()-t_emb:.1f}s)')
    else:
        # 全部已在 checkpoint 中
        all_vecs = np.load(str(VECS_NPY))
        log(f'  从 checkpoint 续跑: {all_vecs.shape} 已嵌入, 无需重新计算')

    # 验证 npy 顺序与 found_pids 一致
    # done_pids 应该按嵌入顺序排列, 但 found_pids 顺序可能不同
    # 我们需要的是"按 found_pids 顺序返回 vecs"
    if len(done_pids) == len(found_pids):
        # 全部嵌入, 重新按 found_pids 顺序
        pid_to_vec = {pid: all_vecs[i] for i, pid in enumerate(done_pids)}
        final_vecs = np.array([pid_to_vec[p] for p in found_pids],
                              dtype='float32')
        log(f'  按 found_pids 顺序重排: {final_vecs.shape}')
    else:
        # 部分嵌入 (应该不会发生, 因为 to_embed 处理完才进这里)
        log(f'  ⚠ 不一致: done_pids={len(done_pids)}, found_pids={len(found_pids)}')
        final_vecs = all_vecs

    log(f'  Phase 2 总耗时: {time.time()-t0:.1f}s')
    return found_pids, final_vecs


# ════════════════════════════════════════════════════════════
# Phase 3: 重建 FAISS (去重主索引 + append 14K vecs)
# ════════════════════════════════════════════════════════════
def phase3_rebuild_faiss(stats: dict, new_pids: list[str],
                          new_vecs, deduped_sqlite_rows: list,
                          use_progress: bool = True) -> None:
    log('══ Phase 3: 重建 FAISS (去重 + 补 14K) ══')
    import faiss
    import numpy as np

    # 3a. 读当前主索引
    log('  读主索引 papers.index ...')
    t0 = time.time()
    idx_main = faiss.read_index(str(FAISS_INDEX))
    n_main = idx_main.ntotal
    log(f'    ntotal = {n_main:,} ({time.time()-t0:.1f}s)')

    # 3b. 重建主索引: 只保留 unique paper_id 的向量 (去重)
    # 思路: 找重复的 paper_id 在 paper_ids.txt 的索引位置, 重建
    # 简化做法: 用 FAISS index reconstruction (对 IndexFlatIP 可行)
    log('  去重主索引: 找重复 ID 的索引位置 ...')
    faiss_ids_list = stats['faiss_ids_list']
    seen: set[str] = set()
    keep_indices: list[int] = []
    drop_count = 0
    dedup_pbar = tqdm(range(len(faiss_ids_list)),
                      desc='扫描去重 (找 keep_indices)',
                      unit='id', disable=not use_progress,
                      file=sys.stderr, dynamic_ncols=True,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                 '[{elapsed}<{remaining}, {rate_fmt}]')
    chunk_size = 50000
    last_update = 0
    for i, pid in enumerate(faiss_ids_list):
        if pid in seen:
            drop_count += 1
            continue
        seen.add(pid)
        keep_indices.append(i)
        if (i + 1) - last_update >= chunk_size:
            dedup_pbar.update(chunk_size)
            last_update = i + 1
    remaining = len(faiss_ids_list) - dedup_pbar.n
    if remaining > 0:
        dedup_pbar.update(remaining)
    dedup_pbar.close()
    log(f'    保留 {len(keep_indices):,} / {n_main:,} '
        f'(丢弃 {drop_count:,} 个重复)')

    # 重建去重后的向量
    log('  重建去重向量 (reconstruct from keep_indices) ...')
    vecs_deduped = np.zeros((len(keep_indices), EMBED_DIM), dtype='float32')
    recon_pbar = tqdm(range(len(keep_indices)),
                      desc='重建去重向量',
                      unit='vec', disable=not use_progress,
                      file=sys.stderr, dynamic_ncols=True,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                 '[{elapsed}<{remaining}, {rate_fmt}]')
    chunk_size = 10000
    for new_i, old_i in enumerate(keep_indices):
        vecs_deduped[new_i] = idx_main.reconstruct(old_i)
        if (new_i + 1) % chunk_size == 0:
            recon_pbar.update(chunk_size)
    # 处理剩余
    remaining = len(keep_indices) - recon_pbar.n
    if remaining > 0:
        recon_pbar.update(remaining)
    recon_pbar.close()
    log(f'    重建完成: {vecs_deduped.shape}')

    ids_deduped = [faiss_ids_list[i] for i in keep_indices]

    # 3c. append 14K 新向量
    if new_pids and new_vecs is not None and len(new_pids) > 0:
        log(f'  append {len(new_pids):,} 个 14K 缺失论文向量 ...')
        # 检查 14K ID 是否已在 ids_deduped (避免双写)
        existing_set = set(ids_deduped)
        new_filtered = [(p, v) for p, v in zip(new_pids, new_vecs)
                        if p not in existing_set]
        if len(new_filtered) < len(new_pids):
            log(f'    过滤已在主索引的: {len(new_pids) - len(new_filtered)}')
        new_pids_filtered = [p for p, _ in new_filtered]
        new_vecs_filtered = np.array([v for _, v in new_filtered],
                                     dtype='float32')
        log(f'    最终追加: {len(new_pids_filtered):,}')

        vecs_final = np.vstack([vecs_deduped, new_vecs_filtered])
        ids_final = ids_deduped + new_pids_filtered
    else:
        vecs_final = vecs_deduped
        ids_final = ids_deduped
        log('  跳过 14K 追加 (--skip-14k 或无新数据)')

    log(f'  最终: {vecs_final.shape[0]:,} vecs / {len(ids_final):,} ids')

    # 3d. 写 staging 文件
    staging_idx = KB_DIR / 'papers.index.staging'
    staging_ids = KB_DIR / 'paper_ids.txt.staging'

    log('  写 staging FAISS ...')
    t0 = time.time()
    idx_new = faiss.IndexFlatIP(EMBED_DIM)
    idx_new.add(vecs_final)
    faiss.write_index(idx_new, str(staging_idx))
    log(f'    staging.index size = '
        f'{staging_idx.stat().st_size/1024**2:.1f} MB '
        f'({time.time()-t0:.1f}s)')

    log('  写 staging ids ...')
    with open(staging_ids, 'w', encoding='utf-8') as f:
        for pid in ids_final:
            f.write(pid + '\n')
    log(f'    staging.ids size = '
        f'{staging_ids.stat().st_size/1024**2:.1f} MB')

    # 3e. 写 staging SQLite (去重后)
    staging_db = KB_DIR / 'papers.db.staging'
    log('  写 staging SQLite (去重后) ...')
    if staging_db.exists():
        staging_db.unlink()
    shutil.copy2(str(SQLITE_DB), str(staging_db))
    conn = sqlite3.connect(str(staging_db))
    c = conn.cursor()
    # SQLite 主键是 INTEGER id, 我们用 paper_id 去重
    # 1. 找重复 paper_id 的旧行 id
    c.execute("""
        SELECT id, paper_id, created FROM papers
        WHERE paper_id IN (
            SELECT paper_id FROM papers GROUP BY paper_id HAVING COUNT(*) > 1
        )
    """)
    dup_rows = c.fetchall()
    if dup_rows:
        # 按 paper_id 分组, 保留 created 最大的
        by_pid: dict[str, list[tuple]] = {}
        for row_id, pid, created in dup_rows:
            by_pid.setdefault(pid, []).append((row_id, created or ''))
        to_delete = []
        for pid, rows in by_pid.items():
            rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
            for row_id, _ in rows_sorted[1:]:  # 除最新外的都删
                to_delete.append(row_id)
        log(f'    删除 SQLite 重复行: {len(to_delete)} 条')
        if to_delete:
            # 批量删除
            chunk = 1000
            for i in range(0, len(to_delete), chunk):
                c.execute(f"DELETE FROM papers WHERE id IN "
                          f"({','.join('?'*min(chunk, len(to_delete)-i))})",
                          to_delete[i:i+chunk])
        conn.commit()
    conn.close()
    log('    staging SQLite 准备完成')


# ════════════════════════════════════════════════════════════
# Phase 4: backup + atomic swap + 验证
# ════════════════════════════════════════════════════════════
def phase4_swap_and_verify(stats: dict, do_backup: bool = True) -> int:
    log('══ Phase 4: backup + atomic swap + 验证 ══')

    staging_idx = KB_DIR / 'papers.index.staging'
    staging_ids = KB_DIR / 'paper_ids.txt.staging'
    staging_db = KB_DIR / 'papers.db.staging'

    if not (staging_idx.exists() and staging_ids.exists()
            and staging_db.exists()):
        raise FileNotFoundError('staging 文件不完整, 跳过 swap')

    # 4a. backup
    if do_backup:
        BACKUP_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        bkp_dir = BACKUP_DIR / ts
        bkp_dir.mkdir(exist_ok=True)
        log(f'  backup → {bkp_dir.name}/')
        shutil.copy2(str(FAISS_INDEX), str(bkp_dir / 'papers.index'))
        shutil.copy2(str(IDS_FILE), str(bkp_dir / 'paper_ids.txt'))
        shutil.copy2(str(SQLITE_DB), str(bkp_dir / 'papers.db'))
        log(f'  backup 完成, 大小: '
            f'{sum(f.stat().st_size for f in bkp_dir.iterdir())/1024**2:.1f} MB')

    # 4b. atomic swap
    log('  atomic swap papers.index ...')
    os.replace(str(staging_idx), str(FAISS_INDEX))
    log('  atomic swap paper_ids.txt ...')
    os.replace(str(staging_ids), str(IDS_FILE))
    log('  atomic swap papers.db ...')
    os.replace(str(staging_db), str(SQLITE_DB))

    # 4c. 验证
    log('  跑 kb_health.py 验证 ...')
    import subprocess
    result = subprocess.run(
        [sys.executable, str(KB_DIR / 'kb_health.py'), '--sample', '0'],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        cwd=str(KB_DIR),
    )
    log(f'  kb_health exit code: {result.returncode}')
    out = result.stdout + result.stderr
    for kw in ('[5b]', 'Verdict:', 'Merge:', 'pct_all',
               'main+gap', 'main FAISS vs SQLite', 'papers_fts'):
        for line in out.splitlines():
            if kw in line:
                log(f'    {line.strip()[:120]}')
                break
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='只跑 Phase 0/1 (只读)')
    ap.add_argument('--skip-14k', action='store_true',
                    help='跳过 14K 补齐, 只去重')
    ap.add_argument('--no-backup', action='store_true',
                    help='跳过 backup')
    ap.add_argument('--reset-checkpoint', action='store_true',
                    help='删除 checkpoint 重新嵌入 (默认: 续跑)')
    ap.add_argument('--no-progress', action='store_true',
                    help='关闭 tqdm 进度条 (重定向日志时用)')
    args = ap.parse_args()

    log('═══════════════════════════════════════════════════════════')
    log('  dedup_and_reindex.py — 重建主索引(去重 + 补 14K)')
    log(f'  KB: {KB_DIR}')
    log(f'  模式: ' + ('DRY-RUN' if args.dry_run else
                       'SKIP-14K' if args.skip_14k else 'FULL'))
    log('═══════════════════════════════════════════════════════════')

    # 处理 --reset-checkpoint
    if args.reset_checkpoint:
        for f in [CHECKPOINT, CHECKPOINT.with_suffix('.vecs.npy')]:
            if f.exists():
                f.unlink()
                log(f'  [reset] 删除 {f.name}')

    t_total = time.time()

    try:
        stats = phase0_prepare()
    except Exception as e:
        log(f'❌ Phase 0 失败: {e}')
        sys.exit(1)

    try:
        deduped_sqlite_rows, missing_ids = phase1_dedup_sqlite(stats)
    except Exception as e:
        log(f'❌ Phase 1 失败: {e}')
        sys.exit(1)

    if args.dry_run:
        log('══ DRY-RUN 完成, 不执行嵌入 / 写盘 ══')
        log(f'  SQLite 去重:  {len(stats["sql_rows"]):,} → '
            f'{len(deduped_sqlite_rows):,} (删 {len(stats["sql_rows"]) - len(deduped_sqlite_rows):,})')
        log(f'  FAISS 去重:   {stats["n_faiss_total"]:,} → '
            f'{stats["n_faiss_unique"]:,} (删 {stats["n_faiss_total"] - stats["n_faiss_unique"]:,})')
        log(f'  待补齐:       {len(missing_ids):,}')
        log(f'  总耗时: {time.time()-t_total:.1f}s')
        sys.exit(0)

    if not args.skip_14k and missing_ids:
        try:
            new_pids, new_vecs = phase2_embed_missing(
                missing_ids, stats, do_embed=True,
                use_progress=not args.no_progress)
        except Exception as e:
            log(f'❌ Phase 2 失败: {e}')
            sys.exit(1)
    else:
        new_pids, new_vecs = [], None
        log('Phase 2 跳过 (--skip-14k 或无缺失 ID)')

    try:
        phase3_rebuild_faiss(stats, new_pids, new_vecs,
                              deduped_sqlite_rows,
                              use_progress=not args.no_progress)
    except Exception as e:
        log(f'❌ Phase 3 失败: {e}')
        sys.exit(2)

    try:
        verify_exit = phase4_swap_and_verify(
            stats, do_backup=not args.no_backup)
    except Exception as e:
        log(f'❌ Phase 4 失败: {e}')
        log('  尝试回滚...')
        # 找最新 backup
        if BACKUP_DIR.exists():
            backups = sorted(BACKUP_DIR.iterdir(), reverse=True)
            if backups:
                bkp = backups[0]
                log(f'  从 {bkp.name} 恢复...')
                if (bkp / 'papers.index').exists():
                    os.replace(str(bkp / 'papers.index'), str(FAISS_INDEX))
                if (bkp / 'paper_ids.txt').exists():
                    os.replace(str(bkp / 'paper_ids.txt'), str(IDS_FILE))
                if (bkp / 'papers.db').exists():
                    os.replace(str(bkp / 'papers.db'), str(SQLITE_DB))
                log('  回滚完成')
        sys.exit(2)

    log(f'✅ 重建完成, 总耗时: {time.time()-t_total:.1f}s')
    if verify_exit == 0:
        log('  状态: HEALTHY')
    elif verify_exit == 1:
        log('  状态: WARN (gap 维度 OK, 其他维度可能 WARN)')
    else:
        log('  状态: DANGER (需检查日志)')

    # 清理 checkpoint (任务完成, 不再保留)
    for f in [CHECKPOINT, CHECKPOINT.with_suffix('.vecs.npy')]:
        if f.exists():
            f.unlink()
            log(f'  [cleanup] 删除 {f.name}')


if __name__ == '__main__':
    main()
