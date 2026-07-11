#!/usr/bin/env python3
"""rebuild_kb_clean.py — 从零重建干净的 FAISS 索引

目标: 消除 FAISS 索引中累积的重复向量, 把搜索质量提升 10x

工作流:
1. 备份当前 papers.index / papers.db / paper_ids.txt
2. 从 paper_ids.txt 提取唯一 paper_id 集合 (~376K)
3. 一次性查 SQLite 拿 (paper_id, title, summary) 对
4. 大 batch 嵌入 (BATCH_SIZE=256)
5. 构建全新 FAISS IndexFlatIP (内积 = cosine, 因为向量已 L2-normalize)
6. 写新 papers.index / paper_ids.txt (覆盖)
7. 验证

时间预估 (CPU, all-MiniLM-L6-v2, 376K 篇):
- 编码: ~30-60 min @ batch 256
- 索引构建: <1 min
- 写盘: 1-2 min
- 总计: ~45-75 min

特性:
- 续跑: 已嵌入的 paper_id 记录在 rebuild_state.json
- 进度: 每 batch 报告速率 + ETA
- 安全: 写之前自动备份
- 验证: 完成后用 sample query 验证搜索质量

用法:
    python rebuild_kb_clean.py --dry-run                # 只看不写
    python rebuild_kb_clean.py --sample 1000            # 只重建 1000 篇做测试
    python rebuild_kb_clean.py                         # 全量重建
    python rebuild_kb_clean.py --reset                 # 重置续跑状态
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# Use all CPU cores for torch
import torch
torch.set_num_threads(16)

# ════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════
KB_DIR = Path(r'E:/peS2o_kb_faiss')
FAISS_INDEX = KB_DIR / 'papers.index'
SQLITE_DB = KB_DIR / 'papers.db'
IDS_FILE = KB_DIR / 'paper_ids.txt'

NEW_FAISS_INDEX = KB_DIR / 'papers_clean.index'
NEW_IDS_FILE = KB_DIR / 'paper_ids_clean.txt'

BACKUP_DIR = KB_DIR / 'rebuild_backups'
STATE_FILE = KB_DIR / 'rebuild_state.json'
LOG_FILE = KB_DIR / 'rebuild.log'

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBED_DIM = 384
BATCH_SIZE = 256

# ════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════

def log(msg: str, *, also_print: bool = True):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    if also_print:
        print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ════════════════════════════════════════════════════════════════
# State (resumability)
# ════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'embedded_ids': [],
        'embedded_count': 0,
        'started_at': None,
        'finished_at': None,
        'embeddings_path': None,
    }


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)


def clear_state():
    if STATE_FILE.exists():
        STATE_FILE.unlink()


# ════════════════════════════════════════════════════════════════
# Backup
# ════════════════════════════════════════════════════════════════

def backup_current() -> dict[str, Path]:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    paths = {}
    for src in [FAISS_INDEX, IDS_FILE]:
        if not src.exists():
            continue
        dst = BACKUP_DIR / f'{src.stem}_old_{ts}{src.suffix}'
        shutil.copy2(src, dst)
        paths[src.name] = dst
    return paths


# ════════════════════════════════════════════════════════════════
# Load unique IDs + metadata
# ════════════════════════════════════════════════════════════════

def load_unique_ids() -> list[str]:
    """Read unique paper_ids preserving order of first occurrence."""
    seen = set()
    unique = []
    with open(IDS_FILE, encoding='utf-8') as f:
        for line in f:
            pid = line.strip()
            if pid and pid not in seen:
                seen.add(pid)
                unique.append(pid)
    return unique


def load_metadata_from_sqlite(ids: list[str]) -> dict[str, dict]:
    """Bulk fetch title+abstract for all IDs from SQLite."""
    conn = sqlite3.connect(str(SQLITE_DB))
    cur = conn.cursor()

    log(f'fetching metadata for {len(ids):,} IDs from SQLite...')
    t0 = time.time()
    metadata = {}
    BATCH = 5000
    for i in range(0, len(ids), BATCH):
        batch = ids[i:i+BATCH]
        placeholders = ','.join('?' * len(batch))
        cur.execute(
            f'SELECT paper_id, title, text_prefix, abstract FROM papers WHERE paper_id IN ({placeholders})',
            batch,
        )
        for row in cur.fetchall():
            pid, title, text_prefix, abstract = row
            # Prefer abstract over text_prefix
            summary = abstract or text_prefix or ''
            metadata[pid] = {
                'title': title or '',
                'summary': summary,
            }

    conn.close()
    log(f'  metadata loaded: {len(metadata):,}/{len(ids):,} in {time.time()-t0:.1f}s')
    return metadata


# ════════════════════════════════════════════════════════════════
# Embedding with progress
# ════════════════════════════════════════════════════════════════

def embed_all(ids: list[str], metadata: dict[str, dict],
              *, reset: bool = False, dry_run: bool = False) -> tuple[list[str], 'np.ndarray']:
    """Embed all papers, resumable via STATE_FILE."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if reset:
        clear_state()

    state = load_state()
    embedded_ids_set = set(state.get('embedded_ids', []))

    # Filter out already embedded (resumability)
    todo_ids = [pid for pid in ids if pid not in embedded_ids_set and pid in metadata]
    skipped_existing = len(embedded_ids_set)
    skipped_no_meta = len(ids) - len(metadata)

    log(f'embed task: total={len(ids):,} | already_done={skipped_existing:,} | no_metadata={skipped_no_meta:,} | to_do={len(todo_ids):,}')

    if dry_run:
        log('DRY-RUN: not actually embedding')
        # Show first 3
        for pid in todo_ids[:3]:
            m = metadata.get(pid, {})
            log(f'  - {pid}: {m.get("title", "")[:70]}')
        return ids[:1000], np.zeros((min(1000, len(ids)), EMBED_DIM), dtype=np.float32)

    # Load model
    log('Loading embedding model...')
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    log(f'  loaded ({time.time()-t0:.1f}s)')

    # Build embed texts
    log('Building embed texts...')
    embed_texts = []
    valid_ids = []
    for pid in todo_ids:
        m = metadata[pid]
        text = (m.get('title', '') + ' ' + m.get('summary', ''))[:512]
        if text.strip():
            embed_texts.append(text)
            valid_ids.append(pid)

    log(f'  texts to embed: {len(embed_texts):,}')

    # Embed in batches
    log(f'Embedding in batches of {BATCH_SIZE}...')
    t_start = time.time()
    state['started_at'] = state.get('started_at') or datetime.now().isoformat()

    all_embeddings = []  # will collect as we go
    accumulated_ids = []

    for batch_idx in range(0, len(embed_texts), BATCH_SIZE):
        batch_texts = embed_texts[batch_idx:batch_idx + BATCH_SIZE]
        batch_ids = valid_ids[batch_idx:batch_idx + BATCH_SIZE]

        t0 = time.time()
        embs = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embs = np.array(embs, dtype=np.float32)
        elapsed = time.time() - t0

        all_embeddings.append(embs)
        accumulated_ids.extend(batch_ids)

        # Save state every batch
        state['embedded_ids'] = list(embedded_ids_set) + accumulated_ids
        state['embedded_count'] = len(state['embedded_ids'])
        save_state(state)

        done = batch_idx + len(batch_texts)
        rate = len(batch_texts) / max(elapsed, 0.001)
        remaining = (len(embed_texts) - done) / max(rate, 0.001)
        eta_min = remaining / 60
        log(f'  batch {done // BATCH_SIZE:>4} | '
            f'done: {done:>7,}/{len(embed_texts):,} | '
            f'rate: {rate:>5.1f}/s | '
            f'ETA: {eta_min:>5.1f} min')

    log(f'embedding complete in {(time.time() - t_start) / 60:.1f} min')

    embeddings_array = np.vstack(all_embeddings)
    final_ids = accumulated_ids

    log(f'final: {len(final_ids):,} embeddings, shape={embeddings_array.shape}')

    return final_ids, embeddings_array


# ════════════════════════════════════════════════════════════════
# Build clean FAISS index
# ════════════════════════════════════════════════════════════════

def build_clean_index(ids: list[str], embeddings, *, dry_run: bool = False):
    """Build a fresh FAISS index from embeddings and write to disk."""
    import faiss
    import numpy as np

    n = len(ids)
    log(f'building fresh FAISS index for {n:,} papers...')

    # Verify embeddings are normalized (cosine = inner product)
    norms = np.linalg.norm(embeddings, axis=1)
    log(f'  embedding norms: min={norms.min():.3f}, max={norms.max():.3f}, mean={norms.mean():.3f}')
    if not np.allclose(norms, 1.0, atol=0.01):
        log('  re-normalizing...')
        faiss.normalize_L2(embeddings)

    if dry_run:
        log('DRY-RUN: not writing')
        return

    # Build index
    index = faiss.IndexFlatIP(EMBED_DIM)  # Inner Product = cosine when normalized
    index.add(embeddings)

    log(f'  index built: {index.ntotal:,} vectors, dim={index.d}')

    # Write FAISS
    log(f'writing: {NEW_FAISS_INDEX}')
    t0 = time.time()
    faiss.write_index(index, str(NEW_FAISS_INDEX))
    log(f'  written ({time.time()-t0:.1f}s, {NEW_FAISS_INDEX.stat().st_size / 1e9:.2f} GB)')

    # Write clean paper_ids
    log(f'writing: {NEW_IDS_FILE}')
    with open(NEW_IDS_FILE, 'w', encoding='utf-8') as f:
        for pid in ids:
            f.write(pid + '\n')
    log(f'  written ({len(ids):,} IDs)')

    return index


# ════════════════════════════════════════════════════════════════
# Atomic swap
# ════════════════════════════════════════════════════════════════

def atomic_swap():
    """Replace old index/ids with clean version (atomic via os.replace)."""
    log('atomic swap: replacing old files...')
    t0 = time.time()
    os.replace(NEW_FAISS_INDEX, FAISS_INDEX)
    os.replace(NEW_IDS_FILE, IDS_FILE)
    log(f'  swapped in {time.time()-t0:.1f}s')


# ════════════════════════════════════════════════════════════════
# Verification
# ════════════════════════════════════════════════════════════════

def verify_clean():
    """Run a quality check: same query should return top-N unique papers."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    log('VERIFY: loading clean index...')
    index = faiss.read_index(str(FAISS_INDEX))
    with open(IDS_FILE, encoding='utf-8') as f:
        paper_ids = [line.strip() for line in f if line.strip()]

    model = SentenceTransformer(MODEL_NAME)
    log(f'  index: {index.ntotal:,} vectors, ids: {len(paper_ids):,}')

    # Test queries
    queries = [
        'calibration in large language models',
        'graph neural network for molecules',
        'reinforcement learning for robotics',
        'transformer architecture efficiency',
    ]

    log('VERIFY: sample queries...')
    for q in queries:
        q_emb = model.encode([q], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        scores, idxs = index.search(q_emb, 10)
        unique_pids = set()
        log(f'\n  Q: "{q}"')
        log(f'  scores: {[f"{s:.4f}" for s in scores[0]]}')
        for s, idx in zip(scores[0], idxs[0]):
            if idx >= 0 and idx < len(paper_ids):
                pid = paper_ids[idx]
                unique_pids.add(pid)
                log(f'    {s:.4f} | {pid}')
        log(f'  unique results: {len(unique_pids)}/10')


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='rebuild_kb_clean — 重建干净 FAISS')
    parser.add_argument('--dry-run', action='store_true', help='只看不写')
    parser.add_argument('--sample', type=int, default=0, help='只重建 N 篇 (测试)')
    parser.add_argument('--reset', action='store_true', help='重置续跑状态')
    parser.add_argument('--no-swap', action='store_true', help='不替换老文件, 只生成新文件')
    parser.add_argument('--verify-only', action='store_true', help='只跑验证')
    args = parser.parse_args()

    log('=' * 70)
    log('REBUILD-KB-CLEAN START')
    log(f'  dry-run: {args.dry_run}')
    log(f'  sample: {args.sample}')
    log(f'  reset: {args.reset}')
    log(f'  no-swap: {args.no_swap}')

    if args.verify_only:
        verify_clean()
        return

    # Step 1: Backup
    log('Step 1: backup current files...')
    backups = backup_current()
    for name, path in backups.items():
        log(f'  backup: {path}')

    # Step 2: Load unique IDs
    log('Step 2: loading unique paper IDs...')
    ids = load_unique_ids()
    log(f'  unique IDs: {len(ids):,}')

    if args.sample > 0:
        ids = ids[:args.sample]
        log(f'  sample limited to: {len(ids):,}')

    # Step 3: Load metadata
    log('Step 3: loading metadata from SQLite...')
    metadata = load_metadata_from_sqlite(ids)

    # Step 4: Embed
    log('Step 4: embedding papers...')
    final_ids, embeddings = embed_all(ids, metadata, reset=args.reset, dry_run=args.dry_run)

    if args.dry_run:
        log('DRY-RUN: done, not writing')
        return

    # Step 5: Build clean index
    log('Step 5: building clean FAISS index...')
    index = build_clean_index(final_ids, embeddings)

    # Step 6: Swap (or leave for manual review)
    if args.no_swap:
        log(f'Step 6: SKIPPED swap')
        log(f'  new files: {NEW_FAISS_INDEX}, {NEW_IDS_FILE}')
        log(f'  manually swap when ready:')
        log(f'    move {NEW_FAISS_INDEX} {FAISS_INDEX}')
        log(f'    move {NEW_IDS_FILE} {IDS_FILE}')
    else:
        log('Step 6: atomic swap...')
        atomic_swap()

    # Step 7: Verify
    log('Step 7: verifying...')
    verify_clean()

    # Step 8: Cleanup state
    clear_state()

    log('REBUILD DONE!')
    log('=' * 70)


if __name__ == '__main__':
    main()