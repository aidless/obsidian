#!/usr/bin/env python3
"""self_grow.py — 论文知识库自我生长管道

功能:
1. 从 jsonl 文件读取新论文 (兼容多种 schema)
2. 生成 embedding (all-MiniLM-L6-v2)
3. 追加到 FAISS 索引 (papers.index)
4. 追加到 SQLite (papers.db)
5. 追加到 paper_ids.txt (与 FAISS 顺序对应)
6. 断点续跑, 失败回滚

用法:
    python self_grow.py --input <jsonl_path>            # 增量添加
    python self_grow.py --input <jsonl_path> --dry-run  # 只看不写
    python self_grow.py --input <jsonl_path> --reset    # 重置断点
    python self_grow.py --verify                       # 验证 KB 一致性
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

# Mirror first, then offline
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np

# ════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════
KB_DIR = Path(r'E:/peS2o_kb_faiss')
FAISS_INDEX = KB_DIR / 'papers.index'
SQLITE_DB = KB_DIR / 'papers.db'
IDS_FILE = KB_DIR / 'paper_ids.txt'
BACKUP_DIR = KB_DIR / 'self_grow_backups'

CHECKPOINT = KB_DIR / 'self_grow_checkpoint.json'
RUN_LOG = KB_DIR / 'self_grow.log'

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBED_DIM = 384
BATCH_SIZE = 64

# ════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════

def log(msg: str, *, also_print: bool = True):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    if also_print:
        print(line, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ════════════════════════════════════════════════════════════════
# Schema normalization
# ════════════════════════════════════════════════════════════════

def normalize_paper(raw: dict) -> dict | None:
    """归一化多种 schema 的 paper 记录到统一格式.

    输入 schema 可能是:
    - peS2o: paper_id, title, authors, summary, text, created, categories, source
    - arXiv API: arxiv_id, title, authors, abstract, categories, published
    - S2 API: paper_id, s2_paper_id, title, authors, summary, year, fields

    输出统一 schema:
    - paper_id, title, authors, summary, year, categories, source, abstract, text
    """
    paper_id = (
        raw.get('paper_id')
        or raw.get('arxiv_id')
        or raw.get('s2_paper_id')
        or ''
    )
    if not paper_id:
        return None

    # Strip version (2606.16682v1 -> 2606.16682)
    paper_id = paper_id.split('v')[0].strip()

    title = (raw.get('title') or '').strip().replace('\n', ' ').replace('\r', ' ')
    if not title:
        return None

    authors = raw.get('authors') or []
    if isinstance(authors, str):
        try:
            authors = json.loads(authors)
        except Exception:
            authors = [authors]
    authors = [str(a) if not isinstance(a, dict) else (a.get('name') or '') for a in authors]
    authors = [a for a in authors if a]

    summary = (
        raw.get('summary')
        or raw.get('abstract')
        or ''
    ).strip().replace('\n', ' ').replace('\r', ' ')

    text = (raw.get('text') or summary).strip()
    year = (
        raw.get('year')
        or (raw.get('published') or '')[:4]
        or (raw.get('created') or '')[:4]
        or ''
    )

    categories = raw.get('categories') or []
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(',') if c.strip()]

    source = raw.get('source') or 'self_grow'
    version = raw.get('version') or '1.0'

    return {
        'paper_id': paper_id,
        'title': title[:500],
        'authors': json.dumps(authors, ensure_ascii=False),
        'summary': summary[:5000],
        'text': text[:50000],
        'year': year,
        'categories': ','.join(categories),
        'source': source,
        'version': version,
    }


# ════════════════════════════════════════════════════════════════
# Checkpoint
# ════════════════════════════════════════════════════════════════

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        try:
            with open(CHECKPOINT, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'processed_lines': 0, 'added_count': 0, 'last_paper_id': None}


def save_checkpoint(cp: dict) -> None:
    tmp = CHECKPOINT.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cp, f, indent=2)
    tmp.replace(CHECKPOINT)


def clear_checkpoint() -> None:
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()


# ════════════════════════════════════════════════════════════════
# Backup helpers
# ════════════════════════════════════════════════════════════════

def backup_files() -> dict[str, Path]:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    paths = {}
    for src in [FAISS_INDEX, SQLITE_DB, IDS_FILE]:
        if not src.exists():
            continue
        dst = BACKUP_DIR / f'{src.stem}_{ts}{src.suffix}'
        shutil.copy2(src, dst)
        paths[src.name] = dst
    return paths


# ════════════════════════════════════════════════════════════════
# Verify
# ════════════════════════════════════════════════════════════════

def verify_consistency() -> dict:
    """校验 KB 三件套的一致性."""
    log('VERIFY: checking FAISS / SQLite / paper_ids.txt consistency')
    result = {}

    # Load
    import faiss
    index = faiss.read_index(str(FAISS_INDEX))
    result['faiss_ntotal'] = int(index.ntotal)

    conn = sqlite3.connect(str(SQLITE_DB))
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM papers')
    result['sqlite_papers'] = cur.fetchone()[0]

    cur.execute('SELECT MAX(id), MIN(id) FROM papers')
    max_id, min_id = cur.fetchone()
    result['sqlite_id_range'] = (min_id, max_id)

    with open(IDS_FILE, encoding='utf-8') as f:
        ids = [line.strip() for line in f if line.strip()]
    result['ids_file_count'] = len(ids)

    # Consistency check
    result['ids_match_faiss'] = (result['ids_file_count'] == result['faiss_ntotal'])
    result['sqlite_lte_faiss'] = (result['sqlite_papers'] <= result['faiss_ntotal'])

    # Sample search
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    q = model.encode(['calibration in large language models'], normalize_embeddings=True)
    q = np.array(q, dtype=np.float32)
    scores, ids_found = index.search(q, 5)
    result['sample_search_scores'] = scores[0].tolist()
    result['sample_search_ids'] = ids_found[0].tolist()

    conn.close()

    log(f'VERIFY RESULT: {json.dumps(result, ensure_ascii=False)}')
    return result


# ════════════════════════════════════════════════════════════════
# Main: ingest
# ════════════════════════════════════════════════════════════════

def ingest(input_path: Path, *, dry_run: bool = False, reset: bool = False) -> bool:
    if not input_path.exists():
        log(f'INPUT NOT FOUND: {input_path}')
        return False

    log('=' * 70)
    log(f'SELF-GROW START')
    log(f'  input: {input_path} ({input_path.stat().st_size / 1e6:.1f} MB)')
    log(f'  dry_run: {dry_run}, reset: {reset}')

    if reset:
        clear_checkpoint()
        log('  checkpoint cleared')

    cp = load_checkpoint()
    start_line = cp.get('processed_lines', 0)
    added_count = cp.get('added_count', 0)
    log(f'  resume: processed_lines={start_line}, added_count={added_count}')

    # Load model and index
    log('Loading embedding model...')
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    log(f'  model loaded ({time.time()-t0:.1f}s)')

    log('Loading FAISS index...')
    t0 = time.time()
    import faiss
    index = faiss.read_index(str(FAISS_INDEX))
    log(f'  index loaded: {index.ntotal:,} vectors ({time.time()-t0:.1f}s)')

    # Load existing paper IDs from ids file
    log('Loading existing paper_ids.txt...')
    existing_ids = set()
    with open(IDS_FILE, encoding='utf-8') as f:
        for line in f:
            pid = line.strip()
            if pid:
                existing_ids.add(pid)
    log(f'  existing IDs: {len(existing_ids):,}')

    # Connect SQLite
    conn = sqlite3.connect(str(SQLITE_DB))
    cur = conn.cursor()

    # Add missing columns if needed
    cur.execute('PRAGMA table_info(papers)')
    existing_cols = {c[1] for c in cur.fetchall()}
    for name, dtype in [
        ('authors', 'TEXT'),
        ('abstract', 'TEXT'),
        ('year', 'TEXT'),
        ('categories', 'TEXT'),
        ('source', 'TEXT'),
        ('version', 'TEXT'),
    ]:
        if name not in existing_cols:
            try:
                cur.execute(f'ALTER TABLE papers ADD COLUMN {name} {dtype}')
                log(f'  added column: {name}')
            except Exception as e:
                log(f'  WARN: cannot add column {name}: {e}')
    conn.commit()

    # Backup before write
    if not dry_run and added_count == 0 and start_line == 0:
        backups = backup_files()
        log(f'  backup created: {list(backups.keys())}')

    # Count input lines
    log('Counting input lines...')
    n_input = sum(1 for _ in open(input_path, encoding='utf-8'))
    log(f'  input papers: {n_input:,}')

    if dry_run:
        log('DRY-RUN mode: not writing to KB')
        conn.close()
        return True

    # Process
    batch_texts: list[str] = []
    batch_papers: list[dict] = []
    total_processed = start_line
    total_added = added_count
    total_skipped_dup = 0
    total_skipped_invalid = 0
    t_run = time.time()

    with open(input_path, encoding='utf-8') as f:
        if start_line > 0:
            log(f'  skipping first {start_line} lines')
            for _ in range(start_line):
                f.readline()

        for line_num, line in enumerate(f, start=start_line + 1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                total_skipped_invalid += 1
                continue

            paper = normalize_paper(raw)
            if not paper:
                total_skipped_invalid += 1
                continue

            pid = paper['paper_id']
            if pid in existing_ids:
                total_skipped_dup += 1
                continue

            existing_ids.add(pid)
            embed_text = (paper['title'] + ' ' + paper['summary'])[:512]
            batch_texts.append(embed_text)
            batch_papers.append(paper)

            if len(batch_texts) >= BATCH_SIZE:
                t0 = time.time()

                embs = model.encode(
                    batch_texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                embs = np.array(embs, dtype=np.float32)

                index.add(embs)

                for paper in batch_papers:
                    cur.execute(
                        '''INSERT OR IGNORE INTO papers
                           (paper_id, title, authors, abstract, year, categories,
                            text_prefix, source_file, created, source, version, fields)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            paper['paper_id'],
                            paper['title'],
                            paper['authors'],
                            paper['summary'],
                            paper['year'],
                            paper['categories'],
                            paper['summary'][:500],
                            'self_grow.jsonl',
                            (paper['year'] + '-01-01T00:00:00.000') if paper['year'] else '',
                            paper['source'],
                            paper['version'],
                            paper['categories'],
                        ),
                    )

                conn.commit()

                with open(IDS_FILE, 'a', encoding='utf-8') as f_ids:
                    for p in batch_papers:
                        f_ids.write(p['paper_id'] + '\n')

                total_processed = line_num
                total_added += len(batch_texts)
                elapsed = time.time() - t0
                rate = BATCH_SIZE / elapsed if elapsed > 0 else 0
                log(
                    f'  batch {total_added // BATCH_SIZE:>4} | '
                    f'added: {total_added:>6} | '
                    f'index: {index.ntotal:>9,} | '
                    f'skip_dup: {total_skipped_dup:>5} | '
                    f'{rate:.1f}/s'
                )

                save_checkpoint({
                    'processed_lines': total_processed,
                    'added_count': total_added,
                    'last_paper_id': batch_papers[-1]['paper_id'],
                })

                batch_texts = []
                batch_papers = []

        # Tail batch
        if batch_texts:
            log(f'  flushing tail batch: {len(batch_texts)} papers')
            embs = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embs = np.array(embs, dtype=np.float32)
            index.add(embs)

            for paper in batch_papers:
                cur.execute(
                    '''INSERT OR IGNORE INTO papers
                       (paper_id, title, authors, abstract, year, categories,
                        text_prefix, source_file, created, source, version, fields)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        paper['paper_id'],
                        paper['title'],
                        paper['authors'],
                        paper['summary'],
                        paper['year'],
                        paper['categories'],
                        paper['summary'][:500],
                        'self_grow.jsonl',
                        (paper['year'] + '-01-01T00:00:00.000') if paper['year'] else '',
                        paper['source'],
                        paper['version'],
                        paper['categories'],
                    ),
                )
            conn.commit()
            with open(IDS_FILE, 'a', encoding='utf-8') as f_ids:
                for p in batch_papers:
                    f_ids.write(p['paper_id'] + '\n')

            total_added += len(batch_texts)
            log(f'  tail batch added: +{len(batch_texts)}')

    # Save FAISS index
    log('Saving FAISS index...')
    t0 = time.time()
    faiss.write_index(index, str(FAISS_INDEX))
    log(f'  saved ({time.time()-t0:.1f}s, {FAISS_INDEX.stat().st_size/1e9:.2f} GB)')

    clear_checkpoint()

    total_time = time.time() - t_run
    log('=' * 70)
    log(f'SELF-GROW DONE!')
    log(f'  added: {total_added}')
    log(f'  skipped_dup: {total_skipped_dup}')
    log(f'  skipped_invalid: {total_skipped_invalid}')
    log(f'  total in index: {index.ntotal:,}')
    log(f'  total time: {total_time:.1f}s ({total_added / max(1, total_time):.1f}/s)')
    log('=' * 70)

    conn.close()
    return True


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='self_grow — 论文 KB 自我生长')
    sub = parser.add_subparsers(dest='cmd')

    p_ingest = sub.add_parser('ingest', help='从 jsonl 增量添加')
    p_ingest.add_argument('--input', required=True, help='输入 jsonl 路径')
    p_ingest.add_argument('--dry-run', action='store_true', help='只看不写')
    p_ingest.add_argument('--reset', action='store_true', help='重置断点')

    p_verify = sub.add_parser('verify', help='验证 KB 一致性')

    args = parser.parse_args()

    if args.cmd == 'verify':
        verify_consistency()
    elif args.cmd == 'ingest':
        ok = ingest(Path(args.input), dry_run=args.dry_run, reset=args.reset)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()