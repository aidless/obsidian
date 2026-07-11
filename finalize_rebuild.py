#!/usr/bin/env python3
"""finalize_rebuild.py — 完成重建的最后一步: swap + verify

用法: py -3 finalize_rebuild.py [--auto]

手动跑: 等 rebuild_kb_clean.py 跑完后, 再跑这个
自动跑: 加 --auto, 它会每 30s 检查一次, 完成后自动 swap
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import faiss
import numpy as np

KB_DIR = Path(r'E:/peS2o_kb_faiss')
OLD_FAISS = KB_DIR / 'papers.index'
OLD_IDS = KB_DIR / 'paper_ids.txt'
NEW_FAISS = KB_DIR / 'papers_clean.index'
NEW_IDS = KB_DIR / 'paper_ids_clean.txt'
STATE_FILE = KB_DIR / 'rebuild_state.json'
BACKUP_DIR = KB_DIR / 'rebuild_backups'


def is_rebuild_done() -> bool:
    """Check if rebuild has finished (state cleaned up or files ready)."""
    if not STATE_FILE.exists():
        return NEW_FAISS.exists() and NEW_IDS.exists()

    try:
        with open(STATE_FILE, encoding='utf-8') as f:
            s = json.load(f)
        return s.get('finished_at') is not None
    except Exception:
        return False


def wait_for_rebuild(timeout_min: int = 240, poll_sec: int = 30) -> bool:
    print(f'waiting for rebuild to complete (timeout {timeout_min} min)...')
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        if is_rebuild_done():
            print('  rebuild appears complete!')
            return True

        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, encoding='utf-8') as f:
                    s = json.load(f)
                count = s.get('embedded_count', 0)
                total_estimate = 362074  # from log
                pct = count / total_estimate * 100
                print(f'  [{time.strftime("%H:%M:%S")}] still running: {count:,}/{total_estimate:,} ({pct:.1f}%)')
            except Exception:
                pass
        time.sleep(poll_sec)
    return False


def backup_old_files() -> dict:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    paths = {}
    for src in [OLD_FAISS, OLD_IDS]:
        if src.exists():
            dst = BACKUP_DIR / f'{src.stem}_pre_swap_{ts}{src.suffix}'
            shutil.copy2(src, dst)
            paths[src.name] = dst
    return paths


def atomic_swap() -> bool:
    print('atomic swap:')
    print(f'  {NEW_FAISS.name} -> {OLD_FAISS.name}')
    print(f'  {NEW_IDS.name} -> {OLD_IDS.name}')
    try:
        os.replace(NEW_FAISS, OLD_FAISS)
        os.replace(NEW_IDS, OLD_IDS)
        print('  swap OK')
        return True
    except Exception as e:
        print(f'  swap FAILED: {e}')
        return False


def verify_clean() -> bool:
    print('\nverify clean index:')
    if not OLD_FAISS.exists():
        print('  papers.index missing!')
        return False
    if not OLD_IDS.exists():
        print('  paper_ids.txt missing!')
        return False

    index = faiss.read_index(str(OLD_FAISS))
    with open(OLD_IDS, encoding='utf-8') as f:
        ids = [line.strip() for line in f if line.strip()]

    print(f'  index: {index.ntotal:,} vectors')
    print(f'  ids: {len(ids):,} entries')

    if index.ntotal != len(ids):
        print(f'  MISMATCH: vectors={index.ntotal} vs ids={len(ids)}')
        return False

    print('  OK: vectors match ids')

    # Sample search
    from sentence_transformers import SentenceTransformer
    print('\n  sample queries:')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    for q in ['calibration in large language models',
              'graph neural network for molecules',
              'reinforcement learning for robotics']:
        q_emb = model.encode([q], normalize_embeddings=True)
        scores, idxs = index.search(np.array(q_emb, dtype=np.float32), 5)
        unique_pids = set()
        for s, i in zip(scores[0], idxs[0]):
            if 0 <= i < len(ids):
                unique_pids.add(ids[i])
        print(f'    Q: "{q[:50]}"')
        print(f'      scores: {[round(float(s), 4) for s in scores[0]]}')
        print(f'      unique results: {len(unique_pids)}/5')

    return True


def main():
    parser = argparse.ArgumentParser(description='finalize_rebuild — 完成重建收尾')
    parser.add_argument('--auto', action='store_true', help='自动等重建完成')
    parser.add_argument('--timeout-min', type=int, default=240, help='自动等多久')
    args = parser.parse_args()

    if args.auto:
        if not wait_for_rebuild(timeout_min=args.timeout_min):
            print('TIMEOUT: rebuild not finished in time')
            sys.exit(1)

    if not NEW_FAISS.exists():
        print(f'new index not found: {NEW_FAISS}')
        print('rebuild may not have finished yet')
        sys.exit(1)

    print('\nbacking up old files...')
    backups = backup_old_files()
    for name, path in backups.items():
        print(f'  backup: {path}')

    if not atomic_swap():
        sys.exit(1)

    if verify_clean():
        print('\nFINALIZE OK! clean KB is active.')
        print('  run kb_search.py to test')
    else:
        print('\nFINALIZE completed but verify FAILED — manual check needed')
        sys.exit(1)


if __name__ == '__main__':
    main()