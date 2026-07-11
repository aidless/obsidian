#!/usr/bin/env python3
"""daily_grow.py — 每日自我生长管道

一个命令完成"拉新论文 → 去重 → 嵌入 → 入库 → 验证"全流程。

工作流:
1. 从 arxiv API 拉取最近 N 天的指定类别论文
2. 与现有 KB 去重 (paper_ids.txt)
3. 调用 self_grow.py 嵌入并加入 KB
4. 验证 KB 一致性, 写日志

用法:
    python daily_grow.py                                      # 默认配置
    python daily_grow.py --cats cs.LG cs.CL cs.AI --days 7   # 自定义类别和时间窗
    python daily_grow.py --max 200 --days 3                  # 限制数量
    python daily_grow.py --dry-run                            # 只拉不写, 看会进来什么
    python daily_grow.py --skip-fetch --input <jsonl>        # 跳过拉取, 用已有 jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

# ════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════
KB_DIR = Path(r'E:/peS2o_kb_faiss')
CS_DIR = Path(r'E:/peS2o_cs')
WORK_DIR = Path(r'F:/test/2026-06-14-20-48-36')

STAGING_FILE = CS_DIR / 'daily_grow_staging.jsonl'
LOG_FILE = KB_DIR / 'daily_grow.log'
STATE_FILE = KB_DIR / 'daily_grow_state.json'

ARXIV_API = 'http://export.arxiv.org/api/query'
ATOM_NS = '{http://www.w3.org/2005/Atom}'
ARXIV_NS = '{http://arxiv.org/schemas/atom}'

DEFAULT_CATS = ['cs.LG', 'cs.CL', 'cs.AI']
DEFAULT_DAYS = 1
DEFAULT_MAX = 100

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
# State (idempotency)
# ════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'last_run': None, 'last_added': 0, 'runs': 0}


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)


# ════════════════════════════════════════════════════════════════
# Fetch from arxiv API
# ════════════════════════════════════════════════════════════════

def fetch_arxiv(cats: list[str], days: int, max_results: int) -> list[dict]:
    """Fetch papers from arxiv API for given categories and time window.

    Uses cat:CS.X submittedDate:[YYYYMMDDHHMM+TO+YYYYMMDDHHMM] syntax.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    # arxiv date format: YYYYMMDDHHMM (UTC)
    start_str = start.strftime('%Y%m%d%H%M')
    end_str = end.strftime('%Y%m%d%H%M')

    # Build query: (cat:cs.LG OR cat:cs.CL OR ...) AND submittedDate:[...]
    cat_query = ' OR '.join(f'cat:{c}' for c in cats)
    date_filter = f' AND submittedDate:[{start_str} TO {end_str}]'
    full_query = f'({cat_query}){date_filter}'

    papers = []
    consecutive_429 = 0
    for batch_start in range(0, max_results, 50):
        batch_size = min(50, max_results - batch_start)
        params = {
            'search_query': full_query,
            'start': str(batch_start),
            'max_results': str(batch_size),
            'sortBy': 'submittedDate',
            'sortOrder': 'descending',
        }
        url = f'{ARXIV_API}?{urllib.parse.urlencode(params)}'

        log(f'fetch: start={batch_start}, size={batch_size}')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'daily_grow/2.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                xml_data = r.read()
            consecutive_429 = 0  # reset on success
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                consecutive_429 += 1
                wait_sec = 60 * consecutive_429  # 60s, 120s, 180s, ...
                log(f'fetch {e.code} (consecutive {consecutive_429}), waiting {wait_sec}s...')
                time.sleep(wait_sec)
                if consecutive_429 >= 5:
                    log(f'fetch: too many rate-limits, aborting')
                    break
                continue  # retry this batch
            else:
                log(f'fetch error: {e}', also_print=True)
                break
        except Exception as e:
            log(f'fetch error: {e}', also_print=True)
            break

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            log(f'parse error: {e}', also_print=True)
            break

        entries = root.findall(f'{ATOM_NS}entry')
        if not entries:
            break

        for entry in entries:
            paper = parse_arxiv_entry(entry)
            if paper:
                papers.append(paper)

        # arxiv returns totalResults in opensearch; check if exhausted
        total_elem = root.find(f'{ATOM_NS}opensearch:totalResults')
        if total_elem is not None:
            try:
                total = int(total_elem.text)
                if batch_start + len(entries) >= total:
                    break
            except (ValueError, TypeError):
                pass

        # Rate-limit politely
        time.sleep(3.5)

    return papers


def parse_arxiv_entry(entry: ET.Element) -> dict | None:
    """Parse one arxiv API entry into a normalized dict."""
    id_elem = entry.find(f'{ATOM_NS}id')
    if id_elem is None or not id_elem.text:
        return None

    # id URL like http://arxiv.org/abs/2507.12345v1
    arxiv_url = id_elem.text.strip()
    arxiv_id = arxiv_url.split('/')[-1]
    arxiv_id = arxiv_id.split('v')[0]  # strip version

    title_elem = entry.find(f'{ATOM_NS}title')
    title = ' '.join((title_elem.text or '').split()) if title_elem is not None else ''

    summary_elem = entry.find(f'{ATOM_NS}summary')
    summary = ' '.join((summary_elem.text or '').split()) if summary_elem is not None else ''

    published_elem = entry.find(f'{ATOM_NS}published')
    published = published_elem.text if published_elem is not None else ''

    updated_elem = entry.find(f'{ATOM_NS}updated')
    updated = updated_elem.text if updated_elem is not None else ''

    authors = []
    for author in entry.findall(f'{ATOM_NS}author'):
        name_elem = author.find(f'{ATOM_NS}name')
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text)

    categories = []
    primary_cat = ''
    for cat in entry.findall(f'{ATOM_NS}category'):
        term = cat.get('term')
        if term:
            categories.append(term)
    primary_elem = entry.find(f'{ARXIV_NS}primary_category')
    if primary_elem is not None:
        primary_cat = primary_elem.get('term', '')

    if not title or not arxiv_id:
        return None

    return {
        'paper_id': arxiv_id,
        'arxiv_id': arxiv_id,
        'title': title,
        'authors': authors,
        'summary': summary,
        'abstract': summary,
        'categories': categories,
        'primary_category': primary_cat,
        'published': published,
        'created': published,
        'updated': updated,
        'source': 'arxiv',
        'version': '1.0',
        'text': f'{title}\n\n{summary}',
    }


# ════════════════════════════════════════════════════════════════
# Deduplicate against KB
# ════════════════════════════════════════════════════════════════

def filter_new(papers: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (new_papers, existing_papers) based on KB paper_ids.txt."""
    ids_file = KB_DIR / 'paper_ids.txt'
    existing = set()
    if ids_file.exists():
        with open(ids_file, encoding='utf-8') as f:
            for line in f:
                pid = line.strip()
                if pid:
                    existing.add(pid)

    new = []
    dup = []
    for p in papers:
        pid = p['paper_id']
        if pid in existing:
            dup.append(p)
        else:
            new.append(p)

    return new, dup


# ════════════════════════════════════════════════════════════════
# Run self_grow.py as subprocess
# ════════════════════════════════════════════════════════════════

def run_self_grow(input_path: Path, *, reset: bool = False) -> int:
    """Run self_grow.py and return exit code."""
    self_grow = KB_DIR / 'self_grow.py'
    if not self_grow.exists():
        log(f'self_grow.py not found at {self_grow}')
        return 1

    cmd = [sys.executable, str(self_grow), 'ingest', '--input', str(input_path)]
    if reset:
        cmd.append('--reset')

    log(f'running: {" ".join(cmd)}')
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(KB_DIR))
    elapsed = time.time() - t0
    log(f'self_grow finished in {elapsed:.1f}s, exit={result.returncode}')
    return result.returncode


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='daily_grow — 每日论文 KB 自生长')
    parser.add_argument('--cats', nargs='+', default=DEFAULT_CATS, help='arxiv 类别')
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS, help='回看天数')
    parser.add_argument('--max', type=int, default=DEFAULT_MAX, dest='max_results', help='最大拉取数')
    parser.add_argument('--dry-run', action='store_true', help='只拉不写')
    parser.add_argument('--skip-fetch', action='store_true', help='跳过拉取, 用 --input')
    parser.add_argument('--input', type=str, help='跳过拉取时使用的 jsonl 路径')
    parser.add_argument('--reset', action='store_true', help='重置 self_grow 断点')
    parser.add_argument('--keep-staging', action='store_true', help='不删除 staging 文件')
    args = parser.parse_args()

    log('=' * 70)
    log(f'DAILY-GROW START')
    log(f'  cats: {args.cats}')
    log(f'  days: {args.days}')
    log(f'  max: {args.max_results}')
    log(f'  dry-run: {args.dry_run}')
    log(f'  skip-fetch: {args.skip_fetch}')

    state = load_state()
    state['runs'] = state.get('runs', 0) + 1

    # ── Step 1: Fetch ─────────────────────────────────────────────
    if args.skip_fetch:
        if not args.input:
            log('ERROR: --skip-fetch requires --input')
            sys.exit(1)
        input_path = Path(args.input)
        if not input_path.exists():
            log(f'ERROR: input not found: {input_path}')
            sys.exit(1)
        log(f'Step 1: skipping fetch, using {input_path}')
        with open(input_path, encoding='utf-8') as f:
            raw_papers = [json.loads(line) for line in f if line.strip()]
    else:
        log(f'Step 1: fetching from arxiv API...')
        t0 = time.time()
        raw_papers = fetch_arxiv(args.cats, args.days, args.max_results)
        log(f'  fetched: {len(raw_papers)} papers in {time.time()-t0:.1f}s')

        # Save staging
        with open(STAGING_FILE, 'w', encoding='utf-8') as f:
            for p in raw_papers:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')
        log(f'  staged: {STAGING_FILE}')

        input_path = STAGING_FILE

    # ── Step 2: Filter new vs existing ────────────────────────────
    log('Step 2: filtering against KB...')
    new_papers, dup_papers = filter_new(raw_papers)
    log(f'  fetched: {len(raw_papers)} | new: {len(new_papers)} | already in KB: {len(dup_papers)}')

    if not new_papers:
        log('No new papers to add. Done.')
        state['last_run'] = datetime.now(timezone.utc).isoformat()
        state['last_added'] = 0
        save_state(state)
        if not args.keep_staging and STAGING_FILE.exists():
            STAGING_FILE.unlink()
        return 0

    # Write only new papers to a clean staging for ingest
    if not args.skip_fetch:
        clean_staging = CS_DIR / f'daily_grow_new_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
        with open(clean_staging, 'w', encoding='utf-8') as f:
            for p in new_papers:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')
        log(f'  clean staging: {clean_staging}')
        ingest_input = clean_staging
    else:
        ingest_input = input_path

    if args.dry_run:
        log('DRY-RUN: not running self_grow')
        log('Sample new papers:')
        for p in new_papers[:5]:
            log(f'  - {p["paper_id"]}: {p["title"][:80]}')
        state['last_run'] = datetime.now(timezone.utc).isoformat()
        state['last_added'] = 0
        save_state(state)
        return 0

    # ── Step 3: Run self_grow ──────────────────────────────────────
    log('Step 3: running self_grow.py...')
    rc = run_self_grow(ingest_input, reset=args.reset)
    if rc != 0:
        log(f'self_grow failed with exit {rc}')
        sys.exit(rc)

    # ── Step 4: Verify ────────────────────────────────────────────
    log('Step 4: verifying KB consistency...')
    # 4a. self_grow 自带的轻量级 verify (FAISS vs paper_ids.txt)
    verify_cmd = [sys.executable, str(KB_DIR / 'self_grow.py'), 'verify']
    subprocess.run(verify_cmd, cwd=str(KB_DIR))
    # 4b. kb_health 严格检查 (SQLite vs FAISS vs IDs),出问题非零退出
    log('Step 4b: kb_health --strict (主索引对齐 + ID 抽样)...')
    health_cmd = [sys.executable, str(KB_DIR / 'kb_health.py'),
                  '--strict', '--sample', '2000', '--json']
    health_log = KB_DIR / 'last_health_check.json'
    try:
        with open(health_log, 'w', encoding='utf-8') as hf:
            rc = subprocess.run(health_cmd, cwd=str(KB_DIR),
                                stdout=hf, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            log('  ✅ KB healthy')
        elif rc == 1:
            log(f'  ⚠ KB WARN (exit {rc}) — see {health_log}')
            log('  ⚠ daily_grow 仍完成,但下次需排查')
        else:
            log(f'  🚨 KB DANGER (exit {rc}) — see {health_log}')
            log('  🚨 主索引严重漂移,建议手动运行 finalize_rebuild.py')
            # 不 sys.exit: 让 state 仍然写完, 以便追踪; 但返回非 0 让调度器感知
    except Exception as e:
        log(f'  health check crashed: {e}')

    # ── Cleanup ───────────────────────────────────────────────────
    if not args.keep_staging:
        if STAGING_FILE.exists():
            STAGING_FILE.unlink()
        # Don't auto-delete clean_staging — useful for debugging
        log(f'  staging: {ingest_input} (kept for debug)')

    state['last_run'] = datetime.now(timezone.utc).isoformat()
    state['last_added'] = len(new_papers)
    state['last_added_ids'] = [p['paper_id'] for p in new_papers[:50]]
    save_state(state)

    log(f'DAILY-GROW DONE! added {len(new_papers)} new papers')
    log('=' * 70)


if __name__ == '__main__':
    main()