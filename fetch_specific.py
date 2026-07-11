#!/usr/bin/env python3
"""fetch_specific.py — 精准拉取指定 arxiv 论文并入 KB

用途: 写 TMLR 论文时遇到"我需要引这篇",一行命令搞定。

功能:
1. 给一个或多个 arxiv ID, 从 arxiv API 拉元数据
2. 自动去重 (跳过已在 KB 的)
3. 嵌入并加入 FAISS + SQLite + paper_ids.txt
4. 自动生成 BibTeX (直接可粘贴)
5. 备份已有文件 (rollback-friendly)

用法:
    # 基本: 拉 1-2 篇 + 入库 + 生成 BibTeX
    py -3 fetch_specific.py 2606.28661 2606.27288 --bibtex out.bib

    # 只拉不写
    py -3 fetch_specific.py 2606.28661 --no-ingest

    # 拉但不查重 (强制入库, 用于修复损坏记录)
    py -3 fetch_specific.py 2606.28661 --force

    # 一次拉 10 篇 + 自动放对应 paper 目录
    py -3 fetch_specific.py 2601.07367 2508.02694 2501.10069 ^
        --bibtex F:\Research\PAPER4_CONSOLIDATED\must_cite.bib
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

ATOM_NS = '{http://www.w3.org/2005/Atom}'

KB_DIR = Path(r'E:/peS2o_kb_faiss')
STAGING_DIR = Path(r'E:/peS2o_cs')


# ════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════
def log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


# ════════════════════════════════════════════════════════════════
# Arxiv API fetch
# ════════════════════════════════════════════════════════════════
def fetch_arxiv_by_id(arxiv_id: str) -> dict | None:
    """Fetch single arxiv paper by ID."""
    clean_id = arxiv_id.split('v')[0]
    url = f'http://export.arxiv.org/api/query?id_list={clean_id}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'fetch_specific/2.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            xml_data = r.read()
    except Exception as e:
        log(f'  fetch error for {clean_id}: {e}')
        return None

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        log(f'  parse error for {clean_id}: {e}')
        return None

    entries = root.findall(f'{ATOM_NS}entry')
    if not entries:
        log(f'  no entry returned for {clean_id}')
        return None

    entry = entries[0]
    id_elem = entry.find(f'{ATOM_NS}id')
    if id_elem is None or not id_elem.text:
        return None

    arxiv_url = id_elem.text.strip()
    full_id = arxiv_url.split('/')[-1].split('v')[0]

    title = ''
    if entry.find(f'{ATOM_NS}title') is not None:
        title = ' '.join((entry.find(f'{ATOM_NS}title').text or '').split())

    summary = ''
    if entry.find(f'{ATOM_NS}summary') is not None:
        summary = ' '.join((entry.find(f'{ATOM_NS}summary').text or '').split())

    published = ''
    if entry.find(f'{ATOM_NS}published') is not None:
        published = entry.find(f'{ATOM_NS}published').text or ''

    authors = []
    for a in entry.findall(f'{ATOM_NS}author'):
        n = a.find(f'{ATOM_NS}name')
        if n is not None and n.text:
            authors.append(n.text)

    categories = []
    for cat in entry.findall(f'{ATOM_NS}category'):
        t = cat.get('term')
        if t:
            categories.append(t)

    return {
        'paper_id': full_id,
        'arxiv_id': full_id,
        'title': title,
        'authors': authors,
        'summary': summary,
        'abstract': summary,
        'categories': categories,
        'primary_category': categories[0] if categories else '',
        'published': published,
        'created': published,
        'source': 'arxiv',
        'version': '1.0',
        'text': f'{title}\n\n{summary}',
    }


# ════════════════════════════════════════════════════════════════
# Dedup against KB
# ════════════════════════════════════════════════════════════════
def load_kb_ids() -> set[str]:
    ids_file = KB_DIR / 'paper_ids.txt'
    ids = set()
    if ids_file.exists():
        with open(ids_file, encoding='utf-8') as f:
            for line in f:
                p = line.strip()
                if p:
                    ids.add(p)
    return ids


# ════════════════════════════════════════════════════════════════
# BibTeX (inline, no cross-file import)
# ════════════════════════════════════════════════════════════════
def make_bibtex_key(paper: dict) -> str:
    authors = paper.get('authors') or []
    first_author = ''
    if authors:
        a = authors[0]
        first_author = re.sub(r'[^a-z]', '', a.lower())
        if not first_author:
            first_author = 'anon'
    year = (paper.get('published') or '')[:4]
    if not year:
        year = 'nd'
    title_words = re.findall(r'[a-z]{4,}', (paper.get('title') or '').lower())
    keyword = title_words[0] if title_words else 'ref'
    return f'{first_author}{year}{keyword}'


def format_bibtex(paper: dict) -> str:
    pid = paper.get('paper_id') or ''
    title = (paper.get('title') or '').strip()
    authors = paper.get('authors') or []
    if isinstance(authors, str):
        try:
            authors = json.loads(authors)
        except Exception:
            authors = [authors]
    authors_str = ' and '.join(
        a if isinstance(a, str) else (a.get('name') or '') for a in authors
    )
    year = (paper.get('published') or '')[:4] or 'n.d.'
    pid_clean = pid.split('v')[0]

    categories = paper.get('categories', []) or []
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(',') if c.strip()]
    primary_cat = categories[0] if categories else 'cs.LG'
    categories_str = ','.join(categories) if isinstance(categories, list) else str(categories)

    venue = 'arXiv preprint arXiv:' + pid_clean if any(
        c in categories_str for c in ['cs.CL', 'cs.LG', 'cs.AI']
    ) else 'arXiv:' + pid_clean

    key = make_bibtex_key(paper)
    return f"""@misc{{{key},
  title     = {{{title}}},
  author    = {{{authors_str}}},
  year      = {{{year}}},
  eprint    = {{{pid_clean}}},
  archivePrefix = {{arXiv}},
  primaryClass = {{{primary_cat}}}
}}"""


# ════════════════════════════════════════════════════════════════
# Ingest via self_grow.py subprocess
# ════════════════════════════════════════════════════════════════
def ingest_papers(papers: list[dict]) -> bool:
    if not papers:
        log('No papers to ingest')
        return False

    staging = STAGING_DIR / f'fetch_specific_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    with open(staging, 'w', encoding='utf-8') as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    log(f'Staged: {staging}')

    self_grow = KB_DIR / 'self_grow.py'
    cmd = [sys.executable, str(self_grow), 'ingest', '--input', str(staging)]
    log(f'Running: {" ".join(cmd)}')
    import subprocess
    result = subprocess.run(cmd, cwd=str(KB_DIR))
    return result.returncode == 0


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='fetch_specific — 精准拉取 arxiv 论文')
    parser.add_argument('ids', nargs='+', help='arxiv ID(s) to fetch')
    parser.add_argument('--bibtex', help='导出 BibTeX 到指定文件')
    parser.add_argument('--no-ingest', action='store_true', help='只拉不入库')
    parser.add_argument('--force', action='store_true', help='强制入库 (跳过去重)')
    parser.add_argument('--sleep', type=float, default=3.0, help='API 请求间隔(秒)')
    args = parser.parse_args()

    log(f'fetch_specific — {len(args.ids)} IDs requested')

    # Step 1: Fetch all papers
    log('Step 1: fetching from arxiv API...')
    papers = []
    for i, arxiv_id in enumerate(args.ids, 1):
        clean_id = arxiv_id.split('v')[0]
        log(f'  [{i}/{len(args.ids)}] {clean_id}...')
        paper = fetch_arxiv_by_id(clean_id)
        if paper is None:
            log(f'    FAILED')
            continue
        log(f'    OK: {paper["title"][:70]}')
        papers.append(paper)
        if i < len(args.ids):
            time.sleep(args.sleep)

    log(f'\nFetched {len(papers)}/{len(args.ids)} papers')
    if not papers:
        log('Nothing to do.')
        sys.exit(1)

    # Step 2: Dedup (unless --force)
    if not args.force:
        log('Step 2: dedup against KB...')
        existing = load_kb_ids()
        new_papers = []
        dup_papers = []
        for p in papers:
            if p['paper_id'] in existing:
                dup_papers.append(p)
                log(f'  {p["paper_id"]}: ALREADY IN KB')
            else:
                new_papers.append(p)
                log(f'  {p["paper_id"]}: NEW')
        log(f'  new: {len(new_papers)}, already in KB: {len(dup_papers)}')
    else:
        new_papers = papers
        dup_papers = []
        log('Step 2: SKIPPED (--force)')

    # Step 3: Generate BibTeX (always, for all papers including dups)
    if args.bibtex:
        log(f'Step 3: generating BibTeX -> {args.bibtex}')
        Path(args.bibtex).parent.mkdir(parents=True, exist_ok=True)
        with open(args.bibtex, 'w', encoding='utf-8') as f:
            f.write(f'% Generated by fetch_specific.py on {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
            f.write(f'% Total: {len(papers)} papers ({len(new_papers)} new, {len(dup_papers)} already in KB)\n')
            f.write(f'% Source: arxiv API\n\n')
            for p in papers:
                f.write(format_bibtex(p) + '\n\n')
        log(f'  BibTeX saved: {args.bibtex} ({len(papers)} entries)')

    # Step 4: Ingest (unless --no-ingest)
    if args.no_ingest:
        log('Step 4: SKIPPED (--no-ingest)')
    elif new_papers:
        log(f'Step 4: ingesting {len(new_papers)} new papers...')
        if ingest_papers(new_papers):
            log(f'  ingest OK')
        else:
            log(f'  ingest FAILED')
            sys.exit(1)
    else:
        log('Step 4: no new papers to ingest')

    log('\nDone.')


if __name__ == '__main__':
    main()