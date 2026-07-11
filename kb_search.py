#!/usr/bin/env python3
"""kb_search.py — 论文知识库搜索器 (TMLR 写作专用)

设计目标: 把 36 万 CS 论文 KB 变成 TMLR 投稿前的"必引发现引擎"

功能:
1. 语义搜索: 给定 query, 返回 top-N 相关论文
2. 必引发现: 给定 query, 返回 KB 中相关但**不在你现有 .bib 里**的论文
3. BibTeX 导出: 一键生成可粘贴进 refs.bib 的条目
4. 过滤: 按年份 / 类别 / 来源 / 是否已读过滤
5. 与 search_faiss_v2 共享同一套 dedup + bridge 逻辑

用法:
    # 基本搜索
    py -3 kb_search.py "calibration in large language models" -n 10

    # 必引发现 (排除现有引用)
    py -3 kb_search.py "agent evaluation" --existing-refs refs.bib -n 5 --must-cite

    # 导出 BibTeX
    py -3 kb_search.py "TTRL" --bibtex out.bib -n 20

    # 过滤年份
    py -3 kb_search.py "graph neural network" --year-min 2024 --year-max 2026

    # 交互模式
    py -3 kb_search.py

依赖: sentence_transformers, faiss, numpy
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sqlite3
import sys
import time
from contextlib import contextmanager
from typing import Optional
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import faiss

# ════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════
KB_DIR = Path(r'E:/peS2o_kb_faiss')
FAISS_INDEX = KB_DIR / 'papers.index'
FAISS_GAP_INDEX = KB_DIR / 'papers_gap.index'  # 增量索引(daily_grow 期间生成的补丁)
SQLITE_DB = KB_DIR / 'papers.db'
IDS_FILE = KB_DIR / 'paper_ids.txt'
IDS_GAP_FILE = KB_DIR / 'papers_gap_ids.txt'    # gap 索引对应的 ID
MODEL_NAME = 'all-MiniLM-L6-v2'

sys.path.insert(0, str(KB_DIR))
from smart_rerank import expand_query, detect_category_preference, rerank_score

# ════════════════════════════════════════════════════════════════
# ID source prefix: 区分 peS2o 原生语料 vs arxiv daily_grow
#   pes2o:64690691    ← peS2o 8位数字 / SHA1 / 短数字 (paper_ids.txt)
#   arxiv:2502.19154  ← arxiv 标准编号 (papers_gap_ids.txt)
# 选 prefix 的好处:
#   1. 搜索结果一眼区分来源
#   2. must-cite 知道哪些能从 arxiv 下 PDF,哪些只能看摘要
#   3. 未来 rebuild 时不会把两类论文混在一起
# ════════════════════════════════════════════════════════════════
PREFIX_PES2O = 'pes2o:'
PREFIX_ARXIV = 'arxiv:'

# Lazy-loaded resources
_model = None
_index = None
_paper_ids = None          # 带前缀的 ID 列表 (供搜索结果展示)
_paper_ids_raw = None      # 原始 ID 列表 (供 SQLite 查询 paper_id 列)
_id_source = None          # 与 _paper_ids 等长的 'pes2o' / 'arxiv' 标签
_conn = None


# ════════════════════════════════════════════════════════════════
# Timeout utilities (跨平台, 不依赖 signal)
# ════════════════════════════════════════════════════════════════

class TimeoutError_(Exception):
    """用户级超时异常, 不与 builtin TimeoutError 冲突"""
    pass


class _Stopwatch:
    """轻量计时器 + 阶段剩余预算管理"""
    def __init__(self, total_timeout: float):
        self.start = time.time()
        self.total_timeout = total_timeout

    def remaining(self) -> float:
        """剩余总时间 (秒)"""
        return max(0.0, self.total_timeout - (time.time() - self.start))

    def check(self, stage: str = '', min_remaining: float = 0.0) -> None:
        """检查是否超时. raise TimeoutError_ if so."""
        remain = self.remaining()
        if remain < min_remaining:
            raise TimeoutError_(
                f'{stage} 超时 (剩余 {remain:.1f}s < 最低 {min_remaining:.1f}s, '
                f'总预算 {self.total_timeout:.0f}s)')


@contextmanager
def time_limit(total_timeout: float, stage: str = 'search'):
    """总超时 context manager.

    用法:
        with time_limit(30, stage='search') as sw:
            do_something()
            if sw.remaining() < 5:
                break   # 提前终止

    超时不会 kill 线程, 只能由合作方检查 sw.remaining() / sw.check() 主动退出.
    这是 Python 跨平台最稳的做法 (signal.SIGALRM 在 Windows / 多线程下不可靠).
    """
    sw = _Stopwatch(total_timeout)
    try:
        yield sw
    except TimeoutError_:
        raise
    finally:
        # 不做事, 仅作为 API 锚点
        pass


def _set_sqlite_busy_timeout(conn, seconds: float) -> None:
    """SQLite PRAGMA busy_timeout: 等待锁的最大秒数.

    注意: 这只在数据库被 LOCKED 时有效, 不会中断 long-running query.
    但能避免 "database is locked" 报错.
    """
    try:
        # 单位: 毫秒
        conn.execute(f'PRAGMA busy_timeout = {int(seconds * 1000)}')
    except sqlite3.OperationalError:
        pass


# 默认超时配置 (conservative)
DEFAULT_TOTAL_TIMEOUT = 30.0      # 总超时 (秒)
DEFAULT_MODEL_TIMEOUT = 60.0      # 模型加载超时
DEFAULT_SQLITE_BUSY_TIMEOUT = 5.0 # SQLite 锁等待


def load_resources(model_timeout: float = DEFAULT_MODEL_TIMEOUT,
                   sqlite_busy_timeout: float = DEFAULT_SQLITE_BUSY_TIMEOUT):
    global _model, _index, _paper_ids, _paper_ids_raw, _id_source, _conn
    from sentence_transformers import SentenceTransformer

    if _model is None:
        print(f'Loading model: {MODEL_NAME}...', end=' ', flush=True)
        t0 = time.time()
        _model = SentenceTransformer(MODEL_NAME)
        model_load_time = time.time() - t0
        if model_load_time > model_timeout:
            raise TimeoutError_(
                f'模型加载超时 ({model_load_time:.1f}s > {model_timeout:.0f}s). '
                f'如果是网络问题, 试试设置 HF_HUB_OFFLINE=1 (已经默认) 或增加 --model-timeout')
        print(f'done ({model_load_time:.1f}s)')

    if _index is None:
        print('Loading FAISS index...', end=' ', flush=True)
        t0 = time.time()
        _index = faiss.read_index(str(FAISS_INDEX))
        gap_count = 0
        if FAISS_GAP_INDEX.exists():
            _gap = faiss.read_index(str(FAISS_GAP_INDEX))
            _index.merge_from(_gap)
            gap_count = _gap.ntotal
        print(f'done ({time.time()-t0:.1f}s) | {_index.ntotal:,} vectors '
              f'(+{gap_count:,} gap)' if gap_count else
              f'done ({time.time()-t0:.1f}s) | {_index.ntotal:,} vectors')

    if _paper_ids is None:
        print('Loading paper_ids (with source prefix)...', end=' ', flush=True)
        t0 = time.time()
        raw_main = []
        with open(IDS_FILE, encoding='utf-8') as f:
            raw_main = [line.strip() for line in f if line.strip()]
        raw_gap = []
        if IDS_GAP_FILE.exists():
            with open(IDS_GAP_FILE, encoding='utf-8') as f:
                raw_gap = [line.strip() for line in f if line.strip()]

        # ID 分类启发式 (不只看文件来源,看 ID 本身格式):
        #   - 含 '.' 且形如 NNNN.NNNN → arxiv (无论在哪个文件里)
        #   - 40 位 hex → SHA1 (peS2o 内部)
        #   - 其他纯数字 → peS2o 内部 ID (8位为主)
        # 这样能修复 daily_grow 把 arxiv 论文塞进 paper_ids.txt 时的归类错误
        import re
        arxiv_pat = re.compile(r'^\d{4}\.\d{4,5}(v\d+)?$')

        def classify(pid: str) -> str:
            return 'arxiv' if arxiv_pat.match(pid) else 'pes2o'

        _paper_ids = []
        _paper_ids_raw = []
        _id_source = []
        n_arxiv_from_main = n_pes2o_from_gap = 0
        for x in raw_main:
            kind = classify(x)
            if kind == 'arxiv':
                n_arxiv_from_main += 1
            _paper_ids.append(f'{PREFIX_ARXIV if kind=="arxiv" else PREFIX_PES2O}{x}')
            _paper_ids_raw.append(x)
            _id_source.append(kind)
        for x in raw_gap:
            kind = classify(x)
            if kind == 'pes2o':
                n_pes2o_from_gap += 1
            _paper_ids.append(f'{PREFIX_ARXIV if kind=="arxiv" else PREFIX_PES2O}{x}')
            _paper_ids_raw.append(x)
            _id_source.append(kind)

        n_pes2o = _id_source.count('pes2o')
        n_arxiv = _id_source.count('arxiv')
        print(f'done ({time.time()-t0:.1f}s) | {len(_paper_ids):,} entries '
              f'({n_pes2o:,} pes2o + {n_arxiv:,} arxiv)')
        if n_arxiv_from_main or n_pes2o_from_gap:
            print(f'  ↳ reclassified: {n_arxiv_from_main} arxiv-style from main file, '
                  f'{n_pes2o_from_gap} non-arxiv from gap file')

    if _conn is None:
        _conn = sqlite3.connect(str(SQLITE_DB))
        _conn.row_factory = sqlite3.Row
        _set_sqlite_busy_timeout(_conn, sqlite_busy_timeout)

    return _model, _index, _paper_ids, _conn


# ════════════════════════════════════════════════════════════════
# Existing .bib parser (for must-cite mode)
# ════════════════════════════════════════════════════════════════

def parse_existing_bib(bib_paths: list[str]) -> set[str]:
    """Extract paper IDs / titles from existing .bib files.

    Returns set of normalized identifiers (arXiv IDs, titles lowercased).
    """
    ids = set()
    titles = set()
    for path in bib_paths:
        if not Path(path).exists():
            print(f'  WARN: bib not found: {path}', file=sys.stderr)
            continue
        with open(path, encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Find arXiv IDs (format: 1234.5678 or 1234.5678vN)
        for m in re.finditer(r'(\d{4}\.\d{4,5})(v\d+)?', content):
            ids.add(m.group(1))

        # Find titles in @TYPE{key, title={...}}
        for m in re.finditer(r'title\s*=\s*\{([^}]+)\}', content, re.IGNORECASE):
            t = m.group(1).strip().lower()
            titles.add(t[:80])  # first 80 chars as signature

    return ids, titles


def is_already_cited(paper_id: str, title: str, existing_ids: set, existing_titles: set) -> bool:
    pid_clean = paper_id.split('v')[0]
    if pid_clean in existing_ids:
        return True
    title_lower = (title or '').strip().lower()[:80]
    if title_lower and title_lower in existing_titles:
        return True
    return False


# ════════════════════════════════════════════════════════════════
# BibTeX formatter
# ════════════════════════════════════════════════════════════════

def make_bibtex_key(paper: dict) -> str:
    """Generate a BibTeX key like 'firstauthor2024keyword'."""
    # Author
    authors_raw = paper.get('authors') or '[]'
    if isinstance(authors_raw, str):
        try:
            authors = json.loads(authors_raw)
        except Exception:
            authors = [authors_raw]
    else:
        authors = authors_raw
    first_author = ''
    if authors:
        a = authors[0] if isinstance(authors[0], str) else authors[0].get('name', '')
        first_author = re.sub(r'[^a-z]', '', a.lower())
        if not first_author:
            first_author = 'anon'

    # Year
    year = paper.get('year') or 'n.d.'
    if 'T' in year:
        year = year[:4]
    elif len(year) >= 4:
        year = year[:4]

    # Keyword from title (first significant word)
    title_words = re.findall(r'[a-z]{4,}', (paper.get('title') or '').lower())
    keyword = title_words[0] if title_words else 'ref'

    return f'{first_author}{year}{keyword}'


def format_bibtex(paper: dict) -> str:
    """Format a paper as a BibTeX entry."""
    pid = paper.get('paper_id') or ''
    title = (paper.get('title') or '').strip()
    authors_raw = paper.get('authors') or '[]'
    if isinstance(authors_raw, str):
        try:
            authors = json.loads(authors_raw)
        except Exception:
            authors = [authors_raw]
    else:
        authors = authors_raw
    authors_str = ' and '.join(
        a if isinstance(a, str) else (a.get('name') or '') for a in authors
    )

    year = (paper.get('year') or '')[:4] or 'n.d.'
    pid_clean = pid.split('v')[0]
    src_tag = paper.get('id_source', 'arxiv')

    # Try to determine venue from categories
    categories = paper.get('categories', '') or ''
    if isinstance(categories, list):
        categories_str = ','.join(categories)
        primary_cat = categories[0] if categories else 'cs.LG'
    else:
        categories_str = categories
        primary_cat = categories.split(',')[0] if categories else 'cs.LG'
    venue = ''
    if 'cs.CL' in categories_str or 'cs.LG' in categories_str or 'cs.AI' in categories_str:
        venue = 'arXiv preprint arXiv:' + pid_clean
    else:
        venue = 'arXiv:' + pid_clean

    key = make_bibtex_key(paper)

    # peS2o 论文: eprint 用 peS2o ID (非 arxiv 编号),加 note 说明来源
    note_line = ''
    if src_tag == 'pes2o':
        note_line = f"  note      = {{peS2o ID {pid_clean}; full text only, no arXiv eprint}},\n"
        # peS2o 没 arxiv eprint,把 eprint 字段去掉换成 howpublished
        return f"""@misc{{{key},
  title     = {{{title}}},
  author    = {{{authors_str}}},
  year      = {{{year}}},
  howpublished = {{peS2o corpus (id={pid_clean})}},
  primaryClass = {{{primary_cat}}},
{note_line}}}"""

    return f"""@misc{{{key},
  title     = {{{title}}},
  author    = {{{authors_str}}},
  year      = {{{year}}},
  eprint    = {{{pid_clean}}},
  archivePrefix = {{arXiv}},
  primaryClass = {{{primary_cat}}}
}}"""


# ════════════════════════════════════════════════════════════════
# Search
# ════════════════════════════════════════════════════════════════

def search(
    query: str,
    n: int = 10,
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    source: str | None = None,
    category: str | None = None,
    must_cite: bool = False,
    smart: bool = True,
    existing_refs: list[str] | None = None,
    bibtex_out: str | None = None,
    json_out: str | None = None,
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
    model_timeout: float = DEFAULT_MODEL_TIMEOUT,
    sqlite_busy_timeout: float = DEFAULT_SQLITE_BUSY_TIMEOUT,
):
    """Main search function.

    超时保护:
      - total_timeout: 整个 search() 流程总预算 (默认 30s)
      - model_timeout: 模型加载超时 (默认 60s)
      - sqlite_busy_timeout: SQLite 锁等待 (默认 5s)

    超时后, 优雅降级: 返回已 fetch 的部分 top-K + 提示. 不抛异常.
    """
    with time_limit(total_timeout, stage='search') as sw:
        return _search_impl(
            query, n, sw,
            year_min=year_min, year_max=year_max,
            source=source, category=category,
            must_cite=must_cite, smart=smart,
            existing_refs=existing_refs,
            bibtex_out=bibtex_out, json_out=json_out,
            model_timeout=model_timeout,
            sqlite_busy_timeout=sqlite_busy_timeout,
        )


def _search_impl(
    query: str,
    n: int,
    sw: _Stopwatch,
    *,
    year_min: int | None = None,
    year_max: int | None = None,
    source: str | None = None,
    category: str | None = None,
    must_cite: bool = False,
    smart: bool = True,
    existing_refs: list[str] | None = None,
    bibtex_out: str | None = None,
    json_out: str | None = None,
    model_timeout: float = DEFAULT_MODEL_TIMEOUT,
    sqlite_busy_timeout: float = DEFAULT_SQLITE_BUSY_TIMEOUT,
):
    """内部实现: 由 search() 包装 time_limit() 调用."""
    try:
        model, index, paper_ids, conn = load_resources(
            model_timeout=model_timeout,
            sqlite_busy_timeout=sqlite_busy_timeout,
        )
    except TimeoutError_ as e:
        print(f'  ⚠ {e}', file=sys.stderr)
        print(f'  → 退化为: 仅返回已加载的部分 (空结果)')
        return []

    existing_ids = set()
    existing_titles = set()
    if must_cite and existing_refs:
        existing_ids, existing_titles = parse_existing_bib(existing_refs)
        print(f'Loaded existing refs: {len(existing_ids)} arXiv IDs, {len(existing_titles)} titles')

    # Smart query expansion
    cat_hint = []
    actual_query = query
    if smart:
        actual_query, _rules = expand_query(query)
        cat_hint = detect_category_preference(query)

    q_emb = model.encode([actual_query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)

    # Over-fetch for filtering (smart needs many more candidates)
    if smart:
        fetch_n = min(max(n * 15, 200), 800)
    else:
        fetch_n = max(n * 10, 50) if (must_cite or year_min or year_max or category) else n * 3
        fetch_n = min(fetch_n, 500)

    scores, idxs = index.search(q_emb, fetch_n)
    search_time = time.time()

    results = []
    seen_pids = set()
    timed_out = False
    for score, idx in zip(scores[0], idxs[0]):
        # 超时检查: 每条结果处理前检查, 优雅降级
        if sw.remaining() < 0.5:
            timed_out = True
            print(f'\n  ⚠ Timeout: 已用 {time.time()-sw.start:.1f}s / '
                  f'总预算 {sw.total_timeout:.0f}s. 提前终止, '
                  f'返回已找到的 {len(results)} 条结果.',
                  file=sys.stderr)
            break
        if idx < 0 or idx >= len(paper_ids):
            continue
        pid_display = paper_ids[idx]        # 带前缀, 用于展示
        pid_raw = _paper_ids_raw[idx]       # 原始 ID, 用于 SQLite 查询
        pid_source = _id_source[idx]        # 'pes2o' / 'arxiv' 标签
        if pid_raw in seen_pids:
            continue

        # Lookup SQLite by paper_id (string) — 必须用原始 ID
        cur = conn.cursor()
        cur.execute(
            'SELECT paper_id, title, fields, text_prefix, abstract, created, year, categories, authors, source, version FROM papers WHERE paper_id = ?',
            (pid_raw,),
        )
        row = cur.fetchone()
        if not row:
            paper = {'paper_id': pid_raw, 'title': '(not in SQLite)', 'authors': '[]',
                     'year': '', 'categories': '', 'source': '', 'text_prefix': ''}
        else:
            paper = dict(row)
            paper['text_prefix'] = row['text_prefix'] or row['abstract'] or ''
        # 加来源标签和展示用 ID
        paper['id_display'] = pid_display
        paper['id_source'] = pid_source
        paper['score'] = float(score)
        pid = pid_raw  # 后续用 pid 兼容旧逻辑

        # Year filter
        paper_year = (paper.get('year') or '')[:4]
        if year_min and paper_year and int(paper_year) < year_min:
            continue
        if year_max and paper_year and int(paper_year) > year_max:
            continue

        # Source filter
        if source and source.lower() not in (paper.get('source') or '').lower():
            continue

        # Category filter
        if category and category not in (paper.get('categories') or ''):
            continue

        # Must-cite filter
        if must_cite and existing_refs:
            if is_already_cited(pid, paper.get('title', ''), existing_ids, existing_titles):
                continue

        # Smart rerank score
        if smart:
            base = float(score)
            final_score, _ = rerank_score(paper, base, cat_hint)
            paper['base_score'] = round(base, 4)
        else:
            final_score = float(score)

        seen_pids.add(pid)
        paper['score'] = round(final_score, 4)
        results.append(paper)

        if len(results) >= n * 3:  # collect more, then filter+rerank
            break

    # Smart filter: prefer cs.* + recent papers when category_hint present
    if smart and cat_hint:
        filtered = []
        for r in results:
            cats = r.get('categories', '') or ''
            year = (r.get('year') or '')[:4]
            in_hint = any(c in cats for c in cat_hint)
            recent = year in {'2024', '2025', '2026'}
            if in_hint or recent:
                filtered.append(r)
        if filtered:
            results = filtered

    # Rerank by smart score
    if smart:
        results.sort(key=lambda r: r.get('score', 0), reverse=True)

    results = results[:n]

    # Print results
    elapsed = time.time() - search_time
    print(f'\n{"="*78}')
    print(f'  Query: "{query}"')
    if year_min or year_max:
        print(f'  Year filter: {year_min or "*"} - {year_max or "*"}')
    if category:
        print(f'  Category filter: {category}')
    if must_cite:
        print(f'  Must-cite mode: excluding {len(existing_ids)} existing IDs / {len(existing_titles)} titles')
    print(f'  KB: {index.ntotal:,} FAISS vectors / {len(paper_ids):,} unique IDs')
    print(f'  Top-{len(results)} results in {elapsed:.3f}s')
    print(f'{"="*78}\n')

    for i, p in enumerate(results, 1):
        print(f'  #{i:>2} | score={p.get("score", 0):.4f}')
        title = (p.get('title') or '')[:100]
        print(f'      {title}')
        pid = p.get('id_display', p.get('paper_id', '?'))  # 带前缀展示
        src_tag = p.get('id_source', '?')
        year = (p.get('year') or '')[:4] or '????'
        cats = (p.get('categories') or '')[:50]
        ksrc = (p.get('source') or '')[:15]
        # 标签: [arXiv] 可下PDF,  [peS2o] 仅摘要/正文
        tag = '[arXiv] ' if src_tag == 'arxiv' else '[peS2o]'
        print(f'      {tag}ID: {pid:<28} | year: {year} | src: {ksrc} | cats: {cats}')
        abstract = (p.get('text_prefix') or p.get('abstract') or '')[:200]
        if abstract:
            print(f'      abstract: {abstract}...')
        print()

    # Export BibTeX
    if bibtex_out and results:
        bibtex_path = Path(bibtex_out)
        bibtex_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'\nGenerating BibTeX for {len(results)} papers...')
        bib_entries = []
        for p in results:
            bib_entries.append(format_bibtex(p))
        with open(bibtex_path, 'w', encoding='utf-8') as f:
            f.write('% Auto-generated by kb_search.py\n')
            f.write(f'% Query: {query}\n')
            f.write(f'% Date: {time.strftime("%Y-%m-%d")}\n\n')
            f.write('\n\n'.join(bib_entries))
        print(f'  BibTeX saved to: {bibtex_path}')
        print(f'  {len(bib_entries)} entries')

    # Export JSON
    if json_out and results:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'query': query,
                'filters': {'year_min': year_min, 'year_max': year_max,
                            'category': category, 'source': source, 'must_cite': must_cite},
                'total': len(results),
                'results': results,
            }, f, ensure_ascii=False, indent=2)
        print(f'  JSON saved to: {json_path}')

    return results


# ════════════════════════════════════════════════════════════════
# Interactive mode
# ════════════════════════════════════════════════════════════════

def interactive_mode():
    load_resources()
    print(f'\n{"="*78}')
    print(f'  KB Search — Interactive Mode')
    print(f'  Index: {Path(FAISS_INDEX).stat().st_size / 1e9:.2f} GB | {Path(SQLITE_DB).stat().st_size / 1e9:.2f} GB')
    print(f'{"="*78}')
    print('  Commands:')
    print('    <query>            : search (top 10)')
    print('    /n <N>             : set result count')
    print('    /y <min> [<max>]   : filter by year (e.g. /y 2024 2026)')
    print('    /c <category>      : filter by arxiv category (e.g. /c cs.LG)')
    print('    /must <refs.bib>   : exclude papers in this .bib')
    print('    /bib <out.bib>     : export next results to BibTeX')
    print('    /json <out.json>   : export next results to JSON')
    print('    q                  : quit')
    print()

    n = 10
    year_min = None
    year_max = None
    category = None
    source = None
    must_cite = False
    existing_refs = None
    bibtex_out = None
    json_out = None

    while True:
        try:
            line = input('🔍 > ').strip()
        except (KeyboardInterrupt, EOFError):
            print('\nBye!')
            break
        if not line:
            continue
        if line.lower() in ('q', 'quit', 'exit'):
            print('Bye!')
            break

        if line.startswith('/n '):
            n = int(line[3:].strip())
            print(f'  result count: {n}')
            continue
        if line.startswith('/y '):
            parts = line[3:].strip().split()
            year_min = int(parts[0]) if parts else None
            year_max = int(parts[1]) if len(parts) > 1 else None
            print(f'  year: {year_min} - {year_max}')
            continue
        if line.startswith('/c '):
            category = line[3:].strip()
            print(f'  category: {category}')
            continue
        if line.startswith('/must '):
            existing_refs = line[6:].strip().split(',')
            must_cite = True
            print(f'  must-cite: exclude {existing_refs}')
            continue
        if line.startswith('/bib '):
            bibtex_out = line[5:].strip()
            print(f'  next results → BibTeX: {bibtex_out}')
            continue
        if line.startswith('/json '):
            json_out = line[6:].strip()
            print(f'  next results → JSON: {json_out}')
            continue

        # Treat as query
        results = search(
            line,
            n=n,
            year_min=year_min,
            year_max=year_max,
            category=category,
            source=source,
            must_cite=must_cite,
            existing_refs=existing_refs,
            bibtex_out=bibtex_out,
            json_out=json_out,
        )
        # Reset one-shot exports
        bibtex_out = None
        json_out = None


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='kb_search — 论文 KB 搜索 (TMLR 写作专用)')
    parser.add_argument('query', nargs='*', help='搜索 query')
    parser.add_argument('-n', type=int, default=10, help='返回数量')
    parser.add_argument('--year-min', type=int, help='最小年份')
    parser.add_argument('--year-max', type=int, help='最大年份')
    parser.add_argument('--source', help='按 source 过滤 (arxiv / supplement / ...)' )
    parser.add_argument('--category', help='按 arxiv 类别过滤 (cs.LG / cs.CL / ...)')
    parser.add_argument('--must-cite', action='store_true', help='必引发现模式')
    parser.add_argument('--no-smart', action='store_true', help='禁用 query 扩展 + rerank (用 raw 搜索)')
    parser.add_argument('--existing-refs', help='现有 .bib 路径(逗号分隔),必引模式用')
    parser.add_argument('--bibtex', help='导出到 BibTeX 文件')
    parser.add_argument('--json', help='导出到 JSON 文件')
    # 超时配置 (保守默认值)
    parser.add_argument('--timeout', type=float, default=DEFAULT_TOTAL_TIMEOUT,
                        help=f'整个 search 流程总超时 (秒, 默认 {DEFAULT_TOTAL_TIMEOUT})')
    parser.add_argument('--model-timeout', type=float, default=DEFAULT_MODEL_TIMEOUT,
                        help=f'模型加载超时 (秒, 默认 {DEFAULT_MODEL_TIMEOUT})')
    parser.add_argument('--sqlite-busy-timeout', type=float,
                        default=DEFAULT_SQLITE_BUSY_TIMEOUT,
                        help=f'SQLite 锁等待 (秒, 默认 {DEFAULT_SQLITE_BUSY_TIMEOUT})')
    args = parser.parse_args()

    if args.query:
        q = ' '.join(args.query)
        existing_refs = None
        if args.existing_refs:
            existing_refs = [p.strip() for p in args.existing_refs.split(',')]

        search(
            q,
            n=args.n,
            year_min=args.year_min,
            year_max=args.year_max,
            source=args.source,
            category=args.category,
            must_cite=args.must_cite,
            existing_refs=existing_refs,
            bibtex_out=args.bibtex,
            json_out=args.json,
            smart=not args.no_smart,
            total_timeout=args.timeout,
            model_timeout=args.model_timeout,
            sqlite_busy_timeout=args.sqlite_busy_timeout,
        )
    else:
        interactive_mode()


if __name__ == '__main__':
    main()