#!/usr/bin/env python3
"""kb_server.py — PaperKB HTTP API (供 AI 工具调用)

启动:
    py -3 E:\peS2o_kb_faiss\kb_server.py

API:
    GET  /health                 — 健康检查
    POST /search                 — 语义搜索
    POST /must_cite              — 必引发现 (排除现有引用)
    POST /fetch                  — 拉取指定 arxiv ID 入库
    GET  /stats                  — KB 统计

调用示例:
    # 基本搜索
    curl -X POST http://localhost:8001/search \
        -H "Content-Type: application/json" \
        -d '{"query": "calibration LLM", "n": 5}'

    # 必引发现
    curl -X POST http://localhost:8001/must_cite \
        -H "Content-Type: application/json" \
        -d '{"query": "TTRL", "existing_refs": ["F:/Research/PAPER5_CONSOLIDATED/refs.bib"], "n": 5}'

    # 拉取特定论文
    curl -X POST http://localhost:8001/fetch \
        -H "Content-Type: application/json" \
        -d '{"arxiv_ids": ["2606.28661", "2606.27288"], "bibtex_path": "out.bib"}'

    # 健康检查
    curl http://localhost:8001/health
"""
from __future__ import annotations
import json
import os
import sys
import time
import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Move smart_rerank import below KB_DIR definition
from typing import Optional

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

KB_DIR = Path(r'E:/peS2o_kb_faiss')
FAISS_INDEX = KB_DIR / 'papers.index'
SQLITE_DB = KB_DIR / 'papers.db'
IDS_FILE = KB_DIR / 'paper_ids.txt'
STAGING_DIR = Path(r'E:/peS2o_cs')
MODEL_NAME = 'all-MiniLM-L6-v2'

HOST = '0.0.0.0'
PORT = 8765

sys.path.insert(0, str(KB_DIR))
from smart_rerank import expand_query, detect_category_preference, rerank_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--host', default=HOST)
    return parser.parse_args()


class Resources:
    model = None
    index = None
    paper_ids = None
    conn = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load KB resources on startup."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import sqlite3

    print('[KB] Loading model...')
    Resources.model = SentenceTransformer(MODEL_NAME)
    print('[KB] Loading FAISS index...')
    Resources.index = faiss.read_index(str(FAISS_INDEX))
    print(f'[KB] Index size: {Resources.index.ntotal}')
    print('[KB] Loading paper IDs...')
    with open(IDS_FILE, encoding='utf-8') as f:
        Resources.paper_ids = [line.strip() for line in f if line.strip()]
    print(f'[KB] IDs: {len(Resources.paper_ids)}')
    Resources.conn = sqlite3.connect(str(SQLITE_DB), check_same_thread=False)
    Resources.conn.row_factory = sqlite3.Row
    print(f'[KB] Ready at http://{HOST}:{PORT}')
    yield
    print('[KB] Shutting down...')
    if Resources.conn:
        Resources.conn.close()


app = FastAPI(
    title='PaperKB API',
    version='1.0',
    description='TMLR paper writing knowledge base (363K+ CS papers)',
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ════════════════════════════════════════════════════════════════
# Request/Response models
# ════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str
    n: int = 10
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    category: Optional[str] = None
    dedup: bool = True
    smart: bool = True  # use query expansion + reranking


class MustCiteRequest(BaseModel):
    query: str
    existing_refs: list[str] = Field(default_factory=list)
    n: int = 10
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    category: Optional[str] = None


class FetchRequest(BaseModel):
    arxiv_ids: list[str]
    force: bool = False
    bibtex_path: Optional[str] = None


class SearchResult(BaseModel):
    paper_id: str
    title: str
    authors: str
    year: str
    abstract: str
    categories: str
    score: float


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SearchResult]


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def dedup_by_paper_id(results: list[dict]) -> list[dict]:
    """Remove duplicates by paper_id."""
    seen = set()
    out = []
    for r in results:
        pid = r.get('paper_id', '')
        if pid and pid not in seen:
            seen.add(pid)
            out.append(r)
    return out


def lookup_sqlite(paper_id: str) -> dict:
    """Fetch paper metadata from SQLite by paper_id."""
    cur = Resources.conn.cursor()
    cur.execute(
        'SELECT paper_id, title, authors, year, abstract, categories, source, fields FROM papers WHERE paper_id = ?',
        (paper_id,),
    )
    row = cur.fetchone()
    result = {'paper_id': paper_id, 'title': '', 'authors': '',
              'year': '', 'abstract': '', 'categories': '', 'source': '', 'fields': ''}
    if not row:
        result['title'] = '(not in SQLite)'
        return result
    for key in row.keys():
        val = row[key]
        if val is None:
            val = ''
        result[key] = val
    return result


def apply_filters(results: list[dict], year_min=None, year_max=None, category=None) -> list[dict]:
    out = []
    for r in results:
        y = (r.get('year') or '')[:4]
        if year_min and y and int(y) < year_min:
            continue
        if year_max and y and int(y) > year_max:
            continue
        cats = r.get('categories') or ''
        if category and category not in cats:
            continue
        out.append(r)
    return out


# ════════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════════

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'kb_vectors': int(Resources.index.ntotal) if Resources.index else 0,
        'paper_ids': len(Resources.paper_ids) if Resources.paper_ids else 0,
        'model': MODEL_NAME,
    }


@app.get('/stats')
def stats():
    if Resources.conn is None:
        return {'error': 'db not initialized', 'model': MODEL_NAME,
                'kb_vectors': int(Resources.index.ntotal) if Resources.index else 0}
    try:
        cur = Resources.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM papers')
        sqlite_n = cur.fetchone()[0]
        return {
            'faiss_vectors': int(Resources.index.ntotal),
            'paper_ids_unique': len(Resources.paper_ids),
            'sqlite_papers': sqlite_n,
            'faiss_size_gb': round(FAISS_INDEX.stat().st_size / 1e9, 3),
            'db_size_gb': round(SQLITE_DB.stat().st_size / 1e9, 3),
            'model': MODEL_NAME,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'type': type(e).__name__}


@app.post('/search', response_model=SearchResponse)
def search(req: SearchRequest):
    """Semantic search over the KB."""
    import numpy as np
    import traceback
    if not Resources.index or not Resources.model:
        raise HTTPException(503, 'KB not loaded')
    if Resources.conn is None:
        raise HTTPException(503, 'SQLite not connected')

    debug_log = KB_DIR / 'search_debug.log'
    try:
        # Query expansion (only if smart=True)
        actual_query = req.query
        cat_hint = []
        if req.smart:
            actual_query, _rules = expand_query(req.query)
            cat_hint = detect_category_preference(req.query)

        q_emb = Resources.model.encode([actual_query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        fetch_n = req.n * 15 if req.smart else req.n * 3
        fetch_n = min(fetch_n, 800)

        scores, idxs = Resources.index.search(q_emb, fetch_n)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(Resources.paper_ids):
                continue
            pid = Resources.paper_ids[idx]
            meta = lookup_sqlite(pid)
            base = float(score)

            if req.smart:
                final_score, _ = rerank_score(meta, base, cat_hint)
            else:
                final_score = base

            meta['score'] = round(final_score, 4)
            meta['base_score'] = round(base, 4)
            results.append(meta)

        if req.smart:
            if cat_hint:
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
            results.sort(key=lambda r: r.get('score', 0), reverse=True)

        if req.dedup:
            results = dedup_by_paper_id(results)
        results = apply_filters(results, req.year_min, req.year_max, req.category)
        results = results[:req.n]

        return SearchResponse(query=req.query, total=len(results),
                             results=[SearchResult(**r) for r in results])
    except HTTPException:
        raise
    except Exception as e:
        with open(debug_log, 'a', encoding='utf-8') as f:
            f.write(f'\n=== {req.query} ===\n')
            traceback.print_exc(file=f)
            f.write(f'{type(e).__name__}: {e}\n')
        raise HTTPException(500, f'{type(e).__name__}: {e}')


@app.post('/must_cite', response_model=SearchResponse)
def must_cite(req: MustCiteRequest):
    """Find papers in KB similar to query but NOT in existing refs."""
    import re
    existing_ids = set()
    existing_titles = set()
    for path in req.existing_refs:
        p = Path(path)
        if not p.exists():
            continue
        try:
            content = p.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        # Extract arXiv IDs
        for m in re.finditer(r'(\d{4}\.\d{4,5})(?:v\d+)?', content):
            existing_ids.add(m.group(1))
        # Extract titles
        for m in re.finditer(r'title\s*=\s*\{([^}]+)\}', content, re.IGNORECASE):
            existing_titles.add(m.group(1).strip().lower()[:80])

    # Reuse search
    results = search(SearchRequest(
        query=req.query, n=req.n * 5,
        year_min=req.year_min, year_max=req.year_max,
        category=req.category, dedup=True,
    ))

    # Filter out already-cited
    filtered = []
    for r in results.results:
        pid = r.paper_id.split('v')[0]
        if pid in existing_ids:
            continue
        title_lower = (r.title or '').lower()[:80]
        if title_lower in existing_titles:
            continue
        filtered.append(r)

    return SearchResponse(query=req.query, total=len(filtered[:req.n]),
                         results=filtered[:req.n])


@app.post('/fetch')
def fetch(req: FetchRequest):
    """Fetch specific arxiv papers and add to KB."""
    import subprocess
    cmd = [
        sys.executable, str(KB_DIR / 'fetch_specific.py'),
        *req.arxiv_ids,
    ]
    if req.force:
        cmd.append('--force')
    if req.bibtex_path:
        cmd.extend(['--bibtex', req.bibtex_path])

    result = subprocess.run(cmd, cwd=str(KB_DIR), capture_output=True, text=True, timeout=300)
    return {
        'cmd': ' '.join(cmd),
        'returncode': result.returncode,
        'stdout_tail': result.stdout[-1000:] if result.stdout else '',
        'stderr_tail': result.stderr[-1000:] if result.stderr else '',
    }


if __name__ == '__main__':
    args = parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)