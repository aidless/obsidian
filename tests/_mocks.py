"""_mocks.py — 为 CI 生成 mock 的 KB 资源 (不需要真实数据)

设计:
  - 100 个 mock papers (10 个 arxiv + 90 个 peS2o 数字 ID)
  - 100 个 mock 384 维向量 (IndexFlatIP 索引)
  - mock SentenceTransformer 返回固定长度的 query 向量
  - 内存 sqlite + 100 行 papers 表

用法:
  from tests._mocks import setup_mock_kb
  setup_mock_kb()  # 注入到 kb_search 全局变量
  # 之后 kb_search.search() 跑的就是 mock 数据
"""
from __future__ import annotations
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import List

import numpy as np

KB_DIR = Path(r'E:/peS2o_kb_faiss')
MOCK_DIM = 384
MOCK_N = 100

# Mock 数据生成
MOCK_PAPERS = []  # list of (paper_id, title, year, cats, source, abstract)
for i in range(MOCK_N):
    if i < 10:
        # arxiv-style
        pid = f'25{10 + i // 10}.{10000 + i:05d}'
        title = f'Mock arxiv paper #{i}: '
        if i % 2 == 0:
            title += 'Calibration in Large Language Models'
        else:
            title += f'Attention Mechanism Variant {i}'
        year = str(2024 + (i % 3))
        cats = 'cs.LG,cs.CL' if i % 2 == 0 else 'cs.CV'
        source = 'arxiv'
    else:
        # peS2o-style (8位数字)
        pid = f'{10000000 + i:08d}'
        title = f'Mock peS2o paper #{i}: '
        if i % 3 == 0:
            title += 'Reinforcement Learning Survey'
        elif i % 3 == 1:
            title += 'Graph Neural Network Design'
        else:
            title += 'Transformer Architecture Analysis'
        year = str(2018 + (i % 7))
        cats = ''
        source = 'pes2o'
    abstract = f'Abstract for {title}. ' * 5
    MOCK_PAPERS.append((pid, title, year, cats, source, abstract))


def _build_mock_paper_ids() -> tuple[list, list, list]:
    """返回 (paper_ids_with_prefix, paper_ids_raw, id_source)"""
    ids_with_prefix = []
    ids_raw = []
    id_source = []
    for pid, _, _, _, source, _ in MOCK_PAPERS:
        prefix = 'arxiv:' if source == 'arxiv' else 'pes2o:'
        ids_with_prefix.append(f'{prefix}{pid}')
        ids_raw.append(pid)
        id_source.append(source)
    return ids_with_prefix, ids_raw, id_source


def _build_mock_index():
    """返回 IndexFlatIP 包含 100 个 mock 向量"""
    import faiss
    rng = np.random.default_rng(seed=42)
    vecs = rng.standard_normal((MOCK_N, MOCK_DIM)).astype('float32')
    faiss.normalize_L2(vecs)  # L2 归一化, 使 IP = cosine
    idx = faiss.IndexFlatIP(MOCK_DIM)
    idx.add(vecs)
    return idx


class MockSentenceTransformer:
    """mock SentenceTransformer: encode() 返回固定长度的随机向量

    关键: encode(query) 返回的向量跟 _build_mock_index 里的向量一致空间
    (即, 对应 query 关键词的 paper 应该排前)
    """

    def __init__(self, model_name: str = 'mock'):
        self.model_name = model_name
        self.rng = np.random.default_rng(seed=123)

    def encode(self, sentences, batch_size: int = 32,
                show_progress_bar: bool = False, **kwargs):
        """返回 shape (n_sentences, MOCK_DIM) 的 float32 数组

        策略: 根据 query 关键词挑出"相关"向量子集, 让搜索有意义
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        n = len(sentences)
        out = np.zeros((n, MOCK_DIM), dtype='float32')

        for i, q in enumerate(sentences):
            q_lower = q.lower()
            # 找匹配的 paper index
            matching = []
            for j, (_, title, _, _, _, _) in enumerate(MOCK_PAPERS):
                # 简单匹配: query 中的关键词在 title 中
                title_lower = title.lower()
                for kw in q_lower.split():
                    if len(kw) > 3 and kw in title_lower:
                        matching.append(j)
                        break
            if not matching:
                matching = [0]  # fallback: 用第 0 个

            # 构造 query 向量: 大部分是平均 + 加点噪声, 但偏向 matching
            base = self.rng.standard_normal(MOCK_DIM).astype('float32')
            base /= (np.linalg.norm(base) + 1e-9)
            # 把 base 投影偏向 matching paper 的真实向量
            import faiss
            tmp_idx = _build_mock_index()
            matching_vecs = np.array(
                [tmp_idx.reconstruct(m) for m in matching[:3]],
                dtype='float32',
            )
            if len(matching_vecs) > 0:
                avg_matching = matching_vecs.mean(axis=0)
                avg_matching /= (np.linalg.norm(avg_matching) + 1e-9)
                # 70% matching + 30% random
                out[i] = 0.7 * avg_matching + 0.3 * base
                out[i] /= (np.linalg.norm(out[i]) + 1e-9)
            else:
                out[i] = base
        return out


def _build_mock_sqlite() -> sqlite3.Connection:
    """返回内存 sqlite + 100 行 mock papers"""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY,
            paper_id TEXT UNIQUE,
            title TEXT,
            authors TEXT,
            year TEXT,
            categories TEXT,
            source TEXT,
            text_prefix TEXT,
            abstract TEXT,
            created TEXT,
            version TEXT,
            fields TEXT
        )
    ''')
    for i, (pid, title, year, cats, source, abstract) in enumerate(MOCK_PAPERS):
        c.execute(
            '''INSERT INTO papers
            (id, paper_id, title, authors, year, categories, source,
             text_prefix, abstract, created, version, fields)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (i + 1, pid, title, '[]', year, cats, source,
             title + ' ' + abstract, abstract, '2024-01-01', '', '[]'),
        )
    conn.commit()
    conn.execute('PRAGMA busy_timeout = 5000')
    return conn


def setup_mock_kb():
    """注入 mock 资源到 kb_search 全局变量

    Returns: (model, index, paper_ids, conn) — 跟 load_resources 同样的接口
    """
    import kb_search

    kb_search._model = MockSentenceTransformer('mock-model')
    kb_search._index = _build_mock_index()
    ids_with_prefix, ids_raw, id_source = _build_mock_paper_ids()
    kb_search._paper_ids = ids_with_prefix
    kb_search._paper_ids_raw = ids_raw
    kb_search._id_source = id_source
    kb_search._conn = _build_mock_sqlite()

    return (kb_search._model, kb_search._index,
            kb_search._paper_ids, kb_search._conn)


def teardown_mock_kb():
    """清空 kb_search 全局变量"""
    import kb_search
    kb_search._model = None
    kb_search._index = None
    kb_search._paper_ids = None
    kb_search._paper_ids_raw = None
    kb_search._id_source = None
    kb_search._conn = None
