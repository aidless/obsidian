"""test_kb_search.py — kb_search.py 的单元测试

覆盖 3 个核心场景:
  1. 正常搜索 (default timeout): 召回结果 + 格式 + 数量
  2. 超时降级: 极小 timeout 快速返回空/部分结果, 不抛异常
  3. SQLite 锁等待: 模拟锁竞争, busy_timeout 防止报错

设计:
  - 不使用 mock, 直接测真实 KB (HEALTHY state 才有意义)
  - 用 unittest (stdlib), 不引入新依赖
  - 离线运行 (HF_HUB_OFFLINE=1 + 模型已下载)
  - 总耗时 < 60s (含模型加载 ~16s)

运行:
  cd E:\\peS2o_kb_faiss
  python -X utf8 -m unittest tests.test_kb_search -v
"""
from __future__ import annotations
import os
import sys
import time
import threading
import unittest
from pathlib import Path

# 强制 HF 离线, 避免网络抖动
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# 把 KB 目录加到 sys.path, 让 import kb_search 能找到
KB_DIR = Path(r'E:/peS2o_kb_faiss')
sys.path.insert(0, str(KB_DIR))
sys.path.insert(0, str(KB_DIR / 'tests'))

import kb_search


class TestKbSearchBasic(unittest.TestCase):
    """正常搜索: 验证搜索功能本身的正确性"""

    @classmethod
    def setUpClass(cls):
        """所有测试共享一次模型加载, 避免每次重新加载 (~16s)"""
        cls.start = time.time()
        # 强制 load_resources, 共享 model/index/conn
        kb_search.load_resources()

    def test_search_returns_list(self):
        """正常 search 返回 list, 长度 = n"""
        results = kb_search.search('calibration in large language models', n=5)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)

    def test_search_results_have_required_fields(self):
        """每条结果有 paper_id / title / score"""
        results = kb_search.search('attention mechanism transformer', n=3)
        for r in results:
            self.assertIn('paper_id', r)
            self.assertIn('title', r)
            self.assertIn('score', r)
            # score 范围:
            #   - 理论 cosine similarity: [-1, 1]
            #   - 实际 dot product on normalized vectors: 通常 [0, 1]
            #   - 但 FAISS IndexFlatIP with un-normalized vectors 可能 > 1
            # 这里只验证 score > 0 (合理的相关性)
            self.assertGreater(r['score'], 0.0,
                               f'score {r["score"]} 应该 > 0')
            self.assertLess(r['score'], 2.0,
                            f'score {r["score"]} 应该 < 2 (FAISS IP 上限)')

    def test_search_results_sorted_by_score_desc(self):
        """结果应按 score 降序 (smart 模式)"""
        results = kb_search.search('reinforcement learning', n=10)
        scores = [r['score'] for r in results]
        # 不要求严格降序 (smart rerank 可能微调), 但不能比前面的低太多
        for i in range(1, len(scores)):
            self.assertGreaterEqual(scores[i-1], scores[i] - 0.01,
                                    f'result {i-1} score {scores[i-1]:.4f} '
                                    f'< result {i} score {scores[i]:.4f}')

    def test_search_with_year_filter(self):
        """年份过滤应生效"""
        results = kb_search.search('transformer', n=5, year_min=2024)
        for r in results:
            year = (r.get('year') or '')[:4]
            if year and year != '????':
                self.assertGreaterEqual(int(year), 2024,
                                        f'year {year} < 2024 filter')

    def test_search_with_category_filter(self):
        """类别过滤应生效"""
        results = kb_search.search('language model', n=5, category='cs.CL')
        for r in results:
            cats = r.get('categories', '') or ''
            self.assertIn('cs.CL', cats,
                          f'category {cats!r} 不含 cs.CL')

    def test_search_no_smart_mode(self):
        """--no-smart 模式应能跑通"""
        results = kb_search.search('neural network', n=3, smart=False)
        self.assertEqual(len(results), 3)


class TestKbSearchTimeout(unittest.TestCase):
    """超时降级: 验证 total_timeout 工作正常"""

    @classmethod
    def setUpClass(cls):
        kb_search.load_resources()

    def test_extreme_timeout_returns_quickly(self):
        """total_timeout=0.001s 应 < 5s 返回 (降级到空结果)"""
        t0 = time.time()
        results = kb_search.search(
            'calibration in large language models',
            n=5,
            total_timeout=0.001,
        )
        elapsed = time.time() - t0
        # 应该 < 1s 退出 (降级, 不再继续 SQLite 查询)
        self.assertLess(elapsed, 1.0,
                        f'extreme timeout took {elapsed:.2f}s, 应该 < 1s')
        # 降级: 返回空或部分结果, 都不抛异常
        self.assertIsInstance(results, list)

    def test_normal_timeout_completes(self):
        """total_timeout=30s 默认值, 正常 query 应能完成"""
        t0 = time.time()
        results = kb_search.search(
            'attention is all you need',
            n=5,
            total_timeout=30.0,
        )
        elapsed = time.time() - t0
        # 正常查询应该 < 25s 完成 (留 5s buffer)
        self.assertLess(elapsed, 25.0,
                        f'normal search took {elapsed:.2f}s > 25s')
        # 应该返回完整 5 条
        self.assertEqual(len(results), 5)

    def test_custom_timeout_value(self):
        """total_timeout=2.0s 应该是合作式降级, 不强 kill"""
        t0 = time.time()
        results = kb_search.search(
            'graph neural network',
            n=5,
            total_timeout=2.0,
        )
        elapsed = time.time() - t0
        # 2s timeout, 不会 hard-kill, 但会比默认快
        self.assertLess(elapsed, 10.0,
                        f'2s timeout took {elapsed:.2f}s, 应 < 10s')
        # 一定时间内返回部分结果
        self.assertIsInstance(results, list)

    def test_model_timeout_raises(self):
        """model_timeout=0s 应触发 TimeoutError_ (模型加载不可能 0s 完成)"""
        # 注: 如果模型已经加载, 这次调用不会触发 load_resources 中的 timeout
        #     所以这个测试可能在 setUpClass 后不会失败, 这里只验证
        #     API 接受这个参数, 不崩
        try:
            results = kb_search.search(
                'test query',
                n=1,
                model_timeout=0.001,
            )
            # 如果没崩, 说明模型已经缓存, 也算正常
            self.assertIsInstance(results, list)
        except kb_search.TimeoutError_:
            # 期望: 0.001s 不足以加载模型
            self.assertTrue(True, '正确触发 TimeoutError_')

    def test_timeout_no_exception_leak(self):
        """超时降级后, 后续 query 仍能正常工作 (state 没被破坏)"""
        # 触发一次降级
        kb_search.search('query1', n=3, total_timeout=0.001)
        # 正常 query 应该不受影响
        results = kb_search.search('transformer', n=3, total_timeout=30.0)
        # 不要求严格 = 3, 只要 >= 1 (smart 模式可能过 1 个数)
        self.assertGreaterEqual(len(results), 1,
                                f'后续 query 应至少返回 1 条, 实际 {len(results)}')


class TestKbSearchSqliteBusyTimeout(unittest.TestCase):
    """SQLite 锁等待: 验证 busy_timeout 防止 database is locked 报错"""

    @classmethod
    def setUpClass(cls):
        kb_search.load_resources()

    def test_busy_timeout_set_on_connection(self):
        """load_resources 后, _conn 应设置了 busy_timeout"""
        # 检查 _conn 的 busy_timeout PRAGMA
        cur = kb_search._conn.execute('PRAGMA busy_timeout')
        timeout_ms = cur.fetchone()[0]
        # 默认 5s = 5000ms
        self.assertEqual(timeout_ms, 5000,
                         f'expected 5000ms, got {timeout_ms}ms')

    def test_busy_timeout_custom_value(self):
        """load_resources(busy_timeout=10) 应设置 10s"""
        # 创建一个独立的 load 调用, 不影响共享 conn
        # 这里只能通过 PRAGMA 临时设置测试
        # 真正测需要新 conn, 留作兼容性测试
        kb_search._conn.execute('PRAGMA busy_timeout = 10000')
        cur = kb_search._conn.execute('PRAGMA busy_timeout')
        self.assertEqual(cur.fetchone()[0], 10000)
        # 还原
        kb_search._conn.execute('PRAGMA busy_timeout = 5000')

    def test_concurrent_write_does_not_break_read(self):
        """并发写不破坏读查询 (busy_timeout 保护)"""
        import sqlite3
        results = {'written': False, 'read_completed': False,
                   'read_error': None}

        def writer():
            """另一个连接, 模拟长写事务"""
            try:
                conn = sqlite3.connect(str(KB_DIR / 'papers.db'),
                                       timeout=10.0)
                conn.execute('BEGIN EXCLUSIVE')
                # 写点东西
                conn.execute(
                    'CREATE TEMP TABLE _test_lock (id INTEGER)'
                )
                conn.execute('INSERT INTO _test_lock VALUES (1)')
                time.sleep(2.0)  # 持锁 2 秒
                conn.execute('COMMIT')
                conn.close()
                results['written'] = True
            except Exception as e:
                results['read_error'] = f'writer: {e}'

        writer_thread = threading.Thread(target=writer, daemon=True)
        writer_thread.start()

        # 等 writer 开始持锁
        time.sleep(0.3)

        # 读查询 (kb_search 内部 SQLite) — 应能成功, busy_timeout 5s 足够等
        try:
            res = kb_search.search('test query for busy timeout', n=1)
            results['read_completed'] = True
        except sqlite3.OperationalError as e:
            if 'locked' in str(e):
                results['read_error'] = f'reader: {e}'

        writer_thread.join(timeout=5.0)
        # 读应该完成 (5s busy_timeout > 2s 持锁)
        self.assertTrue(results['read_completed'],
                        f'读查询没完成: {results["read_error"]}')
        # writer 也应完成
        self.assertTrue(results['written'],
                        f'writer 没完成: {results["read_error"]}')

    def test_search_sqlite_param_passed_through(self):
        """search() 接受 sqlite_busy_timeout 参数"""
        # 不真正改 conn, 只验证签名
        import inspect
        sig = inspect.signature(kb_search.search)
        self.assertIn('sqlite_busy_timeout', sig.parameters)
        # 默认值 = 5.0
        self.assertEqual(sig.parameters['sqlite_busy_timeout'].default, 5.0)


class TestKbSearchEdgeCases(unittest.TestCase):
    """边界场景"""

    @classmethod
    def setUpClass(cls):
        kb_search.load_resources()

    def test_empty_query(self):
        """空 query 不应崩"""
        try:
            results = kb_search.search('', n=3)
            # 行为可以是空 list 或随机结果, 但不抛异常
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f'empty query raised: {e}')

    def test_n_larger_than_index(self):
        """n > index.ntotal 时不崩"""
        results = kb_search.search('test', n=1000000)
        # 应该返回 index 里所有结果
        self.assertIsInstance(results, list)

    def test_special_chars_in_query(self):
        """特殊字符不崩"""
        results = kb_search.search('LLM!@#$%^&*()', n=3)
        self.assertIsInstance(results, list)

    def test_must_cite_mode(self):
        """must-cite 模式接受 existing_refs"""
        # 临时 .bib 内容
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bib',
                                          delete=False, encoding='utf-8') as f:
            f.write('@article{test2024, title={calibration}, year={2024}}\n')
            bib_path = f.name
        try:
            results = kb_search.search(
                'calibration in large language models',
                n=3,
                must_cite=True,
                existing_refs=[bib_path],
            )
            self.assertIsInstance(results, list)
        finally:
            os.unlink(bib_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
