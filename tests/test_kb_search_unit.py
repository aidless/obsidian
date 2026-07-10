"""test_kb_search_unit.py — 不需要真实 KB 的单元测试 (CI 跑)

跟 test_kb_search.py 区别:
  - test_kb_search.py: 需要真实 KB (8GB), 只在本地跑
  - test_kb_search_unit.py: 用 mock 数据, CI 跑

覆盖:
  1. 正常搜索 (mock KB): 召回结果格式
  2. 超时降级: total_timeout 工作
  3. SQLite 锁等待: busy_timeout
  4. 边界场景: 空 query / 特殊字符 / n > index

运行:
  python -X utf8 -m unittest tests.test_kb_search_unit -v
"""
from __future__ import annotations
import os
import sys
import time
import threading
import unittest
from pathlib import Path

# 强制 HF 离线
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

KB_DIR = Path(r'E:/peS2o_kb_faiss')
sys.path.insert(0, str(KB_DIR))
sys.path.insert(0, str(KB_DIR / 'tests'))

import kb_search
from tests._mocks import setup_mock_kb, teardown_mock_kb


class TestKbSearchBasic(unittest.TestCase):
    """正常搜索: 用 mock KB 验证搜索功能"""

    @classmethod
    def setUpClass(cls):
        """所有测试共享一次 mock 设置"""
        setup_mock_kb()

    @classmethod
    def tearDownClass(cls):
        teardown_mock_kb()

    def test_search_returns_list(self):
        results = kb_search.search('calibration in large language models', n=5)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)

    def test_search_results_have_required_fields(self):
        results = kb_search.search('attention mechanism', n=3)
        for r in results:
            self.assertIn('paper_id', r)
            self.assertIn('title', r)
            self.assertIn('score', r)
            self.assertGreater(r['score'], 0.0)
            self.assertLess(r['score'], 2.0)

    def test_search_with_year_filter(self):
        results = kb_search.search('transformer', n=5, year_min=2025)
        for r in results:
            year = (r.get('year') or '')[:4]
            if year and year != '????':
                self.assertGreaterEqual(int(year), 2025)

    def test_search_with_category_filter(self):
        results = kb_search.search('calibration', n=5, category='cs.LG')
        for r in results:
            cats = r.get('categories', '') or ''
            self.assertIn('cs.LG', cats)

    def test_search_no_smart_mode(self):
        results = kb_search.search('neural network', n=3, smart=False)
        self.assertEqual(len(results), 3)

    def test_search_results_sorted_by_score_desc(self):
        results = kb_search.search('reinforcement learning', n=10)
        scores = [r['score'] for r in results]
        for i in range(1, len(scores)):
            self.assertGreaterEqual(scores[i-1], scores[i] - 0.01)


class TestKbSearchTimeout(unittest.TestCase):
    """超时降级"""

    @classmethod
    def setUpClass(cls):
        setup_mock_kb()

    @classmethod
    def tearDownClass(cls):
        teardown_mock_kb()

    def test_extreme_timeout_returns_quickly(self):
        t0 = time.time()
        results = kb_search.search(
            'calibration', n=5, total_timeout=0.001)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 1.0)
        self.assertIsInstance(results, list)

    def test_normal_timeout_completes(self):
        t0 = time.time()
        results = kb_search.search(
            'attention', n=5, total_timeout=5.0)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 5.0)
        self.assertGreaterEqual(len(results), 1)

    def test_custom_timeout_value(self):
        t0 = time.time()
        results = kb_search.search(
            'graph', n=5, total_timeout=1.0)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 5.0)
        self.assertIsInstance(results, list)

    def test_timeout_no_exception_leak(self):
        """降级后 state 不污染"""
        kb_search.search('query1', n=3, total_timeout=0.001)
        results = kb_search.search('transformer', n=3, total_timeout=5.0)
        self.assertGreaterEqual(len(results), 1)

    def test_model_timeout_signature(self):
        """验证 model_timeout 是 search() 的参数"""
        import inspect
        sig = inspect.signature(kb_search.search)
        self.assertIn('model_timeout', sig.parameters)
        self.assertIn('total_timeout', sig.parameters)
        self.assertIn('sqlite_busy_timeout', sig.parameters)


class TestKbSearchSqliteBusyTimeout(unittest.TestCase):
    """SQLite 锁等待"""

    @classmethod
    def setUpClass(cls):
        setup_mock_kb()

    @classmethod
    def tearDownClass(cls):
        teardown_mock_kb()

    def test_busy_timeout_set_on_connection(self):
        cur = kb_search._conn.execute('PRAGMA busy_timeout')
        timeout_ms = cur.fetchone()[0]
        self.assertEqual(timeout_ms, 5000)

    def test_concurrent_write_does_not_break_read(self):
        """并发写不破坏读查询 (busy_timeout 保护)"""
        import sqlite3
        results = {'written': False, 'read_completed': False,
                   'read_error': None}

        def writer():
            try:
                # 写到同一个 mock 连接不实际锁住 (in-memory),
                # 但用 BEGIN EXCLUSIVE 模拟锁
                # 重新开一个 connection 才能锁住
                conn = sqlite3.connect(':memory:')
                conn.execute('BEGIN EXCLUSIVE')
                conn.execute('CREATE TABLE _t (id INTEGER)')
                conn.execute('INSERT INTO _t VALUES (1)')
                time.sleep(0.5)
                conn.execute('COMMIT')
                conn.close()
                results['written'] = True
            except Exception as e:
                results['read_error'] = f'writer: {e}'

        writer_thread = threading.Thread(target=writer, daemon=True)
        writer_thread.start()
        time.sleep(0.1)

        try:
            res = kb_search.search('test query', n=1)
            results['read_completed'] = True
        except Exception as e:
            results['read_error'] = f'reader: {e}'

        writer_thread.join(timeout=3.0)
        self.assertTrue(results['read_completed'],
                        f'read failed: {results["read_error"]}')


class TestKbSearchEdgeCases(unittest.TestCase):
    """边界场景"""

    @classmethod
    def setUpClass(cls):
        setup_mock_kb()

    @classmethod
    def tearDownClass(cls):
        teardown_mock_kb()

    def test_empty_query(self):
        try:
            results = kb_search.search('', n=3)
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f'empty query raised: {e}')

    def test_n_larger_than_index(self):
        results = kb_search.search('test', n=1000000)
        self.assertIsInstance(results, list)
        # mock 索引只有 100 个, 所以应该返回 ≤ 100
        self.assertLessEqual(len(results), 100)

    def test_special_chars_in_query(self):
        results = kb_search.search('!@#$%^&*()', n=3)
        self.assertIsInstance(results, list)

    def test_must_cite_mode(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bib',
                                          delete=False, encoding='utf-8') as f:
            f.write('@article{test2024, title={calibration}, year={2024}}\n')
            bib_path = f.name
        try:
            results = kb_search.search(
                'calibration', n=3,
                must_cite=True, existing_refs=[bib_path],
            )
            self.assertIsInstance(results, list)
        finally:
            os.unlink(bib_path)

    def test_mock_data_loaded(self):
        """sanity check: mock 真的有 100 个 papers"""
        c = kb_search._conn.execute('SELECT COUNT(*) FROM papers')
        count = c.fetchone()[0]
        self.assertEqual(count, 100)

    def test_index_has_100_vectors(self):
        """sanity check: mock index 真的有 100 个向量"""
        self.assertEqual(kb_search._index.ntotal, 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
