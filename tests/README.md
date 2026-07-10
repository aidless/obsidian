# peS2o KB Tests

## 两种测试套件

### 1. Unit Tests (CI 跑)
**文件**: `test_kb_search_unit.py`
**数据**: Mock (100 个 papers, 内存 sqlite, 假 sentence-transformers)
**特点**: 快速 (~10s), 无外部依赖
**运行**:
```bash
python -X utf8 -m unittest tests.test_kb_search_unit -v
```

### 2. Integration Tests (本地跑)
**文件**: `test_kb_search.py`
**数据**: 真实 KB (8GB, 54万 papers, FAISS 索引)
**特点**: 慢 (~13s 首次 + 模型加载), 需要 KB
**运行**:
```bash
python -X utf8 -m unittest tests.test_kb_search -v
```

## 辅助文件

- `_mocks.py` — Mock 数据生成器 (100 个 papers + 假 sentence-transformers)

## CI 工作流

`.github/workflows/ci.yml`:
- **触发**: PR + push main + 手动
- **平台**: ubuntu-latest
- **运行**: `test_kb_search_unit.py` (mock, ~10s)
- **跳过**: `test_kb_search.py` (需要真实 KB)
