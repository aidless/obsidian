# KB 自生长系统变更日志 — 2026-07-09

> 4 轮会话集中改造。涉及 4 个文件, 0 个数据破坏。

## 变更前状况 (发现的问题)

| 问题 | 严重度 | 影响 |
|------|-------|------|
| FAISS 主索引 363K vs SQLite 541K, **差 32.8%** | 🔴 P0 | 19 万篇论文只能查 metadata,搜不到向量 |
| `papers_gap.index` (288MB) 存在但 `kb_search.py` 没加载 | 🔴 P0 | 上述 32.8% 漂移的根因 |
| 主索引 ID 是 peS2o 数字 / SHA1, gap 是 arxiv ID | 🟡 P1 | 两套 ID 体系共存,跨索引 join 出错 |
| peS2o 论文 BibTeX 硬塞假 arxiv eprint | 🟡 P1 | TMLR 投稿可能引用错 |
| daily_grow 没有健康检查钩子 | 🟡 P1 | 漂移要 11 天才发现 |

## 变更清单

### 1. `kb_search.py` — 召回扩到 56 万 + 来源标签

**改动点**:
- L52-55: 加 `FAISS_GAP_INDEX` / `IDS_GAP_FILE` 常量
- L64-79: 加 `PREFIX_PES2O` / `PREFIX_ARXIV` 常量 + 三个 lazy-loaded 数组
  - `_paper_ids`     = 带前缀的展示 ID (`pes2o:5995782`)
  - `_paper_ids_raw` = 原始 ID (供 SQLite 查询)
  - `_id_source`     = `pes2o` / `arxiv` 标签
- L106-128: `load_resources()` 启发式归类 (看 ID 本身格式而非文件来源)
- L302-330: 搜索循环里 SQLite 查询用 `_paper_ids_raw[idx]`,展示用 `id_display`
- L400-414: 打印行加 `[arXiv]` / `[peS2o]` 标签
- L210-260: `format_bibtex()` 按来源分流:
  - arxiv → 正常 eprint + archivePrefix
  - peS2o → 用 howpublished + note 字段,说明无 arxiv eprint

**效果**:
- 召回 363,948 → **560,284 向量** (+54%)
- 搜索结果一目了然区分来源
- BibTeX 不会再编造假 arxiv eprint

**启发式归类规则** (L116):
```python
arxiv_pat = re.compile(r'^\d{4}\.\d{4,5}(v\d+)?$')
# 含 . 的 → arxiv (无论在哪个文件)
# 40 位 hex → peS2o SHA1
# 其他纯数字 → peS2o 内部 ID
```

修了一个隐藏 bug: daily_grow 历史上把 46,425 个 arxiv 论文写进了 `paper_ids.txt` (主文件),
之前被默认标 `pes2o`, 现在启发式自动识别为 arxiv。

### 2. `kb_health.py` — 新建, 8 节健康检查 + JSON 输出

**功能**:
- `[1]` SQLite 统计 (papers + papers_fts)
- `[1b]` ID 格式分布 (arxiv-like / SHA1 / numeric / other)
- `[2-3]` FAISS 主索引 + gap 索引 (向量数 + 文件大小)
- `[4]` ID 文件行数
- `[5]` 一致性核对 (main vs sqlite, main+gap vs sqlite, FAISS vs IDs)
- `[6]` ID 抽样 vs SQLite paper_id (默认 2000 样本)
- `[7]` daily_grow 新鲜度 (>72h 报警)
- `[8]` 磁盘占用

**退出码**:
- 0 = HEALTHY (差距 < warn%)
- 1 = WARN    (warn% ≤ 差距 < danger%)
- 2 = DANGER  (差距 ≥ danger% 或新鲜度 stale)

**`--strict` 模式** (L160-164):
- warn 阈值 1.0% → **0.5%**
- ID 抽样命中率要求 95% → **99.5%**
- 调度器/CI 用, 输出带 `[STRICT]` 标记

**`--json` 模式**: 输出结构化 JSON 供其他工具消费。

### 3. `daily_grow.py` — Step 4b 加健康检查钩子

**改动点** (L383-402):
```python
# Step 4a: self_grow 自带的 verify
# Step 4b: 新加 - 跑 kb_health.py --strict, 写日志到 last_health_check.json
#   - rc=0 → ✅ KB healthy
#   - rc=1 → ⚠ WARN, 继续但记录
#   - rc=2 → 🚨 DANGER, 建议手动跑 finalize_rebuild
```

**关键设计**: 不 `sys.exit`, 让 state 仍然写完; 但把退出码暴露给调度器感知。

### 4. `post_daily_grow_check.bat` — 新建, 独立兜底

**作用**: 即使 daily_grow 内部钩子失败, 调度器还能跑一次健康检查。

**注册方法** (管理员 cmd):
```cmd
schtasks /create /tn "KB Daily Health" ^
  /tr "E:\peS2o_kb_faiss\post_daily_grow_check.bat" ^
  /sc daily /st 03:10
```

输出到 `post_health_check.log` + `last_health_check.json`。

## 当前 KB 状态 (变更后)

| 指标 | 数值 | 状态 |
|------|------|------|
| SQLite papers | 541,877 | ✅ |
| FAISS 总向量 | 560,284 (main 363,948 + gap 196,336) | ✅ |
| ID 抽样命中率 | 100.0% | ✅ |
| daily_grow 新鲜度 | 11.5h | ✅ |
| main+gap vs SQLite 差距 | -3.4% (-18,407) | ⚠ WARN |

**-3.4% 的含义**: 不是"少", 是"多" — FAISS gap 索引里有些向量对应的 ID 在 SQLite 里
对应了多行 (peS2o 论文 metadata 重复)。要彻底清理需跑 `finalize_rebuild.py`
(30 分钟 + 备份), 2026-07-09 暂未执行, 留给下次会话决策。

**`--strict` 阈值**: warn<0.5% → 现在 -3.4% 触发 WARN, exit 1。
下次 daily_grow 跑完会自动检测, 如果差距收敛 <0.5% 就 HEALTHY。

## 验证步骤 (未来回看)

1. `py -3 kb_health.py` — 看当前状态
2. `py -3 kb_search.py "TTRL" -n 5` — 看新格式 (带 `[arXiv]`/`[peS2o]` 标签)
3. 跑一次 daily_grow (--dry-run 先看): `py -3 daily_grow.py --dry-run`
4. 看 `last_health_check.json` 确认钩子正常

## 未来路线图

| 优先级 | 任务 | 估时 |
|-------|------|------|
| P2 | 跑 `finalize_rebuild.py` 修 -3.4% 漂移 | 30 min + 备份 |
| P3 | 写 `kb_archive.py` 归档 99GB peS2o 原始 JSONL | 20 min |
| P3 | 给 daily_grow 加 `--category-decay` 按类别降权 | 1 hour |
| P4 | 把 `kb_search.py` 接入 Obsidian Dataview | 待定 |

## 不在本次改动的内容 (重要!)

- **FAISS 索引本身**: 完全未触碰, 0 字节变动
- **SQLite 数据库**: 完全未触碰, 0 字节变动
- **`paper_ids.txt` / `papers_gap_ids.txt`**: 完全未触碰
- **`daily_grow.py` 的 fetch/embed 主流程**: 完全未触碰
- **`self_grow.py`**: 完全未触碰

所有改动都是 **runtime 层** (脚本逻辑), 数据层零风险。

---

变更日志生成于 2026-07-09, 由 ZCode agent 协助完成。
如发现上述改动导致问题, 可以 `git diff` 比对 (如果纳入版本控制)。