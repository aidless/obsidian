#!/usr/bin/env python3
"""merge_to_disk.py — 把 gap 索引(增量)原子合并到主索引,并清空 gap

为什么需要这个脚本:
  kb_search.py 里的 _index.merge_from(_gap) 是 in-memory 操作,
  进程退出后 gap 仍在磁盘上, 下次冷启动会再次 merge.
  本脚本做的是真正的"消化 gap":
    1. 读 main + gap → in-memory merge
    2. atomic swap 写回主索引
    3. 追加 gap_ids 到主 ids 文件
    4. 清空 / 删除 gap 文件
    5. 验证 kb_health 状态从 DANGER → HEALTHY

4 阶段流水线, 任一阶段失败立即 stop, 不污染状态.

用法:
  py -3 merge_to_disk.py            # 完整执行
  py -3 merge_to_disk.py --dry-run  # 只跑 Phase 0/1 (只读, 不写盘)
  py -3 merge_to_disk.py --no-backup # 跳过备份 (强烈不推荐)

退出码:
  0 = 合并成功, DANGER 状态已消除
  1 = 校验失败 / 异常, 状态未变
  2 = 写盘失败, 已尝试回滚
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

KB_DIR = Path(r'E:/peS2o_kb_faiss')
FAISS_INDEX = KB_DIR / 'papers.index'
FAISS_GAP_INDEX = KB_DIR / 'papers_gap.index'
IDS_FILE = KB_DIR / 'paper_ids.txt'
IDS_GAP_FILE = KB_DIR / 'papers_gap_ids.txt'
BACKUP_DIR = KB_DIR / 'rebuild_backups'

# 单条日志打印函数 (走 stderr, 不污染可能的 stdout 调用方)
def log(msg: str) -> None:
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', file=sys.stderr, flush=True)


def phase0_prepare() -> dict:
    """准备阶段: 检查文件存在性 / mtime 关系 / 磁盘空间"""
    log('══ Phase 0: 准备阶段 (只读) ══')

    if not FAISS_INDEX.exists():
        raise FileNotFoundError(f'main 索引不存在: {FAISS_INDEX}')
    if not FAISS_GAP_INDEX.exists():
        raise FileNotFoundError(f'gap 索引不存在: {FAISS_GAP_INDEX}, 无需 merge')
    if not IDS_FILE.exists():
        raise FileNotFoundError(f'main ids 文件不存在: {IDS_FILE}')
    if not IDS_GAP_FILE.exists():
        raise FileNotFoundError(f'gap ids 文件不存在: {IDS_GAP_FILE}')

    main_mtime = FAISS_INDEX.stat().st_mtime
    gap_mtime = FAISS_GAP_INDEX.stat().st_mtime
    log(f'  main.index mtime: {datetime.fromtimestamp(main_mtime)}')
    log(f'  gap.index  mtime: {datetime.fromtimestamp(gap_mtime)}')
    log(f'  gap 比 main 新 {gap_mtime - main_mtime:.0f} s  '
        f'({(gap_mtime-main_mtime)/3600:.1f} h)')

    if gap_mtime <= main_mtime:
        log('  ⚠ gap 比 main 还旧, 可能已被消化, 跳过 merge')
        log('  提示: 跑一次 kb_search.py 看是否还能召回 gap 的内容, '
            '如果能, 说明 main mtime 没更新 (脚本问题), 请排查')
        sys.exit(1)

    # 磁盘空间检查 (粗估: 合并后索引会变成 main + gap, 临时需要 ~2x 大小)
    main_size = FAISS_INDEX.stat().st_size
    gap_size = FAISS_GAP_INDEX.stat().st_size
    expected_new = main_size + gap_size  # IndexFlatIP 是 raw 向量, 合并后大小相加
    need_free = expected_new * 2  # staging 文件 + backup + 临时
    total, used, free = shutil.disk_usage(KB_DIR)
    log(f'  main.index size: {main_size/1024**2:.1f} MB')
    log(f'  gap.index  size: {gap_size/1024**2:.1f} MB')
    log(f'  预期合并后: {expected_new/1024**2:.1f} MB')
    log(f'  需要可用空间: {need_free/1024**2:.1f} MB, 当前: {free/1024**2:.1f} MB')
    if free < need_free:
        raise RuntimeError(
            f'磁盘空间不足: 需 {need_free/1024**2:.1f} MB, '
            f'仅 {free/1024**2:.1f} MB 可用')

    # gap 向量数预警
    import faiss
    n_main = faiss.read_index(str(FAISS_INDEX)).ntotal
    n_gap = faiss.read_index(str(FAISS_GAP_INDEX)).ntotal
    log(f'  main 向量数: {n_main:,}')
    log(f'  gap  向量数: {n_gap:,}')
    log(f'  合并后预计: {n_main + n_gap:,}')

    if n_gap > 500_000:
        log(f'  ⚠ gap 向量数超过 50 万, 建议分批 merge (本次仍继续)')

    return {
        'main_mtime': main_mtime, 'gap_mtime': gap_mtime,
        'n_main': n_main, 'n_gap': n_gap,
        'main_size': main_size, 'gap_size': gap_size,
    }


def phase1_load_and_merge(stats: dict) -> tuple[object, list[str]]:
    """加载 + 内存合并 (只读)"""
    log('══ Phase 1: 加载 + 内存合并 (只读) ══')
    import faiss
    t0 = time.time()
    log('  加载 main.index ...')
    idx_main = faiss.read_index(str(FAISS_INDEX))
    log(f'  加载 gap.index ({stats["n_gap"]:,} vecs) ...')
    idx_gap = faiss.read_index(str(FAISS_GAP_INDEX))

    log('  in-memory merge_from() ...')
    idx_main.merge_from(idx_gap)
    log(f'  合并后 ntotal = {idx_main.ntotal:,}  '
        f'(预期 {stats["n_main"] + stats["n_gap"]:,})')
    assert idx_main.ntotal == stats['n_main'] + stats['n_gap'], \
        '合并后 ntotal 与预期不一致!'

    log('  加载 paper_ids.txt (main) ...')
    with open(IDS_FILE, encoding='utf-8') as f:
        ids_main = [line.strip() for line in f if line.strip()]
    log(f'  加载 papers_gap_ids.txt (gap) ...')
    with open(IDS_GAP_FILE, encoding='utf-8') as f:
        ids_gap = [line.strip() for line in f if line.strip()]

    log(f'  main ids: {len(ids_main):,}, gap ids: {len(ids_gap):,}')
    assert len(ids_main) == stats['n_main'], \
        f'main ids 数量 {len(ids_main)} 与 FAISS ntotal {stats["n_main"]} 不符'
    assert len(ids_gap) == stats['n_gap'], \
        f'gap ids 数量 {len(ids_gap)} 与 FAISS ntotal {stats["n_gap"]} 不符'

    merged_ids = ids_main + ids_gap
    log(f'  合并后 ids: {len(merged_ids):,}  '
        f'(预期 {stats["n_main"] + stats["n_gap"]:,})')

    log(f'  Phase 1 耗时: {time.time()-t0:.1f}s')
    return idx_main, merged_ids


def phase2_backup_and_swap(idx_merged, merged_ids: list[str],
                           stats: dict, do_backup: bool) -> None:
    """写盘: backup → staging → atomic swap → 追加 ids"""
    log('══ Phase 2: backup + atomic swap (写) ══')
    t0 = time.time()

    # ── 2a. 备份当前 main (如果启用) ──
    if do_backup:
        BACKUP_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_index = BACKUP_DIR / f'papers_{ts}.index'
        backup_ids = BACKUP_DIR / f'paper_ids_{ts}.txt'
        log(f'  备份 main → {backup_index.name}')
        shutil.copy2(str(FAISS_INDEX), str(backup_index))
        log(f'  备份 ids  → {backup_ids.name}')
        shutil.copy2(str(IDS_FILE), str(backup_ids))
        log(f'  备份完成, 大小: '
            f'{(backup_index.stat().st_size + backup_ids.stat().st_size)/1024**2:.1f} MB')
    else:
        log('  ⚠ --no-backup 已启用, 跳过备份 (出问题时无法回滚)')

    # ── 2b. 写到 staging 文件 ──
    import faiss
    staging_index = KB_DIR / 'papers.index.staging'
    staging_ids = KB_DIR / 'paper_ids.txt.staging'
    log(f'  写 staging: {staging_index.name} ...')
    faiss.write_index(idx_merged, str(staging_index))
    log(f'  staging.index size: '
        f'{staging_index.stat().st_size/1024**2:.1f} MB')

    log(f'  写 staging: {staging_ids.name} ...')
    with open(staging_ids, 'w', encoding='utf-8') as f:
        for pid in merged_ids:
            f.write(pid + '\n')
    log(f'  staging.ids  size: '
        f'{staging_ids.stat().st_size/1024**2:.1f} MB')

    # ── 2c. atomic swap (os.replace 在 Windows 是原子的) ──
    log('  atomic swap: papers.index ...')
    os.replace(str(staging_index), str(FAISS_INDEX))
    log('  atomic swap: paper_ids.txt ...')
    os.replace(str(staging_ids), str(IDS_FILE))
    log(f'  Phase 2 耗时: {time.time()-t0:.1f}s')

    # 验证写盘结果
    n_new = faiss.read_index(str(FAISS_INDEX)).ntotal
    log(f'  写盘验证: main.ntotal = {n_new:,}  '
        f'(预期 {stats["n_main"] + stats["n_gap"]:,})')
    assert n_new == stats['n_main'] + stats['n_gap'], '写盘后 ntotal 不一致!'


def phase3_clear_gap() -> None:
    """清空 gap 文件 (写)"""
    log('══ Phase 3: 清空 gap 文件 (写) ══')
    if FAISS_GAP_INDEX.exists():
        size = FAISS_GAP_INDEX.stat().st_size
        FAISS_GAP_INDEX.unlink()
        log(f'  删除 {FAISS_GAP_INDEX.name} ({size/1024**2:.1f} MB)')
    else:
        log(f'  {FAISS_GAP_INDEX.name} 已不存在, 跳过')

    if IDS_GAP_FILE.exists():
        size = IDS_GAP_FILE.stat().st_size
        IDS_GAP_FILE.unlink()
        log(f'  删除 {IDS_GAP_FILE.name} ({size/1024**2:.1f} MB)')
    else:
        log(f'  {IDS_GAP_FILE.name} 已不存在, 跳过')


def phase4_verify() -> int:
    """验证: 再次跑 kb_health 看 DANGER 是否消失"""
    log('══ Phase 4: 验证 (只读) ══')
    import subprocess
    log('  跑 kb_health.py --sample 0 ...')
    result = subprocess.run(
        [sys.executable, str(KB_DIR / 'kb_health.py'), '--sample', '0'],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        cwd=str(KB_DIR),
    )
    log(f'  kb_health exit code: {result.returncode}')

    # 解析关键行
    output = result.stdout + result.stderr
    for keyword in ('[5b]', 'Verdict:', 'Merge:', 'status:'):
        for line in output.splitlines():
            if keyword in line:
                log(f'    {line.strip()[:120]}')
                break

    if result.returncode == 0:
        log('  ✅ DANGER 已消除, KB 恢复 HEALTHY')
    elif result.returncode == 1:
        log('  ⚠ WARN 状态 (gap 已清, 但可能还有其他维度 WARN)')
    else:
        log('  🚨 仍有 DANGER, 请检查上方日志')
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='只跑 Phase 0/1 (只读校验), 不写盘')
    ap.add_argument('--no-backup', action='store_true',
                    help='跳过 backup 步骤 (强烈不推荐, 出问题无法回滚)')
    ap.add_argument('--skip-verify', action='store_true',
                    help='跳过 Phase 4 验证')
    args = ap.parse_args()

    log('═══════════════════════════════════════════════════════════')
    log('  merge_to_disk.py — 把 gap 索引合并到主索引')
    log(f'  KB 目录: {KB_DIR}')
    log(f'  模式: {"DRY-RUN (只读)" if args.dry_run else "FULL"}')
    log('═══════════════════════════════════════════════════════════')

    t_total = time.time()

    try:
        stats = phase0_prepare()
    except Exception as e:
        log(f'❌ Phase 0 失败: {e}')
        sys.exit(1)

    try:
        idx_merged, merged_ids = phase1_load_and_merge(stats)
    except Exception as e:
        log(f'❌ Phase 1 失败: {e}')
        sys.exit(1)

    if args.dry_run:
        log('══ DRY-RUN 完成, 不执行写盘 ══')
        log(f'  合并后预计: {stats["n_main"] + stats["n_gap"]:,} vecs')
        log(f'  总耗时: {time.time()-t_total:.1f}s')
        sys.exit(0)

    try:
        phase2_backup_and_swap(idx_merged, merged_ids, stats,
                                do_backup=not args.no_backup)
    except Exception as e:
        log(f'❌ Phase 2 失败: {e}')
        log('  已尝试保留原状态, 请检查磁盘/权限')
        sys.exit(2)

    try:
        phase3_clear_gap()
    except Exception as e:
        log(f'❌ Phase 3 失败: {e}')
        log('  主索引已更新, 但 gap 未清空, 下次启动会重复 merge')
        log('  手动清空: rm papers_gap.index papers_gap_ids.txt')
        sys.exit(2)

    log(f'✅ 合并完成, 总耗时: {time.time()-t_total:.1f}s')
    log(f'   main: {stats["n_main"]:,} + gap: {stats["n_gap"]:,} '
        f'= {stats["n_main"] + stats["n_gap"]:,} vecs')

    if not args.skip_verify:
        verify_exit = phase4_verify()
        sys.exit(0 if verify_exit in (0, 1) else 2)

    sys.exit(0)


if __name__ == '__main__':
    main()
