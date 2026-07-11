#!/usr/bin/env python3
"""Batch run must-cite scan for all papers.

For each PAPER_CONSOLIDATED:
- Find largest refs.bib
- Run kb_search with must-cite + BibTeX export
- Output: <paper>_must_cite.bib in the same folder
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PAPERS = {
    'PAPER1_CONSOLIDATED': 'calibration LLM uncertainty evaluation confidence',
    'PAPER2_CONSOLIDATED': 'multi-agent LLM evaluation framework calibration contagion',
    'PAPER3_CONSOLIDATED': 'calibration fatigue LLM benchmark spurious features prompt',
    'PAPER4_CONSOLIDATED': 'co-failure ceiling multi-LM routing voting mixture-of-agents impossibility',
    'PAPER5_CONSOLIDATED': 'modal ceiling test-time scaling sample budget TTRL simulation',
}

KB = Path(r'E:/peS2o_kb_faiss')
RESEARCH = Path(r'F:/Research')


def find_main_bib(paper_dir: Path):
    """Pick largest .bib file in paper_dir."""
    if not paper_dir.exists():
        return None
    bibs = []
    for root, _, files in os.walk(paper_dir):
        for f in files:
            if f.endswith('.bib') and 'must_cite' not in f and 'test' not in f and 'rebuttal' not in f:
                fp = Path(root) / f
                bibs.append((fp, fp.stat().st_size))
    if not bibs:
        return None
    return max(bibs, key=lambda x: x[1])[0]


def count_existing_refs(bib_path: Path) -> int:
    """Count number of @ entries in bib."""
    if not bib_path or not bib_path.exists():
        return 0
    with open(bib_path, encoding='utf-8', errors='replace') as f:
        content = f.read()
    return content.count('@')


def run_must_cite(paper_name: str, query: str, existing_bib: Path, out_bib: Path):
    cmd = [
        sys.executable, str(KB / 'kb_search.py'),
        query,
        '--must-cite',
        '--existing-refs', str(existing_bib),
        '-n', '10',
        '--bibtex', str(out_bib),
    ]
    print(f'\n{"="*70}')
    print(f'PAPER: {paper_name}')
    print(f'QUERY: {query}')
    print(f'EXISTING: {existing_bib}')
    print(f'OUTPUT: {out_bib}')
    print(f'CMD: {" ".join(cmd)}')
    print(f'{"="*70}')

    result = subprocess.run(cmd, cwd=str(KB))
    return result.returncode == 0


def main():
    summary = []
    for paper_name, query in PAPERS.items():
        paper_dir = RESEARCH / paper_name
        existing_bib = find_main_bib(paper_dir)
        if not existing_bib:
            print(f'\n!!! {paper_name}: no .bib found, skipping')
            summary.append({'paper': paper_name, 'status': 'no_bib'})
            continue

        n_existing = count_existing_refs(existing_bib)
        out_bib = paper_dir / f'{paper_name}_must_cite.bib'
        success = run_must_cite(paper_name, query, existing_bib, out_bib)

        n_new = 0
        if out_bib.exists():
            n_new = count_existing_refs(out_bib)

        summary.append({
            'paper': paper_name,
            'status': 'ok' if success else 'failed',
            'existing_refs': n_existing,
            'new_suggestions': n_new,
            'existing_bib': str(existing_bib),
            'output_bib': str(out_bib),
        })

    print(f'\n\n{"="*70}')
    print('SUMMARY')
    print(f'{"="*70}')
    for s in summary:
        print(f'{s["paper"]}: {s["status"]}')
        if 'existing_refs' in s:
            print(f'  existing refs: {s["existing_refs"]}')
            print(f'  new suggestions: {s["new_suggestions"]}')
            print(f'  output: {s["output_bib"]}')

    with open(KB / 'batch_must_cite_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'\nFull summary: {KB / "batch_must_cite_summary.json"}')


if __name__ == '__main__':
    main()