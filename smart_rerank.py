#!/usr/bin/env python3
"""smart_rerank.py — 智能重排序 + 查询扩展

解决 PE2o 语料中 "modal ceiling" 撞机械工程论文的问题。

两层策略:

1. 查询侧: 检测 query 是否暗含 AI/ML 上下文, 自动扩展
   - "modal ceiling" + AI 上下文 → "modal ceiling language model test-time scaling"
   - "co-failure" + AI 上下文 → "co-failure multi-agent language model ensembles"

2. 结果侧: 多因子重排序
   - FAISS 语义分(基线)
   - 时效性加分 (2024-2026 论文加权)
   - 类别相关性加分 (cs.LG/CL/AI 优先)
   - 可选: 引用数加分 (Semantic Scholar)

API:
    from smart_rerank import expand_query, rerank
"""
from __future__ import annotations
import re
from typing import Optional


# ════════════════════════════════════════════════════════════════
# Query expansion
# ════════════════════════════════════════════════════════════════

# 关键词词典: 当 query 含这些, 暗示是 AI/ML 上下文
AI_HINTS = {
    # ML 基础
    'model', 'models', 'llm', 'llms', 'transformer', 'transformers',
    'language model', 'language models', 'neural', 'neural network',
    'neural networks', 'deep learning', 'machine learning', 'ml',
    'training', 'inference', 'fine-tuning', 'finetune', 'rag',
    'embedding', 'embeddings', 'attention', 'gpt', 'bert', 't5',
    'rlhf', 'prompt', 'prompts', 'token', 'tokens',

    # ML 评测/对齐
    'calibration', 'evaluation', 'eval', 'benchmark', 'benchmarks',
    'hallucination', 'factuality', 'alignment', 'rlhf', 'dpo',
    'preference', 'preferences', 'judge', 'llm-as-judge',

    # Agent / multi-agent
    'agent', 'agents', 'multi-agent', 'multiagent', 'agentic',
    'tool', 'tools', 'tooling', 'routing', 'mixture-of-agents',

    # Reasoning / scaling
    'reasoning', 'chain-of-thought', 'cot', 'scaling', 'test-time',
    'sampling', 'self-consistency', 'consensus', 'voting',

    # Calibration-specific
    'calibration contagion', 'impossibility triangle', 'modal ceiling',
    'co-failure', 'correlation ceiling', 'ece', 'brier',
}

# 不同 AI 主题的扩展词
EXPANSIONS = {
    'modal ceiling': 'modal ceiling language model test-time scaling test-time compute',
    'co-failure': 'co-failure multi-agent language model ensembles',
    'impossibility triangle': 'impossibility triangle multi-agent calibration',
    'calibration contagion': 'calibration contagion language model multi-agent',
    'ttrl': 'ttrl test-time reinforcement learning language model',
    'self-consistency': 'self-consistency language model reasoning sampling',
}

# 关键词 → 类别偏好
CATEGORY_HINTS = {
    'model': ['cs.LG', 'cs.CL', 'cs.AI'],
    'llm': ['cs.CL', 'cs.LG', 'cs.AI'],
    'transformer': ['cs.CL', 'cs.LG', 'cs.AI'],
    'agent': ['cs.MA', 'cs.CL', 'cs.AI'],
    'multi-agent': ['cs.MA', 'cs.CL', 'cs.AI'],
    'training': ['cs.LG', 'cs.CL'],
    'calibration': ['cs.LG', 'cs.CL', 'cs.AI'],
    'ece': ['cs.LG', 'cs.CL'],
}


def detect_ai_context(query: str) -> bool:
    """检查 query 是否在 AI/ML 上下文."""
    q_lower = query.lower()
    q_tokens = set(re.findall(r'[a-z]+', q_lower))
    return len(q_tokens & AI_HINTS) >= 1


def expand_query(query: str) -> tuple[str, list[str]]:
    """扩展 query, 返回 (expanded_query, applied_rules)."""
    q_lower = query.lower()
    applied = []
    expanded = query

    # 1. 命中已知扩展规则
    for trigger, expansion in EXPANSIONS.items():
        if trigger in q_lower:
            expanded = expanded + ' ' + expansion
            applied.append(f'rule:{trigger}')
            break

    # 2. 如果 query 用了 mechainical-engineering 词 + 但没 AI 上下文
    # 检查是否撞名(modal/co-failure/torque/...)
    if not detect_ai_context(query):
        meche_terms = {'modal', 'co-failure', 'resonance', 'vibration',
                       'torque', 'structural', 'mechanical'}
        q_tokens = set(re.findall(r'[a-z]+', q_lower))
        if q_tokens & meche_terms:
            # 强制加 AI context
            expanded = expanded + ' language model neural network AI'
            applied.append('mech+ai-context')

    return expanded, applied


def detect_category_preference(query: str) -> list[str]:
    """根据 query 推荐 arxiv 类别."""
    q_lower = query.lower()
    q_tokens = set(re.findall(r'[a-z]+', q_lower))
    cats = []
    for trigger, preferred in CATEGORY_HINTS.items():
        if trigger in q_tokens or trigger in q_lower:
            for c in preferred:
                if c not in cats:
                    cats.append(c)
    return cats[:3]  # top 3


# ════════════════════════════════════════════════════════════════
# Result rerank
# ════════════════════════════════════════════════════════════════

# 时效性权重: 不同年份加分 (越大越靠前)
YEAR_BONUS = {
    '2026': 0.30,
    '2025': 0.25,
    '2024': 0.10,
    '2023': -0.10,  # 强降权老论文
    '2022': -0.25,
    '2021': -0.40,
    '2020': -0.60,
}

# LLM-relevant 类别加权
LLM_CATEGORY_BONUS = {
    'cs.LG': 0.10,
    'cs.CL': 0.12,
    'cs.AI': 0.10,
    'cs.MA': 0.08,
    'cs.IR': 0.05,
    'cs.CV': 0.05,
    'cs.LO': 0.08,  # formal languages / logic — LLM reasoning
    'cs.DS': 0.04,
}


def rerank_score(paper: dict, base_score: float,
                  category_hint: Optional[list[str]] = None) -> tuple[float, dict]:
    """计算重排序分数, 返回 (final_score, debug_info)."""
    debug = {
        'base': base_score,
        'year_bonus': 0.0,
        'cat_bonus': 0.0,
        'hint_match': 0.0,
    }
    score = base_score

    # Year bonus
    year = (paper.get('year') or '')[:4]
    if year in YEAR_BONUS:
        debug['year_bonus'] = YEAR_BONUS[year]
        score += YEAR_BONUS[year]

    # Category bonus
    cats = (paper.get('categories') or '')
    for cat, bonus in LLM_CATEGORY_BONUS.items():
        if cat in cats:
            debug['cat_bonus'] += bonus
            score += bonus
            break  # only count primary cat bonus

    # Hint match boost
    if category_hint:
        for hint in category_hint:
            if hint in cats:
                debug['hint_match'] = 0.05
                score += 0.05
                break

    return score, debug


if __name__ == '__main__':
    # Quick test
    test_queries = [
        'modal ceiling test-time scaling',
        'co-failure multi-agent',
        'torque analysis',
        'calibration in transformers',
    ]
    for q in test_queries:
        expanded, rules = expand_query(q)
        pref = detect_category_preference(q)
        print(f'\nQ: {q!r}')
        print(f'  AI context: {detect_ai_context(q)}')
        print(f'  expanded: {expanded}')
        print(f'  rules: {rules}')
        print(f'  preferred cats: {pref}')