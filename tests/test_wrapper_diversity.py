from __future__ import annotations

import random

from wq_agent.generator.llm import (
    LLMAlphaGenerator,
    build_proven_wrappers_section,
    overrepresented_families,
)


# --------------------------------------------------------------- sampled pool
def test_proven_wrappers_section_keeps_guidance_and_samples():
    sec = build_proven_wrappers_section(random.Random(0))
    # 仍保留标题与关键观察（教学内容不能丢）
    assert "实测高 Fitness Wrapper" in sec
    assert "关键观察" in sec
    # 至少包含一个 <signal> 占位的 wrapper 样例
    assert "<signal>" in sec


def test_proven_wrappers_section_varies_with_seed():
    a = build_proven_wrappers_section(random.Random(1))
    b = build_proven_wrappers_section(random.Random(2))
    # 不同 seed 抽到的样例集合应不同——这正是打破"每次都推同 3 个壳子"的关键
    assert a != b


# ----------------------------------------------- overrepresented_families
def test_overrepresented_adaptive_large_library():
    # 大库：没人占到 20%，但 decay 是平均家族的十几倍 → 仍要抓出来
    fams = [
        {"signature": "ts_decay_linear", "count": 176},
        {"signature": "rank", "count": 40},
        {"signature": "group_rank", "count": 8},
    ]
    over = overrepresented_families(fams, total=346, num_families=30)
    sigs = {f["signature"] for f in over}
    assert "ts_decay_linear" in sigs   # 平均 ≈11.5，176 远超 2× → 命中
    assert "group_rank" not in sigs    # 8 不算过量


def test_overrepresented_dominant_small_library():
    # 小库：靠 dominant_share（单家族 ≥25%）兜底
    fams = [{"signature": "ts_decay_linear(rank", "count": 12}, {"signature": "rank(ts_delta", "count": 3}]
    over = overrepresented_families(fams, total=20, num_families=2)
    assert any(f["signature"] == "ts_decay_linear(rank" for f in over)


def test_overrepresented_empty_when_diverse():
    fams = [{"signature": f"op{i}", "count": 1} for i in range(30)]
    assert overrepresented_families(fams, total=30, num_families=30) == []
    assert overrepresented_families(None, total=10, num_families=5) == []
    assert overrepresented_families(fams, total=0, num_families=30) == []


# --------------------------------------------------- family saturation steer
def test_family_saturation_section_flags_overrepresented():
    dist = {
        "total_backtested": 100,
        "unique_skeletons": 40,
        "unique_outer1": 20,
        "unique_outer2": 30,
        "top_outer1": [
            {"signature": "ts_decay_linear", "count": 51, "avg_fitness": 0.11, "max_fitness": 1.2},
            {"signature": "rank", "count": 8, "avg_fitness": 0.3, "max_fitness": 0.7},
        ],
        "top_outer2": [
            {"signature": "ts_decay_linear(rank", "count": 30, "avg_fitness": 0.1, "max_fitness": 1.1},
        ],
    }
    sec = LLMAlphaGenerator._build_family_saturation_section(dist)
    assert sec  # 非空
    assert "ts_decay_linear" in sec      # outer-1 霸屏被点名
    assert "最外层" in sec               # 指明是最外层算子过量


def test_family_saturation_section_empty_when_sparse_or_diverse():
    # 数据太少 → 不打扰
    assert LLMAlphaGenerator._build_family_saturation_section(
        {"total_backtested": 4, "unique_outer1": 4, "top_outer1": []}
    ) == ""
    # 没有任何家族集中（都只占很小份额）→ 空
    dist = {
        "total_backtested": 30,
        "unique_outer1": 30,
        "unique_outer2": 30,
        "top_outer1": [{"signature": f"op{i}", "count": 1} for i in range(30)],
        "top_outer2": [{"signature": f"op{i}(rank", "count": 1} for i in range(30)],
    }
    assert LLMAlphaGenerator._build_family_saturation_section(dist) == ""
    # None 安全
    assert LLMAlphaGenerator._build_family_saturation_section(None) == ""
