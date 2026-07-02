from __future__ import annotations

from pathlib import Path

import yaml

from alphagen_agent.wiki.curator import KnowledgeCurator, parse_since
from alphagen_agent.wiki.store import WikiStore


def _write_page(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_curator_reports_findings_and_suggestions(tmp_path: Path):
    root = tmp_path / "wiki"
    _write_page(
        root,
        "patterns/high-turnover.md",
        """---
title: 高换手失败模式
type: pattern
tags: [pattern, turnover]
created: 2026-06-03
---

# 高换手失败模式

修复 [[trade_when]] 和 [[missing_page]]。
""",
    )
    _write_page(
        root,
        "operators/trade_when.md",
        """---
title: trade_when
type: operator
tags: [operator]
created: 2026-06-03
---

# trade_when
""",
    )
    for i in range(2):
        _write_page(
            root,
            f"lessons/2026-06-0{i + 1}-batch-1.md",
            f"""---
title: Lesson {i}
type: lesson
tags: [reject, turnover]
created: 2026-06-0{i + 1}
---

Turnover 太高，`foo_field` 和 `bar_field` 组合过于频繁。
""",
        )
    _write_page(
        root,
        "recipes/test-recipe.md",
        """---
title: Test recipe
type: recipe
tags: [recipe, test]
created: 2026-06-03
---

`foo_field` 与 `bar_field`。
""",
    )
    bench = root / "bench" / "retrieval_golden.yml"
    bench.parent.mkdir(parents=True)
    bench.write_text("- query: trade_when\n  expected: [operators/trade_when]\n", encoding="utf-8")

    report = KnowledgeCurator(WikiStore(root)).audit(max_suggestions=20)

    assert report.broken_links == 1
    assert any(f.category == "broken_wikilink" for f in report.findings)
    assert any(s.action == "update_pattern" and s.target == "patterns/high-turnover" for s in report.suggestions)
    assert any(s.action == "create_field_page" and s.target == "fields/foo_field" for s in report.suggestions)
    assert any(s.action == "add_bench_query" and s.target == "patterns/high-turnover" for s in report.suggestions)
    assert any(s.action == "add_bench_query" and s.target == "recipes/test-recipe" for s in report.suggestions)


def test_curator_apply_appends_bench_queries_and_report(tmp_path: Path):
    root = tmp_path / "wiki"
    _write_page(
        root,
        "patterns/low-coverage-nan.md",
        """---
title: 低覆盖与 NaN 传播
type: pattern
tags: [pattern, coverage, nan]
created: 2026-06-03
---

# 低覆盖与 NaN 传播
""",
    )
    bench = root / "bench" / "retrieval_golden.yml"
    bench.parent.mkdir(parents=True)
    bench.write_text("# bench\n\n[]\n", encoding="utf-8")

    curator = KnowledgeCurator(WikiStore(root))
    report = curator.audit(max_suggestions=10)
    applied = curator.apply(report)

    data = yaml.safe_load(bench.read_text(encoding="utf-8"))
    assert any("patterns/low-coverage-nan" in row.get("expected", []) for row in data)
    assert (root / "curation_report.json").exists()
    assert any("retrieval bench" in item for item in applied.applied)

    second = curator.audit(max_suggestions=10)
    assert not any(s.target == "patterns/low-coverage-nan" for s in second.applyable_suggestions)


def test_parse_since_accepts_iso_date():
    assert str(parse_since("2026-06-03")) == "2026-06-03"
