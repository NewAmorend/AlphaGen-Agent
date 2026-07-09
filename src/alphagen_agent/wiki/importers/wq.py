from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from loguru import logger

from ...wq.client import WQClient


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_MANAGED_MARKER = "<!-- managed by alphagen-agent wiki import-wq; manual edits below -->"
_LEGACY_MANAGED_MARKER = "<!-- managed by wq-agent wiki import-wq; manual edits below -->"


@dataclass
class WQImportStats:
    operators: int = 0
    datasets: int = 0
    fields: int = 0
    skipped: int = 0


def _safe_filename(name: str) -> str:
    name = name.strip().lower()
    name = _FILENAME_SAFE.sub("_", name)
    return name[:120] or "untitled"


def _flatten(value) -> str:
    """WQ API 经常把字段返回成 {id, name} dict。展平成可读字符串。"""
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("id") or value.get("name") or "")
    if isinstance(value, (list, tuple)):
        return ", ".join(_flatten(v) for v in value if v)
    return str(value)


def _safe_dict_value(value):
    """Frontmatter 友好版：dict → id/name 字符串；其他类型原样保留。"""
    if isinstance(value, dict):
        return _flatten(value)
    if isinstance(value, list):
        return [_flatten(v) for v in value]
    return value


def _frontmatter(title: str, type_: str, tags: list[str], extra: dict | None = None) -> str:
    data: dict = {
        "title": title,
        "type": type_,
        "tags": sorted({t for t in tags if t})[:10],
        "sources": ["worldquantbrain-api"],
        "created": date.today().isoformat(),
    }
    if extra:
        data.update(extra)
    return "---\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True) + "---\n"


def _write_if_new_or_managed(path: Path, content: str) -> bool:
    """只覆盖 importer 写过的页（含 _MANAGED_MARKER）；用户手写过的不动。"""
    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace")
        if not any(marker in existing for marker in (_MANAGED_MARKER, _LEGACY_MANAGED_MARKER)):
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


class WQDocImporter:
    def __init__(self, wiki_root: Path, client: WQClient):
        self.wiki_root = Path(wiki_root)
        self.client = client

    async def import_operators(self) -> int:
        ops = await self.client.get_operators()
        out_dir = self.wiki_root / "operators"
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for op in ops:
            name = op.name
            if not name:
                continue
            content = self._render_operator(op.name, op.category, op.type, op.definition, op.description)
            path = out_dir / f"{_safe_filename(name)}.md"
            if _write_if_new_or_managed(path, content):
                written += 1
        return written

    async def import_datasets(
        self,
        region: str | None = None,
        universe: str | None = None,
        delay: int | None = None,
        embed_field_list: bool = True,
    ) -> int:
        """每个 dataset 一页。embed_field_list=True 时把该 dataset 的所有 field 列在页内
        （取代 7000+ 个 TODO 占位的 fields/*.md）。"""
        datasets = await self.client.get_all_datasets(region=region, universe=universe, delay=delay)
        out_dir = self.wiki_root / "datasets"
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        all_field_ids: set[str] = set()
        for ds in datasets:
            ds_id = ds.get("id")
            if not ds_id:
                continue
            fields: list[dict] = []
            if embed_field_list:
                fields = await self.client.get_all_data_fields_paged(
                    dataset_id=ds_id,
                    region=region,
                    universe=universe,
                    delay=delay,
                )
                for f in fields:
                    fid = f.get("id")
                    if fid:
                        all_field_ids.add(_flatten(fid))
                logger.info(f"Dataset {ds_id}: pulled {len(fields)} fields for inline listing")
            content = self._render_dataset(ds, fields=fields)
            path = out_dir / f"{_safe_filename(ds_id)}.md"
            if _write_if_new_or_managed(path, content):
                written += 1
        if all_field_ids:
            self._write_field_dictionary(all_field_ids)
        return written

    def _write_field_dictionary(self, field_ids: set[str]) -> None:
        """所有 field ID 写到 dictionary/auto.txt，让 FMM 分词器能识别（不占独立 wiki 页）。"""
        path = self.wiki_root / "dictionary" / "auto.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: set[str] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    existing.add(line)
        merged = existing | field_ids
        path.write_text(
            "# Auto-generated by alphagen-agent wiki import-wq; merge of all known field IDs.\n"
            + "\n".join(sorted(merged)) + "\n",
            encoding="utf-8",
        )
        logger.info(f"Wrote {len(merged)} field IDs to dictionary/auto.txt")

    async def import_fields(
        self,
        region: str | None = None,
        universe: str | None = None,
        delay: int | None = None,
        limit_per_dataset: int | None = None,
    ) -> int:
        datasets = await self.client.get_all_datasets(region=region, universe=universe, delay=delay)
        out_dir = self.wiki_root / "fields"
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for ds in datasets:
            ds_id = ds.get("id")
            if not ds_id:
                continue
            max_pages = max(1, (limit_per_dataset // 50) + 1) if limit_per_dataset else 100
            fields = await self.client.get_all_data_fields_paged(
                dataset_id=ds_id,
                region=region,
                universe=universe,
                delay=delay,
                max_pages=max_pages,
            )
            if limit_per_dataset:
                fields = fields[:limit_per_dataset]
            for f in fields:
                fid = f.get("id")
                if not fid:
                    continue
                content = self._render_field(f, dataset_meta=ds, region=region, universe=universe, delay=delay)
                path = out_dir / f"{_safe_filename(fid)}.md"
                if _write_if_new_or_managed(path, content):
                    written += 1
            logger.info(f"Imported {len(fields)} fields from dataset {ds_id}")
        return written

    async def import_all(
        self,
        region: str | None = None,
        universe: str | None = None,
        delay: int | None = None,
        limit_per_dataset: int | None = None,
        include_fields: bool = False,
    ) -> WQImportStats:
        """默认：operators + datasets（含内嵌字段列表）+ auto.txt 词典。
        加 include_fields=True 才会额外写 7000+ 个 per-field 页面。"""
        stats = WQImportStats()
        stats.operators = await self.import_operators()
        stats.datasets = await self.import_datasets(
            region=region, universe=universe, delay=delay, embed_field_list=True,
        )
        if include_fields:
            stats.fields = await self.import_fields(
                region=region, universe=universe, delay=delay, limit_per_dataset=limit_per_dataset
            )
        return stats

    @staticmethod
    def _render_operator(name: str, category: str, type_: str, definition: str, description: str) -> str:
        tags = ["operator", category.lower().replace(" ", "_")]
        if name.startswith("ts_"):
            tags.append("time_series")
        if name.startswith("group_"):
            tags.append("group")
        front = _frontmatter(
            title=f"{name} 算子",
            type_="operator",
            tags=tags,
            extra={"operator_name": name, "category": category, "operator_type": type_ or "SCALAR"},
        )
        body_lines = [
            "",
            _MANAGED_MARKER,
            "",
            f"# `{name}`",
            "",
            f"**Category**：{category or 'N/A'}　**Type**：{type_ or 'SCALAR'}",
            "",
        ]
        if definition:
            body_lines += ["## 签名", "", "```", definition, "```", ""]
        if description:
            body_lines += ["## 官方说明", "", description, ""]
        body_lines += [
            "## 使用提示（人工补充）",
            "",
            "- TODO：典型适用场景、常配合的算子、参数范围、踩坑（如需互链，请填写真实页面名）",
            "",
        ]
        return front + "\n".join(body_lines)

    @staticmethod
    def _render_dataset(ds: dict, fields: list[dict] | None = None) -> str:
        ds_id = _flatten(ds.get("id"))
        name = _flatten(ds.get("name")) or ds_id
        category = _flatten(ds.get("category"))
        delay = ds.get("delay", "")
        region = _flatten(ds.get("region"))
        universe = _flatten(ds.get("universe"))
        description = _flatten(ds.get("description"))
        tags = ["dataset"]
        if category:
            tags.append(category.lower())
        if region:
            tags.append(region.lower())
        front = _frontmatter(
            title=f"{ds_id} 数据集",
            type_="entity",
            tags=tags,
            extra={
                "dataset_id": ds_id,
                "category": category,
                "delay": delay,
                "region": region,
                "universe": universe,
            },
        )
        body_lines = [
            "",
            _MANAGED_MARKER,
            "",
            f"# {name}",
            "",
            f"**ID**：`{ds_id}`　**Category**：{category}　**Region**：{region}　**Universe**：{universe}　**Delay**：{delay}",
            "",
        ]
        if description:
            body_lines += ["## 描述", "", description, ""]
        if ds.get("fieldCount") is not None:
            body_lines += [f"**字段数**：{ds.get('fieldCount')}", ""]
        if ds.get("alphaCount") is not None:
            body_lines += [f"**已用 alpha 数**：{ds.get('alphaCount')}", ""]

        if fields:
            body_lines += [
                f"## 字段清单（{len(fields)} 项）",
                "",
                "| id | type | description |",
                "| --- | --- | --- |",
            ]
            for f in fields:
                fid = _flatten(f.get("id"))
                ftype = _flatten(f.get("type")) or "—"
                fdesc = _flatten(f.get("description")).replace("|", " ").replace("\n", " ")
                if len(fdesc) > 120:
                    fdesc = fdesc[:117] + "…"
                body_lines.append(f"| `{fid}` | {ftype} | {fdesc} |")
            body_lines.append("")

        body_lines += [
            "## 用法提示（人工补充）",
            "",
            "- TODO：典型组合方式、相关的 concept 页面",
            "",
        ]
        return front + "\n".join(body_lines)

    @staticmethod
    def _render_field(
        f: dict,
        dataset_meta: dict | None = None,
        region: str | None = None,
        universe: str | None = None,
        delay: int | None = None,
    ) -> str:
        fid = _flatten(f.get("id"))
        description = _flatten(f.get("description"))
        ftype = _flatten(f.get("type"))
        dataset_id = _flatten(f.get("dataset")) or _flatten((dataset_meta or {}).get("id"))
        coverage = f.get("coverage")
        if isinstance(coverage, (dict, list)):
            coverage = _flatten(coverage)
        tags = ["field"]
        if dataset_id:
            tags.append(dataset_id)
        if ftype:
            tags.append(ftype.lower())
        if region:
            tags.append(region.lower())
        front = _frontmatter(
            title=fid,
            type_="field",
            tags=tags,
            extra={
                "field_id": fid,
                "dataset_id": dataset_id,
                "field_type": ftype,
                "region": region,
                "universe": universe,
                "delay": delay,
                "coverage": coverage,
            },
        )
        body_lines = [
            "",
            _MANAGED_MARKER,
            "",
            f"# `{fid}`",
            "",
            f"**Type**：{ftype or 'N/A'}　**Dataset**：[[{dataset_id}]]　**Region**：{region or 'N/A'}　**Universe**：{universe or 'N/A'}　**Delay**：{delay if delay is not None else 'N/A'}",
            "",
        ]
        if description:
            body_lines += ["## 描述", "", description, ""]
        if coverage is not None:
            body_lines += [f"**覆盖率**：{coverage}", ""]
        body_lines += [
            "## 使用提示（人工补充）",
            "",
            "- TODO：与 [[ts_rank]] / [[zscore]] / [[group_neutralize]] 等的常见组合；适合的因子家族",
            "",
        ]
        return front + "\n".join(body_lines)
