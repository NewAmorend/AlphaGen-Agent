from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger


_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _load_dict_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line)
    return terms


@dataclass
class Tokenizer:
    """前向最大匹配 (FMM) 中文 + ASCII 混合分词器。

    词典里有什么就切什么；词典外的中文字符按单字输出，ASCII 子串走正则。
    """

    terms: set[str] = field(default_factory=set)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    _max_len: int = 0

    @classmethod
    def from_paths(
        cls,
        base_dict: Path | None = None,
        auto_dict: Path | None = None,
        synonyms_yaml: Path | None = None,
        extra_terms: list[str] | None = None,
    ) -> "Tokenizer":
        terms: set[str] = set()
        for p in (base_dict, auto_dict):
            if p:
                for t in _load_dict_file(p):
                    terms.add(t)
        if extra_terms:
            for t in extra_terms:
                t = t.strip()
                if t:
                    terms.add(t)

        synonyms: dict[str, list[str]] = {}
        if synonyms_yaml and synonyms_yaml.exists():
            try:
                data = yaml.safe_load(synonyms_yaml.read_text(encoding="utf-8")) or {}
                for k, v in data.items():
                    if isinstance(v, list):
                        synonyms[str(k)] = [str(x) for x in v]
                        terms.add(str(k))
                        for x in v:
                            terms.add(str(x))
            except Exception as exc:
                logger.warning(f"Failed to load synonyms: {exc}")

        max_len = max((len(t) for t in terms), default=0)
        return cls(terms=terms, synonyms=synonyms, _max_len=max_len)

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        tokens: list[str] = []
        i = 0
        n = len(text)
        max_len = max(self._max_len, 16)
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            # ASCII fast path
            if ch.isascii() and (ch.isalnum() or ch == "_"):
                m = _ASCII_WORD.match(text, i)
                if m:
                    tok = m.group(0)
                    tokens.append(tok.lower())
                    i = m.end()
                    continue
            # 标点直接跳
            if not ch.isalnum() and not _is_cjk(ch):
                i += 1
                continue
            # CJK FMM
            matched = None
            upper = min(n, i + max_len)
            for j in range(upper, i, -1):
                candidate = text[i:j]
                if candidate in self.terms:
                    matched = candidate
                    break
            if matched:
                tokens.append(matched)
                i += len(matched)
            else:
                tokens.append(ch)
                i += 1
        return tokens

    def expand_synonyms(
        self, tokens: list[str]
    ) -> tuple[list[tuple[str, float]], list[str]]:
        """Returns (weighted_tokens, original_tokens_dedup)."""
        weighted: list[tuple[str, float]] = []
        originals: list[str] = []
        seen: set[str] = set()
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            originals.append(tok)
            weighted.append((tok, 1.0))
            for syn in self.synonyms.get(tok, []):
                if syn not in seen:
                    seen.add(syn)
                    weighted.append((syn, 0.5))
        return weighted, originals


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF
    )
