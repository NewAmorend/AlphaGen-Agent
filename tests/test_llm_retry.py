from __future__ import annotations

import asyncio

import httpx
import pytest

from wq_agent.llm.base import chat_completion_with_retry


class _Resp:
    def __init__(self, status: int = 200, content: str = "done"):
        self.status_code = status
        self._content = content
        self.text = "error-body"

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _Client:
    """假 httpx client：前 fail_times 次抛瞬时异常，或按 status_seq 返回状态码。"""

    def __init__(self, fail_times: int = 0, exc=None, status_seq=None):
        self.calls = 0
        self.fail_times = fail_times
        self.exc = exc or httpx.ReadTimeout("timeout")
        self.status_seq = status_seq

    async def post(self, url, json=None):
        self.calls += 1
        if self.status_seq is not None:
            s = self.status_seq[min(self.calls - 1, len(self.status_seq) - 1)]
            return _Resp(status=s)
        if self.calls <= self.fail_times:
            raise self.exc
        return _Resp()


async def _noop(*a, **k):
    pass


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _noop)


@pytest.mark.asyncio
async def test_recovers_after_transient_timeouts():
    c = _Client(fail_times=2)
    out = await chat_completion_with_retry(c, "u", {}, provider="T", max_attempts=3)
    assert out == "done"
    assert c.calls == 3  # 失败2次后第3次成功


@pytest.mark.asyncio
async def test_exhausts_and_raises():
    c = _Client(fail_times=5)
    with pytest.raises(httpx.ReadTimeout):
        await chat_completion_with_retry(c, "u", {}, provider="T", max_attempts=3)
    assert c.calls == 3  # 只试 max_attempts 次


@pytest.mark.asyncio
async def test_retries_5xx_then_succeeds():
    c = _Client(status_seq=[503, 200])
    out = await chat_completion_with_retry(c, "u", {}, provider="T", max_attempts=3)
    assert out == "done"
    assert c.calls == 2


@pytest.mark.asyncio
async def test_does_not_retry_4xx():
    c = _Client(status_seq=[400, 400, 400])
    with pytest.raises(Exception):
        await chat_completion_with_retry(c, "u", {}, provider="T", max_attempts=3)
    assert c.calls == 1  # 400 业务错误不重试，立即抛
