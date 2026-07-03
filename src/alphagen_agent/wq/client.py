from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx
from loguru import logger

from ..config import Settings
from ..models import SimulationRequest, WQDataField, WQOperator
from .auth import WQAuth


class WQClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.auth = WQAuth(settings)
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(settings.WQ_MAX_CONCURRENT)
        self._retry_queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()

    async def connect(self) -> None:
        self._client = await self.auth.authenticate()

    async def close(self) -> None:
        await self.auth.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = await self.auth.refresh()
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        max_retries: int = 5,
        **kwargs: Any,
    ) -> httpx.Response:
        client = await self._ensure_client()
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await getattr(client, method)(path, **kwargs)
                if response.status_code == 401:
                    logger.warning("Session expired, re-authenticating...")
                    self._client = await self.auth.refresh()
                    client = self._client
                    response = await getattr(client, method)(path, **kwargs)
                if response.status_code == 429:
                    retry_after = int(float(response.headers.get("Retry-After", "60")))
                    logger.warning(f"Rate limited, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue
                return response
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError, httpx.RemoteProtocolError) as e:
                last_exc = e
                # 之前是 2**attempt → 1+2+4=7s（3 次）, WQ Brain 偶尔慢一拍直接挂。
                # 改 5 次 + 上限 30s：1+2+4+8+16=31s, 足以扛过常见 60s rate-limit 后的恢复。
                wait = min(2 ** attempt, 30)
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                await asyncio.sleep(wait)
        raise last_exc or Exception(f"Request to {path} failed after {max_retries} retries")

    # WQ Brain dataset 类目；之前硬编码只拉 fundamental 导致 LLM 永远只看到 fnd2/fnd6 字段。
    # 这个列表覆盖 wiki/datasets 里出现过的所有类别。
    DEFAULT_CATEGORIES: list[str] = [
        "pv", "fundamental", "analyst", "model",
        "news", "option", "sentiment", "socialmedia",
    ]

    async def get_data_fields(
        self,
        region: str | None = None,
        universe: str | None = None,
        delay: int | None = None,
        limit: int = 20,
        categories: list[str] | None = None,
    ) -> list[WQDataField]:
        region = region or self.settings.WQ_REGION
        universe = universe or self.settings.WQ_UNIVERSE
        delay = delay if delay is not None else self.settings.WQ_DELAY
        cats = categories or self.DEFAULT_CATEGORIES
        all_fields: list[WQDataField] = []

        delays = [delay]
        if region in ("ASI", "CHN", "GLB"):
            delays = [1]

        for d in delays:
            dataset_ids: list[str] = []
            for cat in cats:
                datasets_resp = await self._request(
                    "get",
                    "/data-sets",
                    params={
                        "category": cat,
                        "delay": d,
                        "instrumentType": "EQUITY",
                        "region": region,
                        "universe": universe,
                        "limit": 50,
                    },
                )
                if datasets_resp.status_code == 200:
                    results = datasets_resp.json().get("results", [])
                    dataset_ids.extend(ds["id"] for ds in results if "id" in ds)
            # 去重（不同 category 可能返回重复 dataset.id 不太可能，但保险）
            dataset_ids = list(dict.fromkeys(dataset_ids))
            if not dataset_ids:
                dataset_ids = [
                    "pv1", "fundamental6", "fundamental2", "analyst4",
                    "model16", "model51", "news12", "sentiment1",
                ]

            for ds_id in dataset_ids:
                count_resp = await self._request(
                    "get",
                    "/data-fields",
                    params={
                        "delay": d,
                        "instrumentType": "EQUITY",
                        "limit": 1,
                        "region": region,
                        "universe": universe,
                        "dataset.id": ds_id,
                    },
                )
                if count_resp.status_code != 200:
                    continue
                total = count_resp.json().get("count", 0)
                if total == 0:
                    continue

                max_offset = max(0, total - limit)
                offset = random.randint(0, max_offset)
                fields_resp = await self._request(
                    "get",
                    "/data-fields",
                    params={
                        "delay": d,
                        "instrumentType": "EQUITY",
                        "limit": min(limit, total),
                        "region": region,
                        "universe": universe,
                        "dataset.id": ds_id,
                        "offset": offset,
                    },
                )
                if fields_resp.status_code == 200:
                    for f in fields_resp.json().get("results", []):
                        all_fields.append(
                            WQDataField(
                                id=f["id"],
                                description=f.get("description", ""),
                                dataset=f.get("dataset", {}).get("id") if isinstance(f.get("dataset"), dict) else None,
                                type=f.get("type"),
                            )
                        )

        seen: dict[str, WQDataField] = {f.id: f for f in all_fields}
        logger.info(f"Fetched {len(seen)} unique data fields")
        return list(seen.values())

    async def get_operators(self) -> list[WQOperator]:
        resp = await self._request("get", "/operators")
        if resp.status_code != 200:
            logger.error(f"Failed to get operators: {resp.text}")
            return _DEFAULT_OPERATORS()

        data = resp.json()
        raw_list = data if isinstance(data, list) else data.get("results", [])
        operators = []
        for op in raw_list:
            operators.append(
                WQOperator(
                    name=op.get("name", op.get("id", "")),
                    category=op.get("category", ""),
                    type=op.get("type", "SCALAR"),
                    definition=op.get("definition", ""),
                    description=op.get("description", ""),
                )
            )
        logger.info(f"Fetched {len(operators)} operators")
        return operators if operators else _DEFAULT_OPERATORS()

    async def submit_simulation(
        self,
        expression: str,
        region: str | None = None,
        universe: str | None = None,
        delay: int | None = None,
        decay: int = 0,
        neutralization: str | None = None,
        truncation: float | None = None,
    ) -> dict[str, Any]:
        from ..models import SimulationSettings

        settings = SimulationSettings(
            region=region or self.settings.WQ_REGION,
            universe=universe or self.settings.WQ_UNIVERSE,
            delay=delay if delay is not None else self.settings.WQ_DELAY,
            decay=decay,
            neutralization=neutralization or self.settings.WQ_NEUTRALIZATION,
            truncation=truncation or self.settings.WQ_TRUNCATION,
            pasteurization=self.settings.WQ_PASTEURIZATION,
        )
        sim_req = SimulationRequest(settings=settings, regular=expression)

        resp = await self._request("post", "/simulations", json=sim_req.model_dump())
        if resp.status_code == 401:
            self._client = await self.auth.refresh()
            resp = await self._request("post", "/simulations", json=sim_req.model_dump())

        if resp.status_code != 201:
            error_text = resp.text
            if "SIMULATION_LIMIT_EXCEEDED" in error_text:
                return {"status": "error", "message": "SIMULATION_LIMIT_EXCEEDED"}
            return {"status": "error", "message": error_text}

        progress_url = resp.headers.get("location", "")
        if not progress_url:
            return {"status": "error", "message": "No progress URL received"}

        return {
            "status": "success",
            "progress_url": progress_url,
        }

    async def poll_simulation(
        self,
        progress_url: str,
        timeout: float = 1800.0,
        poll_interval: float = 5.0,
    ) -> dict[str, Any]:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            client = await self._ensure_client()
            resp = await client.get(progress_url)

            if resp.status_code == 429:
                retry_after = int(float(resp.headers.get("Retry-After", "10")))
                logger.debug(f"Rate limited during poll, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                continue

            if "SIMULATION_LIMIT_EXCEEDED" in resp.text:
                return {"status": "error", "message": "SIMULATION_LIMIT_EXCEEDED"}

            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = int(float(retry_after))
                    await asyncio.sleep(wait)
                    continue
                except (ValueError, TypeError):
                    pass

            data = resp.json()
            status = data.get("status", "")

            if status == "COMPLETE":
                alpha_id = data.get("alpha")
                alpha_data: dict[str, Any] = {}
                if alpha_id:
                    alpha_data = await self._get_alpha_details(alpha_id)
                return {"status": "complete", "alpha_id": alpha_id, "alpha_data": alpha_data, "sim_result": data}

            if status == "ERROR":
                return {"status": "error", "message": data.get("error", "Unknown simulation error")}

            await asyncio.sleep(poll_interval)

        return {"status": "error", "message": f"Simulation polling timed out after {timeout}s"}

    async def _get_alpha_details(self, alpha_id: str) -> dict[str, Any]:
        resp = await self._request("get", f"/alphas/{alpha_id}")
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"Failed to get alpha details for {alpha_id}: {resp.status_code}")
        return {}

    async def get_all_datasets(
        self,
        region: str | None = None,
        universe: str | None = None,
        delay: int | None = None,
        instrument_type: str = "EQUITY",
    ) -> list[dict[str, Any]]:
        region = region or self.settings.WQ_REGION
        universe = universe or self.settings.WQ_UNIVERSE
        delay = delay if delay is not None else self.settings.WQ_DELAY
        out: list[dict[str, Any]] = []
        offset = 0
        page_size = 50
        while True:
            resp = await self._request(
                "get",
                "/data-sets",
                params={
                    "delay": delay,
                    "instrumentType": instrument_type,
                    "region": region,
                    "universe": universe,
                    "limit": page_size,
                    "offset": offset,
                },
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            results = data.get("results", [])
            out.extend(results)
            if len(results) < page_size:
                break
            offset += page_size
            if offset > 2000:
                break
        return out

    async def get_all_data_fields_paged(
        self,
        dataset_id: str,
        region: str | None = None,
        universe: str | None = None,
        delay: int | None = None,
        instrument_type: str = "EQUITY",
        page_size: int = 50,
        max_pages: int = 100,
    ) -> list[dict[str, Any]]:
        region = region or self.settings.WQ_REGION
        universe = universe or self.settings.WQ_UNIVERSE
        delay = delay if delay is not None else self.settings.WQ_DELAY
        out: list[dict[str, Any]] = []
        offset = 0
        for _ in range(max_pages):
            resp = await self._request(
                "get",
                "/data-fields",
                params={
                    "delay": delay,
                    "instrumentType": instrument_type,
                    "region": region,
                    "universe": universe,
                    "dataset.id": dataset_id,
                    "limit": page_size,
                    "offset": offset,
                },
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            results = data.get("results", [])
            out.extend(results)
            if len(results) < page_size:
                break
            offset += page_size
        return out

    async def get_submitted_alphas(self, limit: int = 100) -> list[dict[str, Any]]:
        resp = await self._request(
            "get",
            "/users/self/alphas",
            params={
                "limit": limit,
                "offset": 0,
                "status!=": "UNSUBMITTED%1FIS-FAIL",
                "order": "-dateCreated",
                "hidden": "false",
            },
        )
        if resp.status_code == 200:
            return resp.json().get("results", [])
        return []

    async def get_alpha_check(self, wq_alpha_id: str) -> list[dict] | None:
        """拉单个 alpha 的最新 IS check 状态（特别是 self_correlation——WQ 异步算的，
        /users/self/alphas list endpoint 返回的常常还是 PENDING，必须打 /alphas/{id}/check
        才能拿到 FAIL/PASS 终态）。
        """
        resp = await self._request("get", f"/alphas/{wq_alpha_id}/check")
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except Exception:
            return None
        is_data = data.get("is") or {}
        return is_data.get("checks")

    async def get_pnl(
        self, wq_alpha_id: str, max_attempts: int = 4,
    ) -> tuple[list[str], list[float]] | None:
        """拉 alpha 的 PnL recordset → （日期, 每日收益）。失败返回 None（fail-open）。

        WQ 在突发请求下会对 recordset 端点返回 **200 + 空 body**（软节流），recordset 也
        可能在异步生成时短暂为空——这两种都退避重试。真正的非-200 / 解析错误直接 None。
        """
        from ..engine.correlation import parse_pnl_response
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await self._request("get", f"/alphas/{wq_alpha_id}/recordsets/pnl")
            except Exception as exc:
                logger.warning(f"get_pnl({wq_alpha_id}) request failed: {exc}")
                return None
            if resp.status_code != 200:
                logger.warning(f"get_pnl({wq_alpha_id}) status {resp.status_code}")
                return None
            if resp.text.strip():
                try:
                    return parse_pnl_response(resp.json())
                except Exception as exc:
                    logger.warning(f"get_pnl({wq_alpha_id}) parse failed: {exc}")
                    return None
            # 200 + 空 body = 节流/未就绪 → 退避重试
            if attempt < max_attempts:
                await asyncio.sleep(attempt)  # 1s, 2s, 3s
        logger.warning(f"get_pnl({wq_alpha_id}) empty body after {max_attempts} attempts")
        return None


def _DEFAULT_OPERATORS() -> list[WQOperator]:
    defaults = [
        ("ts_mean", "Time Series", "Time series mean"),
        ("ts_std_dev", "Time Series", "Time series standard deviation"),
        ("ts_zscore", "Time Series", "Time series z-score"),
        ("ts_rank", "Time Series", "Time series rank"),
        ("ts_delta", "Time Series", "Time series delta"),
        ("ts_delay", "Time Series", "Time series delay"),
        ("ts_sum", "Time Series", "Time series sum"),
        ("ts_min", "Time Series", "Time series min"),
        ("ts_max", "Time Series", "Time series max"),
        ("ts_corr", "Time Series", "Time series correlation"),
        ("ts_decay_linear", "Time Series", "Linear decay"),
        ("ts_regression", "Time Series", "Time series regression"),
        ("rank", "Cross Sectional", "Cross-sectional rank"),
        ("zscore", "Cross Sectional", "Cross-sectional z-score"),
        ("normalize", "Cross Sectional", "Normalize"),
        ("scale", "Cross Sectional", "Scale"),
        ("group_neutralize", "Group", "Group neutralize"),
        ("group_rank", "Group", "Group rank"),
        ("group_zscore", "Group", "Group z-score"),
    ]
    return [WQOperator(name=n, category=c, description=d) for n, c, d in defaults]
