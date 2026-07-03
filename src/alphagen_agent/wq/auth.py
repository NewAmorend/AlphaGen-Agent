from __future__ import annotations


import httpx
from loguru import logger

from ..config import Settings


class WQAuth:
    BASE_URL = "https://api.worldquantbrain.com"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                auth=(self.settings.WQ_USERNAME, self.settings.WQ_PASSWORD),
                timeout=60.0,
            )
        return self._client

    async def authenticate(self) -> httpx.AsyncClient:
        client = await self._get_client()
        resp = await client.post("/authentication")
        if resp.status_code != 201:
            raise AuthenticationError(f"Authentication failed ({resp.status_code}): {resp.text}")
        logger.info("WQ Brain authenticated successfully")
        return client

    async def refresh(self) -> httpx.AsyncClient:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        return await self.authenticate()

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class AuthenticationError(Exception):
    pass
