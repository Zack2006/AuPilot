"""HTTP gateway used by every Streamlit page.

所有 Streamlit 页面共用的 HTTP 网关。

English: Centralizing transport behavior keeps URLs, timeouts, error handling,
and JSON decoding out of presentation code. The client never contains portfolio
decision logic.

中文：集中封装 URL、超时、异常和 JSON 解析，避免页面重复网络代码；本客户端不包含
任何持仓决策逻辑。
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class AurumPilotAPIError(RuntimeError):
    """Readable frontend exception for backend failures. / 面向前端展示的后端调用异常。"""


class AurumPilotAPIClient:
    """Small typed wrapper around the versioned FastAPI endpoints.

    对带版本号的 FastAPI 接口进行轻量封装，统一连接复用和错误转换。
    """

    def __init__(self, base_url: str | None = None) -> None:
        # English: Environment configuration supports both local development and
        # the Docker service name without changing page code.
        # 中文：环境变量同时支持本地地址和 Docker 服务名，无需修改页面代码。
        self.base_url = (base_url or os.getenv("AURUMPILOT_API_URL", "http://127.0.0.1:8000/api/v1")).rstrip("/")
        self._client = httpx.Client(base_url=f"{self.base_url}/", timeout=15)

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Issue a GET request and return decoded JSON. / 发起 GET 请求并返回解析后的 JSON。"""
        return self._request("GET", path, params=params)

    def post(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Issue a JSON POST request. / 发起 JSON POST 请求。"""
        kwargs: dict[str, Any] = {"json": payload or {}}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return self._request("POST", path, **kwargs)

    def patch(self, path: str, payload: dict[str, Any]) -> Any:
        """Issue a JSON PATCH request. / 发起 JSON PATCH 请求。"""
        return self._request("PATCH", path, json=payload)

    def put(self, path: str, payload: dict[str, Any]) -> Any:
        """Issue a JSON PUT request. / 发起 JSON PUT 请求。"""
        return self._request("PUT", path, json=payload)

    def delete(self, path: str) -> Any:
        """Issue a DELETE request. / 发起 DELETE 请求。"""
        return self._request("DELETE", path)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Normalize transport failures into one UI-safe exception.

        将网络错误和非成功响应统一转换为界面可处理的异常。
        """
        try:
            response = self._client.request(method, path.lstrip("/"), **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            detail = getattr(exc.response, "text", "") if isinstance(exc, httpx.HTTPStatusError) else str(exc)
            raise AurumPilotAPIError(detail) from exc


# English: A process-level client reuses TCP connections; it stores no user data.
# 中文：进程级客户端用于复用 TCP 连接，本身不保存任何用户持仓数据。
api = AurumPilotAPIClient()
