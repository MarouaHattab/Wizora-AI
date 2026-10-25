import asyncio
import json
from typing import Any, ClassVar

import httpx
from pydantic import Field, PrivateAttr

from .base import BaseLLMProvider, ProviderRequestError


def _completion(data):
    if data.get("error"):
        raise ProviderRequestError("OLLAMA")
    content = data.get("message", {}).get("content", "")
    usage = {
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return {
        "choices": [
            {
                "message": {"content": content},
                "delta": {"content": content},
                "finish_reason": data.get("done_reason"),
            }
        ],
        "usage": usage,
    }


class _NativeStream:
    def __init__(self, context, response):
        self.context, self.response = context, response

    def __iter__(self):
        for line in self.response.iter_lines():
            if line:
                yield _completion(json.loads(line))

    def close(self):
        self.context.__exit__(None, None, None)


class _AsyncNativeStream:
    def __init__(self, context, response):
        self.context, self.response = context, response

    async def __aiter__(self):
        async for line in self.response.aiter_lines():
            if line:
                yield _completion(json.loads(line))

    async def close(self):
        await self.context.__aexit__(None, None, None)


class OllamaProvider(BaseLLMProvider):
    provider_name: ClassVar[str] = "OLLAMA"
    base_url: str = "http://127.0.0.1:11434/v1"
    num_ctx: int | None = Field(default=None, ge=1)
    native_json: ClassVar[bool] = True
    _native_client: Any = PrivateAttr(default=None)
    _native_async_client: Any = PrivateAttr(default=None)
    _native_async_loop: Any = PrivateAttr(default=None)

    def _native_payload(self, params):
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.max_tokens,
        }
        if "stop" in params:
            options["stop"] = params["stop"]
        payload = {
            "model": self.model_name,
            "messages": params["messages"],
            "stream": params["stream"],
            "think": False,
            "options": options,
        }
        if "response_format" in params:
            response_format = params["response_format"]
            payload["format"] = response_format.get("json_schema", {}).get("schema", "json")
        return payload

    def _structured_parameters(self, schema):
        return self._schema_parameters(schema)

    def _headers(self):
        key = self.api_key.get_secret_value()
        return {"Authorization": f"Bearer {key}"} if key and key != "ollama" else {}

    def _native_url(self):
        return self.base_url.rstrip("/").removesuffix("/v1") + "/api/chat"

    def _send(self, params, remaining):
        if self.num_ctx is None:
            return super()._send(params, remaining)
        if self._native_client is None:
            self._native_client = httpx.Client(headers=self._headers())
        payload = self._native_payload(params)
        if params["stream"]:
            context = self._native_client.stream(
                "POST", self._native_url(), json=payload, timeout=remaining
            )
            response = context.__enter__()
            try:
                response.raise_for_status()
            except BaseException:
                context.__exit__(None, None, None)
                raise
            return _NativeStream(context, response)
        response = self._native_client.post(self._native_url(), json=payload, timeout=remaining)
        response.raise_for_status()
        return _completion(response.json())

    async def _asend(self, params, remaining):
        if self.num_ctx is None:
            return await super()._asend(params, remaining)
        loop = asyncio.get_running_loop()
        if self._native_async_client is None or self._native_async_loop is not loop:
            self._native_async_client = httpx.AsyncClient(headers=self._headers())
            self._native_async_loop = loop
        payload = self._native_payload(params)
        if params["stream"]:
            context = self._native_async_client.stream(
                "POST", self._native_url(), json=payload, timeout=remaining
            )
            response = await context.__aenter__()
            try:
                response.raise_for_status()
            except BaseException:
                await context.__aexit__(None, None, None)
                raise
            return _AsyncNativeStream(context, response)
        response = await self._native_async_client.post(
            self._native_url(), json=payload, timeout=remaining
        )
        response.raise_for_status()
        return _completion(response.json())
