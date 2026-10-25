import asyncio
import hashlib
import json
import queue
import random
import threading
import time
from abc import ABC
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    SystemMessage,
    convert_to_openai_messages,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from openai import APIConnectionError, APIError, AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field, PrivateAttr, SecretStr


class ProviderRequestError(RuntimeError):
    def __init__(self, provider, status_code=None, retry_after=None):
        self.status_code = status_code
        self.retry_after = retry_after
        reason = {
            400: "rejected the request parameters",
            401: "authentication failed; check its configured API key",
            403: "access denied; check account permissions",
            404: "model or endpoint is unavailable; check the configured model and URL",
            429: "quota or rate limit reached; wait before retrying",
        }.get(status_code, "request failed or timed out")
        super().__init__(f"{provider} {reason}" + (f" (HTTP {status_code})" if status_code else ""))


_LOCK = threading.Lock()
_LIMITERS = {}
_COOLDOWNS = {}


def _retry_after(headers, default=60.0):
    value = headers.get("retry-after") if headers else None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            seconds = default
    return max(1.0, min(seconds, 86400.0))


def _usage(value):
    if not value:
        return None
    return {
        "input_tokens": value.get("prompt_tokens", 0) or 0,
        "output_tokens": value.get("completion_tokens", 0) or 0,
        "total_tokens": value.get("total_tokens", 0) or 0,
    }


class BaseLLMProvider(BaseChatModel, ABC):
    provider_name: ClassVar[str] = "OPENAI_COMPATIBLE"
    native_json: ClassVar[bool] = False
    model_name: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0, allow_inf_nan=False)
    api_key: SecretStr = Field(exclude=True, repr=False)
    base_url: str
    timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    max_tokens: int = Field(default=2048, ge=1)
    max_retries: int = Field(default=2, ge=0, le=3)
    max_concurrency: int = Field(default=2, ge=1)
    retry_max_wait_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    _client: Any = PrivateAttr(default=None)
    _async_client: Any = PrivateAttr(default=None)
    _async_loop: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self):
        return self.provider_name.lower()

    @property
    def _identifying_params(self):
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "temperature": self.temperature,
        }

    def _identity(self):
        # Share limits across different models/agents using the same provider account.
        return (
            self.provider_name,
            self.base_url.rstrip("/"),
            hashlib.sha256(self.api_key.get_secret_value().encode()).hexdigest(),
        )

    def _limiter(self):
        key = self._identity()
        with _LOCK:
            if key not in _LIMITERS:
                _LIMITERS[key] = threading.BoundedSemaphore(self.max_concurrency)
            return _LIMITERS[key]

    def _check_quota(self):
        with _LOCK:
            remaining = _COOLDOWNS.get(self._identity(), 0) - time.monotonic()
        if remaining > 0:
            raise ProviderRequestError(self.provider_name, 429, remaining)

    def _error(self, exc):
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status is None and response is not None:
            status = response.status_code
        body = getattr(exc, "body", None)
        if status is None and isinstance(body, dict):
            details = body.get("error", body)
            if isinstance(details, dict):
                code = details.get("code") or details.get("type")
                if str(code) in {
                    "429",
                    "rate_limit_exceeded",
                    "insufficient_quota",
                    "RESOURCE_EXHAUSTED",
                }:
                    status = 429
                elif str(code) in {"400", "401", "403", "404", "500", "503"}:
                    status = int(code)
        delay = _retry_after(response.headers) if response is not None else 60
        if status == 429:
            with _LOCK:
                key = self._identity()
                _COOLDOWNS[key] = max(_COOLDOWNS.get(key, 0), time.monotonic() + delay)
        return ProviderRequestError(self.provider_name, status, delay if status == 429 else None)

    def _retry_delay(self, exc, attempt):
        status = getattr(exc, "status_code", None)
        if status is None and getattr(exc, "response", None) is not None:
            status = exc.response.status_code
        if status == 429 or attempt >= self.max_retries:
            return None
        if status not in (408, 409, 500, 502, 503, 504) and not isinstance(
            exc, (APIConnectionError, httpx.TransportError)
        ):
            return None
        response = getattr(exc, "response", None)
        backoff = min(self.retry_max_wait_seconds, 0.5 * 2**attempt + random.uniform(0, 0.25))
        if response is not None and response.headers.get("retry-after"):
            required = _retry_after(response.headers)
            # Never sleep less than Retry-After, nor beyond the configured retry budget.
            return required if required <= self.retry_max_wait_seconds else None
        return backoff

    def _sdk(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key.get_secret_value(), base_url=self.base_url, max_retries=0
            )
        return self._client

    def _async_sdk(self):
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_loop is not loop:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key.get_secret_value(), base_url=self.base_url, max_retries=0
            )
            self._async_loop = loop
        return self._async_client

    def _parameters(self, messages, stop, stream, kwargs):
        # Forward only supported standard parameters, never arbitrary SDK kwargs.
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": convert_to_openai_messages(messages),
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if stop:
            params["stop"] = stop
        if self.native_json and kwargs.get("response_format"):
            params["response_format"] = kwargs["response_format"]
        return params

    def _send(self, params, remaining):
        return self._sdk().chat.completions.create(**params, timeout=remaining)

    async def _asend(self, params, remaining):
        return await self._async_sdk().chat.completions.create(**params, timeout=remaining)

    def _request(self, params, deadline, abandoned=None):
        for attempt in range(self.max_retries + 1):
            if abandoned is not None and abandoned.is_set():
                raise ProviderRequestError(self.provider_name)
            self._check_quota()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderRequestError(self.provider_name) from TimeoutError()
            try:
                return self._send(params, remaining)
            except (APIError, httpx.HTTPError) as exc:
                public = self._error(exc)
                delay = self._retry_delay(exc, attempt)
                if delay is None or delay >= deadline - time.monotonic():
                    raise public from exc
                if abandoned is None:
                    time.sleep(delay)
                else:
                    abandoned.wait(delay)

    async def _arequest(self, params, deadline):
        for attempt in range(self.max_retries + 1):
            self._check_quota()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderRequestError(self.provider_name) from TimeoutError()
            try:
                return await self._asend(params, remaining)
            except (APIError, httpx.HTTPError) as exc:
                public = self._error(exc)
                delay = self._retry_delay(exc, attempt)
                if delay is None or delay >= deadline - time.monotonic():
                    raise public from exc
                await asyncio.sleep(delay)

    async def _acquire(self, limiter, deadline):
        while not limiter.acquire(blocking=False):
            if time.monotonic() >= deadline:
                raise TimeoutError("Provider concurrency wait timed out")
            await asyncio.sleep(0.01)

    @staticmethod
    def _data(value):
        return value if isinstance(value, dict) else value.model_dump()

    def _result(self, value):
        data = self._data(value)
        choices = data.get("choices") or []
        if not choices:
            raise OutputParserException("The provider returned no completion")
        choice = choices[0]
        message = choice.get("message") or {}
        if message.get("refusal") or choice.get("finish_reason") in {"length", "content_filter"}:
            raise OutputParserException("The provider refused or truncated the required result")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OutputParserException("The provider returned no answer text")
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=content,
                        usage_metadata=_usage(data.get("usage")),
                        response_metadata={
                            "provider": self.provider_name,
                            "model_name": self.model_name,
                            "finish_reason": choice.get("finish_reason"),
                        },
                    )
                )
            ]
        )

    def _sync_results(self, params, deadline):
        # A timed-out synchronous request keeps its slot until the socket operation finishes.
        limiter = self._limiter()
        if not limiter.acquire(timeout=max(0, deadline - time.monotonic())):
            raise ProviderRequestError(self.provider_name) from TimeoutError()
        events = queue.Queue(maxsize=1)
        abandoned = threading.Event()

        def publish(kind, value=None):
            while not abandoned.is_set():
                try:
                    events.put((kind, value), timeout=0.02)
                    return
                except queue.Full:
                    pass

        def work():
            response = None
            try:
                response = self._request(params, deadline, abandoned)
                if params["stream"]:
                    iterator = iter(response)
                    while not abandoned.is_set():
                        if time.monotonic() >= deadline:
                            raise ProviderRequestError(self.provider_name) from TimeoutError()
                        try:
                            value = next(iterator)
                        except StopIteration:
                            break
                        if abandoned.is_set():
                            break
                        publish("value", self._chunk(value))
                else:
                    publish("value", self._result(response))
            except (APIError, httpx.HTTPError) as exc:
                public = self._error(exc)
                public.__cause__ = exc
                publish("error", public)
            except BaseException as exc:
                publish("error", exc)
            finally:
                try:
                    if params["stream"] and response is not None:
                        with suppress(Exception):
                            response.close()
                finally:
                    limiter.release()
                publish("done")

        threading.Thread(target=work, daemon=True, name="wizora-provider-request").start()
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderRequestError(self.provider_name) from TimeoutError()
                try:
                    kind, value = events.get(timeout=remaining)
                except queue.Empty:
                    raise ProviderRequestError(self.provider_name) from TimeoutError()
                if kind == "done":
                    return
                if kind == "error":
                    raise value
                yield value
        finally:
            abandoned.set()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        deadline = time.monotonic() + self.timeout_seconds
        return list(self._sync_results(self._parameters(messages, stop, False, kwargs), deadline))[
            0
        ]

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        deadline = time.monotonic() + self.timeout_seconds
        limiter = self._limiter()
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                await self._acquire(limiter, deadline)
                try:
                    return self._result(
                        await self._arequest(
                            self._parameters(messages, stop, False, kwargs), deadline
                        )
                    )
                finally:
                    limiter.release()
        except TimeoutError as exc:
            raise ProviderRequestError(self.provider_name) from exc

    def _chunk(self, value):
        data = self._data(value)
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        if choice.get("finish_reason") in {"length", "content_filter"}:
            raise OutputParserException("The provider truncated the streamed result")
        delta = choice.get("delta") or {}
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content=delta.get("content") or "", usage_metadata=_usage(data.get("usage"))
            )
        )

    def _stream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        deadline = time.monotonic() + self.timeout_seconds
        results = self._sync_results(self._parameters(messages, stop, True, kwargs), deadline)
        try:
            for chunk in results:
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk
        finally:
            results.close()

    async def _astream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> AsyncIterator[ChatGenerationChunk]:
        deadline = time.monotonic() + self.timeout_seconds
        limiter = self._limiter()
        acquired = False
        stream = None
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                await self._acquire(limiter, deadline)
                acquired = True
                stream = await self._arequest(
                    self._parameters(messages, stop, True, kwargs), deadline
                )
            iterator = stream.__aiter__()
            while True:
                try:
                    # Exit the timeout context before yielding: consumer/UI work
                    # must never receive cancellation scheduled by this generator.
                    value = await asyncio.wait_for(
                        anext(iterator), max(0, deadline - time.monotonic())
                    )
                except StopAsyncIteration:
                    break
                chunk = self._chunk(value)
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk
        except TimeoutError as exc:
            raise ProviderRequestError(self.provider_name) from exc
        except (APIError, httpx.HTTPError) as exc:
            raise self._error(exc) from exc
        finally:
            try:
                if stream is not None:
                    with suppress(Exception):
                        await asyncio.wait_for(stream.close(), timeout=1)
            finally:
                if acquired:
                    limiter.release()

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        if kwargs:
            raise ValueError("Unsupported structured-output options")
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("Structured output requires the application's Pydantic model class")
        instructions = (
            "Return exactly one JSON object matching this JSON schema. "
            "Do not use markdown, prose, or extra fields.\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )

        def prepare(value):
            return [SystemMessage(content=instructions), *self._convert_input(value).to_messages()]

        def parse(message):
            try:
                parsed = schema.model_validate_json(message.content)
            except (ValueError, TypeError) as exc:
                if include_raw:
                    return {"raw": message, "parsed": None, "parsing_error": exc}
                raise
            return (
                {"raw": message, "parsed": parsed, "parsing_error": None} if include_raw else parsed
            )

        parameters = self._structured_parameters(schema)
        bound = self.bind(**parameters) if parameters else self
        return RunnableLambda(prepare) | bound | RunnableLambda(parse)

    def _structured_parameters(self, schema):
        return {"response_format": {"type": "json_object"}} if self.native_json else {}

    @staticmethod
    def _schema_parameters(schema):
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        }
