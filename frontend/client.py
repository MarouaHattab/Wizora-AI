import asyncio
import json
import math
import mimetypes
import re
from collections.abc import AsyncIterator
from contextlib import ExitStack, aclosing
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import quote, urlsplit

import httpx
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrontendSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env", extra="ignore"
    )

    FRONTEND_BACKEND_URL: str = "http://127.0.0.1:8000"
    FRONTEND_HOST: str = "127.0.0.1"
    FRONTEND_PORT: int = Field(default=7860, ge=1, le=65535)
    FRONTEND_TIMEOUT_SECONDS: float = Field(default=600, gt=0, allow_inf_nan=False)
    FRONTEND_UPLOAD_TIMEOUT_SECONDS: float = Field(default=1800, gt=0, allow_inf_nan=False)
    MAX_UPLOAD_BYTES: int = Field(default=268435456, gt=0)
    MAX_BATCH_FILES: int = Field(default=32, gt=0, le=128)
    MAX_BATCH_BYTES: int = Field(default=1073741824, gt=0)

    @field_validator("FRONTEND_BACKEND_URL")
    @classmethod
    def backend_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        try:
            parsed.port
        except ValueError:
            raise ValueError("Backend URL has an invalid port") from None
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Backend URL must be an HTTP(S) URL without credentials, query, or fragment"
            )
        return value.rstrip("/")


@lru_cache(maxsize=1)
def get_frontend_settings() -> FrontendSettings:
    return FrontendSettings()


Workflow = Literal["question_generation", "summarization", "qa"]
_TERMINAL_EVENTS = {"awaiting_feedback", "completed", "failed"}
_STREAM_EVENTS = _TERMINAL_EVENTS | {"session_started", "stage_started", "text_delta"}
_MAX_EVENT_BYTES = 1024 * 1024


class BackendError(Exception):
    def __init__(self, detail: str, status_code: int | None = None):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


def _error_detail(response: httpx.Response) -> str:
    default = f"Backend request failed (HTTP {response.status_code})"
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return default
    if isinstance(detail, str) and detail.strip():
        return detail[:2000]
    if isinstance(detail, dict):
        reason = detail.get("reason") or detail.get("detail") or detail.get("message")
        if isinstance(reason, str) and reason.strip():
            return reason[:2000]
    if isinstance(detail, list):
        messages = [item.get("msg") for item in detail if isinstance(item, dict)]
        text = "; ".join(item for item in messages if isinstance(item, str))
        if text:
            return text[:2000]
    return default


def _segment(value: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None
    ):
        raise BackendError("An asset or session identifier is required", 422)
    return quote(value, safe="")


async def _sse_data(response: httpx.Response) -> AsyncIterator[str]:
    buffer = ""
    data: list[str] = []
    event_bytes = 0
    first_line = True

    def consume(line: str) -> str | None:
        nonlocal event_bytes, first_line
        if first_line:
            line = line.removeprefix("\ufeff")
            first_line = False
        event_bytes += len(line.encode("utf-8")) + 1
        if event_bytes > _MAX_EVENT_BYTES:
            raise BackendError("The backend stream event exceeded the size limit")
        if not line:
            payload = "\n".join(data) if data else None
            data.clear()
            event_bytes = 0
            return payload
        if line.startswith(":"):
            return None
        name, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if name == "data":
            data.append(value)
        return None

    async for chunk in response.aiter_text():
        buffer += chunk
        while True:
            positions = [
                position for position in (buffer.find("\n"), buffer.find("\r")) if position >= 0
            ]
            if not positions:
                break
            position = min(positions)
            if buffer[position] == "\r" and position == len(buffer) - 1:
                break
            separator_length = 2 if buffer[position : position + 2] == "\r\n" else 1
            line, buffer = buffer[:position], buffer[position + separator_length :]
            payload = consume(line)
            if payload is not None:
                yield payload
        if event_bytes + len(buffer.encode("utf-8")) > _MAX_EVENT_BYTES:
            raise BackendError("The backend stream event exceeded the size limit")
    if buffer:
        payload = consume(buffer.removesuffix("\r"))
        if buffer.endswith("\r") and payload is not None:
            yield payload
    # Incomplete events at EOF are discarded, as required by the SSE framing contract.


class BackendClient:
    def __init__(self, settings: FrontendSettings | None = None):
        self.settings = settings or get_frontend_settings()
        self.base_url = self.settings.FRONTEND_BACKEND_URL

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.FRONTEND_TIMEOUT_SECONDS, connect=10),
            follow_redirects=False,
        )

    def _url(self, path: str) -> str:
        return self.base_url + path

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            async with self._client() as client:
                response = await client.request(method, self._url(path), **kwargs)
        except httpx.TimeoutException:
            raise BackendError(
                "The backend request timed out; check the session before retrying", 504
            ) from None
        except httpx.RequestError:
            raise BackendError(
                "The backend is unavailable; check its address and service status", 503
            ) from None
        if not response.is_success:
            raise BackendError(_error_detail(response), response.status_code)
        try:
            payload = response.json()
        except ValueError:
            raise BackendError("The backend returned an invalid JSON response", 502) from None
        if not isinstance(payload, dict):
            raise BackendError("The backend returned an unexpected response", 502)
        return payload

    async def upload_configuration(self) -> dict[str, Any]:
        return await self._request("GET", "/api/v1/preprocessing/assets/config")

    async def _upload(
        self, path: str | Path, endpoint: str, vlm_mode: str = "disabled"
    ) -> dict[str, Any]:
        if vlm_mode not in {"disabled", "automatic", "enhanced"}:
            raise BackendError("Choose Disabled, Automatic or Enhanced visual understanding", 422)
        upload_path = Path(path)
        try:
            if upload_path.stat().st_size > self.settings.MAX_UPLOAD_BYTES:
                raise BackendError("The selected file exceeds the upload size limit", 413)
            with upload_path.open("rb") as file:
                content_type = (
                    mimetypes.guess_type(upload_path.name)[0] or "application/octet-stream"
                )
                return await self._request(
                    "POST",
                    endpoint,
                    files={"file": (upload_path.name, file, content_type)},
                    data={"vlm_mode": vlm_mode},
                )
        except OSError:
            raise BackendError("The selected upload file is unavailable", 422) from None

    async def upload(self, path: str | Path, vlm_mode: str = "disabled") -> dict[str, Any]:
        return await self._upload(path, "/api/v1/preprocessing/upload_file/", vlm_mode)

    async def queue_upload(self, path: str | Path, vlm_mode: str = "disabled") -> dict[str, Any]:
        return await self._upload(path, "/api/v1/preprocessing/uploads", vlm_mode)

    async def get_upload(self, asset_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/api/v1/preprocessing/uploads/{_segment(asset_id)}")

    async def queue_batch(self, paths, vlm_mode="disabled", file_configs=None):
        if vlm_mode not in {"disabled", "automatic", "enhanced"}:
            raise BackendError("Choose a valid visual processing mode", 422)
        if not paths or len(paths) > self.settings.MAX_BATCH_FILES:
            raise BackendError(f"Select between 1 and {self.settings.MAX_BATCH_FILES} files", 422)
        try:
            uploads = [Path(path) for path in paths]
            sizes = [path.stat().st_size for path in uploads]
            if (
                any(size > self.settings.MAX_UPLOAD_BYTES for size in sizes)
                or sum(sizes) > self.settings.MAX_BATCH_BYTES
            ):
                raise BackendError("The selected files exceed the upload size limit", 413)
            with ExitStack() as stack:
                files = [
                    (
                        "files",
                        (
                            path.name,
                            stack.enter_context(path.open("rb")),
                            mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                        ),
                    )
                    for path in uploads
                ]
                return await self._request(
                    "POST",
                    "/api/v1/preprocessing/batches",
                    files=files,
                    data={
                        "vlm_mode": vlm_mode,
                        "file_configs": json.dumps(file_configs or []),
                    },
                )
        except OSError:
            raise BackendError("A selected upload file is unavailable", 422) from None

    async def get_batch(self, batch_id):
        return await self._request("GET", f"/api/v1/preprocessing/batches/{_segment(batch_id)}")

    async def resume_failed_batch(self, batch_id):
        return await self._request(
            "POST", f"/api/v1/preprocessing/batches/{_segment(batch_id)}/resume"
        )

    async def provider_status(self, refresh=False):
        return await self._request(
            "GET", "/api/v1/providers/status", params={"refresh": str(bool(refresh)).lower()}
        )

    async def batch_events(
        self, paths, *, vlm_mode="disabled", file_configs=None, poll_interval=1, timeout=None
    ):
        timeout = self.settings.FRONTEND_UPLOAD_TIMEOUT_SECONDS if timeout is None else timeout
        if (
            not math.isfinite(poll_interval)
            or poll_interval < 0
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise BackendError("Invalid processing polling interval or timeout", 422)
        batch = await self.queue_batch(paths, vlm_mode, file_configs)
        batch_id = batch.get("id")
        _segment(batch_id)
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if batch.get("id") != batch_id or batch.get("status") not in {
                "queued",
                "processing",
                "ready",
                "partial",
                "failed",
            }:
                raise BackendError("The backend returned an invalid batch status", 502)
            yield batch
            if batch["status"] in {"ready", "partial", "failed"}:
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise BackendError(
                    f"Processing continues for batch {batch_id}. Use Resume Processing Status to check again.",
                    504,
                )
            await asyncio.sleep(min(poll_interval, remaining))
            batch = await self.get_batch(batch_id)

    async def list_jobs(self, file_id):
        return await self._request("GET", f"/api/v1/preprocessing/assets/{_segment(file_id)}/jobs")

    async def get_job(self, file_id, job_id):
        return await self._request(
            "GET", f"/api/v1/preprocessing/assets/{_segment(file_id)}/jobs/{_segment(job_id)}"
        )

    async def _download(self, method, path, **kwargs):
        try:
            async with self._client() as client:
                response = await client.request(method, self._url(path), **kwargs)
        except httpx.RequestError:
            raise BackendError("The results download could not be completed", 503) from None
        if not response.is_success:
            raise BackendError(_error_detail(response), response.status_code)
        return response.content

    async def download_job(self, file_id, job_id):
        return await self._download(
            "GET",
            f"/api/v1/preprocessing/assets/{_segment(file_id)}/jobs/{_segment(job_id)}/download",
        )

    async def export_jobs(self, file_ids):
        for file_id in file_ids:
            _segment(file_id)
        return await self._download(
            "POST", "/api/v1/preprocessing/job-results/export", json={"file_ids": file_ids}
        )

    async def upload_events(
        self,
        path: str | Path,
        *,
        vlm_mode: str = "disabled",
        poll_interval: float = 1,
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        timeout = self.settings.FRONTEND_UPLOAD_TIMEOUT_SECONDS if timeout is None else timeout
        if (
            not math.isfinite(poll_interval)
            or poll_interval < 0
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise BackendError("Invalid upload polling interval or timeout", 422)
        job = await self.queue_upload(path, vlm_mode=vlm_mode)
        asset_id = job.get("id")
        _segment(asset_id)
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if job.get("id") != asset_id:
                raise BackendError("The backend returned a different upload identifier", 502)
            if job.get("status") not in {"queued", "processing", "ready", "failed"}:
                raise BackendError("The backend returned an invalid upload status", 502)
            yield job
            if job["status"] == "ready":
                return
            if job["status"] == "failed":
                error = job.get("error") or {}
                detail = error.get("detail") if isinstance(error, dict) else error
                raise BackendError(
                    str(detail or "File processing failed; check the backend logs"), 422
                )
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise BackendError(
                    f"Processing is still running for Asset ID {asset_id}. Keep this ID and use Load Extracted Text to check again.",
                    504,
                )
            await asyncio.sleep(min(poll_interval, remaining))
            job = await self.get_upload(asset_id)

    async def get_asset(self, asset_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/api/v1/preprocessing/assets/{_segment(asset_id)}")

    async def extracted_text(self, asset_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"/api/v1/preprocessing/get_extracted_text/{_segment(asset_id)}"
        )

    async def get_content(self, asset_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"/api/v1/preprocessing/assets/{_segment(asset_id)}/content"
        )

    async def get_session(self, workflow: Workflow, thread_id: str) -> dict[str, Any]:
        if workflow not in {"question_generation", "summarization", "qa"}:
            raise BackendError("Unsupported workflow", 422)
        return await self._request("GET", f"/api/v1/sessions/{workflow}/{_segment(thread_id)}")

    async def saved_questions(self, asset_id: str) -> dict[str, Any]:
        _segment(asset_id)
        return await self._request(
            "GET", "/api/v1/sessions/question_generation", params={"asset_id": asset_id}
        )

    async def start_question(
        self,
        asset_id: str,
        question_type: str = "MCQ",
        difficulty: str = "medium",
        question_count: int = 1,
    ) -> dict[str, Any]:
        _segment(asset_id)
        if question_type not in {"MCQ", "T/F"}:
            raise BackendError("Choose MCQ or T/F", 422)
        if (
            difficulty not in {"easy", "medium", "hard"}
            or type(question_count) is not int
            or not 1 <= question_count <= 20
        ):
            raise BackendError("Choose a difficulty and between 1 and 20 questions", 422)
        return await self._request(
            "POST",
            "/api/v1/graph/qg/start_session",
            json={
                "asset_id": asset_id,
                "question_type": question_type,
                "difficulty": difficulty,
                "question_count": question_count,
            },
        )

    async def feedback_question(
        self,
        thread_id: str,
        feedback: str,
        review_id: str | None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/graph/qg/provide_feedback",
            json={
                "thread_id": thread_id,
                "feedback": feedback,
                "review_id": review_id,
            },
        )

    async def _events(self, path: str, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        try:
            async with self._client() as client:
                async with client.stream(
                    "POST", self._url(path), json={**payload, "event_version": "v1"}
                ) as response:
                    if not response.is_success:
                        await response.aread()
                        raise BackendError(_error_detail(response), response.status_code)
                    if (
                        response.headers.get("content-type", "").split(";", 1)[0].strip()
                        != "text/event-stream"
                    ):
                        raise BackendError("The backend did not return an event stream", 502)
                    terminal = False
                    sequence = 0
                    thread_id = None
                    run_id = None
                    async for data in _sse_data(response):
                        try:
                            event = json.loads(data)
                        except ValueError:
                            raise BackendError(
                                "The backend stream contained an invalid event", 502
                            ) from None
                        if (
                            not isinstance(event, dict)
                            or event.get("version") != 1
                            or isinstance(event.get("version"), bool)
                            or not isinstance(event.get("event"), str)
                            or event.get("event") not in _STREAM_EVENTS
                            or not isinstance(event.get("data"), dict)
                            or not isinstance(event.get("sequence"), int)
                            or isinstance(event.get("sequence"), bool)
                            or event["sequence"] != sequence + 1
                            or not isinstance(event.get("thread_id"), str)
                            or not event["thread_id"]
                            or not isinstance(event.get("run_id"), str)
                            or not event["run_id"]
                        ):
                            raise BackendError(
                                "The backend stream violated its event contract", 502
                            )
                        if terminal:
                            raise BackendError(
                                "The backend stream emitted data after completion", 502
                            )
                        if sequence == 0:
                            if event["event"] != "session_started":
                                raise BackendError(
                                    "The backend stream did not identify its session", 502
                                )
                            thread_id, run_id = event["thread_id"], event["run_id"]
                            if payload.get("thread_id") and thread_id != payload["thread_id"]:
                                raise BackendError(
                                    "The backend stream returned a different session", 502
                                )
                            if (
                                payload.get("asset_id")
                                and event["data"].get("asset_id", payload["asset_id"])
                                != payload["asset_id"]
                            ):
                                raise BackendError(
                                    "The backend stream returned a different file", 502
                                )
                        if event["thread_id"] != thread_id or event["run_id"] != run_id:
                            raise BackendError(
                                "The backend stream changed session identifiers", 502
                            )
                        sequence = event["sequence"]
                        terminal = event["event"] in _TERMINAL_EVENTS
                        yield event
                    if not terminal:
                        raise BackendError(
                            "The backend stream ended before the workflow completed; reload the session",
                            502,
                        )
        except httpx.TimeoutException:
            raise BackendError(
                "The backend stream timed out; reload the session before retrying", 504
            ) from None
        except httpx.RequestError:
            raise BackendError(
                "The backend connection was interrupted; reload the session before retrying", 503
            ) from None

    async def summary_events(
        self,
        *,
        asset_id: str | None = None,
        thread_id: str | None = None,
        feedback: str | None = None,
        review_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if thread_id:
            if feedback is None:
                raise BackendError("Feedback is required to resume a summary", 422)
            path = "/api/v1/graph/summarizer/provide_feedback_streaming"
            payload = {"thread_id": thread_id, "feedback": feedback, "review_id": review_id}
        else:
            if not asset_id:
                raise BackendError("Select an educational asset", 422)
            _segment(asset_id)
            path = "/api/v1/graph/summarizer/start_session_streaming"
            payload = {"asset_id": asset_id}
        async with aclosing(self._events(path, payload)) as events:
            async for event in events:
                yield event

    async def qa_events(
        self,
        *,
        asset_id: str | None = None,
        question: str,
        thread_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if not question.strip():
            raise BackendError("Enter a question", 422)
        if thread_id:
            path = "/api/v1/graph/qa/continue_conversation_stream"
            payload = {"thread_id": thread_id, "next_question": question}
        else:
            if not asset_id:
                raise BackendError("Select an educational asset", 422)
            _segment(asset_id)
            path = "/api/v1/graph/qa/start_session_stream"
            payload = {"asset_id": asset_id, "initial_question": question}
        async with aclosing(self._events(path, payload)) as events:
            async for event in events:
                yield event

    def source_url(
        self,
        asset_id: str,
        reference: str,
        *,
        page: int | None = None,
        timestamp: float | None = None,
    ) -> str:
        if (
            not isinstance(reference, str)
            or not reference
            or "\\" in reference
            or reference.startswith("/")
            or any(ord(char) < 32 for char in reference)
            or any(part in {"", ".", ".."} for part in reference.split("/"))
        ):
            raise BackendError("The source reference is invalid", 422)
        path = str(PurePosixPath(reference))
        url = self._url(
            f"/api/v1/preprocessing/assets/{_segment(asset_id)}/files/{quote(path, safe='/')}"
        )
        if page is not None:
            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise BackendError("The source page is invalid", 422)
            return f"{url}#page={page}"
        if timestamp is not None:
            if (
                not isinstance(timestamp, (int, float))
                or isinstance(timestamp, bool)
                or not math.isfinite(timestamp)
                or timestamp < 0
            ):
                raise BackendError("The source timestamp is invalid", 422)
            return f"{url}#t={timestamp:g}"
        return url
