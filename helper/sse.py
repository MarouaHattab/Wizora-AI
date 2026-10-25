import asyncio
import json
import math
import sqlite3
import time
import uuid
from contextlib import suppress
from typing import Literal

import anyio
from fastapi import HTTPException
from google.api_core.exceptions import (
    DeadlineExceeded,
    Forbidden,
    ServiceUnavailable,
    TooManyRequests,
    Unauthorized,
)
from httpx import TimeoutException
from pydantic import ValidationError

from chains import ChainInputError, ProviderExecutionError, StructuredOutputError
from helper.failures import StorageUnavailable

EventVersion = Literal["legacy", "v1"]
TERMINAL_EVENTS = {"awaiting_feedback", "completed", "failed"}


def encode_event(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class EventStream:
    def __init__(self, thread_id, event_version: EventVersion = "legacy"):
        self.thread_id = thread_id
        self.event_version = event_version
        self.run_id = str(uuid.uuid4())
        self.sequence = 0
        self.terminal = False

    def emit(self, event, *, stage=None, data=None, legacy=None):
        if self.terminal:
            return ""
        terminal = event in TERMINAL_EVENTS
        if self.event_version == "legacy" and legacy is None:
            return ""
        self.sequence += 1
        self.terminal = terminal
        if self.event_version == "v1":
            return encode_event(
                {
                    "version": 1,
                    "event": event,
                    "thread_id": self.thread_id,
                    "run_id": self.run_id,
                    "sequence": self.sequence,
                    "stage": stage,
                    "data": data or {},
                }
            )
        return encode_event({**legacy, "terminal": terminal})

    def transport_closed(self, payload):
        if self.event_version == "legacy":
            return encode_event({**payload, "terminal": False})
        return ""


def public_error(error):
    from helper.provider_status import ProviderUnavailableError

    cause = error
    for _ in range(8):
        response = getattr(cause, "response", None)
        status = getattr(cause, "status_code", None) or getattr(response, "status_code", None)
        if status == 429:
            result = {
                "category": "quota",
                "detail": "The configured model's rate limit or quota has been reached. Wait for it to reset before trying again.",
            }
            try:
                delay = float(
                    getattr(cause, "retry_after", None) or response.headers.get("retry-after", "")
                )
                if math.isfinite(delay) and 0 < delay <= 604800:
                    result["retry_after_seconds"] = math.ceil(delay)
            except (ValueError, TypeError, AttributeError):
                pass
            return (
                result  # Never expose provider payloads, credentials or organization identifiers.
            )
        if status in {401, 403}:
            return {
                "category": "provider_auth",
                "detail": "The configured model rejected its credentials or permissions. Check the backend API key and model access.",
            }
        if status == 404:
            return {
                "category": "provider_unavailable",
                "detail": "The configured model or endpoint is unavailable. Check its model ID and backend provider URL.",
            }
        if isinstance(cause, ProviderUnavailableError):
            return {"category": "provider_unavailable", "detail": str(cause)}
        if isinstance(cause, (TimeoutError, TimeoutException)):
            return {"category": "timeout", "detail": "The workflow timed out"}
        if isinstance(cause, TooManyRequests):
            return {
                "category": "quota",
                "detail": "The Google model quota has been reached. Wait for it to reset or use another configured model.",
            }
        if isinstance(cause, DeadlineExceeded):
            return {
                "category": "timeout",
                "detail": "The Google model request timed out. Please try again.",
            }
        if isinstance(cause, ServiceUnavailable):
            return {
                "category": "provider_unavailable",
                "detail": "The Google model is temporarily unavailable. Please try again.",
            }
        if isinstance(cause, (Unauthorized, Forbidden)):
            return {
                "category": "provider_auth",
                "detail": "Google rejected the model credentials or permissions. Check the backend API key and model access.",
            }
        cause = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
        if cause is None:
            break
    if isinstance(error, (StorageUnavailable, sqlite3.OperationalError)):
        return {"category": "storage", "detail": "Workflow storage is unavailable"}
    if isinstance(error, HTTPException) and error.status_code == 503:
        return {"category": "storage", "detail": "Workflow storage is unavailable"}
    if isinstance(error, HTTPException) and error.status_code == 409:
        return {
            "category": "conflict",
            "detail": "The session request conflicted with the current state",
        }
    if isinstance(error, (HTTPException, ChainInputError)):
        return {"category": "input", "detail": "The workflow input could not be accepted"}
    if isinstance(error, StructuredOutputError):
        return {"category": "structured_output", "detail": "The model returned an invalid result"}
    if isinstance(error, (ValidationError, ValueError, TypeError)):
        return {"category": "validation", "detail": "The workflow returned an invalid result"}
    if isinstance(error, ProviderExecutionError):
        return {
            "category": "provider",
            "detail": "The model provider could not complete the workflow",
        }
    return {"category": "workflow", "detail": "The workflow could not be completed"}


async def before_deadline(awaitable, deadline):
    return await asyncio.wait_for(awaitable, timeout=max(0, deadline - time.monotonic()))


async def events_before(iterator, deadline):
    while True:
        try:
            event = await before_deadline(anext(iterator), deadline)
        except StopAsyncIteration:
            return
        yield event


async def close_events(iterator):
    if iterator is not None:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            with anyio.move_on_after(5, shield=True):
                with suppress(Exception):
                    await close()
