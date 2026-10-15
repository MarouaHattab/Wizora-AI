import asyncio
import json
import time
import uuid
from contextlib import suppress
from typing import Literal

import anyio
from fastapi import HTTPException
from httpx import TimeoutException
from pydantic import ValidationError

from chains.contracts import ChainInputError

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
            return encode_event({
                "version": 1, "event": event, "thread_id": self.thread_id,
                "run_id": self.run_id, "sequence": self.sequence,
                "stage": stage, "data": data or {},
            })
        return encode_event({**legacy, "terminal": terminal})

    def transport_closed(self, payload):
        if self.event_version == "legacy":
            return encode_event({**payload, "terminal": False})
        return ""


def public_error(error):
    cause = error
    for _ in range(4):
        if isinstance(cause, (TimeoutError, TimeoutException)):
            return {"category": "timeout", "detail": "The workflow timed out"}
        cause = getattr(cause, "__cause__", None)
        if cause is None:
            break
    if isinstance(error, (HTTPException, ChainInputError)):
        return {"category": "input", "detail": "The workflow input could not be accepted"}
    if isinstance(error, (ValidationError, ValueError, TypeError)):
        return {"category": "validation", "detail": "The workflow returned an invalid result"}
    return {"category": "provider", "detail": "The model provider could not complete the workflow"}


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
