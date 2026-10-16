import logging
import sqlite3
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from chains.contracts import text_fragment, text_output
from controllers.GraphState import SummaryGenState
from helper import get_settings
from helper.asset_http import ReviewFeedbackRequest, SingleAssetRequest, read_asset_text, validate_session_asset
from helper.failures import StorageUnavailable
from helper.session_review import (
    SessionStreamingResponse, accept_feedback, acquire_session,
    review_revision, session_status, validate_feedback,
)
from helper.sse import EventStream, EventVersion, before_deadline, close_events, events_before, public_error
from helper.workflow_metrics import metrics
from helper.workflow_runtime import note_snapshot, require_graph

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


class SummaryRequest(SingleAssetRequest):
    event_version: EventVersion = "legacy"


class SummaryFeedbackRequest(ReviewFeedbackRequest):
    event_version: EventVersion = "legacy"


def summary_feedback(snapshot):
    payload = {
        "table_of_contents": snapshot.values.get("Main_Points", ""),
        "summary": snapshot.values.get("summary", ""),
        "message": "Provide feedback or type 'save' to finish",
        "waiting_for_input": True,
    }
    if snapshot.values.get("review_notice"):
        payload["revision_notice"] = snapshot.values["review_notice"]
    if snapshot.values.get("revision_kind"):
        payload["revision_kind"] = snapshot.values["revision_kind"]
    return payload


async def stream_session(
    thread_id, graph_input, config, lock, initial_status, graph, review=None,
    event_version: EventVersion = "legacy",
) -> AsyncGenerator[str, None]:
    public_nodes = {
        "main_point_summarizer", "summarizer_writer", "summarizer_rewriter",
        "summarizer_structure", "user_feedback",
    }
    token_nodes = public_nodes - {"user_feedback", "summarizer_structure"}
    stream = EventStream(thread_id, event_version)
    snapshot = None
    status = "failed"
    iterator = None
    stage_text = {}
    stage_started = {}
    failure = None
    started = time.perf_counter()
    try:
        yield stream.emit(
            "session_started", data={"status": initial_status},
            legacy={"thread_id": thread_id, "status": initial_status},
        )
        deadline = time.monotonic() + settings.GRAPH_TIMEOUT_SECONDS
        if review is not None:
            accept_feedback("summarization", thread_id, *review)
        iterator = graph.astream_events(graph_input, config=config, version="v2")
        async for event in events_before(iterator, deadline):
            event_type = event.get("event")
            event_name = event.get("name", "")
            if event_type == "on_chain_start" and event_name in public_nodes:
                stage_started[event_name] = time.perf_counter()
                if event_name in token_nodes:
                    stage_text[event_name] = ""
                yield stream.emit(
                    "stage_started", stage=event_name,
                    legacy={
                        "event": event_type, "name": event_name, "thread_id": thread_id,
                        "status_update": f"Starting: {event_name}",
                    },
                )
            elif event_type == "on_chat_model_stream":
                node = event.get("metadata", {}).get("langgraph_node")
                if node not in token_nodes:
                    continue
                content = text_fragment(event.get("data", {}).get("chunk"))
                if content:
                    stage_text[node] = stage_text.get(node, "") + content
                    yield stream.emit(
                        "text_delta", stage=node, data={"text": content},
                        legacy={
                            "event": "token", "name": node, "thread_id": thread_id,
                            "token": content, "status_update": node,
                        },
                    )
            elif event_type == "on_chain_end" and event_name in public_nodes:
                if event_name in stage_started:
                    metrics.observe(
                        "node_ms", (time.perf_counter() - stage_started[event_name]) * 1000,
                        workflow="summarization", node=event_name,
                    )
                if event_name not in token_nodes:
                    continue
                output = event.get("data", {}).get("output")
                field = "Main_Points" if event_name == "main_point_summarizer" else "summary"
                if not isinstance(output, dict) or field not in output:
                    raise ValueError("The summarization stage returned an invalid result")
                final = text_output(output[field])
                sent = stage_text.get(event_name, "")
                if not final.startswith(sent):
                    raise ValueError("The completed draft does not match the stream")
                suffix = final[len(sent):]
                if suffix:
                    stage_text[event_name] = sent + suffix
                    yield stream.emit(
                        "text_delta", stage=event_name, data={"text": suffix},
                        legacy={
                            "event": "token", "name": event_name, "thread_id": thread_id,
                            "token": suffix, "status_update": event_name,
                        },
                    )

        snapshot = await before_deadline(graph.aget_state(config), deadline)
        status = session_status(snapshot)
        if status == "interrupted_for_feedback":
            draft = summary_feedback(snapshot)
            text_output(draft["summary"])
            yield stream.emit(
                "awaiting_feedback", stage="user_feedback",
                data={"status": status, "review_id": review_revision(snapshot), **draft},
                legacy={
                    "event": "review", "thread_id": thread_id, "status": status,
                    "review_id": review_revision(snapshot), "data_for_feedback": draft,
                },
            )
        elif status == "completed":
            summary = text_output(snapshot.values.get("summary"))
            yield stream.emit(
                "completed", data={"status": status, "summary": summary},
                legacy={
                    "event": "stream_end", "thread_id": thread_id,
                    "status_update": "Stream ended", "status": status,
                    "review_id": None, "summary": summary,
                },
            )
        else:
            raise ValueError("Summarization did not reach a review or completion")
    except Exception as error:
        failure = error
        status = "failed"
        public = public_error(error)
        logger.warning(
            "Summary stream failed (category=%s, exception_type=%s, run_id=%s)",
            public["category"], type(error).__name__, stream.run_id,
        )
        frame = stream.emit(
            "failed", data=public,
            legacy={"event": "error", "thread_id": thread_id, "status": status, **public},
        )
        if frame:
            yield frame
    finally:
        try:
            await close_events(iterator)
        finally:
            try:
                note_snapshot(
                    "summarization", thread_id, snapshot, started=started,
                    error=failure, request_id=stream.run_id,
                )
            finally:
                lock.release()

    if status != "completed":
        frame = stream.transport_closed({
            "event": "stream_end", "thread_id": thread_id, "status_update": "Stream ended",
            "status": status, "review_id": review_revision(snapshot),
        })
        if frame:
            yield frame


@router.post("/start_session_streaming")
async def start_summarization_session_streaming(
    request: SummaryRequest,
) -> StreamingResponse:
    thread_id = str(uuid.uuid4())
    context = await read_asset_text(request.asset_id)
    initial_state = SummaryGenState(
        asset_id=request.asset_id, context=context, summary="", feedback="", Main_Points="",
        human_feedback_text="", critic_feedback=None, feedback_review_id=None,
        history=[], revision_requirements=[], state_version=4,
    )
    config = {"configurable": {"thread_id": thread_id}}
    graph = require_graph("summarization")
    lock = acquire_session("summarization", thread_id)
    return SessionStreamingResponse(
        stream_session(
            thread_id, initial_state, config, lock, "starting_session", graph,
            event_version=request.event_version,
        ),
        session_lock=lock, media_type="text/event-stream",
    )


@router.post("/provide_feedback_streaming")
async def provide_feedback_to_summarization_streaming(
    request: SummaryFeedbackRequest,
) -> StreamingResponse:
    thread_id, feedback, review_id = request.thread_id, request.feedback, request.review_id
    config = {"configurable": {"thread_id": thread_id}}
    graph = require_graph("summarization")
    lock = acquire_session("summarization", thread_id)
    try:
        snapshot = await graph.aget_state(config)
        validate_feedback("summarization", thread_id, snapshot, feedback, review_id)
        await validate_session_asset(snapshot.values)
        metrics.increment("workflow_resume", workflow="summarization")
    except HTTPException:
        lock.release()
        raise
    except (StorageUnavailable, sqlite3.OperationalError):
        lock.release()
        raise HTTPException(status_code=503, detail="Workflow storage is unavailable") from None
    except Exception:
        lock.release()
        raise HTTPException(status_code=500, detail="Unable to validate summarization session") from None
    except BaseException:
        lock.release()
        raise
    return SessionStreamingResponse(
        stream_session(
            thread_id,
            Command(resume={"feedback": feedback, "review_id": review_revision(snapshot)}),
            config, lock, "resuming_with_feedback", graph,
            review=(snapshot, feedback, review_id),
            event_version=request.event_version,
        ),
        session_lock=lock, media_type="text/event-stream",
    )
