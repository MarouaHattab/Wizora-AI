import uuid
import json
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from controllers.GraphState import SummaryGenState
from controllers.SummarizerGenGraphController import stream_summarizer_graph as summarizer_graph
from helper.session_review import acquire_session, accept_feedback, validate_feedback, review_revision, session_status, SessionStreamingResponse
from helper.asset_http import read_asset_text

router = APIRouter()

def summary_feedback(snapshot):
    return {
        "table_of_contents": snapshot.values.get("Main_Points", ""),
        "summary": snapshot.values.get("summary", ""),
        "message": "Provide feedback or type 'save' to finish",
        "waiting_for_input": True,
    }


async def stream_session(thread_id, graph_input, config, lock, initial_status, review=None) -> AsyncGenerator[str, None]:
    public_nodes = {"main_point_summarizer", "summarizer_writer", "summarizer_rewriter", "user_feedback"}
    token_nodes = public_nodes - {"user_feedback"}
    status = "failed"
    snapshot = None
    try:
        yield f'data: {json.dumps({"thread_id": thread_id, "status": initial_status})}\n\n'
        if review is not None:
            accept_feedback("summarization", thread_id, *review)
        async for event in summarizer_graph.astream_events(graph_input, config=config, version="v2"):
            event_type = event["event"]
            event_name = event.get("name", "")
            if event_type == "on_chain_start" and event_name in public_nodes:
                payload = {
                    "event": event_type, "name": event_name, "thread_id": thread_id,
                    "status_update": f"Starting: {event_name}",
                }
                yield f'data: {json.dumps(payload)}\n\n'
            elif event_type == "on_chat_model_stream":
                node = event.get("metadata", {}).get("langgraph_node")
                if node not in token_nodes:
                    continue
                chunk = event.get("data", {}).get("chunk")
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    content = "".join(
                        block if isinstance(block, str) else block.get("text", "")
                        for block in content
                        if isinstance(block, str) or isinstance(block, dict) and block.get("type") == "text"
                    )
                if isinstance(content, str) and content:
                    payload = {
                        "event": "token", "name": event_name, "thread_id": thread_id,
                        "token": content, "status_update": node,
                    }
                    yield f'data: {json.dumps(payload)}\n\n'
        snapshot = await summarizer_graph.aget_state(config)
        status = session_status(snapshot)
        if status == "interrupted_for_feedback":
            payload = {
                "event": "review", "thread_id": thread_id, "status": status,
                "review_id": review_revision(snapshot), "data_for_feedback": summary_feedback(snapshot),
            }
            yield f'data: {json.dumps(payload)}\n\n'
        elif status != "completed":
            status = "failed"
            yield f'data: {json.dumps({"event": "error", "thread_id": thread_id, "status": status, "detail": "Summarization did not reach a review or completion"})}\n\n'
    except Exception as exc:
        status = "failed"
        yield f'data: {json.dumps({"event": "error", "thread_id": thread_id, "status": status, "detail": "Summarization execution failed", "error_type": type(exc).__name__})}\n\n'
    finally:
        lock.release()

    payload = {
        "event": "stream_end", "thread_id": thread_id, "status_update": "Stream ended",
        "status": status, "review_id": review_revision(snapshot),
    }
    if status == "completed":
        payload["summary"] = snapshot.values.get("summary", "")
    yield f'data: {json.dumps(payload)}\n\n'


@router.post("/start_session_streaming")
async def start_summarization_session_streaming(asset_id: str = Body(..., embed=True)) -> StreamingResponse:
    thread_id = str(uuid.uuid4())
    context = await read_asset_text(asset_id)

    initial_state = SummaryGenState(context=context, summary="", old_summary="", feedback="", Main_Points="")
    config = {"configurable": {"thread_id": thread_id}}
    lock = acquire_session("summarization", thread_id)
    return SessionStreamingResponse(
        stream_session(thread_id, initial_state, config, lock, "starting_session"),
        session_lock=lock, media_type="text/event-stream",
    )


@router.post("/provide_feedback_streaming")
async def provide_feedback_to_summarization_streaming(
    thread_id: str = Body(..., embed=True),
    feedback: str = Body(..., embed=True),
    review_id: str | None = Body(None, embed=True),
) -> StreamingResponse:
    config = {"configurable": {"thread_id": thread_id}}
    lock = acquire_session("summarization", thread_id)
    try:
        snapshot = await summarizer_graph.aget_state(config)
        validate_feedback("summarization", thread_id, snapshot, feedback, review_id)
    except HTTPException:
        lock.release()
        raise
    except Exception:
        lock.release()
        raise HTTPException(status_code=500, detail="Unable to validate summarization session")
    except BaseException:
        lock.release()
        raise
    return SessionStreamingResponse(
        stream_session(
            thread_id, Command(resume=feedback), config, lock, "resuming_with_feedback",
            review=(snapshot, feedback, review_id),
        ),
        session_lock=lock, media_type="text/event-stream",
    )
