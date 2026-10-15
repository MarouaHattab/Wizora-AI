import logging
import time
import uuid

from fastapi import APIRouter, Body, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from controllers.GraphState import QuestionAnswerState
from controllers.QuestionAnswerGraphController import message_text, stream_qa_graph as qa_graph
from helper import get_settings
from helper.asset_http import read_asset_text
from helper.session_review import SessionStreamingResponse, acquire_session, session_status
from helper.sse import EventStream, EventVersion, before_deadline, close_events, events_before, public_error
from helper.workflow_history import canonical_messages

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)


def _stream_response(payload, config, metadata, session_lock, event_version: EventVersion = "legacy"):
    async def generate():
        stream = EventStream(config["configurable"]["thread_id"], event_version)
        iterator = None
        try:
            yield stream.emit(
                "session_started", data=metadata,
                legacy={"type": "metadata", **metadata},
            )
            deadline = time.monotonic() + settings.GRAPH_TIMEOUT_SECONDS
            streamed_text = ""
            iterator = qa_graph.astream_events(payload, config=config, version="v2")
            async for event in events_before(iterator, deadline):
                event_type = event.get("event")
                node = event.get("metadata", {}).get("langgraph_node")
                if event_type == "on_chain_start" and event.get("name") in {
                    "generate_answer_streaming", "off_topic_response",
                }:
                    stage = event["name"]
                    frame = stream.emit("stage_started", stage=stage)
                    if frame:
                        yield frame
                elif event_type == "on_chat_model_stream" and node == "generate_answer_streaming":
                    text = message_text(event.get("data", {}).get("chunk"))
                    if text:
                        streamed_text += text
                        yield stream.emit(
                            "text_delta", stage=node, data={"text": text},
                            legacy={"type": "token", "content": text},
                        )

            snapshot = await before_deadline(qa_graph.aget_state(config), deadline)
            status = session_status(snapshot)
            if status == "interrupted_for_feedback":
                yield stream.emit(
                    "awaiting_feedback", data={"status": status},
                    legacy={"type": "interrupted", "status": status},
                )
                return
            messages = canonical_messages(snapshot.values)
            if status != "completed" or not messages or not isinstance(messages[-1], AIMessage):
                raise ValueError("Question answering did not complete")
            final_text = message_text(messages[-1])
            if not final_text.strip() or not final_text.startswith(streamed_text):
                raise ValueError("The completed answer does not match the stream")
            if len(final_text) > len(streamed_text):
                suffix = final_text[len(streamed_text):]
                streamed_text += suffix
                yield stream.emit(
                    "text_delta", stage="answer", data={"text": suffix},
                    legacy={"type": "token", "content": suffix},
                )
            if final_text != streamed_text:
                raise ValueError("The completed answer does not match the public text")
            yield stream.emit(
                "completed", data={"status": "completed"}, legacy={"type": "complete"},
            )
        except Exception as error:
            public = public_error(error)
            logger.warning(
                "QA stream failed (category=%s, exception_type=%s, run_id=%s)",
                public["category"], type(error).__name__, stream.run_id,
            )
            frame = stream.emit(
                "failed", data=public,
                legacy={"type": "error", "status": "failed", **public},
            )
            if frame:
                yield frame
        finally:
            try:
                await close_events(iterator)
            finally:
                session_lock.release()

    return SessionStreamingResponse(
        generate(), session_lock=session_lock, media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache", "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*",
        },
    )


@router.post("/start_session_stream")
async def start_qa_session_stream(
    asset_id: str = Body(..., embed=True),
    initial_question: str = Body(..., embed=True),
    event_version: EventVersion = Body("legacy", embed=True),
):
    if not initial_question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty")
    context = await read_asset_text(asset_id)
    thread_id = str(uuid.uuid4())
    initial_state = QuestionAnswerState(
        asset_id=asset_id,
        messages=[HumanMessage(content=initial_question, id=str(uuid.uuid4()))],
        relevant_text=[], on_topic="", context=context, state_version=4, history_trimmed=0,
    )
    config = {"configurable": {"thread_id": thread_id}}
    session_lock = acquire_session("qa", thread_id)
    return _stream_response(
        initial_state, config, {"thread_id": thread_id, "asset_id": asset_id},
        session_lock, event_version,
    )


@router.post("/continue_conversation_stream")
async def continue_qa_conversation_stream(
    thread_id: str = Body(..., embed=True),
    next_question: str = Body(..., embed=True),
    event_version: EventVersion = Body("legacy", embed=True),
):
    if not next_question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty")
    config = {"configurable": {"thread_id": thread_id}}
    session_lock = acquire_session("qa", thread_id)
    try:
        snapshot = await qa_graph.aget_state(config)
        status = session_status(snapshot)
        if status == "not_found":
            raise HTTPException(status_code=404, detail="Session not found. Start a new session.")
        if status != "completed":
            raise HTTPException(status_code=409, detail="Session has not completed its previous question")
        state = snapshot.values
        if not state.get("asset_id") or not state.get("context"):
            raise HTTPException(status_code=409, detail="Session has no asset context. Start a new session.")
        messages = canonical_messages(state)
        if not messages or not isinstance(messages[-1], AIMessage):
            raise HTTPException(status_code=409, detail="Session has no completed answer")
        payload = {"messages": [HumanMessage(content=next_question, id=str(uuid.uuid4()))]}
        return _stream_response(payload, config, {"thread_id": thread_id}, session_lock, event_version)
    except BaseException:
        session_lock.release()
        raise
