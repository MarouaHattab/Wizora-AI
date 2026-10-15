import uuid
import json
from fastapi import APIRouter, HTTPException, Body
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from controllers.GraphState import QuestionAnswerState
from controllers.QuestionAnswerGraphController import stream_qa_graph as qa_graph, message_text
from helper.session_review import acquire_session, session_status, SessionStreamingResponse
from helper.asset_http import read_asset_text

router = APIRouter()

def _event(payload):
    return f"data: {json.dumps(payload)}\n\n"


def _stream_response(payload, config, metadata, session_lock):
    async def generate():
        try:
            yield _event({"type": "metadata", **metadata})
            streamed_text = ""
            async for event in qa_graph.astream_events(payload, config=config, version="v2"):
                if (
                    event["event"] == "on_chat_model_stream"
                    and event.get("metadata", {}).get("langgraph_node") == "generate_answer_streaming"
                ):
                    text = message_text(event["data"].get("chunk"))
                    if text:
                        streamed_text += text
                        yield _event({"type": "token", "content": text})

            snapshot = await qa_graph.aget_state(config)
            status = session_status(snapshot)
            if status == "interrupted_for_feedback":
                yield _event({"type": "interrupted", "status": status})
                return
            messages = snapshot.values.get("messages", [])
            if status != "completed" or not messages or not isinstance(messages[-1], AIMessage):
                raise RuntimeError("Question answering did not complete")
            final_text = message_text(messages[-1])
            if not final_text.strip() or not final_text.startswith(streamed_text):
                raise RuntimeError("The completed answer does not match the stream")
            if len(final_text) > len(streamed_text):
                yield _event({"type": "token", "content": final_text[len(streamed_text):]})
            yield _event({"type": "complete"})
        except Exception:
            yield _event({"type": "error", "status": "failed", "detail": "Question answering failed"})
        finally:
            session_lock.release()

    return SessionStreamingResponse(
        generate(),
        session_lock=session_lock,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


@router.post("/start_session_stream")
async def start_qa_session_stream(asset_id: str = Body(..., embed=True), initial_question: str = Body(..., embed=True)):
    if not initial_question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty")
    context = await read_asset_text(asset_id)

    thread_id = str(uuid.uuid4())
    initial_state = QuestionAnswerState(
        asset_id=asset_id,
        messages=[HumanMessage(content=initial_question)],
        relevant_text=[],
        on_topic="",
        context=context,
        conversation_history=[],
    )
    config = {"configurable": {"thread_id": thread_id}}
    session_lock = acquire_session("qa", thread_id)
    return _stream_response(initial_state, config, {"thread_id": thread_id, "asset_id": asset_id}, session_lock)


@router.post("/continue_conversation_stream")
async def continue_qa_conversation_stream(
    thread_id: str = Body(..., embed=True),
    next_question: str = Body(..., embed=True),
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
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage) or not all(isinstance(message, BaseMessage) for message in messages):
            raise HTTPException(status_code=409, detail="Session has no completed answer")
        payload = {"messages": messages + [HumanMessage(content=next_question)]}
        return _stream_response(payload, config, {"thread_id": thread_id}, session_lock)
    except BaseException:
        session_lock.release()
        raise
