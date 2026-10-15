import uuid
import os
from fastapi import APIRouter, HTTPException, Body
from typing import Literal
from langgraph.types import Command

from controllers.GraphState import QuestionGenState
from controllers.QuestionGenGraphController import qg_graph
from helper.session_review import acquire_session, accept_feedback, review_revision, session_status

router = APIRouter()

ASSETS_DIR = "assets"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_BASE_PATH = os.path.join(PROJECT_ROOT, ASSETS_DIR)


def question_response(thread_id, snapshot):
    status = session_status(snapshot)
    current_graph_state = snapshot.values
    if status == "completed":
        return {"thread_id": thread_id, "status": status, "final_state": current_graph_state}
    if status != "interrupted_for_feedback":
        raise HTTPException(status_code=500, detail="Question generation did not reach a review or completion")
    return {
        "thread_id": thread_id,
        "status": status,
        "review_id": review_revision(snapshot),
        "data_for_feedback": {
            "generated_question": current_graph_state.get("question"),
            "options": current_graph_state.get("options"),
            "answer": current_graph_state.get("answer"),
            "explanation": current_graph_state.get("explanation"),
            "message": "Provide feedback or type 'auto' to automatically generate a new question or 'save' to finish",
        },
        "current_state": current_graph_state,
    }


@router.post("/start_session")
async def start_question_generation_session(
    asset_id: str = Body(..., embed=True),
    question_type: Literal["T/F", "MCQ"] = Body(..., embed=True)
):
    thread_id = str(uuid.uuid4())
    context_file_path = os.path.join(ASSETS_BASE_PATH, asset_id, "extracted_text.txt")
    if not os.path.exists(context_file_path):
        raise HTTPException(status_code=404, detail=f"Asset text file not found for id: {asset_id}")
    try:
        with open(context_file_path, "r", encoding="utf-8") as f:
            context = f.read()
    except OSError:
        raise HTTPException(status_code=500, detail="Failed to read asset text")

    initial_state = QuestionGenState(
        context=context, question_type=question_type, question="", options=None,
        answer="", explanation="", history=[], feedback=""
    )
    config = {"configurable": {"thread_id": thread_id}}
    lock = acquire_session("question_generation", thread_id)
    try:
        await qg_graph.ainvoke(initial_state, config=config)
        return question_response(thread_id, await qg_graph.aget_state(config))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Error invoking question generation graph")
    finally:
        lock.release()


@router.post("/provide_feedback")
async def provide_feedback_to_question_generation(
    thread_id: str = Body(..., embed=True),
    feedback: str = Body(..., embed=True),
    review_id: str | None = Body(None, embed=True),
):
    config = {"configurable": {"thread_id": thread_id}}
    lock = acquire_session("question_generation", thread_id)
    try:
        snapshot = await qg_graph.aget_state(config)
        accept_feedback("question_generation", thread_id, snapshot, feedback, review_id)
        await qg_graph.ainvoke(Command(resume=feedback), config=config)
        return question_response(thread_id, await qg_graph.aget_state(config))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Error resuming question generation graph")
    finally:
        lock.release()
