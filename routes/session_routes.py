from typing import Literal
from uuid import UUID

import anyio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from chains.contracts import StructuredOutputError
from helper.asset_paths import validate_asset_id
from helper.ingestion_errors import InvalidAssetError
from helper.question_review import validate_question_draft
from helper.session_review import review_revision, session_status
from helper.session_store import completed_sessions
from helper.workflow_history import canonical_messages
from helper.workflow_runtime import require_graph

router = APIRouter()
Workflow = Literal["question_generation", "summarization", "qa"]


@router.get("/question_generation")
async def saved_questions(asset_id: str | None = None):
    if asset_id is not None:
        try:
            validate_asset_id(asset_id)
        except InvalidAssetError:
            raise HTTPException(status_code=400, detail="Invalid asset identifier") from None
    graph = require_graph("question_generation")
    sessions = await anyio.to_thread.run_sync(completed_sessions, "question_generation", asset_id)
    questions = []
    for session in sessions:
        snapshot = await graph.aget_state({"configurable": {"thread_id": session["thread_id"]}})
        if session_status(snapshot) != "completed":
            continue
        if snapshot.values.get("asset_id") != session["asset_id"]:
            raise HTTPException(status_code=502, detail="Saved question does not match its source")
        try:
            question = validate_question_draft(snapshot.values)
        except (StructuredOutputError, KeyError):
            raise HTTPException(status_code=502, detail="A saved question could not be validated") from None
        questions.append({**session, "result": question.model_dump(mode="json")})
    return {"questions": questions}


@router.get("/{workflow}/{thread_id}")
async def read_session(workflow: Workflow, thread_id: str):
    try:
        UUID(thread_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session identifier") from None
    snapshot = await require_graph(workflow).aget_state({"configurable": {"thread_id": thread_id}})
    status = session_status(snapshot)
    if status == "not_found":
        raise HTTPException(status_code=404, detail="Session not found")
    state = snapshot.values
    keys = ({"question", "question_type", "options", "answer", "explanation", "critic_feedback", "review_notice", "revision_limit_reached"}
            if workflow == "question_generation" else {"Main_Points", "summary", "review_notice", "revision_kind"}
            if workflow == "summarization" else {"answer_status", "answer_citations", "grounding_status"})
    result = {key: value.model_dump(mode="json") if isinstance(value, BaseModel) else value
              for key, value in state.items() if key in keys}
    if workflow == "qa":
        result["messages"] = [{"role": "user" if message.type == "human" else "assistant", "content": message.content}
                              for message in canonical_messages(state)]
    return {"thread_id": thread_id, "workflow": workflow, "asset_id": state.get("asset_id"),
            "status": status, "review_id": review_revision(snapshot), "result": result}
