from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from helper.session_review import review_revision, session_status
from helper.workflow_history import canonical_messages
from helper.workflow_runtime import require_graph

router = APIRouter()
Workflow = Literal["question_generation", "summarization", "qa"]


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
