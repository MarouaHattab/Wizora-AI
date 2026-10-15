import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import get_settings
from helper.question_review import (
    REVIEW_MESSAGE,
    accepted_feedback,
    invalid_critic_feedback,
    needs_revision,
    parse_review_action,
    refine_budget_exhausted,
    structural_feedback,
    validate_question_draft,
)
from helper.workflow_history import (
    critic_feedback_value,
    human_feedback_value,
    revision_context,
    revision_delta,
)
from chains import Question
from chains.contracts import (
    ChainInputError, InsufficientSourceError, ProviderExecutionError,
    StructuredOutputError, structured_output,
)
from chains.runtime import get_task_chain
from .GraphState import QuestionGenState
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

settings = get_settings()


def current_question(state):
    return structured_output(Question, {
        key: state.get(key) for key in ("question", "question_type", "options", "answer", "explanation")
    })


def _generation_inputs(state):
    if not state["context"].strip():
        raise InsufficientSourceError("The source contains no educational material.")
    return {"context": state["context"], "question_type": state["question_type"]}


def _generation_delta(result):
    if result.status == "insufficient_source":
        raise InsufficientSourceError(result.reason)
    question = validate_question_draft({"question_type": result.question.question_type}, result.question)
    return {
        **question.model_dump(),
        "workflow_action": None,
        "auto_revision_count": 0,
        "review_model_calls": 0,
        "revision_limit_reached": False,
        "review_notice": None,
    }


def QuestionGenerator(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    inputs = _generation_inputs(state)
    result = get_task_chain("question_generation", settings).invoke(inputs, config=config)
    return _generation_delta(result)


async def QuestionGeneratorStream(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    inputs = _generation_inputs(state)
    result = await get_task_chain("question_generation", settings).ainvoke(inputs, config=config)
    return _generation_delta(result)


def _human_feedback_delta(value):
    if isinstance(value, str):
        feedback, review_id = value, None
    elif isinstance(value, dict) and isinstance(value.get("feedback"), str):
        feedback, review_id = value["feedback"], value.get("review_id")
        if review_id is not None and not isinstance(review_id, str):
            raise ValueError("A feedback review_id must be text")
    else:
        raise ValueError("Human feedback must be text or a feedback review object")
    return {
        "feedback": feedback,
        "human_feedback_text": feedback,
        "critic_feedback": None,
        "feedback_review_id": review_id,
        "state_version": 4,
        "workflow_action": parse_review_action(feedback),
        "auto_revision_count": 0,
        "review_model_calls": 0,
        "revision_limit_reached": False,
        "review_notice": None,
    }


def HumanFeedback(state: QuestionGenState) -> dict:
    critic = critic_feedback_value(state)
    payload = {
        "generated_question": state["question"],
        "gernerated_question": state["question"],
        "options": state["options"],
        "answer": state["answer"],
        "explanation": state["explanation"],
        "message": REVIEW_MESSAGE,
    }
    if critic is not None:
        payload["critic_verdict"] = critic.verdict
        payload["critic_feedback"] = critic.feedback
    if state.get("review_notice"):
        payload["revision_notice"] = state["review_notice"]
    return _human_feedback_delta(interrupt(payload))


async def HumanFeedbackStream(state: QuestionGenState) -> dict:
    return HumanFeedback(state)


def Router(state: QuestionGenState) -> str:
    return parse_review_action(human_feedback_value(state))


def _review_inputs(state):
    return {"context": state["context"], **current_question(state).model_dump()}


def _critic_delta(state, critic, *, model_calls=0, notice=None, limit=False):
    calls = (state.get("review_model_calls", 0) if type(state.get("review_model_calls", 0)) is int else 0) + model_calls
    delta = {
        "critic_feedback": critic,
        "workflow_action": "refine",
        "review_model_calls": calls,
        "revision_limit_reached": limit,
        "review_notice": notice,
    }
    if accepted_feedback(critic) or limit:
        reason = notice or (
            "The critic accepted the draft; no rewrite was required."
            if accepted_feedback(critic) else
            "Automatic refinement reached its configured limit."
        )
        delta.update(revision_delta(
            state, settings, draft=current_question(state), status="unchanged",
            reason=reason, critic=critic,
        ))
        delta["review_notice"] = reason
    return delta


def _refiner_precheck(state):
    structural = structural_feedback(state)
    if structural is not None:
        return _critic_delta(state, structural, limit=refine_budget_exhausted(state, settings, extra_calls=1))
    if refine_budget_exhausted(state, settings, extra_calls=1):
        return _critic_delta(
            state, critic_feedback_value(state) or structural,
            notice="Automatic refinement reached its configured model-call limit.",
            limit=True,
        )
    return None


def _invalid_critic_delta(state):
    return _critic_delta(
        state, invalid_critic_feedback(), model_calls=1,
        notice="The critic result could not be validated, so the draft was not rewritten.",
        limit=True,
    )


def QuestionRefiner(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    prepared = _refiner_precheck(state)
    if prepared is not None:
        return prepared
    try:
        response = get_task_chain("question_review", settings).invoke(_review_inputs(state), config=config)
    except (StructuredOutputError, ProviderExecutionError):
        return _invalid_critic_delta(state)
    return _critic_delta(state, response, model_calls=1)


async def QuestionRefinerStream(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    prepared = _refiner_precheck(state)
    if prepared is not None:
        return prepared
    try:
        response = await get_task_chain("question_review", settings).ainvoke(_review_inputs(state), config=config)
    except (StructuredOutputError, ProviderExecutionError):
        return _invalid_critic_delta(state)
    return _critic_delta(state, response, model_calls=1)


def route_after_critique(state: QuestionGenState) -> str:
    if state.get("revision_limit_reached") or refine_budget_exhausted(state, settings, extra_calls=1):
        return "review"
    critic = critic_feedback_value(state)
    if critic is None:
        raise StructuredOutputError("Automatic refinement requires a validated critic decision")
    if needs_revision(critic):
        return "rewrite"
    return "review"


def _revision_inputs(state):
    action = state.get("workflow_action") or parse_review_action(human_feedback_value(state))
    if action not in {"refine", "revise"}:
        action = "revise"
    return {
        "context": state["context"],
        "question_type": state["question_type"],
        "current_question": current_question(state),
        "action": action,
        "feedback": {
            "action": action,
            "human_feedback": human_feedback_value(state),
            "critic_feedback": critic_feedback_value(state),
        },
        "history": revision_context(state, settings),
    }


def _revision_delta(state, response):
    action = state.get("workflow_action") or parse_review_action(human_feedback_value(state))
    calls = (state.get("review_model_calls", 0) if type(state.get("review_model_calls", 0)) is int else 0) + 1
    revisions = state.get("auto_revision_count", 0) if type(state.get("auto_revision_count", 0)) is int else 0
    if action == "refine":
        revisions += 1
    question = validate_question_draft(state, response.question)
    notice = None
    limit = action == "refine" and refine_budget_exhausted(
        {**state, "auto_revision_count": revisions, "review_model_calls": calls}, settings,
    )
    if limit:
        notice = "Automatic refinement reached its configured limit."
    return {
        **question.model_dump(),
        "workflow_action": action if action in {"refine", "revise"} else "revise",
        "review_model_calls": calls,
        "auto_revision_count": revisions,
        "revision_limit_reached": limit,
        "review_notice": notice,
        **revision_delta(
            state, settings, draft=question, status=response.status,
            reason=response.reason or notice, critic=critic_feedback_value(state),
        ),
    }


def _rewrite_precheck(state):
    if (state.get("workflow_action") or parse_review_action(human_feedback_value(state))) != "refine":
        return None
    if not refine_budget_exhausted(state, settings, extra_calls=1):
        return None
    return _critic_delta(
        state, critic_feedback_value(state),
        notice="Automatic refinement reached its configured limit.",
        limit=True,
    )


def _unrevised_delta(state, notice):
    action = state.get("workflow_action") or parse_review_action(human_feedback_value(state))
    if action not in {"refine", "revise"}:
        action = "revise"
    calls = (state.get("review_model_calls", 0) if type(state.get("review_model_calls", 0)) is int else 0) + 1
    revisions = state.get("auto_revision_count", 0) if type(state.get("auto_revision_count", 0)) is int else 0
    if action == "refine":
        revisions += 1
    return {
        "workflow_action": action,
        "review_model_calls": calls,
        "auto_revision_count": revisions,
        "revision_limit_reached": action == "refine",
        "review_notice": notice,
        **revision_delta(
            state, settings, draft=current_question(state), status="unchanged",
            reason=notice, critic=critic_feedback_value(state),
        ),
    }


def _rewrite(state, response):
    try:
        return _revision_delta(state, response)
    except (ChainInputError, StructuredOutputError):
        return _unrevised_delta(state, "The revised question failed validation, so the current question was kept.")


def QuestionRewriter(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    prepared = _rewrite_precheck(state)
    if prepared is not None:
        return prepared
    try:
        response = get_task_chain("question_revision", settings).invoke(_revision_inputs(state), config=config)
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unrevised_delta(state, "The revision could not be validated, so the current question was kept.")
    return _rewrite(state, response)


async def QuestionRewriterStream(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    prepared = _rewrite_precheck(state)
    if prepared is not None:
        return prepared
    try:
        response = await get_task_chain("question_revision", settings).ainvoke(_revision_inputs(state), config=config)
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unrevised_delta(state, "The revision could not be validated, so the current question was kept.")
    return _rewrite(state, response)


graph = StateGraph(QuestionGenState)
graph.add_node("question_gen", RunnableLambda(QuestionGenerator, afunc=QuestionGeneratorStream))
graph.add_node("question_refiner", RunnableLambda(QuestionRefiner, afunc=QuestionRefinerStream))
graph.add_node("question_rewriter", RunnableLambda(QuestionRewriter, afunc=QuestionRewriterStream))
graph.add_node("human_feedback", RunnableLambda(HumanFeedback, afunc=HumanFeedbackStream))

graph.add_edge(START, "question_gen")
graph.add_edge("question_gen", "human_feedback")
graph.add_conditional_edges(
    "human_feedback", Router,
    {"accept": END, "refine": "question_refiner", "revise": "question_rewriter"},
)
graph.add_conditional_edges(
    "question_refiner", route_after_critique,
    {"review": "human_feedback", "rewrite": "question_rewriter"},
)
graph.add_edge("question_rewriter", "human_feedback")

checkpointer = MemorySaver()
qg_graph = graph.compile(checkpointer=checkpointer)
graph = graph.compile()
