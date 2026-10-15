import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import get_settings
from helper.workflow_history import (
    critic_feedback_value,
    human_feedback_value,
    revision_context,
    revision_delta,
)
from chains import Question
from chains.contracts import InsufficientSourceError, structured_output
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
    return result.question.model_dump()


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
    }


def HumanFeedback(state: QuestionGenState) -> dict:
    return _human_feedback_delta(interrupt({
        "gernerated_question": state["question"],
        "options": state["options"],
        "answer": state["answer"],
        "explanation": state["explanation"],
        "message": "Provide feedback or type 'auto' to automatically generate a new question or 'save' to finish",
    }))


async def HumanFeedbackStream(state: QuestionGenState) -> dict:
    return HumanFeedback(state)


def Router(state: QuestionGenState) -> str:
    feedback = human_feedback_value(state).strip().casefold()
    if feedback == "save":
        return "save"
    if feedback == "auto":
        return "auto"
    return "feedback"


def _review_inputs(state):
    return {"context": state["context"], **current_question(state).model_dump()}


def QuestionRefiner(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    inputs = _review_inputs(state)
    response = get_task_chain("question_review", settings).invoke(inputs, config=config)
    return {"critic_feedback": response}


async def QuestionRefinerStream(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    inputs = _review_inputs(state)
    response = await get_task_chain("question_review", settings).ainvoke(inputs, config=config)
    return {"critic_feedback": response}


def _revision_inputs(state):
    return {
        "context": state["context"],
        "question_type": state["question_type"],
        "current_question": current_question(state),
        "feedback": {
            "human_feedback": human_feedback_value(state),
            "critic_feedback": critic_feedback_value(state),
        },
        "history": revision_context(state, settings),
    }


def _revision_delta(state, response):
    return {
        **response.question.model_dump(),
        **revision_delta(
            state, settings, draft=response.question, status=response.status,
            reason=response.reason, critic=critic_feedback_value(state),
        ),
    }


def QuestionRewriter(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    inputs = _revision_inputs(state)
    response = get_task_chain("question_revision", settings).invoke(inputs, config=config)
    return _revision_delta(state, response)


async def QuestionRewriterStream(state: QuestionGenState, config: RunnableConfig = None) -> dict:
    inputs = _revision_inputs(state)
    response = await get_task_chain("question_revision", settings).ainvoke(inputs, config=config)
    return _revision_delta(state, response)


graph = StateGraph(QuestionGenState)
graph.add_node("question_gen", RunnableLambda(QuestionGenerator, afunc=QuestionGeneratorStream))
graph.add_node("question_refiner", RunnableLambda(QuestionRefiner, afunc=QuestionRefinerStream))
graph.add_node("question_rewriter", RunnableLambda(QuestionRewriter, afunc=QuestionRewriterStream))
graph.add_node("human_feedback", RunnableLambda(HumanFeedback, afunc=HumanFeedbackStream))

graph.add_edge(START, "question_gen")
graph.add_edge("question_gen", "human_feedback")
graph.add_conditional_edges("human_feedback", Router, {"save": END, "auto": "question_refiner", "feedback": "question_rewriter"})
graph.add_edge("question_refiner", "question_rewriter")
graph.add_edge("question_rewriter", "human_feedback")

checkpointer = MemorySaver()
qg_graph = graph.compile(checkpointer=checkpointer)
graph = graph.compile()
