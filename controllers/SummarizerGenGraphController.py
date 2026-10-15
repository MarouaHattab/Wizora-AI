import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import get_settings
from helper.summary_revision import classify_summary_feedback
from helper.workflow_history import human_feedback_value, revision_context, revision_delta
from chains.contracts import ChainInputError, ProviderExecutionError, StructuredOutputError
from chains.runtime import get_task_chain
from .GraphState import SummaryGenState
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

settings = get_settings()


def SummarizerMainPointNode(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    response = get_task_chain("main_points", settings).invoke({"context": state["context"]}, config=config)
    return {"Main_Points": response}


async def SummarizerMainPointNodeStream(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    response = await get_task_chain("main_points", settings).ainvoke({"context": state["context"]}, config=config)
    return {"Main_Points": response}


def _writer_inputs(state):
    return {"context": state["context"], "table_of_contents": state["Main_Points"]}


def SummarizerWriterNode(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    response = get_task_chain("summary", settings).invoke(_writer_inputs(state), config=config)
    return {"summary": response}


async def SummarizerWriterNodeStream(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    response = await get_task_chain("summary", settings).ainvoke(_writer_inputs(state), config=config)
    return {"summary": response}


def _summary_feedback_delta(value):
    if isinstance(value, str):
        feedback, review_id = value, None
    elif isinstance(value, dict) and isinstance(value.get("feedback"), str):
        feedback, review_id = value["feedback"], value.get("review_id")
        if review_id is not None and not isinstance(review_id, str):
            raise ValueError("A feedback review_id must be text")
    else:
        raise ValueError("Human feedback must be text or a feedback review object")
    accepted = feedback.strip().casefold() in {"save", "accept"}
    return {
        "feedback": feedback,
        "human_feedback_text": feedback,
        "critic_feedback": None,
        "feedback_review_id": review_id,
        "state_version": 4,
        "workflow_action": "accept" if accepted else "revise",
        "revision_kind": None if accepted else classify_summary_feedback(feedback),
        "review_notice": None,
    }


def UserFeedbackNode(state: SummaryGenState) -> dict:
    payload = {
        "table_of_contents": state["Main_Points"],
        "summary": state["summary"],
        "message": "Provide feedback or type 'save' to finish",
        "waiting_for_input": True,
    }
    if state.get("review_notice"):
        payload["revision_notice"] = state["review_notice"]
    if state.get("revision_kind"):
        payload["revision_kind"] = state["revision_kind"]
    return _summary_feedback_delta(interrupt(payload))


async def UserFeedbackNodeStream(state: SummaryGenState) -> dict:
    return UserFeedbackNode(state)


def Router(state: SummaryGenState) -> str:
    if state.get("workflow_action") == "accept" or human_feedback_value(state).strip().casefold() in {"save", "accept"}:
        return "save"
    kind = state.get("revision_kind") or classify_summary_feedback(human_feedback_value(state))
    return "structural" if kind == "structural" else "wording"


def _revision_inputs(state):
    kind = state.get("revision_kind") if state.get("revision_kind") in {"wording", "unsupported"} else "wording"
    return {
        "context": state["context"],
        "original_summary": state["summary"],
        "user_feedback": {
            "kind": kind,
            "current": human_feedback_value(state),
            "prior": revision_context(state, settings),
        },
    }


def _revision_delta(state, response):
    kind = state.get("revision_kind") if state.get("revision_kind") in {"wording", "unsupported"} else "wording"
    return {
        "summary": response,
        "revision_kind": kind,
        "review_notice": None,
        **revision_delta(state, settings, draft=response, status="revised", reason=None),
    }


def _unchanged_summary(state, notice):
    return {
        "summary": state["summary"],
        "Main_Points": state["Main_Points"],
        "revision_kind": state.get("revision_kind"),
        "review_notice": notice,
        **revision_delta(state, settings, draft=state["summary"], status="unchanged", reason=notice),
    }


def SummarizerRewriterNode(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    try:
        response = get_task_chain("summary_revision", settings).invoke(_revision_inputs(state), config=config)
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unchanged_summary(state, "The summary revision could not be validated, so the current draft was kept.")
    return _revision_delta(state, response)


async def SummarizerRewriterNodeStream(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    try:
        response = await get_task_chain("summary_revision", settings).ainvoke(_revision_inputs(state), config=config)
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unchanged_summary(state, "The summary revision could not be validated, so the current draft was kept.")
    return _revision_delta(state, response)


def _structure_inputs(state):
    return {
        "context": state["context"],
        "outline": state["Main_Points"],
        "original_summary": state["summary"],
        "user_feedback": human_feedback_value(state),
    }


def _structure_delta(state, result):
    notice = result.limitation
    return {
        "Main_Points": result.outline,
        "summary": result.summary,
        "revision_kind": "structural",
        "review_notice": notice,
        **revision_delta(
            state, settings, draft=result.summary, status="revised", reason=notice,
        ),
    }


def SummarizerStructureNode(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    try:
        result = get_task_chain("summary_structure_revision", settings).invoke(_structure_inputs(state), config=config)
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unchanged_summary(state, "The structural revision could not be validated, so the outline and summary were left unchanged.")
    return _structure_delta(state, result)


async def SummarizerStructureNodeStream(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    try:
        result = await get_task_chain("summary_structure_revision", settings).ainvoke(_structure_inputs(state), config=config)
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unchanged_summary(state, "The structural revision could not be validated, so the outline and summary were left unchanged.")
    return _structure_delta(state, result)


stream_graph = StateGraph(SummaryGenState)
stream_graph.add_node("main_point_summarizer", RunnableLambda(SummarizerMainPointNode, afunc=SummarizerMainPointNodeStream))
stream_graph.add_node("summarizer_writer", RunnableLambda(SummarizerWriterNode, afunc=SummarizerWriterNodeStream))
stream_graph.add_node("user_feedback", RunnableLambda(UserFeedbackNode, afunc=UserFeedbackNodeStream))
stream_graph.add_node("summarizer_rewriter", RunnableLambda(SummarizerRewriterNode, afunc=SummarizerRewriterNodeStream))
stream_graph.add_node("summarizer_structure", RunnableLambda(SummarizerStructureNode, afunc=SummarizerStructureNodeStream))
stream_graph.add_edge(START, "main_point_summarizer")
stream_graph.add_edge("main_point_summarizer", "summarizer_writer")
stream_graph.add_edge("summarizer_writer", "user_feedback")
stream_graph.add_conditional_edges(
    "user_feedback", Router,
    {"save": END, "wording": "summarizer_rewriter", "structural": "summarizer_structure"},
)
stream_graph.add_edge("summarizer_rewriter", "user_feedback")
stream_graph.add_edge("summarizer_structure", "user_feedback")

checkpointer = MemorySaver()
stream_summarizer_graph = stream_graph.compile(checkpointer=checkpointer)
graph = stream_graph
summarizer_graph = graph.compile()
