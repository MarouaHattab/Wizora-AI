import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import get_settings
from helper.workflow_history import human_feedback_value, revision_context, revision_delta
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


def UserFeedbackNode(state: SummaryGenState) -> dict:
    user_input = interrupt({
        "table_of_contents": state["Main_Points"],
        "summary": state["summary"],
        "message": "Provide feedback or type 'save' to finish",
        "waiting_for_input": True,
    })
    if isinstance(user_input, str):
        feedback, review_id = user_input, None
    elif isinstance(user_input, dict) and isinstance(user_input.get("feedback"), str):
        feedback, review_id = user_input["feedback"], user_input.get("review_id")
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


async def UserFeedbackNodeStream(state: SummaryGenState) -> dict:
    return UserFeedbackNode(state)


def Router(state: SummaryGenState) -> str:
    return "save" if human_feedback_value(state).strip().casefold() == "save" else "feedback"


def _revision_inputs(state):
    return {
        "context": state["context"],
        "original_summary": state["summary"],
        "user_feedback": {
            "current": human_feedback_value(state),
            "prior": revision_context(state, settings),
        },
    }


def _revision_delta(state, response):
    return {"summary": response, **revision_delta(state, settings, draft=response, status="revised")}


def SummarizerRewriterNode(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    response = get_task_chain("summary_revision", settings).invoke(_revision_inputs(state), config=config)
    return _revision_delta(state, response)


async def SummarizerRewriterNodeStream(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    response = await get_task_chain("summary_revision", settings).ainvoke(_revision_inputs(state), config=config)
    return _revision_delta(state, response)


stream_graph = StateGraph(SummaryGenState)
stream_graph.add_node("main_point_summarizer", RunnableLambda(SummarizerMainPointNode, afunc=SummarizerMainPointNodeStream))
stream_graph.add_node("summarizer_writer", RunnableLambda(SummarizerWriterNode, afunc=SummarizerWriterNodeStream))
stream_graph.add_node("user_feedback", RunnableLambda(UserFeedbackNode, afunc=UserFeedbackNodeStream))
stream_graph.add_node("summarizer_rewriter", RunnableLambda(SummarizerRewriterNode, afunc=SummarizerRewriterNodeStream))
stream_graph.add_edge(START, "main_point_summarizer")
stream_graph.add_edge("main_point_summarizer", "summarizer_writer")
stream_graph.add_edge("summarizer_writer", "user_feedback")
stream_graph.add_conditional_edges("user_feedback", Router, {"save": END, "feedback": "summarizer_rewriter"})
stream_graph.add_edge("summarizer_rewriter", "user_feedback")

checkpointer = MemorySaver()
stream_summarizer_graph = stream_graph.compile(checkpointer=checkpointer)
graph = stream_graph
summarizer_graph = graph.compile()
