import logging

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from chains import ChainInputError, ProviderExecutionError, StructuredOutputError, get_task_chain
from helper import get_settings
from helper.content_context import join_sections, map_config, prepare_source
from helper.summary_revision import classify_summary_feedback
from helper.workflow_history import human_feedback_value, revision_context, revision_delta
from helper.workflow_metrics import metrics

from .GraphState import SummaryGenState

settings = get_settings()
logger = logging.getLogger("wizora.summary")


def SummarizerMainPointNode(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    prepared = prepare_source(state, settings)
    parts = prepared["source_chunks"]
    outlines = [
        get_task_chain("main_points", settings).invoke(
            {"context": part["context"]}, config=map_config(config, len(parts))
        )
        for part in parts
    ]
    return {**prepared, "Main_Points": join_sections(outlines), "source_outlines": outlines}


async def SummarizerMainPointNodeStream(
    state: SummaryGenState, config: RunnableConfig = None
) -> dict:
    import anyio

    prepared = await anyio.to_thread.run_sync(prepare_source, state, settings)
    parts, outlines = prepared["source_chunks"], []
    for part in parts:
        outlines.append(
            await get_task_chain("main_points", settings).ainvoke(
                {"context": part["context"]}, config=map_config(config, len(parts))
            )
        )
    return {**prepared, "Main_Points": join_sections(outlines), "source_outlines": outlines}


def _writer_inputs(state):
    parts = state.get("source_chunks") or [{"context": state["context"]}]
    outlines = state.get("source_outlines") or [state["Main_Points"]]
    if len(parts) != len(outlines):
        raise ChainInputError("The source parts do not match their outlines; start a new summary")
    return [
        {"context": part["context"], "table_of_contents": outline}
        for part, outline in zip(parts, outlines)
    ]


def SummarizerWriterNode(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    inputs = _writer_inputs(state)
    sections = [
        get_task_chain("summary", settings).invoke(item, config=map_config(config, len(inputs)))
        for item in inputs
    ]
    return {"summary": join_sections(sections), "summary_sections": sections}


async def SummarizerWriterNodeStream(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    inputs, sections = _writer_inputs(state), []
    for item in inputs:
        sections.append(
            await get_task_chain("summary", settings).ainvoke(
                item, config=map_config(config, len(inputs))
            )
        )
    return {"summary": join_sections(sections), "summary_sections": sections}


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
        "source_scope": state.get("source_scope", ""),
        "context_note": state.get("context_note", ""),
        "source_citations": state.get("source_citations", []),
    }
    if state.get("review_notice"):
        payload["revision_notice"] = state["review_notice"]
    if state.get("revision_kind"):
        payload["revision_kind"] = state["revision_kind"]
    return _summary_feedback_delta(interrupt(payload))


async def UserFeedbackNodeStream(state: SummaryGenState) -> dict:
    return UserFeedbackNode(state)


def Router(state: SummaryGenState) -> str:
    if state.get("workflow_action") == "accept" or human_feedback_value(
        state
    ).strip().casefold() in {"save", "accept"}:
        decision = "save"
    else:
        kind = state.get("revision_kind") or classify_summary_feedback(human_feedback_value(state))
        decision = "structural" if kind == "structural" else "wording"
    metrics.increment("graph_route", workflow="summarization", decision=decision)
    logger.info("summary review decision=%s", decision)
    return decision


def _revision_inputs(state):
    kind = (
        state.get("revision_kind")
        if state.get("revision_kind") in {"wording", "unsupported"}
        else "wording"
    )
    parts = state.get("source_chunks") or [{"context": state["context"]}]
    sections = state.get("summary_sections") or [state["summary"]]
    if len(parts) != len(sections):
        raise ChainInputError("The summary sections do not match their source parts")
    return [
        {
            "context": part["context"],
            "original_summary": section,
            "user_feedback": {
                "kind": kind,
                "current": human_feedback_value(state),
                "prior": revision_context(state, settings),
            },
        }
        for part, section in zip(parts, sections)
    ]


def _revision_delta(state, responses):
    response = join_sections(responses)
    kind = (
        state.get("revision_kind")
        if state.get("revision_kind") in {"wording", "unsupported"}
        else "wording"
    )
    return {
        "summary": response,
        "summary_sections": responses,
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
        **revision_delta(
            state, settings, draft=state["summary"], status="unchanged", reason=notice
        ),
    }


def SummarizerRewriterNode(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    try:
        inputs = _revision_inputs(state)
        response = [
            get_task_chain("summary_revision", settings).invoke(
                item, config=map_config(config, len(inputs))
            )
            for item in inputs
        ]
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unchanged_summary(
            state, "The summary revision could not be validated, so the current draft was kept."
        )
    return _revision_delta(state, response)


async def SummarizerRewriterNodeStream(
    state: SummaryGenState, config: RunnableConfig = None
) -> dict:
    try:
        inputs, response = _revision_inputs(state), []
        for item in inputs:
            response.append(
                await get_task_chain("summary_revision", settings).ainvoke(
                    item, config=map_config(config, len(inputs))
                )
            )
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unchanged_summary(
            state, "The summary revision could not be validated, so the current draft was kept."
        )
    return _revision_delta(state, response)


def _structure_inputs(state):
    parts = state.get("source_chunks") or [{"context": state["context"]}]
    sections = state.get("summary_sections") or [state["summary"]]
    outlines = state.get("source_outlines") or [state["Main_Points"]]
    if len(parts) != len(sections) or len(parts) != len(outlines):
        raise ChainInputError("The summary sections do not match their source parts")
    return [
        {
            "context": part["context"],
            "outline": outline,
            "original_summary": section,
            "user_feedback": human_feedback_value(state),
        }
        for part, section, outline in zip(parts, sections, outlines)
    ]


def _structure_delta(state, results):
    notice = (
        " ".join(dict.fromkeys(result.limitation for result in results if result.limitation))
        or None
    )
    outline = join_sections([result.outline for result in results])
    summary = join_sections([result.summary for result in results])
    return {
        "Main_Points": outline,
        "summary": summary,
        "source_outlines": [result.outline for result in results],
        "summary_sections": [result.summary for result in results],
        "revision_kind": "structural",
        "review_notice": notice,
        **revision_delta(
            state,
            settings,
            draft=summary,
            status="revised",
            reason=notice,
        ),
    }


def SummarizerStructureNode(state: SummaryGenState, config: RunnableConfig = None) -> dict:
    try:
        inputs = _structure_inputs(state)
        result = [
            get_task_chain("summary_structure_revision", settings).invoke(
                item, config=map_config(config, len(inputs))
            )
            for item in inputs
        ]
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unchanged_summary(
            state,
            "The structural revision could not be validated, so the outline and summary were left unchanged.",
        )
    return _structure_delta(state, result)


async def SummarizerStructureNodeStream(
    state: SummaryGenState, config: RunnableConfig = None
) -> dict:
    try:
        inputs, result = _structure_inputs(state), []
        for item in inputs:
            result.append(
                await get_task_chain("summary_structure_revision", settings).ainvoke(
                    item, config=map_config(config, len(inputs))
                )
            )
    except (ChainInputError, StructuredOutputError, ProviderExecutionError):
        return _unchanged_summary(
            state,
            "The structural revision could not be validated, so the outline and summary were left unchanged.",
        )
    return _structure_delta(state, result)


stream_graph = StateGraph(SummaryGenState)
stream_graph.add_node(
    "main_point_summarizer",
    RunnableLambda(SummarizerMainPointNode, afunc=SummarizerMainPointNodeStream),
)
stream_graph.add_node(
    "summarizer_writer", RunnableLambda(SummarizerWriterNode, afunc=SummarizerWriterNodeStream)
)
stream_graph.add_node(
    "user_feedback", RunnableLambda(UserFeedbackNode, afunc=UserFeedbackNodeStream)
)
stream_graph.add_node(
    "summarizer_rewriter",
    RunnableLambda(SummarizerRewriterNode, afunc=SummarizerRewriterNodeStream),
)
stream_graph.add_node(
    "summarizer_structure",
    RunnableLambda(SummarizerStructureNode, afunc=SummarizerStructureNodeStream),
)
stream_graph.add_edge(START, "main_point_summarizer")
stream_graph.add_edge("main_point_summarizer", "summarizer_writer")
stream_graph.add_edge("summarizer_writer", "user_feedback")
stream_graph.add_conditional_edges(
    "user_feedback",
    Router,
    {"save": END, "wording": "summarizer_rewriter", "structural": "summarizer_structure"},
)
stream_graph.add_edge("summarizer_rewriter", "user_feedback")
stream_graph.add_edge("summarizer_structure", "user_feedback")


def compile_summary_graph(checkpointer):
    return stream_graph.compile(checkpointer=checkpointer)
