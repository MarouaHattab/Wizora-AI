import logging
from functools import partial
from time import perf_counter

import anyio
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.config import merge_configs
from langgraph.graph import END, START, StateGraph

from helper.workflow_metrics import metrics

from chains.contracts import ChainInputError, StructuredOutputError, structured_output, text_fragment, text_output
from chains.runtime import get_task_chain, get_task_token_counter
from chains.schemes.RAGSchem import EvidenceDecision, GroundedAnswer
from helper import get_settings
from helper.asset_index import RetrievalHit, ensure_asset_index
from helper.asset_storage import load_asset_documents
from helper.rag_context import CallMetrics, bounded_history, conservative_token_count, evidence_budget, fallback_query, needs_context
from helper.rag_evidence import render_cited_answer, select_evidence, validate_evidence
from helper.workflow_history import canonical_messages, message_delta
from .GraphState import QuestionAnswerState

settings = get_settings()
logger = logging.getLogger(__name__)
REFUSAL = "Seems that your question is not related to the uploaded content. I cannot answer this question!"
INSUFFICIENT = "The supplied source evidence is insufficient to answer this question reliably."
UNVERIFIED = "I could not verify an answer against the supplied source evidence. Please narrow the question or provide additional source material."


def _question_text(state):
    messages = canonical_messages(state)
    if not messages or not isinstance(messages[-1], HumanMessage):
        raise ChainInputError("Question answering requires a pending human question")
    return text_output(messages[-1])


def _history(state):
    return bounded_history(canonical_messages(state)[:-1], _question_text(state), settings)


def _call_config(config, metrics):
    return merge_configs(config or {}, {"callbacks": [metrics], "configurable": {
        "rag_input_budget": settings.RAG_CONTEXT_WINDOW_TOKENS - settings.RAG_OUTPUT_RESERVE_TOKENS,
    }})


def _call(task, inputs, config):
    metrics = CallMetrics()
    result = get_task_chain(task, settings).invoke(inputs, config=_call_config(config, metrics))
    return result, metrics.result(task)


async def _acall(task, inputs, config):
    metrics = CallMetrics()
    result = await get_task_chain(task, settings).ainvoke(inputs, config=_call_config(config, metrics))
    return result, metrics.result(task)


def _diagnostics(state, call=None, **updates):
    diagnostics = dict(state.get("rag_diagnostics") or {})
    calls = list(diagnostics.get("model_calls", []))
    if call is not None:
        calls.append(call)
    return {**diagnostics, **updates, "model_calls": calls}


def _chunk_id(document):
    identifier = document.id or document.metadata.get("chunk_id")
    if not isinstance(identifier, str) or not identifier:
        raise ChainInputError("Indexed evidence requires a chunk identifier")
    return identifier


def _decision(state):
    value = state.get("evidence_decision")
    if isinstance(value, EvidenceDecision):
        return value
    return structured_output(EvidenceDecision, value)


def _query_start(state):
    question = _question_text(state)
    if len(question) > settings.RAG_QUERY_MAX_CHARACTERS:
        raise ChainInputError("The question exceeds the configured retrieval query limit")
    return {
        "original_question": question, "retrieval_query": question,
        "retrieval_pass": 0, "retrieved_ids": [], "retrieval_scores": {},
        "retrieval_issue": None, "on_topic": "",
        "relevant_text": [], "evidence_decision": None, "answer_draft": None,
        "grounding_assessment": None, "answer_citations": [], "answer_status": None,
        "grounding_status": "not_applicable", "rag_diagnostics": {"model_calls": [], "rewrite": "unnecessary"},
    }


def _rewrite_inputs(state):
    return {"question": _question_text(state), "history": _history(state),
            "history_messages": settings.RAG_HISTORY_MESSAGES, "history_characters": settings.RAG_HISTORY_CHARACTERS}


def _rewrite_needed(state):
    return needs_context(_question_text(state), canonical_messages(state)[:-1])


def _rewrite_delta(state, delta, result, metrics):
    query = result.query
    if len(query) > settings.RAG_QUERY_MAX_CHARACTERS:
        raise StructuredOutputError("The rewritten query exceeds the configured limit")
    return {**delta, "retrieval_query": query,
            "rag_diagnostics": {"rewrite": "resolved", "model_calls": [metrics]}}


def _rewrite_fallback(state, delta, metrics=None):
    return {**delta, "retrieval_query": fallback_query(
        _question_text(state), _history(state), settings.RAG_QUERY_MAX_CHARACTERS,
    ), "rag_diagnostics": {"rewrite": "fallback" if metrics else "deterministic",
        "model_calls": [metrics.result("query_rewrite", outcome="fallback")] if metrics else []}}


def _unresolved_query(delta):
    return {**delta, "retrieval_issue": "The recent conversational subject does not fit the context budget. Please restate the subject in a standalone question."}


def understand_query(state: QuestionAnswerState, config: RunnableConfig = None):
    delta = _query_start(state)
    if not _rewrite_needed(state):
        return delta
    if not _history(state):
        return _unresolved_query(delta)
    if not settings.RAG_REWRITE_ENABLED:
        return _rewrite_fallback(state, delta)
    metrics = CallMetrics()
    try:
        result, call = _call("query_rewrite", _rewrite_inputs(state), config)
        return _rewrite_delta(state, delta, result, call)
    except Exception:
        return _rewrite_fallback(state, delta, metrics)


async def understand_query_async(state: QuestionAnswerState, config: RunnableConfig = None):
    delta = _query_start(state)
    if not _rewrite_needed(state):
        return delta
    if not _history(state):
        return _unresolved_query(delta)
    if not settings.RAG_REWRITE_ENABLED:
        return _rewrite_fallback(state, delta)
    metrics = CallMetrics()
    try:
        result, call = await _acall("query_rewrite", _rewrite_inputs(state), config)
        return _rewrite_delta(state, delta, result, call)
    except Exception:
        return _rewrite_fallback(state, delta, metrics)


def _retrieval_work(state, corrective):
    documents = load_asset_documents(state["asset_id"], context=state["context"])
    index = ensure_asset_index(state["asset_id"], state["context"], settings, documents=documents)
    if corrective and state.get("index_version") != index.index_version:
        raise ChainInputError("The asset index changed during this question")
    issue = state.get("retrieval_issue")
    if issue:
        return [], index.index_version, issue
    if not index.query_fits(state["retrieval_query"]):
        return [], index.index_version, "The retrieval question exceeds the embedding model's input budget. Please restate the subject in a shorter standalone question."
    hits = index.retrieve_scored(
        state["retrieval_query"], strategy=settings.RAG_RETRIEVAL_STRATEGY,
        k=max(settings.RAG_TOP_K, settings.RAG_CORRECTIVE_TOP_K) if corrective else settings.RAG_TOP_K,
        fetch_k=max(settings.RAG_FETCH_K, settings.RAG_CORRECTIVE_FETCH_K) if corrective else settings.RAG_FETCH_K,
        min_relevance=settings.RAG_MIN_RELEVANCE, lambda_mult=settings.RAG_MMR_LAMBDA,
        exclude_ids=state.get("retrieved_ids", []) if corrective else (), adjacent=corrective,
    )
    return hits, index.index_version, None


def _retrieval_delta(state, hits, version, elapsed, corrective, issue=None):
    scores = dict(state.get("retrieval_scores", {})) if corrective else {}
    combined = [RetrievalHit(document, scores.get(_chunk_id(document), 0), rank)
                for rank, document in enumerate(state.get("relevant_text", []), 1)] if corrective else []
    combined += hits
    counter = get_task_token_counter("grounded_answer", settings, initialize=False) if combined else None
    native_counter = counter is not None
    if combined:
        max_tokens, counter = evidence_budget(state["original_question"], state["retrieval_query"], _history(state), settings, counter)
    else:
        max_tokens, counter = settings.RAG_MAX_EVIDENCE_TOKENS, conservative_token_count
    selected = select_evidence(
        combined, asset_id=state["asset_id"], index_version=version,
        max_documents=settings.RAG_MAX_EVIDENCE_DOCUMENTS,
        max_characters=settings.RAG_MAX_EVIDENCE_CHARACTERS,
        max_tokens=max_tokens, token_count=counter,
        near_duplicate_threshold=settings.RAG_NEAR_DUPLICATE_THRESHOLD,
    )
    for hit in hits:
        scores[_chunk_id(hit.document)] = hit.relevance
    retrieved = list(dict.fromkeys([*(state.get("retrieved_ids", []) if corrective else []), *(_chunk_id(hit.document) for hit in hits)]))
    previous = [_chunk_id(document) for document in state.get("relevant_text", [])] if corrective else []
    logger.info("QA retrieval (pass=%d, candidates=%d, selected=%d, elapsed_ms=%.2f)", int(corrective), len(hits), len(selected), elapsed)
    metrics.increment("rag_retrieval", workflow="qa", corrective=str(bool(corrective)).lower())
    metrics.observe("rag_retrieval_ms", round(elapsed, 3), workflow="qa")
    metrics.increment("rag_chunks_selected", amount=len(selected), workflow="qa")
    return {
        "relevant_text": selected, "index_version": version, "retrieval_pass": int(corrective),
        "retrieval_issue": issue,
        "retrieved_ids": retrieved, "retrieval_scores": scores,
        "evidence_changed": previous != [_chunk_id(document) for document in selected],
        "rag_diagnostics": _diagnostics(state, retrieval_passes=int(corrective) + 1,
            retrieved_count=len(retrieved), selected_count=len(selected),
            retrieval_ms=round(state.get("rag_diagnostics", {}).get("retrieval_ms", 0) + elapsed, 2),
            token_budget_method="model_tokenizer" if native_counter else "conservative_utf8_bytes"),
    }


def _retrieve(state, corrective=False):
    started = perf_counter()
    hits, version, issue = _retrieval_work(state, corrective)
    return _retrieval_delta(state, hits, version, (perf_counter() - started) * 1000, corrective, issue)


async def _aretrieve(state, corrective=False):
    started = perf_counter()
    hits, version, issue = await anyio.to_thread.run_sync(partial(_retrieval_work, state, corrective))
    return _retrieval_delta(state, hits, version, (perf_counter() - started) * 1000, corrective, issue)


def retrieve(state: QuestionAnswerState):
    return _retrieve(state)


async def retrieve_async(state: QuestionAnswerState):
    return await _aretrieve(state)


def corrective_retrieval(state: QuestionAnswerState):
    return _retrieve(state, True)


async def corrective_retrieval_async(state: QuestionAnswerState):
    return await _aretrieve(state, True)


def _evidence_inputs(state):
    return {"question": state["original_question"], "retrieval_query": state["retrieval_query"],
            "documents": state["relevant_text"], "asset_id": state["asset_id"], "index_version": state["index_version"]}


def _deterministic_decision(state):
    if not state["relevant_text"]:
        return EvidenceDecision(status="insufficient", supporting_chunk_ids=[], missing_information=state.get("retrieval_issue") or "No usable source excerpts were retrieved within the configured evidence budget.")
    if state.get("retrieval_pass", 0) and not state.get("evidence_changed") and state.get("grounding_status") != "rejected":
        return _decision(state)
    return None


def _decision_delta(state, decision, call=None):
    metrics.increment("rag_evidence", workflow="qa", status=decision.status)
    return {"evidence_decision": decision, "on_topic": "No" if decision.status == "off_topic" else "Yes",
            "rag_diagnostics": _diagnostics(state, call, evidence_status=decision.status)}


def assess_evidence(state: QuestionAnswerState, config: RunnableConfig = None):
    decision = _deterministic_decision(state)
    if decision is not None:
        return _decision_delta(state, decision)
    decision, call = _call("evidence_assessment", _evidence_inputs(state), config)
    return _decision_delta(state, decision, call)


async def assess_evidence_async(state: QuestionAnswerState, config: RunnableConfig = None):
    decision = _deterministic_decision(state)
    if decision is not None:
        return _decision_delta(state, decision)
    decision, call = await _acall("evidence_assessment", _evidence_inputs(state), config)
    return _decision_delta(state, decision, call)


def evidence_router(state):
    status = _decision(state).status
    if status in {"supported", "partial"}:
        decision = "answer"
    elif status == "insufficient" and not state.get("retrieval_issue") and state["retrieval_pass"] < settings.RAG_CORRECTIVE_PASSES:
        decision = "correct"
    else:
        decision = "abstain"
    metrics.increment("graph_route", workflow="qa", decision=decision)
    logger.info("qa evidence route=%s status=%s", decision, status)
    return decision


def _answer_inputs(state):
    decision = _decision(state)
    approved = set(decision.supporting_chunk_ids)
    documents = [document for document in state["relevant_text"] if _chunk_id(document) in approved]
    if not documents:
        raise ChainInputError("The evidence decision does not match packed source excerpts")
    return {**_evidence_inputs(state), "documents": documents, "history": _history(state), "evidence_decision": decision,
            "history_messages": settings.RAG_HISTORY_MESSAGES, "history_characters": settings.RAG_HISTORY_CHARACTERS}


def _draft_delta(state, draft, call):
    counter = get_task_token_counter("grounded_answer", settings, initialize=False) or conservative_token_count
    if counter(draft.model_dump_json()) > settings.RAG_OUTPUT_RESERVE_TOKENS:
        raise StructuredOutputError("The answer exceeds the configured output reserve")
    return {"answer_draft": draft, "rag_diagnostics": _diagnostics(state, call)}


def generate_answer(state: QuestionAnswerState, config: RunnableConfig = None):
    draft, call = _call("grounded_answer", _answer_inputs(state), config)
    return _draft_delta(state, draft, call)


async def generate_answer_streaming(state: QuestionAnswerState, config: RunnableConfig = None):
    draft, call = await _acall("grounded_answer", _answer_inputs(state), config)
    return _draft_delta(state, draft, call)


def _answer_draft(state):
    value = state.get("answer_draft")
    if isinstance(value, GroundedAnswer):
        return value
    return structured_output(GroundedAnswer, value)


def _grounding_inputs(state):
    return {"question": state["original_question"], "documents": state["relevant_text"],
            "answer": _answer_draft(state), "asset_id": state["asset_id"], "index_version": state["index_version"]}


def _rejected_grounding(state, call=None):
    metrics.increment("rag_grounding", workflow="qa", status="rejected")
    # Corrective retrieval still needs the pending human question as the last
    # message. Publish an abstention only after the bounded recovery is spent.
    final_message = {} if _can_correct_grounding(state) else message_delta(state, UNVERIFIED, settings)
    return {**final_message, "answer_draft": None,
            "answer_citations": [], "answer_status": "insufficient",
            "grounding_status": "rejected", "grounding_assessment": None,
            "rag_diagnostics": _diagnostics(state, call, grounding="rejected")}


def _can_correct_grounding(state):
    return not state.get("retrieval_issue") and state.get("retrieval_pass", 0) < settings.RAG_CORRECTIVE_PASSES


def grounding_router(state):
    decision = "correct" if state.get("grounding_status") == "rejected" and _can_correct_grounding(state) else "finish"
    metrics.increment("graph_route", workflow="qa", decision="grounding_" + decision)
    logger.info("qa grounding route=%s pass=%d", decision, state.get("retrieval_pass", 0))
    return decision


def _grounding_delta(state, assessment, call):
    validate_evidence(state["relevant_text"], asset_id=state["asset_id"], index_version=state["index_version"])
    supported = all(claim.supported for claim in assessment.claims)
    if not supported:
        return {**_rejected_grounding(state, call), "grounding_assessment": assessment}
    metrics.increment("rag_grounding", workflow="qa", status="accepted")
    text, citations = render_cited_answer(
        _answer_draft(state).claims, state["relevant_text"], asset_id=state["asset_id"],
        index_version=state["index_version"], missing_information=_decision(state).missing_information,
    )
    return {**message_delta(state, text, settings), "answer_draft": None,
            "answer_citations": citations, "answer_status": _decision(state).status,
            "grounding_status": "model_assessed", "grounding_assessment": assessment,
            "rag_diagnostics": _diagnostics(state, call, grounding="model_assessed")}


def validate_grounding(state: QuestionAnswerState, config: RunnableConfig = None):
    try:
        assessment, call = _call("grounding_check", _grounding_inputs(state), config)
    except (ChainInputError, StructuredOutputError) as exc:
        logger.warning("Grounding check could not verify the draft answer (exception_type=%s)", type(exc).__name__)
        return _rejected_grounding(state)
    return _grounding_delta(state, assessment, call)


async def validate_grounding_async(state: QuestionAnswerState, config: RunnableConfig = None):
    try:
        assessment, call = await _acall("grounding_check", _grounding_inputs(state), config)
    except (ChainInputError, StructuredOutputError) as exc:
        logger.warning("Grounding check could not verify the draft answer (exception_type=%s)", type(exc).__name__)
        return _rejected_grounding(state)
    return _grounding_delta(state, assessment, call)


def off_topic_response(state: QuestionAnswerState):
    decision = _decision(state)
    text = REFUSAL if decision.status == "off_topic" else INSUFFICIENT
    if decision.status == "insufficient" and decision.missing_information:
        text += " Missing evidence: " + decision.missing_information
    return {**message_delta(state, text, settings), "answer_status": decision.status,
            "answer_citations": [], "grounding_status": "not_applicable", "answer_draft": None}


def message_text(message):
    return text_fragment(message)


def _build_graph(answer_name):
    graph = StateGraph(QuestionAnswerState)
    graph.add_node("understand_query", RunnableLambda(understand_query, afunc=understand_query_async))
    graph.add_node("retrieve", RunnableLambda(retrieve, afunc=retrieve_async))
    graph.add_node("assess_evidence", RunnableLambda(assess_evidence, afunc=assess_evidence_async))
    graph.add_node("corrective_retrieval", RunnableLambda(corrective_retrieval, afunc=corrective_retrieval_async))
    graph.add_node(answer_name, RunnableLambda(generate_answer, afunc=generate_answer_streaming))
    graph.add_node("validate_grounding", RunnableLambda(validate_grounding, afunc=validate_grounding_async))
    graph.add_node("off_topic_response", off_topic_response)
    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "retrieve")
    graph.add_edge("retrieve", "assess_evidence")
    graph.add_conditional_edges("assess_evidence", evidence_router,
                                {"answer": answer_name, "correct": "corrective_retrieval", "abstain": "off_topic_response"})
    graph.add_edge("corrective_retrieval", "assess_evidence")
    graph.add_edge(answer_name, "validate_grounding")
    graph.add_conditional_edges("validate_grounding", grounding_router,
                                {"correct": "corrective_retrieval", "finish": END})
    graph.add_edge("off_topic_response", END)
    return graph


stream_graph = _build_graph("generate_answer_streaming")


def compile_qa_graph(checkpointer):
    return stream_graph.compile(checkpointer=checkpointer)
