import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import get_settings
from llm import LLMProviderFactory
from chains import QuestionGenPrompt, QuestionRefinerPrompt, QuestionRewriterPrompt
from chains import Question, Feedback, GenerationResult, RevisionResult
from chains.contracts import InsufficientSourceError, StructuredOutputError, requested_question, structured_output
from .GraphState import QuestionGenState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

config = get_settings()


def current_question(state):
    return structured_output(Question, {
        key: state[key] for key in ("question", "question_type", "options", "answer", "explanation")
    })


def QuestionGenerator(state: QuestionGenState) -> QuestionGenState:
    if not state["context"].strip():
        raise InsufficientSourceError("The source contains no educational material.")
    llm = LLMProviderFactory(config).create(
        config.QUESTION_GENERATER_PROVIDER,
        config.QUESTION_GENERATER_MODEL_ID,
        config.QUESTION_GENERATER_TEMPERATURE,
    )
    chain = QuestionGenPrompt | llm.with_structured_output(GenerationResult)
    result = structured_output(GenerationResult, chain.invoke({
        "context": state["context"], "question_type": state["question_type"]
    }))
    if result.status == "insufficient_source":
        raise InsufficientSourceError(result.reason)
    question = requested_question(result.question, state["question_type"])
    return {**state, **question.model_dump(), "history": state.get("history") or []}


def HumanFeedback(state: QuestionGenState) -> QuestionGenState:
    human_feedback = interrupt({
        "gernerated_question": state["question"],
        "options": state["options"],
        "answer": state["answer"],
        "explanation": state["explanation"],
        "message": "Provide feedback or type 'auto' to automatically generate a new question or 'save' to finish",
    })
    state["feedback"] = human_feedback
    return state


def Router(state: QuestionGenState) -> str:
    if state["feedback"].strip().casefold() == "save":
        return "save"
    elif state["feedback"].strip().casefold() == "auto":
        return "auto"
    return "feedback"


def QuestionRefiner(state: QuestionGenState) -> QuestionGenState:
    question = current_question(state)
    llm = LLMProviderFactory(config).create(
        config.QUESTION_REFINER_PROVIDER,
        config.QUESTION_REFINER_MODEL_ID,
        config.QUESTION_REFINER_TEMPERATURE,
    )
    chain = QuestionRefinerPrompt | llm.with_structured_output(Feedback)
    response = structured_output(Feedback, chain.invoke({
        "context": state["context"],
        **question.model_dump(),
        "options": json.dumps(question.options, ensure_ascii=False),
    }))
    return {**state, "feedback": response.model_dump_json()}


def QuestionRewriter(state: QuestionGenState) -> QuestionGenState:
    draft = current_question(state)
    prior_revisions = [entry for entry in state.get("history", []) if isinstance(entry, dict)][-3:]
    llm = LLMProviderFactory(config).create(
        config.QUESTION_REWRITER_PROVIDER,
        config.QUESTION_REWRITER_MODEL_ID,
        config.QUESTION_REWRITER_TEMPERATURE,
    )
    chain = QuestionRewriterPrompt | llm.with_structured_output(RevisionResult)
    response = structured_output(RevisionResult, chain.invoke({
        "context": state["context"],
        "question_type": state["question_type"],
        "current_question": draft.model_dump_json(),
        "feedback": state["feedback"],
        "history": json.dumps(prior_revisions, ensure_ascii=False),
    }))
    question = requested_question(response.question, state["question_type"])
    if response.status == "unchanged" and question != draft:
        raise StructuredOutputError("An unchanged revision must preserve the current question")
    revision = {
        "feedback": state["feedback"],
        "status": response.status,
        "reason": response.reason,
        "question": question.model_dump(),
    }
    return {**state, **question.model_dump(), "history": (prior_revisions + [revision])[-3:]}

graph = StateGraph(QuestionGenState)

graph.add_node("question_gen", QuestionGenerator)
graph.add_node("question_refiner", QuestionRefiner)
graph.add_node("question_rewriter", QuestionRewriter)
graph.add_node("human_feedback", HumanFeedback)

graph.add_edge(START, "question_gen")
graph.add_edge("question_gen", "human_feedback")
graph.add_conditional_edges("human_feedback", Router, {"save": END, "auto": "question_refiner", "feedback": "question_rewriter"})

graph.add_edge("question_refiner", "question_rewriter")
graph.add_edge("question_rewriter", "human_feedback")

checkpointer = MemorySaver()
qg_graph = graph.compile(checkpointer=checkpointer)

graph = graph.compile() # for langgraph dev 
# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))
