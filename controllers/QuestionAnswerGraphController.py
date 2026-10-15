from functools import partial

import anyio
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from chains import GradeQuestion
from chains.contracts import ChainInputError, text_fragment, text_output
from chains.runtime import get_task_chain
from helper import get_settings
from helper.asset_index import ensure_asset_index
from helper.asset_storage import load_asset_documents
from helper.workflow_history import canonical_messages, message_delta
from .GraphState import QuestionAnswerState

settings = get_settings()
REFUSAL = "Seems that your question is not related to the uploaded content. I cannot answer this question!"


def _question_text(state: QuestionAnswerState) -> str:
    messages = canonical_messages(state)
    if not messages or not isinstance(messages[-1], HumanMessage):
        raise ChainInputError("Question answering requires a pending human question")
    return text_output(messages[-1])


def _retrieve_documents(state: QuestionAnswerState, question: str):
    documents = load_asset_documents(state["asset_id"], context=state["context"])
    asset_index = ensure_asset_index(
        state["asset_id"], state["context"], settings, documents=documents
    )
    return asset_index.retrieve(question)


def _topic_inputs(documents, question):
    return {"question": question, "docs": "\n".join(doc.page_content for doc in documents)}


def question_classifier(state: QuestionAnswerState, config: RunnableConfig = None) -> dict:
    question = _question_text(state)
    documents = _retrieve_documents(state, question)
    result = get_task_chain("topic_grade", settings).invoke(
        _topic_inputs(documents, question), config=config
    )
    return {"relevant_text": documents, "on_topic": result.score}


async def question_classifier_streaming(state: QuestionAnswerState, config: RunnableConfig = None) -> dict:
    question = _question_text(state)
    documents = await anyio.to_thread.run_sync(partial(_retrieve_documents, state, question))
    result = await get_task_chain("topic_grade", settings).ainvoke(
        _topic_inputs(documents, question), config=config
    )
    return {"relevant_text": documents, "on_topic": result.score}


def on_topic_router(state: QuestionAnswerState):
    on_topic = GradeQuestion.model_validate({"score": state["on_topic"]}).score
    return "on_topic" if on_topic == "Yes" else "off_topic"


def retrieve(state: QuestionAnswerState):
    return {}


def _answer_inputs(state: QuestionAnswerState):
    messages = canonical_messages(state)
    return {
        "documents": state["relevant_text"],
        "question": _question_text(state),
        "history": messages[:-1],
        "history_messages": settings.QA_HISTORY_MESSAGES,
        "history_characters": settings.QA_HISTORY_CHARACTERS,
    }


def generate_answer(state: QuestionAnswerState, config: RunnableConfig = None):
    answer = get_task_chain("answer", settings).invoke(_answer_inputs(state), config=config)
    return message_delta(state, text_output(answer), settings)


def message_text(message):
    return text_fragment(message)


async def generate_answer_streaming(state: QuestionAnswerState, config: RunnableConfig = None):
    chunks = []
    async for chunk in get_task_chain("answer", settings).astream(_answer_inputs(state), config=config):
        chunks.append(message_text(chunk))
    return message_delta(state, text_output("".join(chunks)), settings)


def off_topic_response(state: QuestionAnswerState):
    return message_delta(state, REFUSAL, settings)


async def off_topic_response_streaming(state: QuestionAnswerState):
    return off_topic_response(state)


stream_graph = StateGraph(QuestionAnswerState)
stream_graph.add_node("topic_decision", question_classifier_streaming)
stream_graph.add_node("off_topic_response", off_topic_response_streaming)
stream_graph.add_node("retrieve", retrieve)
stream_graph.add_node("generate_answer_streaming", generate_answer_streaming)
stream_graph.add_conditional_edges(
    "topic_decision", on_topic_router,
    {"on_topic": "retrieve", "off_topic": "off_topic_response"},
)
stream_graph.add_edge("retrieve", "generate_answer_streaming")
stream_graph.add_edge("generate_answer_streaming", END)
stream_graph.add_edge("off_topic_response", END)
stream_graph.add_edge(START, "topic_decision")
checkpointer = MemorySaver()
stream_qa_graph = stream_graph.compile(checkpointer=checkpointer)

qa_graph = StateGraph(QuestionAnswerState)
qa_graph.add_node("question_classifier", question_classifier)
qa_graph.add_node("off_topic_response", off_topic_response)
qa_graph.add_node("retrieve", retrieve)
qa_graph.add_node("generate_answer", generate_answer)
qa_graph.add_conditional_edges(
    "question_classifier", on_topic_router,
    {"on_topic": "retrieve", "off_topic": "off_topic_response"},
)
qa_graph.add_edge("retrieve", "generate_answer")
qa_graph.add_edge("generate_answer", END)
qa_graph.add_edge("off_topic_response", END)
qa_graph.add_edge(START, "question_classifier")
qa_graph = qa_graph.compile()
