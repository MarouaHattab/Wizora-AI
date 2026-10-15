import os
import sys
sys.path.append(os.path.dirname(os.path.dirname( os.path.abspath(__file__))))

from helper import get_settings
from helper.asset_index import ensure_asset_index
from helper.asset_storage import load_asset_documents
from llm import LLMProviderFactory

from chains import GradeQuestion, GradeQuestionPrompt
from chains.QuestionAnswerChain import QuestionAnswerPrompt, prepare_question_answer_inputs
from chains.contracts import text_output
from .GraphState import QuestionAnswerState

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

config = get_settings()


def _question_text(state: QuestionAnswerState) -> str:
    message = state["messages"][-1]
    if isinstance(message, dict):
        message = message["content"]
    return text_output(message)

def question_classifier(state: QuestionAnswerState) -> QuestionAnswerState: 
    question = _question_text(state)

    if "conversation_history" not in state:
        state["conversation_history"] = []

    documents = load_asset_documents(state["asset_id"], context=state["context"])
    asset_index = ensure_asset_index(state["asset_id"], state["context"], config, documents=documents)
    state["relevant_text"] = asset_index.retrieve(question)

    docs = "\n".join([doc.page_content for doc in state["relevant_text"]])
    llm = LLMProviderFactory(config).create(
        config.GRADE_QA_PROVIDER, 
        config.GRADE_QA_MODEL_ID, 
        config.GRADE_QA_TEMPERATURE
    )

    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = GradeQuestionPrompt | structured_llm
    result = grader_llm.invoke({"question": question, "docs": docs})
    if isinstance(result, GradeQuestion):
        result = result.model_dump()
    result = GradeQuestion.model_validate(result)
    
    state["on_topic"] = result.score

    return state

def on_topic_router(state: QuestionAnswerState): 
    on_topic = GradeQuestion.model_validate({"score": state["on_topic"]}).score
    if on_topic == "Yes":
        return "on_topic"
    return "off_topic"

def retrieve(state: QuestionAnswerState):
    return state

def generate_answer(state: QuestionAnswerState): 
    question = _question_text(state)
    chat_history = state.get("conversation_history", [])
    inputs = prepare_question_answer_inputs(state["relevant_text"], question, chat_history)
    llm = LLMProviderFactory(config).create(
        config.QUESTION_ANSWERER_PROVIDER, 
        config.QUESTION_ANSWERER_MODEL_ID, 
        config.QUESTION_ANSWERER_TEMPERATURE
    )

    rag_chain = QuestionAnswerPrompt | llm
    generation = AIMessage(content=text_output(rag_chain.invoke(inputs)))
    return {
        **state,
        "messages": state["messages"] + [generation],
        "conversation_history": chat_history + [state["messages"][-1], generation],
    }

def message_text(message):
    if isinstance(message, BaseMessage):
        return message.text()
    return ""


async def generate_answer_streaming(state: QuestionAnswerState): 
    question = _question_text(state)
    chat_history = state.get("conversation_history", [])
    inputs = prepare_question_answer_inputs(state["relevant_text"], question, chat_history)
    llm = LLMProviderFactory(config).create(
        config.QUESTION_ANSWERER_PROVIDER, 
        config.QUESTION_ANSWERER_MODEL_ID, 
        config.QUESTION_ANSWERER_TEMPERATURE
    )

    rag_chain = QuestionAnswerPrompt | llm
    
    full_response = ""
    async for chunk in rag_chain.astream(inputs):
        full_response += message_text(chunk)
    generation = AIMessage(content=text_output(full_response))
    return {
        **state,
        "messages": state["messages"] + [generation],
        "conversation_history": chat_history + [state["messages"][-1], generation],
    }

def off_topic_response(state: QuestionAnswerState): 
    state["messages"].append(AIMessage(content="Seems that your question is not related to the uploaded content. I cannot answer this question!"))
    return state

async def off_topic_response_streaming(state: QuestionAnswerState):
    response_text = "Seems that your question is not related to the uploaded content. I cannot answer this question!"
    generation = AIMessage(content=response_text)
    return {
        **state,
        "messages": state["messages"] + [generation],
        "conversation_history": state.get("conversation_history", []) + [
            state["messages"][-1], generation
        ],
    }


stream_graph = StateGraph(QuestionAnswerState)

stream_graph.add_node("topic_decision", question_classifier)
stream_graph.add_node("off_topic_response", off_topic_response_streaming)
stream_graph.add_node("retrieve", retrieve)
stream_graph.add_node("generate_answer_streaming", generate_answer_streaming)

stream_graph.add_conditional_edges(
    "topic_decision", 
    on_topic_router, 
    {
        "on_topic": "retrieve", 
        "off_topic": "off_topic_response"
    }
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
    "question_classifier", 
    on_topic_router, 
    {
        "on_topic": "retrieve", 
        "off_topic": "off_topic_response"
    }
)

qa_graph.add_edge("retrieve", "generate_answer")
qa_graph.add_edge("generate_answer", END)
qa_graph.add_edge("off_topic_response", END)
qa_graph.add_edge(START, "question_classifier")

qa_graph = qa_graph.compile()

# from IPython.display import display, Image
# display(Image(qa_graph.get_graph().draw_mermaid_png()))
