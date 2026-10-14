import asyncio
import importlib
import uuid

import pytest


@pytest.fixture
def graph_modules(monkeypatch):
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    return {
        "qg": importlib.import_module("controllers.QuestionGenGraphController"),
        "summary": importlib.import_module("controllers.SummarizerGenGraphController"),
        "qa": importlib.import_module("controllers.QuestionAnswerGraphController"),
    }


def _models(monkeypatch, module, scripts):
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    calls = []

    class ScriptedModel(RunnableLambda):
        def __init__(self, name, responses):
            values = iter(responses)

            def respond(prompt, schema=None):
                messages = prompt.to_messages()
                calls.append((name, messages))
                response = next(values)
                return schema(**response) if schema else AIMessage(content=response)

            self.response_fn = respond
            super().__init__(respond)

        def with_structured_output(self, schema):
            respond = self.response_fn
            return RunnableLambda(lambda prompt: respond(prompt, schema))

    models = {}
    for attribute, responses in scripts.items():
        monkeypatch.setattr(module.config, attribute, attribute)
        models[attribute] = ScriptedModel(attribute, responses)

    def create(factory, provider, model_id=None, model_temperature=0.2):
        return models[model_id]

    monkeypatch.setattr(module.LLMProviderFactory, "create", create)
    return calls


def _question(label, question_type="MCQ"):
    return {
        "question": label,
        "question_type": question_type,
        "options": ["A. Ordered delivery", "B. Unordered delivery"] if question_type == "MCQ" else None,
        "answer": "A" if question_type == "MCQ" else "True",
        "explanation": "TCP ensures packets arrive in order.",
    }


def _question_state(question_type="MCQ"):
    return {
        "context": "TCP ensures that packets arrive in order.",
        "question_type": question_type,
        "question": "",
        "options": None,
        "answer": "",
        "explanation": "",
        "history": [],
        "feedback": "",
    }


def _summary_state():
    return {
        "context": "TCP ensures that packets arrive in order.",
        "summary": "",
        "old_summary": "",
        "feedback": "",
        "Main_Points": "",
    }


def _qa_state(question="How does TCP deliver packets?"):
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=question)],
        "context": "TCP ensures that packets arrive in order.",
        "relevant_text": [],
        "on_topic": "",
        "conversation_history": [],
    }


def _thread():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def _retrieval_double(monkeypatch, module):
    from langchain_core.documents import Document

    calls = []
    embedding = object()
    documents = [Document(page_content="TCP ensures that packets arrive in order.")]

    class Retriever:
        def invoke(self, query):
            calls.append(("query", query))
            return documents

    class Store:
        def as_retriever(self, **kwargs):
            calls.append(("retriever", kwargs))
            return Retriever()

    def from_texts(texts, embeddings):
        calls.append(("index", texts, embeddings))
        return Store()

    monkeypatch.setattr(module.EmbeddingProviderFactory, "create", lambda factory, provider, model_name=None: embedding)
    monkeypatch.setattr(module.Chroma, "from_texts", staticmethod(from_texts))
    return calls, documents, embedding


@pytest.mark.parametrize("question_type", ["MCQ", "T/F"])
def test_question_generation_pauses_with_structured_question(graph_modules, monkeypatch, question_type):
    module = graph_modules["qg"]
    calls = _models(monkeypatch, module, {"QUESTION_GENERATER_MODEL_ID": [_question("Initial question", question_type)]})
    config = _thread()

    result = module.qg_graph.invoke(_question_state(question_type), config=config)
    snapshot = module.qg_graph.get_state(config)

    assert result["question"] == "Initial question"
    assert result["question_type"] == question_type
    assert result["answer"] == ("A" if question_type == "MCQ" else "True")
    assert result["history"] == []
    assert snapshot.next == ("human_feedback",)
    assert snapshot.tasks[0].interrupts[0].value["gernerated_question"] == "Initial question"
    assert len(calls) == 1


def test_question_generation_preserves_history_across_auto_and_user_resume(graph_modules, monkeypatch):
    from langgraph.types import Command

    module = graph_modules["qg"]
    calls = _models(monkeypatch, module, {
        "QUESTION_GENERATER_MODEL_ID": [_question("Initial question")],
        "QUESTION_REFINER_MODEL_ID": [{"feedback": "Make the distractor clearer."}],
        "QUESTION_REWRITER_MODEL_ID": [_question("Automatically refined question"), _question("User refined question")],
    })
    config = _thread()
    module.qg_graph.invoke(_question_state(), config=config)

    automatic = module.qg_graph.invoke(Command(resume="auto"), config=config)
    assert automatic["question"] == "Automatically refined question"
    assert automatic["feedback"] == "Make the distractor clearer."
    assert len(automatic["history"]) == 1
    assert module.qg_graph.get_state(config).next == ("human_feedback",)

    rewritten = module.qg_graph.invoke(Command(resume="Use simpler wording"), config=config)
    assert rewritten["question"] == "User refined question"
    assert rewritten["feedback"] == "Use simpler wording"
    assert len(rewritten["history"]) == 2
    assert "Automatically refined question" in rewritten["history"][0]
    assert "Orginal Question:" in rewritten["history"][1]
    rewrite_calls = [messages for name, messages in calls if name == "QUESTION_REWRITER_MODEL_ID"]
    assert "Automatically refined question" in "\n".join(str(message.content) for message in rewrite_calls[1])

    completed = module.qg_graph.invoke(Command(resume="SAVE"), config=config)
    assert completed["question"] == "User refined question"
    assert completed["history"] == rewritten["history"]
    assert module.qg_graph.get_state(config).next == ()
    assert [name for name, _ in calls] == [
        "QUESTION_GENERATER_MODEL_ID", "QUESTION_REFINER_MODEL_ID",
        "QUESTION_REWRITER_MODEL_ID", "QUESTION_REWRITER_MODEL_ID",
    ]


def test_question_sessions_keep_independent_checkpoints(graph_modules, monkeypatch):
    from langgraph.types import Command

    module = graph_modules["qg"]
    _models(monkeypatch, module, {"QUESTION_GENERATER_MODEL_ID": [_question("First session"), _question("Second session")]})
    first, second = _thread(), _thread()
    module.qg_graph.invoke(_question_state(), config=first)
    module.qg_graph.invoke(_question_state(), config=second)
    module.qg_graph.invoke(Command(resume="save"), config=first)

    assert module.qg_graph.get_state(first).values["question"] == "First session"
    assert module.qg_graph.get_state(first).next == ()
    assert module.qg_graph.get_state(second).values["question"] == "Second session"
    assert module.qg_graph.get_state(second).next == ("human_feedback",)


def test_summary_sync_nodes_preserve_source_message_and_string_types(graph_modules, monkeypatch):
    from langchain_core.messages import AIMessage

    module = graph_modules["summary"]
    calls = _models(monkeypatch, module, {
        "SUMMARIZER_MAINPOINT_MODEL_ID": ["1. Reliable delivery"],
        "SUMMARIZER_WRITER_MODEL_ID": ["TCP delivers packets in order."],
        "SUMMARIZER_REWRITER_MODEL_ID": ["TCP guarantees ordered packet delivery."],
    })
    state = _summary_state()
    module.SummarizerMainPointNode(state)
    module.SummarizerWriterNode(state)

    assert isinstance(state["Main_Points"], AIMessage)
    assert isinstance(state["summary"], AIMessage)
    original = state["summary"]
    state["feedback"] = "Make it clearer"
    module.SummarizerRewriterNode(state)

    assert state["old_summary"] is original
    assert state["summary"] == "TCP guarantees ordered packet delivery."
    assert "Make it clearer" in calls[-1][1][-1].content


def test_summary_stream_graph_interrupts_rewrites_and_saves(graph_modules, monkeypatch):
    from langgraph.types import Command

    module = graph_modules["summary"]
    calls = _models(monkeypatch, module, {
        "SUMMARIZER_MAINPOINT_MODEL_ID": ["1. Reliable delivery"],
        "SUMMARIZER_WRITER_MODEL_ID": ["TCP delivers packets in order."],
        "SUMMARIZER_REWRITER_MODEL_ID": ["TCP guarantees ordered packet delivery."],
    })
    config = _thread()

    async def workflow():
        initial = await module.stream_summarizer_graph.ainvoke(_summary_state(), config=config)
        initial_snapshot = module.stream_summarizer_graph.get_state(config)
        rewritten = await module.stream_summarizer_graph.ainvoke(Command(resume="Make it clearer"), config=config)
        rewritten_snapshot = module.stream_summarizer_graph.get_state(config)
        final = await module.stream_summarizer_graph.ainvoke(Command(resume="save"), config=config)
        return initial, initial_snapshot, rewritten, rewritten_snapshot, final

    initial, initial_snapshot, rewritten, rewritten_snapshot, final = asyncio.run(workflow())
    assert initial["Main_Points"] == "1. Reliable delivery"
    assert initial["summary"] == "TCP delivers packets in order."
    assert initial_snapshot.next == ("user_feedback",)
    assert initial_snapshot.tasks[0].interrupts[0].value["waiting_for_input"] is True
    assert rewritten["old_summary"] == initial["summary"]
    assert rewritten["summary"] == "TCP guarantees ordered packet delivery."
    assert rewritten_snapshot.next == ("user_feedback",)
    assert final["summary"] == rewritten["summary"]
    assert module.stream_summarizer_graph.get_state(config).next == ()
    assert len(calls) == 3


def test_qa_graph_grades_retrieved_context_and_records_conversation(graph_modules, monkeypatch):
    module = graph_modules["qa"]
    retrieval, documents, embedding = _retrieval_double(monkeypatch, module)
    calls = _models(monkeypatch, module, {
        "GRADE_QA_MODEL_ID": [{"score": "Yes"}],
        "QUESTION_ANSWERER_MODEL_ID": ["TCP ensures ordered delivery."],
    })
    result = module.qa_graph.invoke(_qa_state())

    assert result["on_topic"] == "Yes"
    assert result["relevant_text"] == documents
    assert retrieval[0][2] is embedding
    assert retrieval[1] == ("retriever", {"search_type": "mmr", "search_kwargs": {"k": 5}})
    assert retrieval[2] == ("query", "How does TCP deliver packets?")
    assert "TCP ensures that packets arrive in order." in calls[0][1][-1].content
    assert result["messages"][-1].content == "TCP ensures ordered delivery."
    assert [message.type for message in result["conversation_history"]] == ["human", "ai"]


def test_qa_graph_refuses_unrelated_questions_without_answer_model(graph_modules, monkeypatch):
    module = graph_modules["qa"]
    _retrieval_double(monkeypatch, module)
    calls = _models(monkeypatch, module, {"GRADE_QA_MODEL_ID": [{"score": "No"}]})

    result = module.qa_graph.invoke(_qa_state("What is today's weather?"))

    assert result["messages"][-1].content == "Seems that your question is not related to the uploaded content. I cannot answer this question!"
    assert result["conversation_history"] == []
    assert [name for name, _ in calls] == ["GRADE_QA_MODEL_ID"]


def test_stream_qa_preserves_history_when_invoked_with_next_question(graph_modules, monkeypatch):
    from langchain_core.messages import HumanMessage

    module = graph_modules["qa"]
    _retrieval_double(monkeypatch, module)
    calls = _models(monkeypatch, module, {
        "GRADE_QA_MODEL_ID": [{"score": "Yes"}, {"score": "yes"}],
        "QUESTION_ANSWERER_MODEL_ID": ["TCP delivers in order.", "Yes, it is ordered."],
    })
    config = _thread()

    async def workflow():
        first = await module.stream_qa_graph.ainvoke(_qa_state(), config=config)
        second = await module.stream_qa_graph.ainvoke({
            "messages": first["messages"] + [HumanMessage(content="Is it ordered?")],
        }, config=config)
        return first, second

    first, second = asyncio.run(workflow())
    assert len(first["conversation_history"]) == 2
    assert [message.type for message in second["conversation_history"]] == ["human", "ai", "human", "ai"]
    assert second["messages"][-1].content == "Yes, it is ordered."
    answer_prompts = [messages for name, messages in calls if name == "QUESTION_ANSWERER_MODEL_ID"]
    assert "TCP delivers in order." in answer_prompts[1][-1].content
    assert module.stream_qa_graph.get_state(config).values["context"] == _qa_state()["context"]


def test_stream_off_topic_preserves_source_single_character_final_message(graph_modules, monkeypatch):
    module = graph_modules["qa"]
    _retrieval_double(monkeypatch, module)
    _models(monkeypatch, module, {"GRADE_QA_MODEL_ID": [{"score": "No"}]})
    config = _thread()

    result = asyncio.run(module.stream_qa_graph.ainvoke(_qa_state("Tell me a joke"), config=config))

    assert result["messages"][-1].content == "!"
    assert result["conversation_history"] == []


def test_classifier_uses_real_chroma_mmr_with_deterministic_embeddings(graph_modules, monkeypatch):
    from chromadb.config import Settings
    from langchain_core.embeddings import Embeddings

    module = graph_modules["qa"]
    _models(monkeypatch, module, {"GRADE_QA_MODEL_ID": [{"score": "Yes"}]})
    monkeypatch.setenv("CHUNK_SIZE", "75")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")

    class DeterministicEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text):
            text = text.lower()
            return [float("tcp" in text or "packet" in text), float("garden" in text or "flower" in text), 0.01]

    embedding = DeterministicEmbeddings()
    monkeypatch.setattr(module.EmbeddingProviderFactory, "create", lambda factory, provider, model_name=None: embedding)
    original_from_texts = module.Chroma.from_texts
    original_as_retriever = module.Chroma.as_retriever
    stores, retrieval = [], []

    def from_texts(texts, embeddings):
        store = original_from_texts(
            texts, embeddings,
            collection_name="test-" + uuid.uuid4().hex,
            client_settings=Settings(anonymized_telemetry=False),
        )
        stores.append(store)
        return store

    def as_retriever(store, **kwargs):
        retrieval.append(kwargs)
        return original_as_retriever(store, **kwargs)

    monkeypatch.setattr(module.Chroma, "from_texts", staticmethod(from_texts))
    monkeypatch.setattr(module.Chroma, "as_retriever", as_retriever)
    state = _qa_state()
    state["context"] = "TCP ensures packets arrive in order. Reliable transport uses acknowledgments.\n\nA garden contains flowers. Flower gardens need soil and water."
    try:
        result = module.question_classifier(state)
        assert result["on_topic"] == "Yes"
        assert any("TCP" in document.page_content for document in result["relevant_text"])
        assert retrieval == [{"search_type": "mmr", "search_kwargs": {"k": 5}}]
        assert stores[0]._collection.count() == len(module.text_splitter(state["context"], chunk_size=75, chunk_overlap=0))
    finally:
        for store in stores:
            store.delete_collection()
