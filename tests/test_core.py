import importlib
from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic import ValidationError


def test_settings_load_original_environment_contract(tmp_path, monkeypatch):
    from helper.Config import get_settings

    template = Path(__file__).resolve().parents[1] / ".env.example"
    for name in dotenv_values(template):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(template.read_text(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = get_settings()

    assert settings.QUESTION_GENERATER_PROVIDER == "OLLAMA"
    assert settings.QUESTION_GENERATER_MODEL_ID == "qwen3:1.7b"
    assert settings.SUMMARIZER_MAINPOINT_PROVIDER == "GOOGLE_GENAI"
    assert settings.EMBEDDING_MODEL_PROVIDER == "LOCAL_EMBEDDING"
    assert (settings.CHUNK_SIZE, settings.CHUNK_OVERLAP) == (350, 50)
    assert settings.LANGSMITH_PROJECT == "Wizora-AI"
    assert settings.LANGSMITH_TRACING is False


def test_missing_required_provider_configuration_is_rejected(monkeypatch):
    from helper.Config import Settings

    monkeypatch.delenv("QUESTION_GENERATER_PROVIDER")

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)

    assert "QUESTION_GENERATER_PROVIDER" in str(error.value)


@pytest.mark.parametrize(
    ("provider", "constructor"),
    [("OLLAMA", "ChatOllama"), ("GOOGLE_GENAI", "ChatGoogleGenerativeAI"), ("GROQ", "ChatGroq")],
)
def test_llm_provider_preserves_model_and_temperature(monkeypatch, provider, constructor):
    module = importlib.import_module("llm.LLMProviderFactory")
    calls = []
    client = object()

    def create_client(**kwargs):
        calls.append(kwargs)
        return client

    monkeypatch.setattr(module, constructor, create_client)

    assert module.LLMProviderFactory({}).create(provider, "original-model", 0.37) is client
    assert calls == [{"model": "original-model", "temperature": 0.37}]
    assert module.LLMProviderFactory({}).create("UNKNOWN") is None


def test_embedding_provider_selection_without_model_download(monkeypatch):
    module = importlib.import_module("llm.EmbeddingProviderFactory")
    local = object()
    hosted = object()
    calls = []
    monkeypatch.setattr(module, "LocalEmbeddingProvider", lambda: local)

    def huggingface(**kwargs):
        calls.append(kwargs)
        return hosted

    monkeypatch.setattr(module, "HuggingFaceEmbeddings", huggingface)
    factory = module.EmbeddingProviderFactory({})

    assert factory.create("LOCAL_EMBEDDING", "unused-model-name") is local
    assert factory.create("HUGGINGFACE", "configured-model") is hosted
    assert calls == [{"model_name": "configured-model"}]
    assert factory.create("UNKNOWN") is None


def test_recursive_splitter_preserves_overlap_and_document_content():
    from helper.text_splitter import text_splitter

    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = text_splitter(text, chunk_size=10, chunk_overlap=3, separators=[""])

    assert chunks[0] == text[:10]
    assert all(len(chunk) <= 10 for chunk in chunks)
    assert all(left[-3:] == right[:3] for left, right in zip(chunks, chunks[1:]))
    assert chunks[0] + "".join(chunk[3:] for chunk in chunks[1:]) == text


def test_question_schema_supports_mcq_and_true_false():
    from chains import Feedback, Question

    common = {"question": "Is the source sorted?", "answer": "True", "explanation": "The source states this."}
    true_false = Question(question_type="T/F", options=None, **common)
    mcq = Question(question_type="MCQ", options=["A. Sorted", "B. Unsorted"], **common)

    assert true_false.model_dump()["options"] is None
    assert mcq.model_dump()["options"] == ["A. Sorted", "B. Unsorted"]
    assert "Answer: True" in true_false.format()
    assert Feedback(feedback="Explain the reason").format() == "Feedback: Explain the reason\n"
    with pytest.raises(ValidationError):
        Question(question_type="essay", options=None, **common)


@pytest.mark.parametrize(
    ("name", "inputs", "expected"),
    [
        ("QuestionGenPrompt", {"context": "Unique source context", "question_type": "MCQ"}, "Unique source context"),
        ("QuestionRefinerPrompt", {"question": "Unique question", "options": None, "answer": "True", "explanation": "Evidence"}, "Unique question"),
        ("QuestionRewriterPrompt", {"history": [("human", "Prior feedback")], "original_question": "Unique original", "feedback": "Rewrite clearly"}, "Prior feedback"),
        ("SummarizerMainPointPrompt", {"context": "Unique source context"}, "Unique source context"),
        ("SummarizerGenPrompt", {"context": "Unique source context", "table_of_contents": "Unique outline"}, "Unique outline"),
        ("SummarizerRewriterPrompt", {"context": "Unique source context", "original_summary": "Unique summary", "user_feedback": "Add detail"}, "Unique summary"),
        ("GradeQuestionPrompt", {"docs": "Unique retrieved context", "question": "Ask about source"}, "Unique retrieved context"),
    ],
)
def test_prompt_inputs_reach_the_model(name, inputs, expected):
    import chains

    messages = getattr(chains, name).format_messages(**inputs)

    assert messages[0].type == "system"
    assert expected in "\n".join(message.content for message in messages)
