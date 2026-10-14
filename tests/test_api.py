import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def api(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    main = importlib.import_module("main")
    routes = {
        name: importlib.import_module(f"routes.{name}_routes")
        for name in ("file_processing", "question_gen", "summarizer", "qa")
    }
    monkeypatch.chdir(tmp_path)
    for name in ("question_gen", "summarizer", "qa"):
        monkeypatch.setattr(routes[name], "ASSETS_BASE_PATH", str(tmp_path / "assets"))
    with TestClient(main.app) as client:
        yield SimpleNamespace(
            client=client, app=main.app, routes=routes, assets=tmp_path / "assets"
        )


@pytest.fixture
def asset(api):
    asset_id = "existing-asset"
    directory = api.assets / asset_id
    directory.mkdir(parents=True)
    (directory / "extracted_text.txt").write_text("Lesson context", encoding="utf-8")
    return asset_id


@pytest.fixture
def pdf_bytes():
    import fitz

    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Agent workflows use lesson context.")
        return pdf.tobytes()


def sse_payloads(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


class EventGraph:
    def __init__(self, events):
        self.events = events
        self.calls = []

    async def astream_events(self, payload, *, config, version):
        self.calls.append((payload, config, version))
        for event in self.events:
            yield event


def test_root_and_registered_routes(api):
    assert api.client.get("/").json() == {"message": "Welcome to Wizora-AI API."}
    assert api.app.title == "Wizora-AI API"
    paths = api.client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/",
        "/upload_file/",
        "/get_extracted_text/{asset_id}",
        "/api/v1/preprocessing/upload_file/",
        "/api/v1/preprocessing/get_extracted_text/{asset_id}",
        "/api/v1/graph/qg/start_session",
        "/api/v1/graph/qg/provide_feedback",
        "/api/v1/graph/summarizer/start_session_streaming",
        "/api/v1/graph/summarizer/provide_feedback_streaming",
        "/api/v1/graph/qa/start_session_stream",
        "/api/v1/graph/qa/continue_conversation_stream",
    }


@pytest.mark.parametrize(
    ("prefix", "filename", "content_type"),
    [("", "lesson.pdf", "application/octet-stream"),
     ("/api/v1/preprocessing", "lesson.bin", "application/pdf")],
)
def test_pdf_upload_and_retrieval_aliases(api, pdf_bytes, prefix, filename, content_type):
    response = api.client.post(
        f"{prefix}/upload_file/", files={"file": (filename, pdf_bytes, content_type)}
    )
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"message", "id", "text_path", "original_file_path"}
    assert result["message"] == "File processed successfully"
    assert Path(result["original_file_path"]).read_bytes() == pdf_bytes
    assert Path(result["text_path"]).is_file()
    for retrieval_prefix in ("", "/api/v1/preprocessing"):
        response = api.client.get(f"{retrieval_prefix}/get_extracted_text/{result['id']}")
        assert response.status_code == 200
        assert response.json() == {
            "asset_id": result["id"],
            "extracted_text": Path(result["text_path"]).read_text(encoding="utf-8"),
        }
        assert "Agent workflows use lesson context." in response.json()["extracted_text"]


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [("lecture.mp4", "application/octet-stream"), ("lecture.bin", "audio/wav")],
)
def test_media_upload_uses_transcription(api, monkeypatch, filename, content_type):
    calls = []

    def transcribe(path):
        calls.append(path)
        return "A transcribed lesson"

    monkeypatch.setattr(api.routes["file_processing"], "transcript_loader", transcribe)
    response = api.client.post(
        "/upload_file/", files={"file": (filename, b"media", content_type)}
    )
    assert response.status_code == 200
    assert calls == [response.json()["original_file_path"]]
    assert Path(response.json()["text_path"]).read_text(encoding="utf-8") == "A transcribed lesson"


def test_unsupported_upload_preserves_error_status_and_cleans_directory(api):
    response = api.client.post(
        "/upload_file/", files={"file": ("notes.txt", b"lesson", "text/plain")}
    )
    assert response.status_code == 500
    assert response.json()["detail"] == (
        "An error occurred during file processing: 400: Unsupported file type: text/plain. "
        "Only PDF, audio, or video files are allowed."
    )
    assert list(api.assets.iterdir()) == []


def test_processing_failure_cleans_uploaded_file(api, monkeypatch, pdf_bytes):
    def fail_extraction(path):
        raise RuntimeError("invalid PDF")

    monkeypatch.setattr(api.routes["file_processing"], "extract_text_from_pdf", fail_extraction)
    response = api.client.post(
        "/upload_file/", files={"file": ("lesson.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "An error occurred during file processing: invalid PDF"}
    assert list(api.assets.iterdir()) == []


@pytest.mark.parametrize("prefix", ["", "/api/v1/preprocessing"])
def test_preprocessing_missing_file_and_asset_validation(api, prefix):
    assert api.client.post(f"{prefix}/upload_file/").status_code == 422
    response = api.client.get(f"{prefix}/get_extracted_text/missing")
    assert response.status_code == 404
    assert response.json() == {"detail": "Extracted text not found for asset ID: missing"}
    assert api.client.get(f"{prefix}/get_extracted_text/%20").status_code == 400


@pytest.mark.parametrize(
    ("path", "payload"),
    [("qg/start_session", {"asset_id": "missing", "question_type": "MCQ"}),
     ("summarizer/start_session_streaming", {"asset_id": "missing"}),
     ("qa/start_session_stream", {"asset_id": "missing", "initial_question": "What?"})],
)
def test_agent_start_rejects_missing_assets(api, path, payload):
    response = api.client.post(f"/api/v1/graph/{path}", json=payload)
    assert response.status_code == 404
    assert response.json() == {"detail": "Asset text file not found for id: missing"}


@pytest.mark.parametrize(
    "path",
    ["qg/start_session", "qg/provide_feedback", "summarizer/start_session_streaming",
     "summarizer/provide_feedback_streaming", "qa/start_session_stream", "qa/continue_conversation_stream"],
)
def test_graph_routes_require_json_fields(api, path):
    assert api.client.post(f"/api/v1/graph/{path}", json={}).status_code == 422


def test_question_generation_start_and_feedback_shapes(api, asset, monkeypatch):
    result = {"question": "What is an agent?", "options": ["A", "B"],
              "answer": "A", "explanation": "It follows the workflow."}
    calls = []

    class Graph:
        def invoke(self, state, *, config):
            calls.append((state, config))
            return result

    monkeypatch.setattr(api.routes["question_gen"], "qg_graph", Graph())
    response = api.client.post(
        "/api/v1/graph/qg/start_session", json={"asset_id": asset, "question_type": "MCQ"}
    )
    assert response.status_code == 200
    started = response.json()
    thread_id = started["thread_id"]
    assert started["status"] == "interrupted_for_feedback"
    assert started["current_state"] == result
    assert started["data_for_feedback"] == {
        "generated_question": result["question"], "options": result["options"],
        "answer": result["answer"], "explanation": result["explanation"],
        "message": "Provide feedback or type 'auto' to automatically generate a new question or 'save' to finish",
    }
    assert calls[0] == ({
        "context": "Lesson context", "question_type": "MCQ", "question": "", "options": None,
        "answer": "", "explanation": "", "history": [], "feedback": "",
    }, {"configurable": {"thread_id": thread_id}})

    response = api.client.post(
        "/api/v1/graph/qg/provide_feedback", json={"thread_id": thread_id, "feedback": "auto"}
    )
    assert response.status_code == 200
    assert response.json() == started
    assert calls[-1] == ({"feedback": "auto"}, {"configurable": {"thread_id": thread_id}})

    response = api.client.post(
        "/api/v1/graph/qg/provide_feedback", json={"thread_id": thread_id, "feedback": "SAVE"}
    )
    assert response.json() == {"thread_id": thread_id, "status": "completed", "final_state": result}
    assert calls[-1][0] == {"feedback": "SAVE"}
    assert api.client.post(
        "/api/v1/graph/qg/start_session", json={"asset_id": asset, "question_type": "essay"}
    ).status_code == 422


def test_question_generation_graph_error_response(api, asset, monkeypatch):
    class Graph:
        def invoke(self, state, *, config):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(api.routes["question_gen"], "qg_graph", Graph())
    response = api.client.post(
        "/api/v1/graph/qg/start_session", json={"asset_id": asset, "question_type": "T/F"}
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Error invoking question generation graph: provider unavailable"}


def test_summarizer_start_stream_schema(api, asset, monkeypatch):
    from langchain_core.messages import AIMessageChunk

    graph = EventGraph([
        {"event": "on_chain_start", "name": "main_point_summarizer", "data": {"input": {"context": "Lesson context"}}},
        {"event": "on_chat_model_stream", "name": "Model", "data": {"chunk": AIMessageChunk(content="Main points")}},
        {"event": "on_chain_start", "name": "summarizer_writer", "data": {"input": {"Main_Points": "Main points"}}},
        {"event": "on_chat_model_stream", "name": "Model", "data": {"chunk": AIMessageChunk(content="Summary")}},
    ])
    monkeypatch.setattr(api.routes["summarizer"], "summarizer_graph", graph)
    payloads = sse_payloads(api.client.post(
        "/api/v1/graph/summarizer/start_session_streaming", json={"asset_id": asset}
    ))
    thread_id = payloads[0]["thread_id"]
    assert payloads[0] == {"thread_id": thread_id, "status": "starting_session"}
    assert payloads[1] == {
        "event": "on_chain_start", "name": "main_point_summarizer", "thread_id": thread_id,
        "status_update": "Starting: main_point_summarizer", "data": {"context": "Lesson context"},
    }
    assert [(item["token"], item["status_update"]) for item in payloads if item.get("event") == "token"] == [
        ("Main points", "main_point_summarizer"), ("Summary", "summarizer_writer")
    ]
    assert payloads[-1] == {"event": "stream_end", "thread_id": thread_id, "status_update": "Stream ended"}
    assert graph.calls == [({"context": "Lesson context", "summary": "", "old_summary": "", "feedback": "", "Main_Points": ""},
                            {"configurable": {"thread_id": thread_id}}, "v2")]


def test_summarizer_feedback_uses_resume_command(api, monkeypatch):
    from langchain_core.messages import AIMessageChunk
    from langgraph.types import Command

    graph = EventGraph([
        {"event": "on_chain_start", "name": "summarizer_rewriter", "data": {"input": Command(resume="Shorter")}},
        {"event": "on_chat_model_stream", "name": "Model", "data": {"chunk": AIMessageChunk(content="Revised")}},
    ])
    monkeypatch.setattr(api.routes["summarizer"], "summarizer_graph", graph)
    payloads = sse_payloads(api.client.post(
        "/api/v1/graph/summarizer/provide_feedback_streaming", json={"thread_id": "summary-thread", "feedback": "Shorter"}
    ))
    assert payloads[0] == {"thread_id": "summary-thread", "status": "resuming_with_feedback"}
    assert payloads[1]["data"] == {"resume_value": "Shorter"}
    assert payloads[2] == {"event": "token", "name": "Model", "thread_id": "summary-thread", "token": "Revised", "status_update": "summarizer_rewriter"}
    assert payloads[-1]["event"] == "stream_end"
    command, config, version = graph.calls[0]
    assert isinstance(command, Command)
    assert command.resume == "Shorter"
    assert config == {"configurable": {"thread_id": "summary-thread"}}
    assert version == "v2"


def test_summarizer_stream_errors_are_sse_events(api, asset, monkeypatch):
    class Graph:
        async def astream_events(self, payload, *, config, version):
            raise RuntimeError("provider unavailable")
            yield

    monkeypatch.setattr(api.routes["summarizer"], "summarizer_graph", Graph())
    payloads = sse_payloads(api.client.post(
        "/api/v1/graph/summarizer/start_session_streaming", json={"asset_id": asset}
    ))
    assert payloads[1] == {
        "event": "error", "thread_id": payloads[0]["thread_id"],
        "detail": "Error in summarization stream: provider unavailable", "error_type": "RuntimeError",
    }
    assert payloads[-1]["event"] == "stream_end"


def test_qa_start_stream_schema_and_initial_state(api, asset, monkeypatch):
    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

    graph = EventGraph([
        {"event": "on_chat_model_stream", "data": {"chunk": AIMessageChunk(content="Answer")}},
        {"event": "on_chain_stream", "name": "off_topic_response", "data": {"chunk": {"messages": [AIMessage(content="Stay on topic")]}}},
    ])
    monkeypatch.setattr(api.routes["qa"], "qa_graph", graph)
    response = api.client.post(
        "/api/v1/graph/qa/start_session_stream", json={"asset_id": asset, "initial_question": "What is an agent?"}
    )
    payloads = sse_payloads(response)
    thread_id = payloads[0]["thread_id"]
    assert payloads == [
        {"type": "metadata", "thread_id": thread_id, "asset_id": asset},
        {"type": "token", "content": "Answer"},
        {"type": "token", "content": "Stay on topic"},
        {"type": "complete"},
    ]
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["connection"] == "keep-alive"
    assert response.headers["access-control-allow-origin"] == "*"
    state, config, version = graph.calls[0]
    assert state == {"messages": [HumanMessage(content="What is an agent?")], "relevant_text": [],
                     "on_topic": "", "context": "Lesson context", "conversation_history": []}
    assert config == {"configurable": {"thread_id": thread_id}}
    assert version == "v2"


def test_qa_continuation_preserves_history_and_stream_schema(api, monkeypatch):
    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

    class Graph(EventGraph):
        def get_state(self, config):
            assert config == {"configurable": {"thread_id": "qa-thread"}}
            return SimpleNamespace(values={"messages": [
                HumanMessage(content="First question"), {"type": "ai", "content": "First answer"}
            ]})

    graph = Graph([{"event": "on_chat_model_stream", "data": {"chunk": AIMessageChunk(content="Next answer")}}])
    monkeypatch.setattr(api.routes["qa"], "qa_graph", graph)
    payloads = sse_payloads(api.client.post(
        "/api/v1/graph/qa/continue_conversation_stream", json={"thread_id": "qa-thread", "next_question": "Explain more"}
    ))
    assert payloads == [{"type": "metadata", "thread_id": "qa-thread"},
                        {"type": "token", "content": "Next answer"}, {"type": "complete"}]
    assert graph.calls == [({"messages": [HumanMessage(content="First question"), AIMessage(content="First answer"),
                                          HumanMessage(content="Explain more")]},
                            {"configurable": {"thread_id": "qa-thread"}}, "v2")]


def test_qa_missing_snapshot_response(api, monkeypatch):
    monkeypatch.setattr(api.routes["qa"], "qa_graph", SimpleNamespace(get_state=lambda config: None))
    response = api.client.post(
        "/api/v1/graph/qa/continue_conversation_stream", json={"thread_id": "missing", "next_question": "What?"}
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Session (thread_id: missing) not found or state is empty. Start a new session."}
