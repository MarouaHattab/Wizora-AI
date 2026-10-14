import importlib

import pytest


def test_pdf_extraction_preserves_page_order(tmp_path):
    import fitz

    from tools.pdf_extractor_tool import extract_text_from_pdf, save_extracted_text

    pdf_path = tmp_path / "lesson.pdf"
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "First page: agent workflows.")
        pdf.new_page().insert_text((72, 72), "Second page: retrieval context.")
        pdf.save(pdf_path)

    documents = extract_text_from_pdf(str(pdf_path))
    assert len(documents) == 2
    assert "First page: agent workflows." in documents[0].page_content
    assert "Second page: retrieval context." in documents[1].page_content

    output_path = tmp_path / "extracted_text.txt"
    save_extracted_text(documents, str(output_path))
    assert output_path.read_text(encoding="utf-8") == "".join(
        document.page_content for document in documents
    )


def test_whisper_uses_tiny_model_for_each_transcription(monkeypatch):
    transcript_tool = importlib.import_module("tools.transcript_tool")
    loaded_models = []
    transcribed_paths = []

    class Model:
        def transcribe(self, path):
            transcribed_paths.append(path)
            return {"text": f"Transcript of {path}"}

    def load_model(name):
        loaded_models.append(name)
        return Model()

    monkeypatch.setattr(transcript_tool.whisper, "load_model", load_model)
    assert transcript_tool.transcript_loader("lecture.wav") == "Transcript of lecture.wav"
    assert transcript_tool.transcript_loader("lecture.mp4") == "Transcript of lecture.mp4"
    assert loaded_models == ["tiny", "tiny"]
    assert transcribed_paths == ["lecture.wav", "lecture.mp4"]


def test_whisper_transcription_failure_returns_error_text(monkeypatch):
    transcript_tool = importlib.import_module("tools.transcript_tool")

    class Model:
        def transcribe(self, path):
            raise RuntimeError("decoder failed")

    monkeypatch.setattr(transcript_tool.whisper, "load_model", lambda name: Model())
    assert transcript_tool.transcript_loader("lecture.wav") == (
        "Error transcribing audio: decoder failed"
    )


def test_whisper_model_loading_failure_propagates(monkeypatch):
    transcript_tool = importlib.import_module("tools.transcript_tool")

    def load_model(name):
        raise RuntimeError("weights unavailable")

    monkeypatch.setattr(transcript_tool.whisper, "load_model", load_model)
    with pytest.raises(RuntimeError, match="weights unavailable"):
        transcript_tool.transcript_loader("lecture.wav")
