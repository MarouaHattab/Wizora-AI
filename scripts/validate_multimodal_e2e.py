"""Live API/ASR/LLM/vision validation, with per-step evidence and no model doubles.

Run only with deliberately configured providers; --configured-google uses the
already configured Google summary model for missing local Ollama tasks in this
process only. Results and isolated checkpoints/assets stay under --output.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default="output/multimodal-fixtures")
    parser.add_argument("--output", default="output/multimodal-validation")
    parser.add_argument("--configured-google", action="store_true")
    parser.add_argument("--vision", action="store_true", help="Use the explicitly configured Google summary model for vision")
    parser.add_argument("--only", nargs="*", default=["pdf", "docx", "txt", "mp3", "wav", "m4a", "mp4", "mov", "webm", "visual"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reuse-from", help="Reuse already validated assets and successful earlier stages from a recorded live run")
    parser.add_argument("--model-calls-per-minute", type=float, default=12, help="Shared rate limit on every real model call; no responses are substituted")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous_report = json.loads(Path(args.reuse_from).read_text(encoding="utf-8")) if args.reuse_from else {}
    source_output = Path(args.reuse_from).resolve().parent if args.reuse_from else output
    fixtures = Path(args.fixtures).resolve()
    os.environ["CHECKPOINT_DATABASE"] = str(source_output / "checkpoints.sqlite")
    os.environ["MEDIA_TIMEOUT_SECONDS"] = "600"
    os.environ["GRAPH_TIMEOUT_SECONDS"] = "300"
    os.environ["SESSION_LEASE_SECONDS"] = "330"
    os.environ["AUDIO_CHUNK_SECONDS"] = "20"
    os.environ["INDEX_ON_UPLOAD"] = "false"
    os.environ["LOG_LEVEL"] = "WARNING"
    os.environ["LANGSMITH_TRACING"] = "false"
    model = os.environ.get("SUMMARIZER_WRITER_MODEL_ID", "").strip('" ')
    if args.configured_google:
        if os.environ.get("SUMMARIZER_WRITER_PROVIDER", "").strip('" ').upper() != "GOOGLE_GENAI":
            raise SystemExit("--configured-google requires an existing configured Google summary provider")
        for prefix in ("QUESTION_GENERATER", "QUESTION_REFINER", "QUESTION_REWRITER"):
            os.environ[prefix + "_PROVIDER"] = "GOOGLE_GENAI"
            os.environ[prefix + "_MODEL_ID"] = model
            os.environ[prefix + "_TEMPERATURE"] = "0"
    if args.vision:
        os.environ["VISION_PROVIDER"] = "GOOGLE_GENAI"
        os.environ["VISION_MODEL"] = model
    from fastapi.testclient import TestClient
    from helper import asset_paths, get_settings
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from llm.LLMProviderFactory import LLMProviderFactory
    live_limiter = InMemoryRateLimiter(requests_per_second=args.model_calls_per_minute / 60,
                                      check_every_n_seconds=0.05, max_bucket_size=1)
    original_create = LLMProviderFactory.create
    def create_limited(self, *arguments, **keywords):
        real_model = original_create(self, *arguments, **keywords)
        real_model.rate_limiter = live_limiter
        return real_model
    # Only scheduling is wrapped. Extraction, model requests and responses remain real.
    LLMProviderFactory.create = create_limited
    asset_paths.ASSETS_ROOT = source_output / "assets"
    from main import app
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "real_models": True,
              "configured_google_override": args.configured_google, "question_test_temperature": 0 if args.configured_google else "configured", "vision_enabled": args.vision,
              "model": model, "reused_evidence_from": args.reuse_from, "formats": {}, "negative_cases": {}}
    results_path = output / "results.json"
    def save():
        results_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    def required(response):
        if not response.is_success:
            raise AssertionError(f"HTTP {response.status_code}: {response.text[:1200]}")
        return response.json()
    def stream(client, endpoint, payload):
        response = client.post(endpoint, json={**payload, "event_version": "v1"})
        if not response.is_success:
            raise AssertionError(f"HTTP {response.status_code}: {response.text[:1000]}")
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        terminals = [event for event in events if event.get("event") in {"completed", "awaiting_feedback", "failed"}]
        if not terminals:
            raise AssertionError("No terminal SSE event")
        last = terminals[-1]
        if last["event"] == "failed":
            raise AssertionError("Workflow failed: " + json.dumps(last["data"]))
        return last, events
    with TestClient(app) as client:
        report["health"] = required(client.get("/health/ready"))
        report["capabilities"] = required(client.get("/api/v1/preprocessing/assets/config"))
        for name, expected in (("empty.txt", 422), ("corrupt.pdf", 422), ("corrupt.docx", 422),
                               ("invalid.wav", 422), ("unsupported.exe", 415)):
            with (fixtures / name).open("rb") as file:
                response = client.post("/api/v1/preprocessing/upload_file/", files={"file": (name, file, mimetypes.guess_type(name)[0] or "application/octet-stream")})
            report["negative_cases"][name] = {"passed": response.status_code == expected, "expected": expected,
                                              "actual": response.status_code, "detail": response.text[:500]}
        response = client.post("/api/v1/preprocessing/upload_file/", files={"file": ("lesson.pdf", b"%PDF-", "audio/mpeg")})
        report["negative_cases"]["mime_mismatch"] = {"passed": response.status_code == 415, "actual": response.status_code}
        save()
        def run_format(extension):
            started = time.monotonic()
            previous = previous_report.get("formats", {}).get(extension, {})
            row = {"steps": dict(previous.get("steps", {})), "evidence": dict(previous.get("evidence", {})), "reused_steps": []}
            skip_summary = bool(previous.get("steps", {}).get("summary_save"))
            skip_questions = bool(previous.get("steps", {}).get("saved_question_library")) and (
                extension != "pdf" or bool(previous.get("steps", {}).get("duplicate_prevention")))
            current = "upload"
            try:
                source = fixtures / ("visual-only.mp4" if extension == "visual" else "lesson-long.txt" if extension == "long" else "lesson." + extension)
                if previous.get("steps", {}).get("upload") and previous.get("asset_id"):
                    asset = required(client.get("/api/v1/preprocessing/uploads/" + previous["asset_id"]))
                    row["reused_steps"].append("upload")
                else:
                    with source.open("rb") as file:
                        upload = required(client.post("/api/v1/preprocessing/uploads", files={"file": (source.name, file, mimetypes.guess_type(source.name)[0] or "application/octet-stream")}))
                    asset = required(client.get("/api/v1/preprocessing/uploads/" + upload["id"]))
                if asset.get("status") != "ready":
                    raise AssertionError("Processing failed: " + json.dumps(asset.get("error", asset)))
                asset_id = asset["id"]
                row["asset_id"] = asset_id
                row["evidence"]["asset"] = asset
                row["steps"][current] = True
                current = "type_detection"
                expected_type = "video" if extension in {"mp4", "mov", "webm", "visual"} else "audio" if extension in {"mp3", "wav", "m4a"} else "text" if extension in {"txt", "long"} else extension
                assert asset["source_type"] == expected_type, asset["source_type"]
                row["steps"][current] = True
                current = "extraction_accuracy"
                content = required(client.get(f"/api/v1/preprocessing/assets/{asset_id}/content"))
                text = "\n".join(d["page_content"] for d in content["documents"])
                row["evidence"]["extracted_text"] = text
                assert len(text.strip()) >= 100, "Extraction is not meaningful"
                if extension == "visual":
                    assert "VIOLET" in text.upper() and "73" in text, "Visual-only access code was not recovered"
                    assert asset["processing_scope"] == "visual_only", asset
                else:
                    assert "solar" in text.lower() and ("battery" in text.lower() or "inverter" in text.lower()), "Lesson facts missing"
                if expected_type == "audio":
                    assert any(d["metadata"].get("start", 0) >= 20 for d in content["documents"]), "Audio chunk timestamp offsets missing"
                row["steps"][current] = True
                if not skip_summary:
                    current = "summary_generation"
                    summary, _ = stream(client, "/api/v1/graph/summarizer/start_session_streaming", {"asset_id": asset_id})
                    assert summary["event"] == "awaiting_feedback"
                    assert len(summary["data"]["summary"]) > 40
                    summary_text = summary["data"]["summary"].casefold()
                    if extension == "visual":
                        assert "violet" in summary_text and "73" in summary_text
                    else:
                        assert "solar" in summary_text and "battery" in summary_text
                    if extension == "long":
                        assert "orchid" in summary_text and "92" in summary_text, "Final source part was not summarized"
                    row["evidence"]["summary"] = summary["data"]["summary"]
                    row["steps"][current] = True
                    current = "summary_feedback"
                    revision, _ = stream(client, "/api/v1/graph/summarizer/provide_feedback_streaming", {"thread_id": summary["thread_id"], "review_id": summary["data"]["review_id"], "feedback": "Use concise wording while keeping the source facts."})
                    assert revision["event"] == "awaiting_feedback"
                    if revision["data"].get("revision_notice"):
                        raise AssertionError("Summary revision was not applied: " + revision["data"]["revision_notice"])
                    row["evidence"]["summary_revision"] = revision["data"]
                    row["steps"][current] = True
                    if extension in {"pdf", "long"}:
                        current = "summary_structural_revision"
                        revision, _ = stream(client, "/api/v1/graph/summarizer/provide_feedback_streaming", {
                            "thread_id": revision["thread_id"], "review_id": revision["data"]["review_id"],
                            "feedback": "Use clear numbered section headings while keeping all source facts and the final checkpoint when present."})
                        assert revision["event"] == "awaiting_feedback"
                        assert not revision["data"].get("revision_notice"), revision["data"]
                        row["evidence"]["structural_revision"] = revision["data"]
                        row["steps"][current] = True
                    current = "summary_save"
                    saved, _ = stream(client, "/api/v1/graph/summarizer/provide_feedback_streaming", {"thread_id": revision["thread_id"], "review_id": revision["data"]["review_id"], "feedback": "save"})
                    assert saved["event"] == "completed"
                    row["steps"][current] = True
                else:
                    row["reused_steps"].extend(name for name, passed in previous["steps"].items() if passed and name.startswith("summary_"))
                if not skip_questions:
                    for kind, label in (("MCQ", "mcq"), ("T/F", "true_false")):
                        current = label + "_generation"
                        question = required(client.post("/api/v1/graph/qg/start_session", json={"asset_id": asset_id, "question_type": kind}))
                        assert question["status"] == "interrupted_for_feedback"
                        row["evidence"][label] = question["data_for_feedback"]
                        row["steps"][current] = True
                        current = label + "_feedback_refinement"
                        question = required(client.post("/api/v1/graph/qg/provide_feedback", json={"thread_id": question["thread_id"], "review_id": question["review_id"], "feedback": "Begin the explanation with exactly 'From the uploaded source: ' and keep it concise. Preserve the question, choices and correct answer."}))
                        assert question["status"] == "interrupted_for_feedback"
                        row["evidence"][label + "_revision"] = question
                        history = question["current_state"].get("history", [])
                        if not history or history[-1].get("status") != "revised":
                            raise AssertionError("Question revision did not produce a validated revision")
                        row["steps"][current] = True
                        if extension in {"pdf", "wav", "visual"} and kind == "MCQ":
                            current = "automatic_question_refinement"
                            question = required(client.post("/api/v1/graph/qg/provide_feedback", json={
                                "thread_id": question["thread_id"], "review_id": question["review_id"], "feedback": "refine"}))
                            assert question["status"] == "interrupted_for_feedback"
                            assert question["current_state"].get("critic_feedback"), "Automatic refinement did not produce a critic result"
                            row["evidence"][current] = question["data_for_feedback"]
                            row["steps"][current] = True
                        current = label + "_manual_update"
                        draft = question["data_for_feedback"]
                        edited = {"question_type": kind, "question": draft["generated_question"], "options": draft.get("options"), "answer": draft["answer"], "explanation": draft["explanation"] + " This is supported by the uploaded lesson."}
                        question = required(client.post("/api/v1/graph/qg/edit_question", json={"thread_id": question["thread_id"], "review_id": question["review_id"], "question": edited}))
                        assert question["data_for_feedback"]["explanation"] == edited["explanation"]
                        row["steps"][current] = True
                        current = label + "_save"
                        question = required(client.post("/api/v1/graph/qg/provide_feedback", json={"thread_id": question["thread_id"], "review_id": question["review_id"], "feedback": "accept"}))
                        assert question["status"] == "completed"
                        row["steps"][current] = True
                    current = "saved_question_library"
                    library = required(client.get("/api/v1/sessions/question_generation", params={"asset_id": asset_id}))
                    assert len(library["questions"]) >= 2
                    row["evidence"]["saved_questions"] = library["questions"]
                    row["steps"][current] = True
                    if extension == "pdf":
                        current = "duplicate_prevention"
                        from helper.content_context import duplicate_question
                        extra = required(client.post("/api/v1/graph/qg/start_session", json={"asset_id": asset_id, "question_type": "MCQ"}))
                        stems = [item["result"]["question"] for item in library["questions"]]
                        assert not duplicate_question(extra["data_for_feedback"]["generated_question"], stems)
                        row["evidence"][current] = extra["data_for_feedback"]
                        row["steps"][current] = True
                else:
                    row["reused_steps"].extend(name for name, passed in previous["steps"].items() if passed and (name.startswith(("mcq_", "true_false_")) or name in {"saved_question_library", "automatic_question_refinement", "duplicate_prevention"}))
                current = "grounded_qa"
                query = "What is the experiment access code shown on the slide?" if extension == "visual" else "How many solar panels does the laboratory have?"
                qa, events = stream(client, "/api/v1/graph/qa/start_session_stream", {"asset_id": asset_id, "initial_question": query})
                answer = "".join(event["data"].get("text", "") for event in events if event["event"] == "text_delta")
                row["evidence"]["qa"] = {**qa, "answer": answer}
                assert qa["data"].get("citations"), "Grounded answer has no citations"
                if extension == "visual":
                    assert "VIOLET" in answer.upper() and "73" in answer
                else:
                    assert "six" in answer.lower() or "6" in answer
                row["steps"][current] = True
                current = "conversation_followup"
                next_question = "Was that information spoken, or shown visually?" if extension == "visual" else "What is their total power in the test conditions?"
                followup, followup_events = stream(client, "/api/v1/graph/qa/continue_conversation_stream", {"thread_id": qa["thread_id"], "next_question": next_question})
                followup_text = "".join(event["data"].get("text", "") for event in followup_events if event["event"] == "text_delta")
                row["evidence"]["followup"] = {**followup, "answer": followup_text}
                assert followup["event"] == "completed" and followup["data"].get("citations")
                if extension != "visual":
                    assert "1200" in followup_text.replace(",", "") or "1,200" in followup_text or "twelve hundred" in followup_text.lower(), followup_text
                row["steps"][current] = True
                current = "qa_abstention"
                unknown, unknown_events = stream(client, "/api/v1/graph/qa/continue_conversation_stream", {"thread_id": qa["thread_id"], "next_question": "What was the installation cost in euros?"})
                row["evidence"]["unknown_qa"] = unknown
                row["evidence"]["unknown_qa"]["answer"] = "".join(event["data"].get("text", "") for event in unknown_events if event["event"] == "text_delta")
                assert unknown["data"].get("answer_status") in {"insufficient", "off_topic"}, unknown
                assert not unknown["data"].get("citations"), "Abstention should not claim source support for unknown cost"
                row["steps"][current] = True
                assert all(row["steps"].values()), "An earlier failed step has not been verified successfully"
                row["passed"] = True
            except Exception as exc:
                row["steps"][current] = False
                row.update(passed=False, failed_step=current, error=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
            row["seconds"] = round(time.monotonic() - started, 2)
            return extension, row
        with ThreadPoolExecutor(max_workers=max(1, min(3, args.workers))) as pool:
            futures = [pool.submit(run_format, extension) for extension in args.only]
            for future in as_completed(futures):
                extension, row = future.result()
                report["formats"][extension] = row
                save()
                print(extension, "PASS" if row["passed"] else "FAIL " + row.get("failed_step", ""), row["seconds"], flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    print(results_path, flush=True)
    return 0 if all(row["passed"] for row in report["formats"].values()) and all(row["passed"] for row in report["negative_cases"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
