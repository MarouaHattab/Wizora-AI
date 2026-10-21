# Wizora AI

Wizora AI is a local educational assistant that turns uploaded documents and media into
structured study material. It provides a FastAPI backend, a Gradio user interface, and
three LangGraph workflows for question generation, summarization, and document-grounded
question answering.

## Features

- Upload and process PDF, DOCX, TXT, audio, and video files.
- Extract page-aware PDF text, ordered DOCX paragraphs/tables, and encoded plain text.
- Transcribe chunked audio locally with Whisper and preserve global timestamps.
- Optionally analyze sampled video frames with an explicitly configured vision model,
  including videos without speech. Transcript-only processing is clearly disclosed.
- Generate multiple-choice (`MCQ`) and true/false (`T/F`) questions.
- Review, manually edit, refine, rewrite, and save generated questions.
- Generate summaries and main-point outlines with interactive feedback and revision.
- Ask questions about an uploaded asset through a document-grounded chatbot.
- Stream summarization and chatbot progress over Server-Sent Events (SSE).
- Persist workflow checkpoints and session history in SQLite.
- Use configurable Ollama, Google Gemini, or Groq models for the language-model stages.
- Use local Sentence Transformers or Hugging Face embeddings for retrieval.

## Requirements

- Windows, macOS, or Linux
- Python 3.11.x (the project requires `>=3.11,<3.12`)
- `uv` for the documented setup, or another environment manager that can install the
  pinned dependencies from `pyproject.toml` and `uv.lock`
- FFmpeg for audio/video. The existing `imageio-ffmpeg` dependency supplies a binary;
  system FFmpeg/FFprobe or `FFMPEG_BINARY`/`FFPROBE_BINARY` can override it. FFprobe is
  optional because media inspection has an FFmpeg fallback.
- One configured model provider:
  - Ollama running locally, or
  - a Google Generative AI API key, or
  - a Groq API key
- An optional Hugging Face token when using the Hugging Face embedding provider

The application uses local SQLite files for workflow checkpoints and session metadata.
When `INDEX_ON_UPLOAD=true`, it also creates local Chroma indexes for document retrieval.
No hosted database or vector service is required by the repository.

## Installation

```powershell
git clone https://github.com/MarouaHattab/Wizora-AI.git
Set-Location Wizora-AI

uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv sync
Copy-Item .env.example .env
```

Edit `.env` and configure the providers and model identifiers you want to use. Keep
credentials out of source control. The checked-in `.env.example` contains placeholders
only.

At minimum, the model settings must be internally consistent. For example, when using
Ollama for question generation:

```dotenv
QUESTION_GENERATER_PROVIDER=OLLAMA
QUESTION_GENERATER_MODEL_ID=qwen3:1.7b
QUESTION_REFINER_PROVIDER=OLLAMA
QUESTION_REFINER_MODEL_ID=qwen3:1.7b
QUESTION_REWRITER_PROVIDER=OLLAMA
QUESTION_REWRITER_MODEL_ID=qwen3:1.7b
```

The default example also configures Google models for summarization and question
answering. Set `GOOGLE_API_KEY` before starting if those settings remain unchanged.

## Usage

### Start the backend

From the repository root:

```powershell
.\.venv\Scripts\python.exe main.py
```

The FastAPI server listens on `http://127.0.0.1:8000`. Its interactive OpenAPI
documentation is available at `http://127.0.0.1:8000/docs`.

### Start the Gradio frontend

In a second terminal with the same virtual environment:

```powershell
.\.venv\Scripts\python.exe gradio_ui.py
```

The frontend listens on `http://127.0.0.1:7860` by default. It communicates with the
backend through `helper/frontend_client.py` and `FRONTEND_BACKEND_URL`; model provider
credentials remain on the backend.

### Basic workflow

1. Open the Gradio URL and upload a supported file.
2. Select the uploaded asset.
3. Generate an MCQ or T/F question, start a summary, or ask a question in the chatbot.
4. For question generation and summarization, review the interrupted draft and submit
   feedback, an edit, or an acceptance action.
5. Continue a chatbot conversation using the returned thread ID.

The unified uploader queues extraction and displays the actual processing stages.
The source inspector shows page/line/element references or transcript timestamps,
visual frame observations, processing scope, and extraction warnings. It previews
up to eight source records; the combined extracted text remains available in full.

The backend can also be checked without a model call:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

## System Architecture

Wizora AI is a single-process application:

1. **Gradio frontend** — `gradio_ui.py` exposes upload, question, summary, and chatbot
   controls. `helper/frontend_client.py` calls the HTTP API and parses JSON or SSE
   responses.
2. **FastAPI backend** — `main.py` mounts file-processing routes, versioned session
   routes, and the three graph-agent route groups under `/api/v1`.
3. **Ingestion layer** — `helper/ingestion.py` validates sizes, extensions, MIME
   declarations and actual decodability; modality tools produce normalized source
   records. Video can combine timestamped ASR and sampled visual observations.
4. **Asset storage and retrieval** — processed source records and extracted text are
   stored under `assets/`. When indexing is enabled, Chroma and the configured
   embedding provider support retrieval for the QA workflow.
5. **LangGraph orchestration** — the controllers in `controllers/` define the state,
   nodes, conditional routes, interruptions, and checkpointed execution for each agent.
6. **Persistence and operations** — `langgraph-checkpoint-sqlite` stores graph state in
   the configured SQLite checkpoint database. The runtime is intentionally restricted to
   one worker process.

The application does not implement a separate microservice deployment in this
repository. The frontend and backend are separate processes during local development,
but the backend owns the AI providers and workflow state.

See [multimodal architecture and data flow](docs/multimodal-architecture.md) for the
processing diagram, metadata contract, generation budgets, and error handling.

## LangGraph Agent Architecture

The following images are generated by
[`scripts/generate_graph_diagrams.py`](scripts/generate_graph_diagrams.py) from the
compiled graphs exported by the three controller modules. They are not manually drawn.

### 1. Question Generation Workflow

![Question Generation Graph](assets/architecture/question_generation_graph.png)

Source: `controllers/QuestionGenGraphController.py`.

**Nodes and flow**

- `__start__` enters `question_gen`.
- `question_gen` creates the initial MCQ or T/F draft.
- `human_feedback` interrupts for a review decision.
- `question_refiner` applies critique-oriented refinement and routes either back to
  `human_feedback` or to `question_rewriter`.
- `question_rewriter` performs a broader rewrite and returns to `human_feedback`.
- `__end__` is reached when the feedback router returns `accept`.

The conditional feedback routes are `accept -> END`, `refine -> question_refiner`, and
`revise -> question_rewriter`. The graph is checkpointed so the API can resume it with a
LangGraph `Command` after the user supplies feedback. The API records the completed
state as the saved question.

### 2. Content Summarization Workflow

![Summarization Graph](assets/architecture/summarization_graph.png)

Source: `controllers/SummarizerGenGraphController.py`.

**Nodes and flow**

- `__start__` enters `main_point_summarizer`.
- `main_point_summarizer` extracts the main points.
- `summarizer_writer` produces the summary from those points.
- `user_feedback` interrupts for a save or revision decision.
- `summarizer_rewriter` handles wording revisions and returns to `user_feedback`.
- `summarizer_structure` handles structural revisions and returns to `user_feedback`.
- `__end__` is reached when the feedback router returns `save`.

The conditional routes are `save -> END`, `wording -> summarizer_rewriter`, and
`structural -> summarizer_structure`. The route handlers stream public stage events and
resume the checkpointed graph with the user's feedback.

### 3. Question Answering Workflow

![Question Answering Graph](assets/architecture/question_answering_graph.png)

Source: `controllers/QuestionAnswerGraphController.py`.

**Nodes and flow**

- `__start__` enters `understand_query`.
- `understand_query` classifies and prepares the incoming question.
- `retrieve` obtains relevant asset context.
- `assess_evidence` decides whether the evidence is sufficient.
- `generate_answer_streaming` drafts a grounded answer when the route is `answer`.
- `validate_grounding` validates the generated answer. A rejected first draft can
  use the existing bounded corrective retrieval pass before a final answer or abstention.
- `corrective_retrieval` performs one corrective retrieval pass when the route is
  `correct`, then loops back to `assess_evidence`.
- `off_topic_response` handles `abstain` and then reaches `__end__`.

The graph has no human review interruption. Conversation turns are persisted in the
checkpointed state, and the API resumes a completed QA thread for the next question.
Answers are streamed as SSE events and include answer status, citations, and grounding
metadata when available.

Grounding recovery shares the same single corrective-pass budget as insufficient
initial retrieval. Exact arithmetic and defined unit conversions may be explained
from fully cited operands; unsupported factors or conditions remain disallowed.

Regenerate the images after changing a workflow:

```powershell
.\.venv\Scripts\python.exe scripts\generate_graph_diagrams.py
```

## API Documentation

All versioned endpoints below are prefixed with `/api/v1`. The unversioned upload routes
are retained by the backend for compatibility.

### Health and root

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Return the API welcome message. |
| `GET` | `/health/live` | Liveness check: `{"status":"live"}`. |
| `GET` | `/health/ready` | Check SQLite workflow readiness. Returns `503` when unavailable. |
| `GET` | `/health/metrics` | Return runtime metrics; returns `503` until the runtime is ready. |

### File processing

| Method | Path | Body / response |
| --- | --- | --- |
| `POST` | `/api/v1/preprocessing/upload_file/` | Multipart field `file`; synchronously processes and returns an asset job object. |
| `POST` | `/api/v1/preprocessing/uploads` | Multipart field `file`; queues processing and returns `202` with job metadata. |
| `GET` | `/api/v1/preprocessing/uploads/{asset_id}` | Return processing status and metadata. |
| `GET` | `/api/v1/preprocessing/assets` | Return `{"assets":[...]}`. |
| `GET` | `/api/v1/preprocessing/assets/config` | Return max upload size, supported extensions, and index mode. |
| `GET` | `/api/v1/preprocessing/assets/{asset_id}` | Return asset metadata. |
| `GET` | `/api/v1/preprocessing/assets/{asset_id}/content` | Inspect validated source records and their original locators, scope, and warnings. |
| `GET` | `/api/v1/preprocessing/assets/{asset_id}/files/{reference}` | Download an original or processed source file. |
| `GET` | `/api/v1/preprocessing/get_extracted_text/{asset_id}` | Return `{"asset_id": "...", "extracted_text": "..."}`. |

Supported extensions are `.pdf`, `.docx`, `.txt`, `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`,
`.aac`, `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, and `.flv`. Unsupported extensions or
MIME mismatches return `415`; oversized files return `413`; invalid file contents return
`422`.

Example upload:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/preprocessing/upload_file/ `
  -F "file=@C:\path\to\lesson.pdf"
```

### Question generation

`POST /api/v1/graph/qg/start_session`

Request:

```json
{
  "asset_id": "asset-id",
  "question_type": "MCQ"
}
```

`question_type` must be `MCQ` or `T/F`. A successful response contains a `thread_id`,
the workflow `status`, and either `data_for_feedback` for an interrupted draft or
`final_state` when the graph has completed.

`POST /api/v1/graph/qg/provide_feedback`

Request:

```json
{
  "thread_id": "thread-id",
  "feedback": "Make the distractors more precise",
  "review_id": "review-id"
}
```

`POST /api/v1/graph/qg/edit_question` accepts `thread_id`, `review_id`, and a `question`
object matching the current draft schema. It persists a manual draft edit without
invoking a model.

Typical errors are `422` for invalid question input or insufficient source content,
`409` for a stale/conflicting review, `502` for provider or structured-output failures,
`503` for unavailable workflow storage, and `504` for a timeout.

### Summarization

`POST /api/v1/graph/summarizer/start_session_streaming`

Request:

```json
{
  "asset_id": "asset-id",
  "event_version": "v1"
}
```

`POST /api/v1/graph/summarizer/provide_feedback_streaming`

Request:

```json
{
  "thread_id": "thread-id",
  "feedback": "Shorten the second section",
  "review_id": "review-id",
  "event_version": "v1"
}
```

Both endpoints return `Content-Type: text/event-stream`. For `event_version: "v1"`,
each event is JSON framed as SSE data with this shape:

```json
{
  "version": 1,
  "event": "text_delta",
  "thread_id": "thread-id",
  "run_id": "run-id",
  "sequence": 2,
  "stage": "summarizer_writer",
  "data": {"text": "next text"}
}
```

The stream uses `session_started`, `stage_started`, `text_delta`,
`awaiting_feedback`, `completed`, and `failed` events. A terminal event contains the
final status. `event_version: "legacy"` is supported for the existing frontend client.

### Question answering

`POST /api/v1/graph/qa/start_session_stream`

Request:

```json
{
  "asset_id": "asset-id",
  "initial_question": "What are the three main concepts?",
  "event_version": "v1"
}
```

`POST /api/v1/graph/qa/continue_conversation_stream`

Request:

```json
{
  "thread_id": "thread-id",
  "next_question": "Can you explain the second concept?",
  "event_version": "v1"
}
```

The QA stream uses the same versioned SSE envelope. `text_delta` events carry answer
text; `completed` includes `answer_status`, `citations`, and `grounding`. Empty
questions return `422`, unknown threads return `404`, and attempting to continue a
thread before its previous question completes returns `409`.

### Sessions

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/sessions/question_generation` | List completed saved questions; optional `asset_id` filter. |
| `GET` | `/api/v1/sessions/{workflow}/{thread_id}` | Read a checkpointed session for `question_generation`, `summarization`, or `qa`. |

Invalid UUIDs return `400`; missing sessions return `404`.

## Project Structure

```text
Wizora-AI/
├── chains/                         # LangChain chain definitions and contracts
├── controllers/                    # LangGraph state types and graph topology
├── helper/                         # Settings, ingestion, storage, SSE, and frontend client
├── llm/                            # LLM and embedding provider factories
├── routes/                         # FastAPI route groups
├── tools/                          # PDF, DOCX, audio, and video processing tools
├── assets/
│   └── architecture/               # Native LangGraph PNG diagrams
├── scripts/
│   └── generate_graph_diagrams.py  # Reproducible graph renderer
├── gradio_ui.py                    # Gradio application
├── main.py                         # FastAPI application entry point
├── .env.example                    # Configuration template
├── pyproject.toml                  # Pinned project dependencies
└── uv.lock                         # Resolved dependency lockfile
```

Runtime uploads, indexes, logs, and SQLite files are intentionally not described as
source files in this tree.

## Multimodal configuration

Defaults keep local Whisper `tiny` on CPU and video vision disabled. Whisper's
first use downloads the configured weights if they are not already cached.
Choose a larger multilingual `WHISPER_MODEL` to improve recognition, and leave
`WHISPER_LANGUAGE` blank for automatic language detection. `WHISPER_DEVICE` may be
`cpu`, `cuda` when available, or `auto`. Recordings are normalized to mono 16 kHz
and processed in `AUDIO_CHUNK_SECONDS` windows (default 300, maximum 600 seconds).

To analyze video images with local Ollama, explicitly configure a locally installed
vision-capable model:

```dotenv
VISION_PROVIDER=OLLAMA
VISION_MODEL=your-installed-vision-model
VISION_BASE_URL=http://127.0.0.1:11434
```

Alternatively use `VISION_PROVIDER=GOOGLE_GENAI`, `VISION_MODEL` set to a model
available to your account with image input, and your configured `GOOGLE_API_KEY`.
This sends sampled frames to that provider and may consume its quota or billing.
No external vision API is enabled automatically.

`VIDEO_MAX_FRAMES` (default 12) caps inference cost. Sample timestamps span the whole
video using interval-based midpoint coverage; `VIDEO_FRAME_INTERVAL_SECONDS`
defaults to 30. Only exact decoded-image duplicates are removed, so a changed
number or equation is not deliberately discarded as a near duplicate.

Useful limits include `MAX_UPLOAD_BYTES` (256 MiB), `VIDEO_MAX_DURATION_SECONDS`
(7200, also applies to audio), `MEDIA_TIMEOUT_SECONDS` (180),
`VISION_TIMEOUT_SECONDS` (60), `FRONTEND_UPLOAD_TIMEOUT_SECONDS` (1800),
`GENERATION_CHUNK_CHARACTERS`, `GENERATION_CONTEXT_CHARACTERS`, and
`GENERATION_MAX_CHUNKS`. Despite their historical names, generation text budgets
are conservatively measured in UTF-8 bytes. Large/slow models may need higher
media and graph deadlines; `SESSION_LEASE_SECONDS` must exceed the graph deadline.
An oversized source is rejected explicitly rather than silently losing its tail.

