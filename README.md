# Wizora AI

Wizora AI turns documents and media into study questions, summaries and answers grounded
in the uploaded sources. It runs a FastAPI backend and a Gradio frontend locally.

## Setup

Use Python 3.11 and `uv`:

```powershell
uv sync --frozen
Copy-Item .env.example .env
```

Copy the example only when creating a new configuration. Keep an existing `.env` and
its credentials when updating the application.

Configure the provider keys, models and task assignments in `.env`. The current example
uses NVIDIA NIM for question generation, review and rewriting, Google Gemini for
summaries and question answering, and local Ollama for visual extraction. Groq and
generic OpenAI-compatible text endpoints remain supported.

| Provider | Credential | Endpoint setting |
| --- | --- | --- |
| NVIDIA NIM | `NVIDIA_NIM_API_KEY` | `NVIDIA_NIM_BASE_URL` |
| Google Gemini | `GOOGLE_API_KEY` | `GOOGLE_GENAI_BASE_URL` |
| Groq | `GROQ_API_KEY` | `GROQ_BASE_URL` |
| Ollama | Optional `OLLAMA_API_KEY` | `OLLAMA_BASE_URL`, `VISION_BASE_URL` |
| Generic endpoint | `OPENAI_COMPATIBLE_API_KEY` | `OPENAI_COMPATIBLE_BASE_URL` |

Each agent's `*_PROVIDER`, `*_MODEL_ID` and `*_TEMPERATURE` fields control its model.
The existing `QUESTION_GENERATER` spelling is intentional. Choose a model available
to the configured account. **Check Providers** checks metadata; a successful generation
request is needed to verify inference. Provider fallback is disabled by default.

Start the services in separate terminals:

```powershell
uv run python main.py
uv run python gradio_ui.py
```

Open [Wizora](http://127.0.0.1:7860) and the [API documentation](http://127.0.0.1:8000/docs).
`FRONTEND_BACKEND_URL` must point to the running backend.

## Using the application

1. Upload PDF, DOCX, TXT, audio or video files and choose the visual processing mode.
2. Process the files. Multiple uploads automatically share one learning context while
   preserving each source's identity, page references and timestamps.
3. Generate MCQ or True/False questions, choose a difficulty, and review, edit, refine
   or save each result. The question tab also supports practice and scoring.
4. Generate and revise summaries, or ask the chatbot questions about the active sources.
5. Inspect per-file progress, resume interrupted processing and export job results.

Visual modes are Disabled, Automatic and Enhanced. Disabled makes no visual-model
calls. Automatic selects meaningful artifacts; Enhanced analyzes all meaningful
artifacts within configured limits. Both require an image-capable model. The default
local vision model is `qwen3.5:2b`; this application does not pull models automatically.
Small visual models can make factual errors, so inspect source evidence when reviewing
their descriptions.

Audio transcription uses local Whisper. FFmpeg is supplied by `imageio-ffmpeg`, with
optional overrides through `FFMPEG_BINARY` and `FFPROBE_BINARY`. OCR requires suitable
Tesseract language data or an available configured engine. Complex Word rendering
requires LibreOffice. Native text and table extraction remain available without these
optional tools.

Embeddings use local `sentence-transformers/all-MiniLM-L6-v2` by default. Chunking,
indexing and retrieval retain source citations and context limits.

## Project layout

```text
main.py                 FastAPI entry point
gradio_ui.py            Gradio entry point
chains/                 Prompts, validated task chains and model runtime
controllers/            LangGraph workflows and state types
frontend/               HTTP client, callbacks, presentation and styles
helper/                 Configuration, assets, sessions, jobs and retrieval
llm/                    Provider factories and shared model transport
routes/                 API routers
schemas/                Pydantic question, summary and QA contracts
tools/                  Document, audio, video and visual extraction
scripts/check_architecture.py
                        Import-boundary maintenance check
assets/                 Source assets and generated indexes
runtime/                Local SQLite and processing state
```

Use public package imports such as `from llm import LLMProviderFactory,
EmbeddingProviderFactory`, `from chains import get_task_chain`, and
`from frontend import GradioUI`. Exports load implementations lazily. Older import paths
remain available for saved checkpoints and existing integrations.

## Operations

Run one backend worker: SQLite checkpoints, active sessions and Chroma resources are
managed within one process. Preserve `assets/` and `runtime/` when updating the source.
Keep `.env` out of source control and restart the affected service after changing it.

Health endpoints are `/health/live` and `/health/ready`. Provider metadata is available
at `/api/v1/providers/status`. The existing preprocessing, question, summary, QA and
session routes are listed in the API documentation.

Check code and imports without loading models:

```powershell
uv run ruff check .
uv run python -m scripts.check_architecture
```

`controllers/langgraph.json` retains the optional LangGraph entry points. Its file,
dependency and environment paths are relative to the configuration directory.
