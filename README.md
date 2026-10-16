# Wizora-AI

Wizora-AI is an agentic AI educational system built with FastAPI and Gradio, featuring multiformat document and media processing, automated question generation with iterative human feedback, content summarization, and an interactive Q&A assistant.

## Features

- **Multiformat Preprocessing**: Ingest PDF, DOCX, audio (MP3, WAV, M4A), and video (MP4, MOV, MKV, WEBM) files.
- **Question Generation Agent**: Generate MCQs and True/False questions with automated refinement and revision based on user feedback.
- **Summarization Agent**: Generate high-level key points and comprehensive summaries with iterative rewriting capabilities.
- **Interactive Q&A Agent**: Grounded conversational assistant with source citations and session history.
- **Modern Gradio Web UI**: Clean, responsive interface for uploading materials and managing agent workflows.
- **FastAPI REST API**: Asynchronous backend with full OpenAPI / Swagger documentation and health monitoring.

## Getting Started

### Prerequisites

- Python 3.11
- Virtual environment (`venv` or `uv`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MarouaHattab/Wizora-AI.git
   cd Wizora-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   # or with uv:
   uv sync
   ```

4. Configure environment variables:
   Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

### Running the Application

1. **Start the FastAPI Backend**:
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
   API Docs available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

2. **Start the Gradio Web Interface**:
   ```bash
   python gradio_ui.py
   ```
   Web interface available at: [http://127.0.0.1:7860](http://127.0.0.1:7860)

## License

BSD-2-Clause License
