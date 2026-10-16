from functools import partial

import gradio as gr

from .callbacks import Callbacks
from .components import OUTPUT_NAMES, fresh_state
from .api import BackendClient
from .settings import browser_secret, get_frontend_settings
from .theme import CSS, theme


class Workspace(gr.Blocks):
    def launch(self, **kwargs):
        kwargs.setdefault("theme", theme())
        kwargs.setdefault("css", CSS)
        return super().launch(**kwargs)


def heading(title, description):
    return gr.HTML(f'<div class="wiz-section"><h2>{title}</h2><p>{description}</p></div>')


def create_app(client=None):
    client = client if client is not None else BackendClient()
    settings = client.settings
    callbacks = Callbacks(client)
    controls = {}
    with Workspace(title="Wizora-AI", analytics_enabled=False, delete_cache=(3600, 3600)) as app:
        controls["state"] = gr.State(fresh_state)
        controls["browser"] = gr.BrowserState(
            {"selected_asset": None, "sessions": {}}, storage_key="wizora-ai-sessions", secret=browser_secret(settings),
        )
        gr.HTML(
            '<header class="wiz-header"><h1 class="wiz-brand">🎓 Wizora-AI — AI Teaching Assistant</h1>'
            '<p class="wiz-subtitle">Upload educational materials, generate questions, create summaries, '
            'and ask source-grounded questions.</p></header>'
        )
        with gr.Row(elem_id="asset-bar"):
            controls["asset"] = gr.Dropdown(choices=[], label="Selected material", value=None,
                                           info="This selection applies to every workspace below.", scale=5, min_width=260)
            refresh = gr.Button("Refresh library & sessions", scale=1, min_width=180)
        controls["metadata"] = gr.HTML('<div class="wiz-metadata">Select an uploaded material to begin.</div>')
        controls["notice"] = gr.Markdown("", elem_classes="wiz-status")

        with gr.Tabs():
            with gr.Tab("📁 Upload Content"):
                heading("Upload your educational materials", "Process PDF and DOCX text, or transcribe speech from audio and video.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=260):
                        upload_file = gr.File(label="Select File", file_types=[".pdf", ".docx", ".mp3", ".wav", ".m4a", ".mp4", ".mov", ".mkv", ".webm"], type="filepath", elem_classes="upload-area")
                        upload = controls["upload"] = gr.Button("🚀 Upload File", variant="primary", interactive=False)
                        controls["asset_id"] = gr.Textbox(label="Generated Asset ID", placeholder="Asset ID will appear here after upload…", interactive=False)
                        preview = gr.Button("📄 Load Extracted Text")
                        gr.Markdown(f"PDF · DOCX · Audio · Video · Up to {settings.MAX_UPLOAD_BYTES / 1048576:g} MiB", elem_classes="wiz-hint")
                    with gr.Column(scale=2, min_width=290):
                        controls["extracted"] = gr.Textbox(label="Extracted Text", placeholder="Extracted text or transcript will appear here after processing…", lines=15, max_lines=24, interactive=False, buttons=["copy"], elem_classes="extracted-text")
                gr.Markdown("Audio and video use speech transcripts. Content shown only in slides or video frames is unavailable. Scanned PDFs require OCR.", elem_classes="wiz-hint")
                with gr.Accordion("Uploaded materials", open=False):
                    controls["library"] = gr.HTML('<div class="wiz-empty">Your library is empty. Upload a document or recording to begin.</div>')

            with gr.Tab("❓ Question Generation"):
                heading("Generate questions from your content", "Create multiple-choice or True/False questions, review the AI critic, and refine the result before saving.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7, min_width=300):
                        question_type = gr.Radio([("Multiple choice", "MCQ"), ("True / False", "T/F")], value="MCQ", label="Question type")
                        controls["qgenerate"] = gr.Button("Generate question", variant="primary", interactive=False)
                        controls["qstatus"] = gr.Markdown("Choose a material, then generate a question.", elem_classes="wiz-status")
                        controls["question"] = gr.Markdown("A generated question will appear here for review.", line_breaks=True, buttons=["copy"])
                    with gr.Column(scale=4, min_width=280, elem_classes="wiz-rule"):
                        gr.Markdown("### Review & refine")
                        qfeedback = gr.Textbox(label="Revision feedback", placeholder="Describe the change you want to make…", lines=4)
                        controls["qrevise"] = gr.Button("Apply feedback", interactive=False)
                        controls["qauto"] = gr.Button("AI review & refine", interactive=False)
                        controls["qaccept"] = gr.Button("Accept & save question", variant="primary", interactive=False)
                        with gr.Accordion("AI review findings", open=True):
                            controls["critic"] = gr.Markdown("Automatic review findings will appear after an AI review.")
                with gr.Accordion("Question session", open=False):
                    controls["qthread"] = gr.Textbox(label="Session ID", interactive=False)
                    controls["qreview"] = gr.Textbox(label="Current review ID", interactive=False)

            with gr.Tab("📝 Content Summarization"):
                heading("Create lecture summaries and main points", "Review the outline, request a wording or structural revision, and save your summary.")
                controls["sgenerate"] = gr.Button("Generate summary", variant="primary", interactive=False)
                controls["sstatus"] = gr.Markdown("Choose a material to summarize.", elem_classes="wiz-status")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=250):
                        gr.Markdown("### Outline & main points")
                        controls["outline"] = gr.Markdown("", line_breaks=True)
                    with gr.Column(scale=7, min_width=300, elem_classes="wiz-rule"):
                        gr.Markdown("### Summary")
                        controls["summary"] = gr.Markdown("", line_breaks=True, buttons=["copy"])
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7, min_width=300):
                        sfeedback = gr.Textbox(label="Summary feedback", placeholder="Describe how the summary should change…", lines=3)
                    with gr.Column(scale=3, min_width=250):
                        revision_kind = gr.Radio(["Wording", "Structure"], label="Revision focus", value="Wording")
                        controls["srevise"] = gr.Button("Revise summary", interactive=False)
                        controls["saccept"] = gr.Button("Accept & save summary", variant="primary", interactive=False)
                with gr.Accordion("Summary session", open=False):
                    controls["sthread"] = gr.Textbox(label="Session ID", interactive=False)
                    controls["sreview"] = gr.Textbox(label="Current review ID", interactive=False)

            with gr.Tab("💬 Question Answering"):
                heading("Ask questions about your materials", "Chat with extracted text and transcripts. Inspect citations for each source-grounded answer.")
                with gr.Row():
                    controls["chat"] = gr.Chatbot(value=[], label="Conversation", height=400, buttons=["copy"],
                                                 placeholder="Select a material and ask your first question.", allow_tags=False)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7, min_width=300):
                        controls["cstatus"] = gr.Markdown("Ask a question grounded in the selected material.", elem_classes="wiz-status")
                        prompt = gr.Textbox(label="Your question", placeholder="What would you like to understand?", lines=2, max_lines=5)
                        with gr.Row():
                            controls["send"] = gr.Button("Send question", variant="primary", interactive=False)
                            new_chat = gr.Button("New conversation")
                    with gr.Column(scale=3, min_width=260, elem_classes="wiz-rule"):
                        controls["citations"] = gr.HTML('<div class="wiz-evidence"><h3>Evidence & sources</h3><p>Verified source references appear here with each grounded answer.</p></div>')
                with gr.Accordion("Conversation session", open=False):
                    controls["cthread"] = gr.Textbox(label="Session ID", interactive=False)

            with gr.Tab("📋 Processing Status"):
                heading("Processing status", "These stages come directly from the backend. Recordings are transcribed from speech; no video frames are analyzed.")
                check_processing = gr.Button("Check processing")
                controls["processing"] = gr.HTML('<div class="wiz-empty">Upload a file to see its processing stages here.</div>')
                gr.Markdown("A ready material may be indexed on the first RAG question, depending on backend configuration.")

        with gr.Accordion("Restore a saved session by ID", open=False):
            gr.Markdown("Select the associated material first. Saved browser session IDs also restore automatically when you return.")
            with gr.Row():
                workflow = gr.Dropdown([("Question review", "question_generation"), ("Summary review", "summarization"), ("Conversation", "qa")],
                                       value="qa", label="Workflow", scale=2)
                resume_id = gr.Textbox(label="Session ID", placeholder="Paste an existing session ID", scale=4)
                resume = gr.Button("Restore session", scale=1)

        outputs = [controls[name] for name in OUTPUT_NAMES]
        state, browser = controls["state"], controls["browser"]
        options = {"outputs": outputs, "trigger_mode": "once", "show_progress": "minimal", "api_visibility": "private"}
        workflow_options = {**options, "concurrency_id": "wizora-workflows", "concurrency_limit": 1}
        app.load(callbacks.load, inputs=[browser], **workflow_options)
        controls["asset"].input(callbacks.select, inputs=[controls["asset"], state, browser], **workflow_options)
        refresh.click(callbacks.refresh, inputs=[state, browser], **workflow_options)
        upload.click(callbacks.upload, inputs=[upload_file, state], **workflow_options)
        preview.click(callbacks.preview, inputs=[state], **workflow_options)
        check_processing.click(callbacks.check_processing, inputs=[state], **workflow_options)
        controls["qgenerate"].click(partial(callbacks.question, action="start"), inputs=[question_type, qfeedback, state], **workflow_options)
        for name, action in (("qaccept", "save"), ("qrevise", "revise"), ("qauto", "auto")):
            controls[name].click(partial(callbacks.question, action=action), inputs=[question_type, qfeedback, state], **workflow_options)
        controls["sgenerate"].click(partial(callbacks.summary, action="start"), inputs=[sfeedback, revision_kind, state], **workflow_options)
        controls["saccept"].click(partial(callbacks.summary, action="save"), inputs=[sfeedback, revision_kind, state], **workflow_options)
        controls["srevise"].click(partial(callbacks.summary, action="revise"), inputs=[sfeedback, revision_kind, state], **workflow_options)
        send = gr.on(triggers=[controls["send"].click, prompt.submit], fn=callbacks.chat, inputs=[prompt, state], **workflow_options)
        send.success(
            lambda value: "" if value.get("sessions", {}).get(value.get("selected_asset"), {}).get("qa", {}).get("status") == "completed" else gr.skip(),
            inputs=state, outputs=prompt, api_visibility="private",
        )
        new_chat.click(callbacks.new_chat, inputs=[state], **workflow_options)
        resume.click(callbacks.resume, inputs=[workflow, resume_id, state], **workflow_options)
    app.queue(default_concurrency_limit=4, max_size=32)
    return app


def main():
    settings = get_frontend_settings()
    create_app().launch(
        server_name=settings.FRONTEND_HOST, server_port=settings.FRONTEND_PORT,
        share=False, inbrowser=False, max_file_size=settings.MAX_UPLOAD_BYTES,
        show_error=False, footer_links=[],
    )


if __name__ == "__main__":
    main()
