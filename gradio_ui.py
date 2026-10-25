import logging

import gradio as gr

from frontend import (
    CUSTOM_CSS,
    SUPPORTED_EXTENSIONS,
    VISUAL_EXTENSIONS,
    VLM_CHOICES,
    VLM_DESCRIPTIONS,
    VLM_DROPDOWN_JS,
    GradioUI,
    PracticeUI,
    asset_details,
    create_theme,
    empty_state,
    get_frontend_settings,
    markdown_text,
    timestamp_text,
    upload_progress,
)

__all__ = [
    "GradioUI",
    "create_gradio_interface",
    "launch",
    "create_theme",
    "CUSTOM_CSS",
    "VLM_DROPDOWN_JS",
    "empty_state",
    "markdown_text",
    "timestamp_text",
    "asset_details",
    "upload_progress",
    "SUPPORTED_EXTENSIONS",
    "VISUAL_EXTENSIONS",
    "VLM_CHOICES",
    "VLM_DESCRIPTIONS",
]

logger = logging.getLogger(__name__)


def create_gradio_interface(client=None):
    from frontend import SAVE_ZIP_JS, ExportBlocks

    ui = GradioUI(client)
    practice = PracticeUI(ui.client)
    with ExportBlocks(title="Wizora-AI — AI Teaching Assistant", analytics_enabled=False) as app:
        state = gr.State(empty_state)
        practice_state = gr.State(None)
        vlm_preference = gr.State({})
        file_overrides = gr.State({})
        with gr.Column():
            gr.Markdown(
                "# 🎓 Wizora-AI\n### AI Teaching Assistant\nUpload content, generate questions, create summaries and chat with your materials.",
                elem_classes="header",
            )
            with gr.Tabs(elem_id="workflow-tabs"):
                with gr.Tab(
                    "📁 Upload Content", elem_id="upload-tab", elem_classes="workflow-panel"
                ):
                    gr.Markdown(
                        "### Upload text, documents, audio or video for processing\n"
                        "TXT · PDF · DOCX · MP3 · WAV · M4A · FLAC · OGG · AAC · MP4 · MOV · MKV · WEBM · AVI · FLV",
                        elem_classes="workflow-description",
                    )
                    with gr.Row(elem_classes="workflow-row"):
                        with gr.Column(
                            scale=1, elem_classes=["workflow-card", "workflow-controls"]
                        ):
                            file_input = gr.File(
                                label="Select Files",
                                file_count="multiple",
                                type="filepath",
                                file_types=SUPPORTED_EXTENSIONS,
                                elem_classes="upload-area",
                                elem_id="upload-file",
                            )
                            vlm_mode = gr.Dropdown(
                                choices=VLM_CHOICES[:1],
                                value="disabled",
                                label="Visual Understanding (VLM)",
                                interactive=False,
                                allow_custom_value=False,
                                elem_id="vlm-mode",
                            )
                            vlm_description = gr.Markdown(
                                "Checking backend visual processing availability…",
                                elem_id="vlm-mode-description",
                            )
                            with gr.Accordion("Provider and tool status", open=False):
                                refresh_providers = gr.Button(
                                    "Check Providers", elem_id="check-providers"
                                )
                                provider_notice = gr.Markdown(
                                    "Check reachability, installed models and metadata capabilities without running inference.",
                                    elem_id="provider-status",
                                )
                                provider_json = gr.JSON(label="Provider check details", open=False)
                            with gr.Accordion("Optional visual mode per file", open=False):
                                override_file = gr.Dropdown(
                                    choices=[],
                                    label="File to configure",
                                    interactive=True,
                                    elem_id="vlm-override-file",
                                )
                                override_mode = gr.Dropdown(
                                    choices=VLM_CHOICES[:1],
                                    value="disabled",
                                    label="This file’s visual mode",
                                    interactive=False,
                                    elem_id="vlm-override-mode",
                                )
                                apply_override = gr.Button("Apply File Mode")
                                override_notice = gr.Markdown(
                                    "Upload files first, choose a file and its visual mode, then click **Apply File Mode**. Files without an override use the main mode. TXT and audio always use Disabled."
                                )
                            upload_btn = gr.Button(
                                "🚀 Process Files",
                                variant="primary",
                                elem_classes="button-primary",
                                elem_id="upload-process",
                            )
                            resume_btn = gr.Button("Resume Processing Status")
                            retry_btn = gr.Button("Retry / Resume Failed Files")
                            asset_id = gr.Textbox(
                                label="Generated Asset ID",
                                placeholder="Asset ID will appear here after upload…",
                            )
                            load_btn = gr.Button("📄 Load Extracted Text")
                            upload_status = gr.Markdown(
                                empty_state()["upload_status"], elem_classes="status"
                            )
                            details = gr.Markdown("", elem_classes="asset-details")
                        with gr.Column(scale=2, elem_classes=["workflow-card", "workflow-results"]):
                            file_progress = gr.Dataframe(
                                headers=[
                                    "File",
                                    "Type",
                                    "Size",
                                    "Status",
                                    "VLM",
                                    "Current stage / error",
                                    "File ID",
                                ],
                                datatype=["str"] * 7,
                                interactive=False,
                                label="Per-file processing",
                                elem_id="file-progress",
                                wrap=False,
                                max_height=320,
                            )
                            extracted = gr.Textbox(
                                label="Extracted Content",
                                placeholder="Extracted text, transcript and available visual descriptions will appear here after processing…",
                                lines=15,
                                max_lines=20,
                                interactive=False,
                                buttons=["copy"],
                                elem_classes="extracted-text",
                            )
                            with gr.Accordion("Source records and timestamps", open=False):
                                preview = gr.Markdown("", elem_classes="source-preview")
                with gr.Tab(
                    "❓ Question Generation", elem_id="question-tab", elem_classes="workflow-panel"
                ):
                    gr.Markdown(
                        "### Generate questions from your uploaded content",
                        elem_classes="workflow-description",
                    )
                    with gr.Row(elem_classes="workflow-row"):
                        with gr.Column(
                            scale=1, elem_classes=["workflow-card", "workflow-controls"]
                        ):
                            q_asset = gr.Textbox(
                                label="Asset ID",
                                placeholder="Populated automatically after upload…",
                            )
                            question_type = gr.Radio(
                                choices=["T/F", "MCQ"], label="Question Type", value="MCQ"
                            )
                            difficulty = gr.Dropdown(
                                choices=["easy", "medium", "hard"],
                                value="medium",
                                label="Difficulty",
                                info="Easy: recall. Medium: apply a rule. Hard: combine evidence through multiple reasoning steps.",
                            )
                            question_count = gr.Number(
                                value=1,
                                minimum=1,
                                maximum=20,
                                precision=0,
                                label="Number of questions",
                            )
                            generate_q = gr.Button(
                                "🎯 Generate Questions",
                                variant="primary",
                                elem_classes="button-primary",
                            )
                            question_selector = gr.Radio(
                                choices=[],
                                label="Generated questions",
                                interactive=True,
                                elem_id="question-selector",
                            )
                        with gr.Column(scale=2, elem_classes=["workflow-card", "workflow-results"]):
                            question = gr.Textbox(
                                label="Generated Question", lines=3, interactive=False
                            )
                            options = gr.Textbox(label="Answer Options", lines=4, interactive=False)
                            answer = gr.Textbox(label="Correct Answer", lines=1, interactive=False)
                            explanation = gr.Textbox(
                                label="Explanation", lines=3, interactive=False
                            )
                    with gr.Row():
                        q_feedback = gr.Textbox(
                            label="Feedback",
                            placeholder="Provide feedback to improve the question…",
                            lines=2,
                            scale=3,
                        )
                        update_q = gr.Button("🔄 Update Question", scale=1, interactive=False)
                    with gr.Row():
                        refine_q = gr.Button("✨ Automatically Refine", interactive=False)
                        accept_q = gr.Button("✅ Accept / Save Question", interactive=False)
                    q_status = gr.Markdown(empty_state()["question_status"], elem_classes="status")
                    with gr.Accordion(
                        "Practice test from saved questions", open=False, elem_id="practice-test"
                    ):
                        gr.Markdown(
                            "Test yourself on saved questions from the active Asset ID. "
                            "Each new attempt uses a snapshot of that source’s saved questions."
                        )
                        practice_start = gr.Button("Start / Restart Test", elem_id="practice-start")
                        practice_status = gr.Markdown(
                            "Save questions for the active source, then start a practice test.",
                            elem_classes="status",
                            elem_id="practice-status",
                        )
                        practice_question = gr.Textbox(
                            label="Practice Question",
                            lines=3,
                            interactive=False,
                            elem_id="practice-question",
                        )
                        practice_answer = gr.Radio(
                            choices=[],
                            label="Your Answer",
                            interactive=False,
                            elem_id="practice-answer",
                        )
                        with gr.Row():
                            practice_previous = gr.Button(
                                "Previous", interactive=False, elem_id="practice-previous"
                            )
                            practice_next = gr.Button(
                                "Next", interactive=False, elem_id="practice-next"
                            )
                            practice_submit = gr.Button(
                                "Submit Test",
                                variant="primary",
                                interactive=False,
                                elem_id="practice-submit",
                            )
                        practice_score = gr.Markdown("", elem_id="practice-score")
                        practice_feedback = gr.Markdown("", elem_id="practice-feedback")
                        practice_cursor = gr.Textbox("", visible=False, elem_id="practice-cursor")
                with gr.Tab(
                    "📝 Content Summarization", elem_id="summary-tab", elem_classes="workflow-panel"
                ):
                    gr.Markdown(
                        "### Create summaries and main points from your content",
                        elem_classes="workflow-description",
                    )
                    with gr.Row(elem_classes="workflow-row"):
                        with gr.Column(
                            scale=1, elem_classes=["workflow-card", "workflow-controls"]
                        ):
                            s_asset = gr.Textbox(
                                label="Asset ID",
                                placeholder="Populated automatically after upload…",
                            )
                            generate_s = gr.Button(
                                "📊 Generate Summary",
                                variant="primary",
                                elem_classes="button-primary",
                            )
                            s_feedback = gr.Textbox(
                                label="Feedback",
                                placeholder="Describe wording changes or how to reorganize the outline and sections…",
                                lines=3,
                            )
                            update_s = gr.Button("🔄 Update Summary", interactive=False)
                            accept_s = gr.Button("✅ Accept / Save Summary", interactive=False)
                        with gr.Column(scale=2, elem_classes=["workflow-card", "workflow-results"]):
                            main_points = gr.Textbox(
                                label="Main Points", lines=6, interactive=False, buttons=["copy"]
                            )
                            summary = gr.Textbox(
                                label="Detailed Summary",
                                lines=8,
                                interactive=False,
                                buttons=["copy"],
                            )
                    s_status = gr.Markdown(empty_state()["summary_status"], elem_classes="status")
                with gr.Tab(
                    "💬 Question Answering", elem_id="qa-tab", elem_classes="workflow-panel"
                ):
                    gr.Markdown(
                        "### Ask questions about your uploaded content",
                        elem_classes="workflow-description",
                    )
                    with gr.Row(elem_classes="workflow-row"):
                        with gr.Column(
                            scale=1, elem_classes=["workflow-card", "workflow-controls"]
                        ):
                            qa_asset = gr.Textbox(
                                label="Asset ID",
                                placeholder="Populated automatically after upload…",
                            )
                            initial_question = gr.Textbox(
                                label="Initial Question",
                                placeholder="Ask your first question about the content…",
                                lines=2,
                            )
                            start_qa = gr.Button(
                                "🚀 Start Q&A Session",
                                variant="primary",
                                elem_classes="button-primary",
                            )
                            qa_status = gr.Markdown(
                                empty_state()["qa_status"], elem_classes="status"
                            )
                        with gr.Column(scale=2, elem_classes=["workflow-card", "workflow-results"]):
                            chatbot = gr.Chatbot(
                                label="Conversation",
                                height=400,
                                elem_classes="chat-container",
                                buttons=["copy_all"],
                                placeholder="Answers and source citations will appear here.",
                            )
                            with gr.Row():
                                message = gr.Textbox(
                                    label="Your Message",
                                    placeholder="Continue the conversation…",
                                    lines=1,
                                    scale=4,
                                )
                                send = gr.Button("📤 Send", scale=1)
                with gr.Tab("🧾 Job Results", elem_id="jobs-tab", elem_classes="workflow-panel"):
                    gr.Markdown(
                        "### Inspect actual processing results\nEach operation records versioned JSON with its status, configuration, source references and results. Failed files remain inspectable."
                    )
                    with gr.Row():
                        job_file = gr.Dropdown(
                            choices=[], label="File or batch", interactive=True, elem_id="job-file"
                        )
                        job_selector = gr.Dropdown(
                            choices=[],
                            label="Recorded operation",
                            interactive=True,
                            elem_id="job-selector",
                        )
                        refresh_jobs = gr.Button("Refresh Job Results")
                    job_status = gr.Markdown("Process files to inspect operation results.")
                    job_json = gr.JSON(label="Job result JSON", elem_id="job-json")
                    with gr.Row():
                        download_json = gr.Button("Download Selected JSON")
                        export_zip = gr.Button("Export All Results (ZIP)")
                    json_file = gr.File(label="JSON download", interactive=False)
                    zip_file = gr.Button(
                        "Download Results ZIP", interactive=False, elem_id="zip-download"
                    )
                    zip_entry = gr.JSON(value=None, visible=False)
                    zip_fallback = gr.DownloadButton(
                        label="Standard ZIP Download", interactive=False
                    )
            gr.Markdown(
                "---\n### 💡 How to Use\n1. **Upload:** Add text, PDF, DOCX, audio and video files together, choose visual processing, then process. Wizora automatically builds one learning context with source citations.\n2. **Review:** Follow each file’s status and inspect extracted content, citations and Job Results JSON.\n3. **Questions:** Choose type, difficulty and count, generate, then select each question for feedback, refinement and save.\n4. **Summary:** Create main points and a detailed summary, revise, then accept/save.\n5. **Chat:** Ask questions about the active learning context."
            )
        outputs = [
            state,
            asset_id,
            q_asset,
            s_asset,
            qa_asset,
            upload_status,
            extracted,
            question,
            options,
            answer,
            explanation,
            q_status,
            main_points,
            summary,
            s_status,
            chatbot,
            qa_status,
            q_feedback,
            s_feedback,
            initial_question,
            message,
            update_q,
            refine_q,
            accept_q,
            update_s,
            accept_s,
            details,
            preview,
        ]
        event_options = dict(
            outputs=outputs,
            concurrency_limit=1,
            concurrency_id="single-file-workflow",
            trigger_mode="once",
            show_progress="minimal",
        )
        capability_options = dict(
            inputs=[file_input, vlm_preference],
            outputs=[vlm_mode, vlm_description, vlm_preference],
            concurrency_id="single-file-workflow",
            concurrency_limit=1,
            trigger_mode="always_last",
            show_progress="hidden",
        )
        app.load(ui.configure_vlm, **capability_options)
        app.load(fn=None, inputs=None, outputs=None, js=VLM_DROPDOWN_JS, queue=False)
        refresh_providers.click(
            ui.refresh_provider_status,
            [],
            [provider_notice, provider_json],
            concurrency_id="provider-status",
            concurrency_limit=1,
            show_progress="minimal",
        ).then(ui.configure_vlm, **capability_options)
        file_input.change(ui.configure_vlm, **capability_options)
        vlm_mode.input(
            ui.apply_vlm_mode,
            [vlm_mode, vlm_preference],
            [vlm_mode, vlm_description, vlm_preference],
            concurrency_id="single-file-workflow",
            concurrency_limit=1,
            trigger_mode="always_last",
            show_progress="hidden",
        )
        batch_options = {**event_options, "outputs": [*outputs, file_progress, job_file]}
        question_options = {**event_options, "outputs": [*outputs, question_selector]}
        file_input.change(ui.pending_files, [file_input], [file_progress], show_progress="hidden")
        file_input.change(
            ui.override_files,
            [file_input, file_overrides],
            [override_file, file_overrides],
            concurrency_id="single-file-workflow",
            concurrency_limit=1,
            show_progress="hidden",
        )
        gr.on(
            [override_file.change, vlm_preference.change],
            ui.configure_override,
            [override_file, vlm_preference, file_overrides],
            [override_mode],
            show_progress="hidden",
        )
        apply_override.click(
            ui.set_override,
            [override_file, override_mode, vlm_preference, file_overrides],
            [file_overrides, override_notice],
            concurrency_id="single-file-workflow",
            concurrency_limit=1,
            show_progress="hidden",
        )
        batch_events = [
            upload_btn.click(
                ui.upload_files,
                [file_input, state, vlm_mode, file_overrides, vlm_preference],
                **batch_options,
            ),
            resume_btn.click(ui.resume_batch, [state], **batch_options),
            retry_btn.click(ui.retry_failed_batch, [state], **batch_options),
            load_btn.click(ui.load_text_batch, [asset_id, state], **batch_options),
        ]
        for field in (asset_id, q_asset, s_asset, qa_asset):
            batch_events.append(field.input(ui.select_asset_batch, [field, state], **batch_options))
        # Source changes can clear the bank. Refresh only after that operation,
        # so an independent state listener cannot reset a user's selection.
        for event in batch_events:
            event.then(
                ui.question_choices,
                [state],
                [question_selector],
                concurrency_id="single-file-workflow",
                concurrency_limit=1,
                show_progress="hidden",
            )
        generate_q.click(
            ui.start_question_set,
            [q_asset, question_type, difficulty, question_count, state],
            **question_options,
        )
        question_type.input(ui.change_question_type, [question_type, state], **question_options)
        difficulty.input(
            ui.change_difficulty,
            [difficulty, state],
            **{**question_options, "trigger_mode": "always_last"},
        )
        question_selector.input(ui.select_question, [question_selector, state], **question_options)
        update_q.click(ui.update_question, [q_feedback, state], **event_options)
        refine_q.click(ui.refine_question, [state], **event_options)
        accept_q.click(ui.accept_question, [state], **event_options)
        practice_outputs = [
            practice_state,
            practice_status,
            practice_question,
            practice_answer,
            practice_feedback,
            practice_score,
            practice_previous,
            practice_next,
            practice_submit,
            practice_cursor,
        ]
        practice_options = dict(
            outputs=practice_outputs,
            concurrency_id="single-file-workflow",
            concurrency_limit=1,
            trigger_mode="once",
            show_progress="minimal",
        )
        practice_start.click(practice.start, [state], **practice_options)
        # Validate canonical answers in PracticeUI after rejecting stale cursors;
        # Radio's current choice list may belong to a newer question already.
        practice_action_options = {**practice_options, "preprocess": False}
        practice_answer.input(
            practice.choose,
            [practice_answer, state, practice_state, practice_cursor],
            **{**practice_action_options, "trigger_mode": "always_last"},
        )
        for button, callback in (
            (practice_previous, practice.previous),
            (practice_next, practice.next),
            (practice_submit, practice.submit),
        ):
            button.click(
                callback,
                [practice_answer, state, practice_state, practice_cursor],
                **practice_action_options,
            )
        state.change(practice.sync_source, [state, practice_state], **practice_options)
        generate_s.click(ui.start_summary, [s_asset, state], **event_options)
        update_s.click(ui.update_summary, [s_feedback, state], **event_options)
        accept_s.click(ui.accept_summary, [state], **event_options)
        start_qa.click(ui.start_qa, [qa_asset, initial_question, state], **event_options)
        gr.on([send.click, message.submit], ui.continue_qa, [message, state], **event_options)
        job_options = dict(
            outputs=[job_selector, job_json, json_file, job_status],
            concurrency_id="job-inspection",
            concurrency_limit=1,
        )
        job_file.change(ui.refresh_jobs, [job_file], **job_options)
        refresh_jobs.click(ui.refresh_jobs, [job_file], **job_options)
        download_options = dict(
            concurrency_id="job-inspection", concurrency_limit=1, trigger_mode="once"
        )
        job_selector.input(
            ui.inspect_job,
            [job_file, job_selector],
            [job_json, json_file, job_status],
            **download_options,
        )
        download_json.click(
            ui.download_job, [job_file, job_selector], [json_file, job_status], **download_options
        )
        export_zip.click(
            ui.prepare_browser_zip,
            [state],
            [zip_file, zip_entry, job_status, zip_fallback],
            **download_options,
        )
        zip_file.click(None, [zip_entry], [job_status], js=SAVE_ZIP_JS, queue=False)
    return app.queue(default_concurrency_limit=1)


def launch():
    settings = get_frontend_settings()
    create_gradio_interface().launch(
        server_name=settings.FRONTEND_HOST,
        server_port=settings.FRONTEND_PORT,
        share=False,
        show_error=False,
        theme=create_theme(),
        css=CUSTOM_CSS,
        max_file_size=settings.MAX_UPLOAD_BYTES,
        footer_links=[],
    )


if __name__ == "__main__":
    launch()
