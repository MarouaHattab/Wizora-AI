import html
from copy import deepcopy

import gradio as gr


WORKFLOWS = ("question_generation", "summarization", "qa")
OUTPUT_NAMES = (
    "state", "browser", "asset", "asset_id", "metadata", "library", "processing", "extracted",
    "question", "critic", "qstatus", "qthread", "qreview", "qgenerate", "qaccept", "qrevise", "qauto",
    "outline", "summary", "sstatus", "sthread", "sreview", "sgenerate", "saccept", "srevise",
    "chat", "citations", "cstatus", "cthread", "send", "notice", "upload",
)


def fresh_state():
    return {"selected_asset": None, "assets": {}, "sessions": {}, "job": None, "notice": "", "busy": None}


def session(state, workflow):
    return state.setdefault("sessions", {}).setdefault(state.get("selected_asset"), {}).setdefault(workflow, {})


def browser_value(state):
    sessions = {}
    for asset_id, workflows in state.get("sessions", {}).items():
        if asset_id:
            values = {name: value["thread_id"] for name, value in workflows.items()
                      if name in WORKFLOWS and value.get("thread_id")}
            if values:
                sessions[asset_id] = values
    return {"selected_asset": state.get("selected_asset"), "sessions": sessions}


def escape(value):
    return html.escape(str(value), quote=True)


def markdown(value):
    value = str(value or "")
    for character in "\\`*_{}[]()#+-.!|<>":
        value = value.replace(character, "\\" + character)
    return value


def time_label(value):
    seconds = max(0, int(value))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def metadata_html(asset):
    if not asset:
        return '<div class="wiz-metadata">Select an uploaded material to begin.</div>'
    details = [str(asset.get("source_type", "document")).upper()]
    for key, label in (("pages", "pages"), ("record_count", "source records")):
        if key == "pages" and asset.get("source_type") != "pdf":
            continue
        value = asset.get(key)
        if isinstance(value, list):
            value = len(value)
        if isinstance(value, (int, float)) and value > 0:
            details.append(f"{value} {label}")
    if isinstance(asset.get("duration_seconds"), (int, float)):
        details.append(time_label(asset["duration_seconds"]))
    if asset.get("source_type") in {"audio", "video"}:
        details.append("Speech transcript only")
    return '<div class="wiz-metadata">' + escape(" · ".join(details)) + '</div>'


def library_html(assets):
    if not assets:
        return '<div class="wiz-empty">Your library is empty. Upload a document or recording to begin.</div>'
    rows = []
    for asset in assets.values():
        rows.append('<article><div><strong>' + escape(asset.get("source_name", asset["id"]))
                    + '</strong><small>' + escape(asset["id"]) + '</small></div><span class="asset-kind">'
                    + escape(str(asset.get("source_type", "document")).upper()) + ' · Ready</span></article>')
    return '<div class="wiz-library">' + "".join(rows) + '</div>'


def processing_html(job):
    if not job:
        return '<div class="wiz-empty">Upload a file to see its processing stages here.</div>'
    stages = []
    for stage in job.get("stages", []):
        name = str(stage.get("name", "Processing")).replace("_", " ").capitalize()
        details = []
        if stage.get("segment") is not None:
            details.append("Segment " + str(stage["segment"]) + (" of " + str(stage["segments"]) if stage.get("segments") else ""))
        if stage.get("duration_seconds") is not None:
            details.append(time_label(stage["duration_seconds"]))
        stages.append('<li>' + escape(name) + '<small>' + escape(" · ".join(details)) + '</small></li>')
    status = str(job.get("status", "processing")).replace("_", " ").capitalize()
    error = job.get("error") or {}
    detail = error.get("detail", "") if isinstance(error, dict) else str(error)
    warnings = "".join('<p>' + escape(item) + '</p>' for item in job.get("warnings", []))
    return ('<div class="wiz-processing"><strong>' + escape(job.get("source_name", "Upload"))
            + '</strong><p>' + escape(status) + '</p><small>Asset ' + escape(job.get("id", "")) + '</small><ol>' + "".join(stages) + '</ol>'
            + ('<p class="error">' + escape(detail) + '</p>' if detail else "") + warnings + '</div>')


def question_markdown(result):
    if not result.get("question"):
        return "A generated question will appear here for review."
    text = "### " + markdown(result["question"])
    options = result.get("options") or []
    if options:
        text += "\n\n" + "\n\n".join(markdown(option) for option in options)
    return text + "\n\n**Correct answer:** " + markdown(result.get("answer", "")) + "\n\n" + markdown(result.get("explanation", ""))


def critic_markdown(result):
    critic = result.get("critic_feedback")
    if not isinstance(critic, dict):
        return "Automatic review findings will appear after an AI review."
    text = "**" + markdown(str(critic.get("verdict", "Reviewed")).replace("_", " ").capitalize()) + "**"
    if critic.get("feedback"):
        text += "\n\n" + markdown(critic["feedback"])
    for issue in critic.get("issues", []):
        text += "\n\n- " + markdown(issue.get("severity", "")) + ": " + markdown(issue.get("correction_hint", ""))
    return text


def citation_html(citations, asset, client):
    prefix = '<div class="wiz-evidence"><h3>Evidence & sources</h3>'
    if not citations:
        return prefix + '<p>Verified source references appear here with each grounded answer.</p></div>'
    rows = []
    files = asset.get("files", []) if asset else []
    original = next((item for item in files if isinstance(item, str) and item.startswith("original.")), None)
    for citation in citations:
        if not isinstance(citation, dict) or not asset or citation.get("asset_id") != asset.get("id"):
            continue
        places = []
        if type(citation.get("page")) is int:
            places.append("Page " + str(citation["page"]))
        if citation.get("section"):
            places.append("Section: " + str(citation["section"]))
        if citation.get("paragraph") is not None:
            places.append("Paragraph " + str(citation["paragraph"]))
        if citation.get("table_index") is not None:
            places.append("Table " + str(citation["table_index"]))
        if isinstance(citation.get("start"), (int, float)):
            location = time_label(citation["start"])
            if isinstance(citation.get("end"), (int, float)):
                location += "–" + time_label(citation["end"])
            places.append(location)
        if citation.get("source_type") == "video":
            places.append("Video transcript")
        label = '[' + str(citation.get("number", "")) + '] ' + str(citation.get("source", asset.get("source_name", "Source")))
        href = client.source_url(asset["id"], original, page=citation.get("page"), timestamp=citation.get("start")) if original else None
        title = '<a target="_blank" rel="noopener noreferrer" href="' + escape(href) + '">' + escape(label) + '</a>' if href else escape(label)
        rows.append('<article><strong>' + title + '</strong><p>' + escape(" · ".join(places)) + '</p><small>Evidence '
                    + escape(citation.get("chunk_id", "")) + '</small></article>')
    return prefix + "".join(rows) + '</div>'


def status_text(value, fallback):
    status = value.get("status")
    if value.get("error"):
        return value["error"]
    text = {"interrupted_for_feedback": "Ready for your review.", "completed": "Saved.",
            "running": "Working…", "failed": "The request did not complete. Refresh the session before retrying."}.get(status, fallback)
    notice = value.get("result", {}).get("review_notice")
    return text + (" " + notice if notice else "")


def render(state, client):
    state = deepcopy(state)
    asset_id = state.get("selected_asset")
    assets = state.get("assets", {})
    asset = assets.get(asset_id, {})
    ready = bool(asset and asset.get("status") == "ready")
    available = ready and not state.get("busy")
    q, s, c = [session(state, name) for name in WORKFLOWS]
    qreview = available and q.get("status") == "interrupted_for_feedback"
    sreview = available and s.get("status") == "interrupted_for_feedback"
    chat = deepcopy(c.get("result", {}).get("messages", []))
    pending = c.get("pending")
    if pending:
        chat.append({"role": "user", "content": pending["question"]})
        if pending.get("text"):
            chat.append({"role": "assistant", "content": pending["text"]})
    values = [
        state, browser_value(state), gr.update(choices=[(item.get("source_name", key), key) for key, item in assets.items()], value=asset_id), asset_id or "",
        metadata_html(asset), library_html(assets), processing_html(state.get("job")), state.get("extracted", ""),
        question_markdown(q.get("result", {})), critic_markdown(q.get("result", {})), status_text(q, "Choose a question type, then generate a question."),
        q.get("thread_id", ""), q.get("review_id", ""), gr.update(interactive=available), gr.update(interactive=qreview),
        gr.update(interactive=qreview), gr.update(interactive=qreview and not q.get("result", {}).get("revision_limit_reached")),
        s.get("preview_outline", s.get("result", {}).get("Main_Points", "")),
        s.get("preview_summary", s.get("result", {}).get("summary", "")),
        status_text(s, "Generate an outline and summary from the selected material."), s.get("thread_id", ""), s.get("review_id", ""),
        gr.update(interactive=available), gr.update(interactive=sreview), gr.update(interactive=sreview), chat,
        citation_html(c.get("result", {}).get("answer_citations", []), asset, client),
        status_text(c, "Ask a question grounded in the selected material.") if c.get("status") != "completed" else c.get("answer_notice", "Answer complete. Ask a follow-up question."),
        c.get("thread_id", ""), gr.update(interactive=available and c.get("status") not in {"running", "failed", "interrupted_for_feedback"}),
        state.get("notice", ""), gr.update(interactive=not state.get("busy")),
    ]
    return tuple(values)
