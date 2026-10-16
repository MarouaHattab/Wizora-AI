import gradio as gr


def theme():
    configured = gr.themes.Soft(
        primary_hue="violet", secondary_hue="indigo", neutral_hue="slate",
        font=["Arial", "Helvetica", "sans-serif"], radius_size="sm", text_size="md",
    ).set(
        body_background_fill="#0B0F19", body_text_color="#F3F4F6",
        background_fill_primary="#1F2937", background_fill_secondary="#374151",
        border_color_primary="#465267", border_color_accent="#8B5CF6",
        body_text_color_subdued="#B6BFCD",
        block_background_fill="#1F2937", block_border_color="#364152",
        block_label_background_fill="#4F46E5", block_label_text_color="#FFFFFF",
        block_label_border_color="#4F46E5",
        block_title_background_fill="#1F2937", block_title_text_color="#F3F4F6",
        panel_background_fill="#1F2937",
        input_background_fill="#374151", input_background_fill_focus="#374151",
        input_background_fill_hover="#3D4859", input_border_color="#465267",
        input_placeholder_color="#9DA8B8",
        button_primary_background_fill="linear-gradient(90deg, #4F46E5, #7C3AED)",
        button_primary_background_fill_hover="linear-gradient(90deg, #5B50EB, #8B46EF)",
        button_primary_text_color="#FFFFFF",
        button_secondary_background_fill="#374151",
        button_secondary_background_fill_hover="#465267",
        button_secondary_text_color="#F3F4F6", button_secondary_border_color="#465267",
        table_text_color="#F3F4F6", table_border_color="#465267",
        table_even_background_fill="#263244", table_odd_background_fill="#1F2937",
        block_shadow="none",
    )
    for name in tuple(vars(configured)):
        if name.endswith("_dark") and hasattr(configured, name[:-5]):
            setattr(configured, name, getattr(configured, name[:-5]))
    return configured


CSS = """
:root { color-scheme:dark; --wiz-ink:#f3f4f6; --wiz-accent:#a78bfa; --wiz-rule:#364152; --wiz-muted:#b6bfcd; }
html, body, gradio-app { min-width:0 !important; max-width:100% !important; background:#0b0f19 !important; }
body, .gradio-container { background:#0b0f19 !important; color:var(--wiz-ink) !important; }
body .gradio-container.gradio-container { width:100% !important; min-width:0 !important; max-width:1200px !important; box-sizing:border-box; margin:0 auto !important; padding:24px 28px 36px !important; }
.gradio-container .main, .gradio-container .row, .gradio-container .column { min-width:0 !important; max-width:100% !important; }
.gradio-container .row { flex-wrap:wrap !important; }
.gradio-container .row > * { min-width:0 !important; max-width:100% !important; }
.wiz-header { padding:0 0 14px; }
.wiz-brand { margin:0; font-size:25px; line-height:1.3; font-weight:700; letter-spacing:-.3px; color:#f3f4f6; }
.wiz-subtitle { margin:8px 0 0; color:#d1d5db; font-size:14px; line-height:1.5; }
#asset-bar { padding:4px 0; align-items:flex-end; }
.wiz-section { padding:12px 0 8px; }
.wiz-section h2 { margin:0 0 6px; font-size:18px; line-height:1.4; font-weight:650; color:#f3f4f6; }
.wiz-section p { margin:0; font-size:13px; line-height:1.5; color:var(--wiz-muted); }
.wiz-rule { border:1px solid var(--wiz-rule); border-radius:8px; background:#1f2937; padding:16px !important; }
.wiz-evidence h3 { margin:0 0 14px; font-size:17px; font-weight:650; }
.wiz-evidence p { color:var(--wiz-muted); font-size:14px; line-height:1.6; }
.wiz-evidence article { border-top:1px solid var(--wiz-rule); padding:12px 0; overflow-wrap:anywhere; }
.wiz-evidence article p { margin:6px 0; }
.wiz-evidence a { color:var(--wiz-accent); text-decoration:underline; text-underline-offset:3px; }
.wiz-evidence small { display:block; font-size:12px; color:var(--wiz-muted); line-height:1.5; }
.wiz-empty { padding:16px 0; color:var(--wiz-muted); font-size:14px; line-height:1.6; }
.wiz-status { min-height:0; font-size:13px; color:var(--wiz-muted); }
.wiz-status:empty { display:none; }
.wiz-library article { display:flex; justify-content:space-between; gap:16px; padding:12px 0; border-bottom:1px solid var(--wiz-rule); }
.wiz-library article strong { display:block; font-weight:600; overflow-wrap:anywhere; }
.wiz-library small { color:var(--wiz-muted); font-size:12px; overflow-wrap:anywhere; }
.wiz-library .asset-kind { font-size:12px; color:#c4b5fd; white-space:nowrap; }
.wiz-metadata { color:var(--wiz-muted); font-size:12px; line-height:1.5; padding:0 0 4px; }
.wiz-hint, .wiz-hint p { font-size:12px !important; line-height:1.5; color:var(--wiz-muted) !important; }
.wiz-processing { background:#1f2937; border:1px solid var(--wiz-rule); padding:16px; border-radius:8px; }
.wiz-processing ol { padding-left:22px; margin:12px 0; }
.wiz-processing li { padding:6px 0; font-size:14px; }
.wiz-processing li small { padding-left:12px; color:var(--wiz-muted); }
.wiz-processing .error { color:#fca5a5; }
.gradio-container button { min-height:38px; font-weight:600 !important; box-shadow:none !important; }
.gradio-container textarea, .gradio-container input { font-size:14px !important; line-height:1.55 !important; }
.gradio-container .tab-nav { border-bottom:1px solid var(--wiz-rule) !important; gap:8px; background:transparent !important; }
.gradio-container .tab-nav button { border:none !important; border-radius:0 !important; padding:10px 12px !important; color:var(--wiz-muted); font-size:13px; }
.gradio-container .tab-nav button.selected { color:#a78bfa !important; box-shadow:inset 0 -2px #8b5cf6 !important; background:transparent !important; }
.gradio-container .block { box-shadow:none !important; }
.upload-area { border:1px dashed #a5b4fc !important; border-radius:8px !important; min-height:175px; }
.upload-area:hover { border-color:#c4b5fd !important; }
.extracted-text textarea { min-height:365px !important; }
.gradio-container a { color:#a78bfa; }
footer { display:none !important; }
@media(max-width:760px) {
 body .gradio-container.gradio-container { padding:16px 12px 28px !important; }
 .gradio-container .row > .column { flex-basis:100% !important; }
 .wiz-brand { font-size:21px; }
 .wiz-subtitle { font-size:13px; }
 .gradio-container textarea, .gradio-container input { font-size:16px !important; }
 .gradio-container .tab-nav { overflow-x:auto; flex-wrap:nowrap; }
 .wiz-library article { flex-wrap:wrap; gap:8px; }
 .wiz-rule { padding:12px !important; }
 .extracted-text textarea { min-height:280px !important; }
}
"""
