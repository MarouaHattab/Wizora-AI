import gradio as gr

VLM_DROPDOWN_JS = """() => {
    const root = document.querySelector('.gradio-container');
    if (!root || root.dataset.wizoraVisualEscape === 'ready') return [];
    root.dataset.wizoraVisualEscape = 'ready';
    root.addEventListener('keyup', (event) => {
        if (event.key !== 'Escape' || !(event.target instanceof HTMLInputElement)) return;
        if (event.target.closest('#vlm-mode, #vlm-override-file, #vlm-override-mode')) {
            // Gradio closes on Escape but keeps focus; its menu opens on focus,
            // so a later click cannot reopen it until the field loses focus.
            event.target.blur();
        }
    });
    return [];
}"""


CUSTOM_CSS = """
/* Preserve Wizora-AI's full-width purple workflow in both color modes. */
body { background: #f8fafc !important; }
body.dark { background: #0f1423 !important; }
.gradio-container { width: 100% !important; max-width: none !important; margin: 0 !important;
    --wiz-page: #f8fafc; --wiz-surface: #ffffff; --wiz-text: #1f2937; --wiz-muted: #4b5563;
    --wiz-border: #e5e7eb; --wiz-heading: #312e81; --wiz-subheading: #6b21a8; --wiz-link: #4338ca;
    --wiz-accent-surface: #eef2ff; --wiz-accent-text: #312e81; --wiz-accent-border: #c7d2fe;
    --wiz-success-bg: #f0fdf4; --wiz-success-text: #166534; --wiz-success-border: #86efac;
    --wiz-error-bg: #fef2f2; --wiz-error-text: #991b1b; --wiz-error-border: #fca5a5;
    padding: clamp(16px, 2.2vw, 28px) clamp(10px, 2.55vw, 32px) clamp(28px, 3.5vw, 44px) !important;
    background: var(--wiz-page) !important;
    color: var(--wiz-text) !important; color-scheme: light; }
.dark .gradio-container, .gradio-container.dark {
    --wiz-page: #0f1423; --wiz-surface: #191f31; --wiz-text: #e5e7eb; --wiz-muted: #b9c3d5;
    --wiz-border: #3d455d; --wiz-heading: #ddd6fe; --wiz-subheading: #d8b4fe; --wiz-link: #c4b5fd;
    --wiz-accent-surface: #25213f; --wiz-accent-text: #ddd6fe; --wiz-accent-border: #65588c;
    --wiz-success-bg: #14322a; --wiz-success-text: #bbf7d0; --wiz-success-border: #34755b;
    --wiz-error-bg: #3c202a; --wiz-error-text: #fecaca; --wiz-error-border: #99535e;
    color-scheme: dark; }
.header { text-align: center; margin: 0 auto 22px !important; padding: 8px 16px 18px !important;
    background: transparent; }
.header h1 { margin-bottom: 4px !important; color: var(--wiz-heading); letter-spacing: -.02em; }
.header h3 { color: var(--wiz-subheading); font-weight: 600; }
.gradio-container textarea, .gradio-container input { color: var(--wiz-text) !important;
    background: var(--wiz-surface) !important; }
.gradio-container label span { color: var(--wiz-text) !important; font-weight: 600; }
.gradio-container #workflow-tabs { overflow: hidden; border-radius: 18px;
    background: var(--wiz-surface); box-shadow: 0 8px 28px rgba(76, 29, 149, .10); }
.gradio-container #workflow-tabs .tab-wrapper { height: auto; min-height: 56px; padding: 0; }
.gradio-container #workflow-tabs .tab-container { display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr)); align-items: stretch;
    box-sizing: border-box; height: auto; gap: 6px; padding: 7px !important; }
.gradio-container #workflow-tabs [role="tablist"] {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 0; }
.gradio-container #workflow-tabs .tab-container button { justify-content: center;
    min-width: 0; min-height: 46px; padding: 10px 8px; white-space: normal;
    line-height: 1.3; text-align: center; }
.gradio-container #workflow-tabs [role="tab"] { color: #f8fafc !important;
    border-radius: 11px !important; border: 1px solid transparent !important;
    background: transparent !important; font-weight: 600; }
.gradio-container #workflow-tabs [role="tab"]:hover:not(:disabled),
.gradio-container #workflow-tabs [role="tab"][aria-selected="true"] {
    background: rgba(49, 46, 129, .72) !important; color: #ffffff !important; }
.gradio-container #workflow-tabs [role="tab"]:focus-visible {
    outline: 3px solid #fef08a !important; outline-offset: -5px; }
.gradio-container #workflow-tabs [role="tab"][aria-selected="true"]::after {
    background: #ffffff !important; }
.workflow-panel { padding: 24px 24px 28px !important; }
.workflow-description { margin: 0 0 18px !important; color: var(--wiz-muted); }
.workflow-row { gap: 22px !important; align-items: start; }
.workflow-card { padding: 18px !important; border-radius: 16px;
    border: 1px solid var(--wiz-border); background: var(--wiz-surface); min-width: 0; }
.workflow-results { background: var(--wiz-surface); }
.upload-area { border: 2px dashed #e0e7ff !important; border-radius: 16px !important;
    background: var(--wiz-surface) !important; transition: border-color .2s ease, background .2s ease !important; }
.upload-area:hover, .upload-area:focus-within { border-color: #4f46e5 !important;
    background: var(--wiz-accent-surface) !important; }
.button-primary { background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important; color: #ffffff !important; border-radius: 11px !important;
    padding: 11px 20px !important; font-weight: 700 !important;
    transition: transform .2s ease, box-shadow .2s ease !important; }
.button-primary:hover:not(:disabled) { transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(79, 70, 229, .28); }
.gradio-container button:focus-visible, .gradio-container input:focus-visible,
.gradio-container textarea:focus-visible { outline: 3px solid #a78bfa !important;
    outline-offset: 2px; }
.extracted-text textarea, .workflow-results textarea { line-height: 1.55 !important; }
.extracted-text textarea { font-family: "Courier New", Courier, monospace !important;
    background: var(--wiz-page) !important; max-height: 600px !important; overflow-y: auto !important; }
.status { border: 1px solid var(--wiz-accent-border); background: var(--wiz-accent-surface); color: var(--wiz-accent-text);
    border-radius: 11px; padding: 12px 14px; margin-top: 16px; }
.status-success { border-color: var(--wiz-success-border); background: var(--wiz-success-bg); color: var(--wiz-success-text); }
.status-error { border-color: var(--wiz-error-border); background: var(--wiz-error-bg); color: var(--wiz-error-text); }
.chat-container { max-height: 500px !important; overflow-y: auto !important; background: var(--wiz-surface) !important;
    border: 1px solid var(--wiz-border) !important; border-radius: 14px !important; }
.asset-details, .source-preview { overflow-wrap: anywhere; color: var(--wiz-text); }
.source-preview { max-height: 440px; overflow-y: auto; }
.source-preview a { color: var(--wiz-link) !important; }
#vlm-mode { border-color: var(--wiz-border) !important; background: var(--wiz-surface) !important; }
#vlm-mode-description { color: var(--wiz-muted); font-size: .9rem; line-height: 1.5; }
#vlm-mode input { min-width: 0; }
#file-progress, #job-json { max-width: 100%; overflow-x: auto; }
#provider-status { max-height: 24rem; max-width: 100%; overflow: auto; }
#provider-status table { min-width: 640px; }
@media (max-width: 720px) {
    .header { padding: 18px 14px !important; border-radius: 16px; }
    #workflow-tabs .tab-container { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .workflow-panel { padding: 18px 12px 22px !important; }
    .workflow-row { gap: 12px !important; }
    .workflow-card { padding: 13px !important; }
}
@media (prefers-reduced-motion: reduce) {
    .button-primary, .upload-area { transition: none !important; }
    .button-primary:hover { transform: none; }
}
"""


def create_theme():
    return gr.themes.Soft(primary_hue="purple").set(
        background_fill_primary="#f8fafc",
        background_fill_primary_dark="#0f1423",
        background_fill_secondary="#ffffff",
        background_fill_secondary_dark="#191f31",
        body_text_color="#1f2937",
        body_text_color_dark="#e5e7eb",
        block_background_fill="#ffffff",
        block_background_fill_dark="#191f31",
    )
