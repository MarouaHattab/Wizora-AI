SUPPORTED_EXTENSIONS = [
    ".txt",
    ".pdf",
    ".docx",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".flv",
]


VISUAL_EXTENSIONS = {".pdf", ".docx", ".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv"}


VLM_CHOICES = [
    ("Disabled — Text / Audio Only", "disabled"),
    ("Automatic — Hybrid Extraction (Recommended)", "automatic"),
    ("Enhanced — Comprehensive Visual Understanding", "enhanced"),
]


VLM_DESCRIPTIONS = {
    "disabled": "Extract text, tables and audio only. No visual descriptions or VLM inference.",
    "automatic": "Extract text, tables and audio, then analyze selected meaningful diagrams, charts and visuals to reduce processing time and cost.",
    "enhanced": "Extract text, tables and audio, then analyze all detected meaningful visuals with detailed relationships and context. Duplicate images and frames are skipped.",
}
