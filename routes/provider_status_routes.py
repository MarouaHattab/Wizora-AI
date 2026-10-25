import shutil
from functools import partial

import anyio
from fastapi import APIRouter

from helper import provider_status
from helper.settings import get_settings
from tools.ocr_support import ocr_dependencies

router = APIRouter()


@router.get("/status")
async def read_provider_status(refresh: bool = False):
    """Inspect metadata only. This endpoint never sends generation requests."""
    settings = get_settings()
    setting = partial(provider_status.setting, settings)
    payload = await anyio.to_thread.run_sync(
        partial(provider_status.provider_status, settings, refresh=refresh)
    )
    payload["tools"] = {
        "ocr": {
            "enabled": setting("PDF_OCR_ENABLED", False),
            "backend": setting("PDF_OCR_BACKEND", "auto"),
            "languages": setting("PDF_OCR_LANGUAGE", "eng"),
            **await anyio.to_thread.run_sync(
                partial(
                    ocr_dependencies,
                    language=setting("PDF_OCR_LANGUAGE", "eng"),
                    binary=setting("OCR_BINARY"),
                )
            ),
            "message": "PyMuPDF OCR needs matching language data (TESSDATA_PREFIX); the CLI fallback also needs a Tesseract executable. These are metadata checks; extraction validates the engine.",
        },
        "word_rendering": {
            "enabled": setting("DOCX_RENDER_ENABLED", True),
            "binary_available": bool(
                shutil.which(setting("LIBREOFFICE_BINARY") or "soffice")
                or shutil.which("libreoffice")
            ),
            "message": "LibreOffice renders complex Word visuals; native text and chart extraction remain available without it.",
        },
    }
    return payload
