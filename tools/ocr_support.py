import hashlib
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path

from helper.ingestion_errors import ProcessingError


def _options(language, backend):
    if backend not in {"auto", "pymupdf", "tesseract"}:
        raise ProcessingError("Choose an enabled OCR backend: auto, pymupdf or tesseract.")
    if (
        not isinstance(language, str)
        or re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", language) is None
    ):
        raise ProcessingError("OCR languages must be Tesseract language codes separated by +.")
    return list(dict.fromkeys(language.split("+")))


def normalize_ocr_text(text):
    # NFC retains mathematical compatibility distinctions (x², x₂, ℝ). Expand
    # only legacy Arabic presentation forms so logical Arabic stays searchable.
    expanded = "".join(
        unicodedata.normalize("NFKC", char)
        if 0xFB50 <= ord(char) <= 0xFDFF or 0xFE70 <= ord(char) <= 0xFEFF
        else char
        for char in text
    )
    return unicodedata.normalize("NFC", expanded).strip()


def _language_files(directory, languages):
    result = {}
    for code in languages:
        path = directory / (code + ".traineddata") if directory else None
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            result[code] = digest.hexdigest()
        except (OSError, AttributeError):
            result[code] = None
    return result


def _remaining(deadline, limit=60):
    remaining = min(limit, deadline - time.monotonic()) if deadline is not None else limit
    if remaining <= 0:
        raise ProcessingError("The OCR processing deadline was exceeded.")
    return remaining


def ocr_dependencies(*, language="eng", binary=None, backend="auto", deadline=None):
    languages = _options(language, backend)
    executable = binary or shutil.which("tesseract")
    directory = None
    try:
        import pymupdf

        directory = Path(pymupdf.get_tessdata())
    except (ImportError, OSError, RuntimeError, TypeError):
        pass
    files = _language_files(directory, languages)
    missing = [code for code in languages if files[code] is None]
    available = not missing
    result = {
        "tesseract_binary_available": bool(executable),
        "language_data_available": available,
        "missing_languages": missing,
        "pymupdf_backend_available": available,
        "native_directory": str(directory.resolve()) if directory else None,
        "native_language_files": files,
        "backend": "pymupdf" if backend != "tesseract" and available else "tesseract",
        "binary": str(executable) if executable else None,
    }
    if backend == "tesseract" or backend == "auto" and not available and executable:
        cli_languages, cli_directory = [], None
        if executable:
            try:
                listed = subprocess.run(
                    [str(executable), "--list-langs"],
                    capture_output=True,
                    timeout=_remaining(deadline, 10),
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                output = (
                    listed.stdout.decode("utf-8", errors="replace")
                    if isinstance(listed.stdout, bytes)
                    else listed.stdout
                )
                if not listed.returncode:
                    cli_languages = [
                        line.strip()
                        for line in output.splitlines()
                        if re.fullmatch(r"[A-Za-z0-9_]+", line.strip())
                    ]
                    match = re.search(r'List of available languages in "([^"]+)"', output)
                    cli_directory = Path(match.group(1)) if match else directory
            except (OSError, subprocess.TimeoutExpired):
                pass
        result.update(
            missing_languages=[code for code in languages if code not in cli_languages],
            language_data_available=all(code in cli_languages for code in languages),
            cli_languages=sorted(cli_languages),
            cli_directory=str(cli_directory.resolve()) if cli_directory else None,
            cli_language_files=_language_files(cli_directory, languages),
        )
    return result


def validate_ocr_configuration(*, language="eng", backend="auto", binary=None, deadline=None):
    status = ocr_dependencies(language=language, backend=backend, binary=binary, deadline=deadline)
    if not status["language_data_available"]:
        missing = ", ".join(status["missing_languages"])
        raise ProcessingError(
            f"OCR is unavailable for requested language data: {missing}. Configure TESSDATA_PREFIX for PyMuPDF, or install Tesseract with every requested language; no language was substituted."
        )
    return status


def ocr_cache_identity(*, language="eng", backend="auto", binary=None, deadline=None):
    status = ocr_dependencies(language=language, backend=backend, binary=binary, deadline=deadline)
    if backend == "auto" and status["binary"]:
        # Runtime native failure can select CLI even with native files present.
        # Include that possible backend so its stale OCR cannot bypass preflight.
        status["cli_fallback"] = ocr_dependencies(
            language=language, backend="tesseract", binary=binary, deadline=deadline
        )
    return status


def recognize_page(page, *, language="eng", backend="auto", binary=None, deadline=None):
    _remaining(deadline)
    status = validate_ocr_configuration(
        language=language, backend=backend, binary=binary, deadline=deadline
    )
    if status["backend"] == "pymupdf":
        try:
            textpage = page.get_textpage_ocr(
                language=language, dpi=150, full=True, tessdata=status["native_directory"]
            )
            # PyMuPDF spatial sorting reverses words in recognized RTL lines.
            return normalize_ocr_text(page.get_text("text", textpage=textpage, sort=False))
        except Exception as error:
            if backend == "pymupdf":
                raise ProcessingError(
                    "PyMuPDF OCR failed. Set TESSDATA_PREFIX to the requested language data directory, or install Tesseract and its language data."
                ) from error
            status = validate_ocr_configuration(
                language=language, backend="tesseract", binary=binary, deadline=deadline
            )
    executable = status["binary"]
    if not executable:
        raise ProcessingError(
            "OCR is unavailable: set TESSDATA_PREFIX for PyMuPDF, or install Tesseract with the requested language data and configure OCR_BINARY."
        )
    remaining = _remaining(deadline)
    with tempfile.TemporaryDirectory(prefix="wizora-ocr-") as directory:
        image = Path(directory) / "page.png"
        page.get_pixmap(dpi=150, alpha=False).save(image)
        try:
            result = subprocess.run(
                [str(executable), str(image), "stdout", "-l", language],
                capture_output=True,
                timeout=remaining,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ProcessingError(
                "Tesseract OCR could not complete within the configured deadline."
            ) from error
        if result.returncode:
            raise ProcessingError(
                "Tesseract OCR failed. Verify installation and the configured language data."
            )
        return normalize_ocr_text(result.stdout.decode("utf-8", errors="replace"))
