import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from tools.document_visuals import setting


def render_word(source, *, settings, deadline):
    executable = (
        setting(settings, "LIBREOFFICE_BINARY")
        or shutil.which("soffice")
        or shutil.which("libreoffice")
    )
    if not executable:
        return (
            None,
            "LibreOffice is unavailable. Native Word text, chart and DrawingML data are retained; complex diagram rendering requires LibreOffice.",
        )
    remaining = min(
        float(setting(settings, "DOCX_RENDER_TIMEOUT_SECONDS", 45)), deadline - time.monotonic()
    )
    if remaining <= 0:
        return None, "The Word rendering deadline was exceeded; native extraction was retained."
    source = Path(source).resolve()
    target = source.parent / "rendered"
    target.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wizora-office-") as profile:
        try:
            result = subprocess.run(
                [
                    str(executable),
                    "-env:UserInstallation=" + Path(profile).as_uri(),
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(target),
                    str(source),
                ],
                capture_output=True,
                timeout=remaining,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            return (
                None,
                "LibreOffice could not render this Word document within the deadline; native extraction was retained.",
            )
    pdf = target / (source.stem + ".pdf")
    if result.returncode or not pdf.is_file():
        return (
            None,
            "LibreOffice did not produce a usable Word rendering; native extraction was retained.",
        )
    return pdf, None


def rendered_candidates(source, pdf, *, settings, deadline):
    import pymupdf

    from tools.document_visuals import check_deadline, store_png
    from tools.pdf_extractor_tool import _hybrid_regions, _hybrid_tables

    candidates = []
    maximum = min(128, int(setting(settings, "VLM_MAX_VISUALS", 32)))
    with pymupdf.open(pdf) as document:
        for page in document:
            check_deadline(deadline)
            for rect, kind, _ in _hybrid_regions(page, _hybrid_tables(page), scanned=False):
                candidate = {
                    "element_id": f"rendered-page-{page.number + 1:06d}",
                    "render_page": page.number + 1,
                    "render_reference": "rendered/" + pdf.name,
                    "artifact_type": kind,
                    "priority": 1.0,
                    "source_locator_precision": "rendered_page",
                    "bbox": list(rect),
                }
                if len(candidates) >= maximum:
                    candidate.update(
                        reference="", processing_status="skipped_budget", reason="render_budget"
                    )
                else:
                    dimension = int(setting(settings, "VLM_IMAGE_MAX_DIMENSION", 1280))
                    scale = min(2.0, dimension / max(rect.width, rect.height))
                    pixmap = page.get_pixmap(
                        matrix=pymupdf.Matrix(scale, scale), clip=rect, alpha=False
                    )
                    candidate["reference"] = store_png(source, pixmap.tobytes("png"))
                candidates.append(candidate)
    return candidates
