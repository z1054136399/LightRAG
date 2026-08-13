"""Document-to-pages rendering helpers for the dots.ocr engine.

Supports:
- PDF / any fitz-supported format: render with pymupdf.
- PPTX / DOCX: convert to PDF via LibreOffice first so embedded images are
  preserved as proper PDF image objects, then render.
- Standalone images (PNG/JPG/…): load as a single-page document.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from lightrag.utils import logger

if TYPE_CHECKING:
    from PIL import Image as PILImage


def convert_to_pdf(doc_path: str, libreoffice_path: str, output_dir: str) -> str:
    """Convert a PPTX/DOCX to PDF via LibreOffice.

    Writes the PDF into *output_dir*. Returns the full path to the resulting
    PDF file.
    """
    cmd = [
        libreoffice_path,
        "--headless",
        "--norestore",
        "--convert-to", "pdf",
        "--outdir", output_dir,
        doc_path,
    ]
    logger.info("[dots] Converting %s to PDF via LibreOffice", doc_path)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    stem = Path(doc_path).stem
    pdf_path = Path(output_dir) / f"{stem}.pdf"
    if result.returncode != 0:
        if pdf_path.exists():
            # LibreOffice exited non-zero (e.g. javaldx Java warning) but still
            # produced the PDF — treat as a non-fatal warning and continue.
            detail = (result.stderr or result.stdout or "").strip()
            logger.warning(
                "[dots] LibreOffice exited rc=%d but PDF was produced (ignoring): %s",
                result.returncode,
                detail,
            )
        else:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"LibreOffice conversion failed (rc={result.returncode}): {detail}"
            )
    elif not pdf_path.exists():
        raise RuntimeError(
            f"LibreOffice did not produce expected PDF: {pdf_path}"
        )
    logger.info("[dots] LibreOffice produced: %s", pdf_path)
    return str(pdf_path)


def render_document_pages(
    doc_path: str, dpi: int = 200
) -> "list[tuple[PILImage.Image, int]]":
    """Render any fitz-supported document into (PIL image, page_idx) tuples."""
    try:
        import fitz
    except ImportError:
        raise RuntimeError(
            "pymupdf (fitz) is required for dots.ocr parsing but not installed"
        )
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "Pillow is required for dots.ocr parsing but not installed"
        )

    pages: list[tuple[PILImage.Image, int]] = []
    doc = fitz.open(doc_path)
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append((img, i))
    finally:
        doc.close()
    return pages


def load_image_pages(
    image_path: str,
) -> "list[tuple[PILImage.Image, int]]":
    """Load a standalone image file as a single-page document."""
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "Pillow is required for dots.ocr parsing but not installed"
        )
    img = Image.open(image_path).convert("RGB")
    return [(img, 0)]


class ImageCounter:
    """Thread-safe sequential counter for unique extracted-image filenames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._n = 0

    def next(self, ext: str = "png") -> str:
        with self._lock:
            name = f"image_{self._n}.{ext}"
            self._n += 1
        return name


_CATEGORY_MAP: dict[str, tuple[str, int | None]] = {
    "Title": ("text", 1),
    "Section-header": ("text", 2),
    "Text": ("text", None),
    "List-item": ("text", None),
    "Caption": ("text", None),
    "Footnote": ("text", None),
    "Page-header": ("text", None),
    "Page-footer": ("text", None),
    "Picture": ("image", None),
    "Table": ("table", None),
    "Formula": ("equation", None),
}


def cells_to_content_list(
    cells: list[dict],
    page_idx: int,
    page_img: "PILImage.Image",
    images_dir: str,
    counter: ImageCounter,
) -> list[dict]:
    """Convert one page's layout cells to content_list items."""
    items: list[dict] = []
    for cell in cells:
        category = cell.get("category", "Text")
        text = cell.get("text", "")
        bbox = cell.get("bbox")
        item_type, level = _CATEGORY_MAP.get(category, ("text", None))

        item: dict = {"page_idx": page_idx}
        if bbox:
            item["bbox"] = bbox

        if item_type == "image":
            rel = _crop_image(page_img, bbox, images_dir, counter)
            if not rel:
                continue
            item["type"] = "image"
            item["img_path"] = rel
            if text:
                item["image_caption"] = [text]
        elif item_type == "table":
            item["type"] = "table"
            item["table_body"] = text
        elif item_type == "equation":
            if not text:
                continue
            item["type"] = "equation"
            item["text"] = text
        else:
            if not text:
                continue
            item["type"] = "text"
            item["text"] = text
            if level is not None:
                item["text_level"] = level
        items.append(item)
    return items


def _crop_image(
    page_img: "PILImage.Image",
    bbox,
    images_dir: str,
    counter: ImageCounter,
) -> str | None:
    if not bbox or len(bbox) < 4:
        return None
    w, h = page_img.size
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    try:
        crop = page_img.crop((x1, y1, x2, y2))
    except Exception as exc:
        logger.warning("[dots] crop failed: %s", exc)
        return None
    fname = counter.next("png")
    crop.save(os.path.join(images_dir, fname), format="PNG")
    return f"images/{fname}"


__all__ = [
    "ImageCounter",
    "cells_to_content_list",
    "convert_to_pdf",
    "load_image_pages",
    "render_document_pages",
]
