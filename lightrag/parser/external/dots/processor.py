"""Runs the dots.ocr processing pipeline and writes a raw bundle.

Implements the ``download_into`` hook for :class:`DotsParser`: given a source
file path and a raw directory, it:

1. Converts PPTX/DOCX to PDF via LibreOffice (to preserve embedded images).
2. Renders pages with pymupdf.
3. Sends each page to the dots.ocr VLM endpoint.
4. Writes ``content_list.json`` + ``images/`` into raw_dir.
5. Writes ``_manifest.json`` for cache validation.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Limit concurrent dots.ocr document processing — the service is not designed
# for parallel load. Default 1; raise with DOTS_MAX_CONCURRENT env var.
_DOTS_SEMAPHORE: asyncio.Semaphore | None = None


def _get_dots_semaphore() -> asyncio.Semaphore:
    global _DOTS_SEMAPHORE
    if _DOTS_SEMAPHORE is None:
        n = max(1, int(os.getenv("DOTS_MAX_CONCURRENT", "1")))
        _DOTS_SEMAPHORE = asyncio.Semaphore(n)
    return _DOTS_SEMAPHORE

from lightrag.parser.external._manifest import (
    Manifest,
    ManifestFile,
    write_manifest,
)
from lightrag.parser.external.dots.cache import (
    CONTENT_LIST_FILENAME,
    DotsParserOptions,
    compute_size_and_hash,
)
from lightrag.parser.external.dots.client import DotsClient
from lightrag.parser.external.dots.converter import (
    ImageCounter,
    cells_to_content_list,
    convert_to_pdf,
    load_image_pages,
    render_document_pages,
)
from lightrag.utils import logger

_LIBREOFFICE_FORMATS = frozenset({".pptx", ".docx"})


class DotsRawProcessor:
    """Processes a source document via dots.ocr and writes the raw bundle.

    Analogous to :class:`MinerURawClient` but runs in-process instead of
    calling an external HTTP service.
    """

    def __init__(self, *, overrides: "Mapping[str, Any] | None" = None) -> None:
        self._overrides = dict(overrides or {})
        self._options = DotsParserOptions.from_env(overrides)
        self._libreoffice_path = os.getenv("LIBREOFFICE_PATH", "soffice")
        self._timeout = int(os.getenv("DOTS_TIMEOUT", "180"))
        host_header = os.getenv("DOTS_API_HOST_HEADER", "")
        token = os.getenv("DOTS_API_TOKEN", "")
        if not self._options.api_url:
            raise ValueError(
                "DOTS_API_URL is required for the dots.ocr parser engine"
            )
        if not token:
            raise ValueError(
                "DOTS_API_TOKEN is required for the dots.ocr parser engine"
            )
        self._client = DotsClient(
            api_url=self._options.api_url,
            token=token,
            host_header=host_header,
            model=self._options.model,
            timeout=self._timeout,
        )

    async def process(
        self,
        raw_dir: Path,
        source_path: Path,
        *,
        upload_name: str | None = None,
    ) -> None:
        """Run the full pipeline. Writes bundle into raw_dir."""
        resolved_name = Path(str(upload_name or "")).name or source_path.name
        async with _get_dots_semaphore():
            await asyncio.to_thread(
                self._process_sync, raw_dir, source_path, resolved_name
            )

    # ------------------------------------------------------------------
    # Synchronous implementation (runs in thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _process_sync(
        self, raw_dir: Path, source_path: Path, upload_name: str
    ) -> None:
        ext = source_path.suffix.lower()

        # Render pages. For PPTX/DOCX: convert to PDF in a temp dir first
        # so LibreOffice-embedded images are preserved as PDF image objects.
        # The temp dir is cleaned up after rendering; PIL images are in memory.
        if ext in _LIBREOFFICE_FORMATS:
            with tempfile.TemporaryDirectory(prefix="dots_lo_") as tmpdir:
                pdf_path = convert_to_pdf(
                    str(source_path), self._libreoffice_path, tmpdir
                )
                pages = render_document_pages(pdf_path, dpi=self._options.render_dpi)
        elif ext == ".pdf":
            pages = render_document_pages(
                str(source_path), dpi=self._options.render_dpi
            )
        else:
            pages = load_image_pages(str(source_path))

        if not pages:
            raise RuntimeError(
                f"[dots] document produced no pages: {source_path.name}"
            )

        logger.info(
            "[dots] Processing %s: %d page(s)", source_path.name, len(pages)
        )

        images_dir = raw_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        counter = ImageCounter()

        # Process pages sequentially — each call is a blocking HTTP request.
        parsed_pages = []
        for page_img, page_idx in pages:
            cells = self._client.parse_image(page_img)
            parsed_pages.append((cells, page_idx, page_img))

        content_list: list[dict] = []
        for cells, page_idx, page_img in parsed_pages:
            content_list.extend(
                cells_to_content_list(
                    cells, page_idx, page_img, str(images_dir), counter
                )
            )

        crit_path = raw_dir / CONTENT_LIST_FILENAME
        crit_path.write_text(
            json.dumps(content_list, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "[dots] wrote %d content items for %s",
            len(content_list),
            source_path.name,
        )

        self._write_manifest(raw_dir, source_path, upload_name)

    def _write_manifest(
        self, raw_dir: Path, source_path: Path, upload_name: str
    ) -> None:
        source_size, source_hash = compute_size_and_hash(source_path)
        crit_path = raw_dir / CONTENT_LIST_FILENAME
        crit_size, crit_hash = compute_size_and_hash(crit_path)

        others: list[ManifestFile] = []
        total = crit_size
        for p in sorted(raw_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.name == "_manifest.json":
                continue
            rel = p.relative_to(raw_dir).as_posix()
            if rel == CONTENT_LIST_FILENAME:
                continue
            size = p.stat().st_size
            others.append(ManifestFile(path=rel, size=size))
            total += size

        manifest = Manifest(
            engine="dots",
            source_content_hash=source_hash,
            source_size_bytes=source_size,
            source_filename_at_parse=upload_name,
            critical_file=ManifestFile(
                path=CONTENT_LIST_FILENAME,
                size=crit_size,
                sha256=crit_hash,
            ),
            files=others,
            total_size_bytes=total,
            task_id="",
            api_mode="",
            engine_version="",
            endpoint_signature=self._options.api_url,
            options_signature=self._options.signature(),
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )
        write_manifest(raw_dir, manifest)


__all__ = ["DotsRawProcessor"]
