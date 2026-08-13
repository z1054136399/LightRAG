"""
Multimodal asset serving routes for LightRAG.

When ``MULTIMODAL_ENABLED`` is set, query responses inline ``<img>`` tags that
point at ``/multimodal/{doc_stem}/{media_id}``. This router serves the original
parsed asset (image / table / equation) so the caller can render it from the
same origin that served the query — no external storage or base64 embedding.

Asset discovery is deliberately tolerant of the parser's naming quirks:
- the parsed directory is ``<parsed_root>/<doc_stem>.parsed`` where
  ``doc_stem`` is the bare source file name (e.g. ``img_doc.pdf``);
- the sidecar file name drops the extension (e.g. ``img_doc.drawings.json``),
  so we glob ``*.drawings.json`` rather than assuming a fixed name;
- the asset ``path`` inside the sidecar is relative to the parsed directory.
"""
import json
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Path as _ApiPath
from fastapi.responses import FileResponse

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.multimodal_utils import (
    _MIME_BY_EXT,
    _SIDECAR_ROOTS,
    _parsed_root,
    doc_stem_from_filepath,
    inject_image_urls,
    is_multimodal_enabled,
    resolve_public_base,
)


def _find_and_serve(parsed_root: Path, doc_stem: str, media_id: str):
    """Locate and return a FileResponse for *media_id* inside *parsed_root*.

    Returns ``None`` if the parsed directory or asset is not found (so the
    caller can fall through to the next candidate root).
    """
    parsed_dir = parsed_root / f"{doc_stem}.parsed"
    if not parsed_dir.is_dir():
        return None

    for root in _SIDECAR_ROOTS:
        for sidecar in sorted(parsed_dir.glob(f"*.{root}.json")):
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            entries = data.get(root)
            if isinstance(entries, dict):
                items = list(entries.values())
            elif isinstance(entries, list):
                items = entries
            else:
                items = []
            for it in items:
                if not isinstance(it, dict) or it.get("id") != media_id:
                    continue
                rel = it.get("path") or it.get("img_path") or it.get("src") or ""
                if not rel:
                    continue
                asset = Path(rel)
                if not asset.is_absolute():
                    asset = parsed_dir / rel
                if asset.is_file():
                    return FileResponse(
                        asset,
                        media_type=_MIME_BY_EXT.get(asset.suffix.lower(), "application/octet-stream"),
                    )
    return None


def create_multimodal_routes(
    api_key: Optional[str] = None,
    kb_input_dir: Optional[Path] = None,
    *,
    kb_registry=None,
    input_dir_base: Optional[str] = None,
):
    """Build the ``/multimodal`` router.

    Parameterized mode (``kb_registry`` + ``input_dir_base``): the handler
    resolves the per-request ``kb_id`` path parameter and serves from
    ``input_dir_base/{kb_id}/__parsed__/``.

    Legacy fixed mode (``kb_input_dir``): serves from ``kb_input_dir/__parsed__/``.

    Fallback: environment-derived ``_parsed_root()``.
    """
    router = APIRouter(tags=["multimodal"])
    combined_auth = get_combined_auth_dependency(api_key)

    if kb_registry is not None and input_dir_base is not None:
        _input_dir_base = Path(input_dir_base)

        def _get_parsed_root(kb_id: str = _ApiPath(...)) -> Path:
            return _input_dir_base / kb_id / "__parsed__"

        ParsedRootDep = Annotated[Path, Depends(_get_parsed_root)]

        @router.get(
            "/multimodal/{doc_stem}/{media_id}",
            dependencies=[Depends(combined_auth)],
            summary="Serve a parsed multimodal asset by id",
            responses={404: {"description": "Asset not found"}},
        )
        async def serve_multimodal(doc_stem: str, media_id: str, parsed_root: ParsedRootDep):
            result = _find_and_serve(parsed_root, doc_stem, media_id)
            if result is not None:
                return result
            raise HTTPException(status_code=404, detail="multimodal asset not found")
    else:
        # Legacy fixed-root mode (used when mounting per-KB with kb_input_dir,
        # or falling back to global parsed root).
        _fixed_root: Path = (kb_input_dir / "__parsed__") if kb_input_dir is not None else _parsed_root()

        @router.get(
            "/multimodal/{doc_stem}/{media_id}",
            dependencies=[Depends(combined_auth)],
            summary="Serve a parsed multimodal asset by id",
            responses={404: {"description": "Asset not found"}},
        )
        async def serve_multimodal(doc_stem: str, media_id: str):  # type: ignore[misc]
            result = _find_and_serve(_fixed_root, doc_stem, media_id)
            if result is not None:
                return result
            raise HTTPException(status_code=404, detail="multimodal asset not found")

    return router
