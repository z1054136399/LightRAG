"""
Shared multimodal helpers for LightRAG.

Pure, dependency-free utilities used both by the API layer (to inline image
URLs into query responses) and by the query/retrieval core (to inline image
URLs into the LLM context so the model can reference them in its answer).

Keeping these here (rather than in the API router) avoids a circular import
between ``lightrag.operate`` (core) and ``lightrag.api.routers.*`` (HTTP).
"""
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

# Sidecar roots that may hold renderable assets, in lookup priority.
_SIDECAR_ROOTS = ("drawings", "tables", "equations")

_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

# Matches a self-closing <drawing ... /> marker and captures its id attribute.
_DRAWING_RE = re.compile(r'<drawing\b[^>]*?\bid="([^"]+)"[^>]*?/>')


def _parsed_root() -> Path:
    """Return ``<INPUT_DIR>/__parsed__`` from the container env (or a default)."""
    base = os.getenv("INPUT_DIR") or os.getenv("LIGHTRAG_INPUT_DIR") or "/app/data/inputs"
    return Path(base) / "__parsed__"


def is_multimodal_enabled() -> bool:
    """Gate for <img> injection. Defaults to off so behaviour is unchanged unless opted in."""
    return os.getenv("MULTIMODAL_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def resolve_public_base(request=None) -> str:
    """Base URL prefix for inlined ``<img>`` tags.

    Priority: explicit ``MULTIMODAL_PUBLIC_BASE`` (e.g. behind a reverse proxy)
    -> otherwise the incoming request's ``base_url`` (same origin that served
    the query). Returns "" when neither is available, making injection a no-op.
    """
    cfg = os.getenv("MULTIMODAL_PUBLIC_BASE", "").strip()
    if cfg:
        return cfg.rstrip("/")
    base_url = getattr(request, "base_url", None)
    if base_url:
        return str(base_url).rstrip("/")
    return ""


def doc_stem_from_filepath(file_path: str) -> str:
    """Extract the parsed-dir stem from a chunk/reference ``file_path``.

    The ``/multimodal/{doc_stem}`` route resolves ``<doc_stem>.parsed`` where
    ``doc_stem`` is the bare source file name (e.g. ``img_doc.pdf``). Stored
    chunk ``file_path`` is already canonicalized to that basename, but taking
    ``Path(...).name`` keeps this correct if a full path ever leaks through
    (idempotent for bare names) and matches ``query_routes._stem_from_filepath``.
    """
    if not file_path:
        return ""
    return Path(file_path).name


def inject_image_urls(content: str, doc_stem: str, public_base: str = "") -> str:
    """Replace ``<drawing id="X" .../>`` markers with a Markdown image link.

    ``![drawing X](.../multimodal/{doc_stem}/X)`` — Markdown (not raw HTML) is
    used on purpose: LLMs reliably echo Markdown image links in their answers,
    and the WebUI renders Markdown ``![]()`` as ``<img>`` (which the
    ``MultimodalImg`` component upgrades to an authenticated ``blob:`` URL). Raw
    ``<img>`` tags are usually dropped or ignored by the model.

    When ``public_base`` is empty (the common case for LLM context injection,
    or when no reverse-proxy base is configured), the URL is a **relative path**
    ``/multimodal/{doc_stem}/{media_id}`` so the caller's origin resolves it
    automatically. When ``public_base`` is set, an absolute URL is produced.

    Returns the content unchanged when content/doc_stem is empty or no markers
    are present, so callers can safely pass every chunk through.
    """
    if not content or not doc_stem:
        return content

    prefix = public_base.rstrip("/") if public_base else ""
    url_prefix = f"{prefix}/multimodal" if prefix else "/multimodal"

    def _repl(m: "re.Match") -> str:
        media_id = m.group(1)
        # URL-encode doc_stem so filenames with spaces (e.g. "My Doc.docx") don't
        # produce invalid Markdown image syntax — CommonMark requires spaces in
        # URLs to be percent-encoded (or the URL wrapped in angle brackets).
        safe_stem = quote(doc_stem, safe=".-_~")
        url = f"{url_prefix}/{safe_stem}/{media_id}"
        return f"![drawing {media_id}]({url})"

    return _DRAWING_RE.sub(_repl, content)


# Appended to the answer-system prompt when the retrieval context already carries
# inlined Multimodal image links, so the model reproduces them in its reply.
_MULTIMODAL_ANSWER_HINT = (
    "\n\nIMPORTANT: Some of the source passages above contain image references "
    "written as Markdown links of the form `![alt](/multimodal/...)`. For every "
    "passage or step you include in your answer, reproduce ALL of its image "
    "Markdown links verbatim at the corresponding point in your response so the "
    "reader can view every figure inline. Do not omit any image from sections you "
    "discuss, and do not replace them with a textual description."
)


def multimodal_answer_hint(context_text: str) -> str:
    """Return an instruction to echo inlined image links, only when present.

    ``context_text`` is the assembled retrieval context (the value fed into the
    ``{context_data}`` / ``{content_data}`` slot of the answer prompt). The hint
    is returned only when multimodal is enabled AND at least one ``/multimodal/``
    link is present, so non-multimodal queries are completely unaffected.
    """
    if is_multimodal_enabled() and context_text and "/multimodal/" in context_text:
        return _MULTIMODAL_ANSWER_HINT
    return ""
