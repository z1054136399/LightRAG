"""Cache validation for ``*.dots_raw/`` bundles.

Policy mirrors MinerU (see ``parser/external/mineru/cache.py``):

1. ``_manifest.json`` exists, parses, ``version=1.0`` ∧ ``engine=dots``.
2. Source size fast-path.
3. Source content sha256.
4. Options signature: api_url + model + render_dpi.
5. Critical file (content_list.json): size + sha256.
6. Other files: size only.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightrag.parser.external._manifest import load_manifest
from lightrag.utils import logger

CONTENT_LIST_FILENAME = "content_list.json"

DEFAULT_DOTS_MODEL = "dotsocr-model"
DEFAULT_DOTS_RENDER_DPI = 200


@dataclass(frozen=True)
class DotsParserOptions:
    """Effective dots.ocr parser options — used for live requests and the
    cache signature so the client and cache validator always agree."""

    api_url: str
    model: str
    render_dpi: int

    @classmethod
    def from_env(
        cls, overrides: "Mapping[str, Any] | None" = None
    ) -> "DotsParserOptions":
        overrides = overrides or {}
        return cls(
            api_url=str(
                overrides.get("api_url", os.getenv("DOTS_API_URL", ""))
            ).strip(),
            model=(
                str(
                    overrides.get(
                        "model", os.getenv("DOTS_MODEL", DEFAULT_DOTS_MODEL)
                    )
                ).strip()
                or DEFAULT_DOTS_MODEL
            ),
            render_dpi=int(
                overrides.get(
                    "render_dpi",
                    os.getenv("DOTS_RENDER_DPI", str(DEFAULT_DOTS_RENDER_DPI)),
                )
            ),
        )

    def signature(self) -> str:
        payload = {
            "signature_version": 1,
            "api_url": self.api_url,
            "model": self.model,
            "render_dpi": self.render_dpi,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_size_and_hash(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, f"sha256:{h.hexdigest()}"


def is_bundle_valid(
    raw_dir: Path,
    source_file: Path,
    *,
    overrides: "Mapping[str, Any] | None" = None,
) -> bool:
    """Return True iff the bundle is intact and matches the current source."""
    if not raw_dir.is_dir():
        return False
    manifest = load_manifest(raw_dir, expected_engine="dots")
    if manifest is None:
        return False

    try:
        cur_size = source_file.stat().st_size
    except OSError:
        return False
    if cur_size != int(manifest.source_size_bytes):
        return False

    _, cur_hash = compute_size_and_hash(source_file)
    if cur_hash != manifest.source_content_hash:
        return False

    if not manifest.options_signature:
        return False
    if DotsParserOptions.from_env(overrides).signature() != manifest.options_signature:
        return False

    crit = manifest.critical_file
    crit_path = raw_dir / crit.path
    try:
        if crit_path.stat().st_size != int(crit.size):
            return False
    except OSError:
        return False
    if crit.sha256:
        _, actual = compute_size_and_hash(crit_path)
        if actual != crit.sha256:
            return False

    for entry in manifest.files:
        ep = raw_dir / entry.path
        try:
            if ep.stat().st_size != int(entry.size):
                return False
        except OSError:
            return False

    return True


__all__ = [
    "CONTENT_LIST_FILENAME",
    "DotsParserOptions",
    "compute_size_and_hash",
    "is_bundle_valid",
]
