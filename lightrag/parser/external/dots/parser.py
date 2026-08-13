"""dots.ocr engine adapter (implements ExternalParserBase hooks)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.constants import DOTS_RAW_DIR_SUFFIX, PARSER_ENGINE_DOTS
from lightrag.parser.external._base import ExternalParserBase

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc


class DotsParser(ExternalParserBase):
    """Parser engine that calls dots.ocr in-process for layout analysis.

    Produces the same ``content_list.json + images/`` bundle format as MinerU
    so :class:`MinerUIRBuilder` is reused for IR construction unchanged.
    """

    engine_name = PARSER_ENGINE_DOTS
    raw_dir_suffix = DOTS_RAW_DIR_SUFFIX
    force_reparse_env = "LIGHTRAG_FORCE_REPARSE_DOTS"

    def is_bundle_valid(
        self,
        raw_dir: Path,
        source_path: Path,
        *,
        engine_params: "Mapping[str, Any] | None" = None,
    ) -> bool:
        from lightrag.parser.external.dots.cache import is_bundle_valid

        return is_bundle_valid(raw_dir, source_path, overrides=engine_params)

    async def download_into(
        self,
        raw_dir: Path,
        source_path: Path,
        *,
        upload_name: str,
        engine_params: "Mapping[str, Any] | None" = None,
    ) -> None:
        from lightrag.parser.external.dots.processor import DotsRawProcessor

        await DotsRawProcessor(overrides=engine_params).process(
            raw_dir, source_path, upload_name=upload_name
        )

    def build_ir(self, raw_dir: Path, document_name: str) -> "IRDoc":
        from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder

        return MinerUIRBuilder().normalize_from_workdir(
            raw_dir, document_name=document_name
        )
