"""dots.ocr parser engine — in-process adapter for the dots.ocr layout model.

Supports PDF, PPTX, DOCX (via LibreOffice → PDF conversion) and standalone
images. Produces the same ``content_list.json + images/`` bundle format as
MinerU so :class:`MinerUIRBuilder` can be reused for IR construction.

Key env vars:
    DOTS_API_URL     : dots.ocr chat/completions endpoint (required)
    DOTS_API_TOKEN   : Bearer token (required)
    DOTS_MODEL       : model name (default: dotsocr-model)
    DOTS_RENDER_DPI  : page render resolution (default: 200)
    DOTS_TIMEOUT     : per-page request timeout in seconds (default: 180)
    DOTS_MAX_WORKERS : unused by integrated engine (kept for compat)
    LIBREOFFICE_PATH : soffice binary (default: soffice)
"""

from lightrag.parser.external.dots.cache import DotsParserOptions, is_bundle_valid
from lightrag.parser.external.dots.processor import DotsRawProcessor

__all__ = ["DotsParserOptions", "DotsRawProcessor", "is_bundle_valid"]
