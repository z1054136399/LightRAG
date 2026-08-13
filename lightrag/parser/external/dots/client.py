"""dots.ocr chat/completions client.

Synchronous HTTP wrapper around the OpenAI-compatible endpoint.
Call from async code via ``asyncio.to_thread``.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import TYPE_CHECKING

from lightrag.utils import logger

if TYPE_CHECKING:
    from PIL import Image as PILImage

LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - Output all the text without missing any words.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


class DotsClient:
    """Thin wrapper over the dots.ocr chat/completions endpoint."""

    def __init__(
        self,
        api_url: str,
        token: str,
        host_header: str = "",
        model: str = "dotsocr-model",
        timeout: int = 180,
    ) -> None:
        self.api_url = api_url
        self.token = token
        self.host_header = host_header
        self.model = model
        self.timeout = timeout

    def parse_image(self, image: "PILImage.Image") -> list[dict]:
        """Send one page image to dots.ocr. Returns layout cell list."""
        try:
            import requests
        except ImportError:
            raise RuntimeError(
                "requests is required for dots.ocr parsing but not installed"
            )

        b64 = _pil_to_b64(image)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": LAYOUT_PROMPT},
                    ],
                }
            ],
            "stream": False,
        }
        headers: dict[str, str] = {
            "Accept": "*/*",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        if self.host_header:
            headers["Host"] = self.host_header

        try:
            resp = requests.post(
                self.api_url, json=payload, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"dots.ocr request failed (url={self.api_url}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        data = resp.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        cells = _parse_cells(content)
        logger.debug("[dots] page parsed: %d layout cells", len(cells))
        return cells


def _pil_to_b64(image: "PILImage.Image") -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_cells(content: str) -> list[dict]:
    """Best-effort layout-cell extraction from model output."""
    if not content or not content.strip():
        return []
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        arr = re.search(r"\[.*\]", text, re.DOTALL)
        if arr:
            try:
                obj = json.loads(arr.group(0))
            except json.JSONDecodeError:
                obj = None
        else:
            obj = None
    if isinstance(obj, list):
        return [_coerce_cell(c) for c in obj if isinstance(c, dict)]
    if isinstance(obj, dict):
        for key in ("layout", "elements", "result", "cells", "data", "content"):
            val = obj.get(key)
            if isinstance(val, list):
                return [_coerce_cell(c) for c in val if isinstance(c, dict)]
        return [_coerce_cell(obj)]
    # JSON parsing failed (e.g. unescaped quotes in table HTML).  Try regex-based
    # extraction before falling back to a raw text block.
    recovered = _extract_cells_via_regex(text)
    if recovered:
        logger.warning(
            "[dots] JSON parse failed; recovered %d cells via regex fallback",
            len(recovered),
        )
        return recovered
    logger.warning("[dots] could not parse layout JSON; emitting raw text block")
    return [{"category": "Text", "text": content.strip()}]


def _extract_cells_via_regex(text: str) -> list[dict]:
    """Best-effort cell extraction when JSON parsing fails due to unescaped quotes.

    Handles the common case where a model returns valid layout cells but embeds
    raw HTML in table ``text`` fields that contains unescaped ``"`` characters.
    Extracts Caption/text cells (short strings, usually safe) and Table cells
    (detected by ``<table>…</table>`` boundaries) via regex instead of JSON.
    """
    cells: list[dict] = []

    # Extract simple string categories: Caption, Title, Section-header, Text, etc.
    # These values are typically short and do not contain embedded quotes.
    simple_re = re.compile(
        r'"category"\s*:\s*"(Caption|Title|Section-header|Text|List-item'
        r'|Footnote|Page-header|Page-footer)"\s*,\s*"text"\s*:\s*"([^"]*)"',
    )
    for m in simple_re.finditer(text):
        cells.append({"category": m.group(1), "text": m.group(2)})

    # Extract Table cells: look for "category":"Table","text":"<table>...</table>"
    # The HTML content may contain unescaped " so we anchor on </table>.
    table_re = re.compile(
        r'"category"\s*:\s*"Table"[^}]{0,200}?"text"\s*:\s*"(<table>.*?</table>)',
        re.DOTALL,
    )
    for m in table_re.finditer(text):
        html = m.group(1).replace('\\"', '"')
        cells.append({"category": "Table", "text": html})

    return cells


def _coerce_cell(cell: dict) -> dict:
    out = dict(cell)
    bbox = out.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            out["bbox"] = [int(round(float(v))) for v in bbox[:4]]
        except (TypeError, ValueError):
            out.pop("bbox", None)
    else:
        out.pop("bbox", None)
    out["category"] = str(out.get("category") or "Text").strip()
    out["text"] = str(out.get("text") or "").strip()
    return out


__all__ = ["DotsClient", "LAYOUT_PROMPT"]
