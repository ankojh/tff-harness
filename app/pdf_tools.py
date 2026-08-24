from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError


MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PAGES_PER_CALL = 20
DEFAULT_TEXT_CHARS = 20_000
MAX_TEXT_CHARS = 50_000
MAX_METADATA_FIELDS = 20
MAX_METADATA_VALUE_CHARS = 500


class PdfToolError(Exception):
    """A PDF tool request failed validation or extraction."""


class PdfTools:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_pdf",
                    "description": (
                        "Extract text and metadata from a PDF in the model workspace. "
                        "Read selected pages for large documents."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Workspace-relative PDF path.",
                            },
                            "pages": {
                                "type": "array",
                                "description": (
                                    "Optional 1-based page numbers. At most 20 pages "
                                    "per call. Defaults to the first 20 pages."
                                ),
                                "items": {"type": "integer", "minimum": 1},
                                "minItems": 1,
                                "maxItems": MAX_PAGES_PER_CALL,
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Maximum extracted text characters.",
                                "minimum": 1000,
                                "maximum": MAX_TEXT_CHARS,
                                "default": DEFAULT_TEXT_CHARS,
                            },
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    @property
    def names(self) -> set[str]:
        return {"read_pdf"}

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        return await asyncio.to_thread(self._execute_sync, name, arguments)

    def _execute_sync(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name != "read_pdf":
                raise PdfToolError(f"Unknown PDF tool: {name}")
            result = self.read_pdf(
                self._string(arguments, "path"),
                self._pages(arguments),
                self._integer(
                    arguments,
                    "max_chars",
                    DEFAULT_TEXT_CHARS,
                    minimum=1000,
                    maximum=MAX_TEXT_CHARS,
                ),
            )
            return json.dumps({"ok": True, **result}, ensure_ascii=False)
        except PdfToolError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        except (OSError, PdfReadError) as exc:
            return json.dumps(
                {"ok": False, "error": f"Could not read PDF: {exc}"},
                ensure_ascii=False,
            )

    def read_pdf(
        self,
        path: str,
        pages: list[int] | None = None,
        max_chars: int = DEFAULT_TEXT_CHARS,
    ) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise PdfToolError(f"PDF does not exist: {path}")
        size = target.stat().st_size
        if size > MAX_PDF_BYTES:
            raise PdfToolError(
                f"PDF is {size} bytes; the file limit is {MAX_PDF_BYTES} bytes."
            )

        with target.open("rb") as stream:
            reader = PdfReader(stream, strict=False)
            if reader.is_encrypted:
                raise PdfToolError("Encrypted PDFs are not supported by this tool.")
            page_count = len(reader.pages)
            selected_pages = pages or list(
                range(1, min(page_count, MAX_PAGES_PER_CALL) + 1)
            )
            invalid_pages = [page for page in selected_pages if page > page_count]
            if invalid_pages:
                raise PdfToolError(
                    f"Requested page exceeds the {page_count}-page document: "
                    f"{invalid_pages[0]}"
                )

            extracted_pages: list[dict[str, Any]] = []
            warnings: list[str] = []
            total_chars = 0
            text_truncated = False
            for page_number in selected_pages:
                if total_chars >= max_chars:
                    text_truncated = True
                    break
                try:
                    text = reader.pages[page_number - 1].extract_text() or ""
                except Exception as exc:
                    warnings.append(
                        f"Page {page_number} text extraction failed: {type(exc).__name__}."
                    )
                    text = ""
                remaining = max_chars - total_chars
                returned_text = text[:remaining]
                if len(text) > len(returned_text):
                    text_truncated = True
                if not text.strip():
                    warnings.append(
                        f"Page {page_number} has no extractable text; it may be scanned or image-only."
                    )
                extracted_pages.append(
                    {
                        "page": page_number,
                        "text": returned_text,
                        "chars": len(returned_text),
                    }
                )
                total_chars += len(returned_text)
                if text_truncated:
                    break

            metadata = self._metadata(reader)

        pages_truncated = pages is None and page_count > len(selected_pages)
        return {
            "path": path,
            "bytes": size,
            "page_count": page_count,
            "pages": extracted_pages,
            "metadata": metadata,
            "warnings": warnings,
            "text_chars": total_chars,
            "text_truncated": text_truncated,
            "pages_truncated": pages_truncated,
        }

    @staticmethod
    def display_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        path = arguments.get("path")
        pages = arguments.get("pages")
        display: dict[str, Any] = {"path": path if isinstance(path, str) else ""}
        if isinstance(pages, list):
            display["pages"] = pages
        return display

    @staticmethod
    def _metadata(reader: PdfReader) -> dict[str, str]:
        raw_metadata = reader.metadata
        if not raw_metadata:
            return {}
        metadata: dict[str, str] = {}
        for raw_key, raw_value in list(raw_metadata.items())[:MAX_METADATA_FIELDS]:
            if raw_value is None:
                continue
            key = str(raw_key).lstrip("/")[:100]
            metadata[key] = str(raw_value)[:MAX_METADATA_VALUE_CHARS]
        return metadata

    def _resolve(self, raw_path: str) -> Path:
        path = PurePosixPath(raw_path)
        if path.is_absolute() or not path.parts:
            raise PdfToolError("Paths must be relative to the model workspace.")
        if any(part in {"", ".", ".."} for part in path.parts):
            raise PdfToolError("Path traversal is not allowed.")
        self.root.mkdir(parents=True, exist_ok=True)
        candidate = (self.root / Path(*path.parts)).resolve()
        try:
            inside_root = os.path.commonpath([self.root, candidate]) == str(self.root)
        except ValueError:
            inside_root = False
        if not inside_root:
            raise PdfToolError("Path escapes the model workspace.")
        return candidate

    @staticmethod
    def _string(arguments: dict[str, Any], key: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value:
            raise PdfToolError(f"{key} must be a non-empty string.")
        return value

    @staticmethod
    def _pages(arguments: dict[str, Any]) -> list[int] | None:
        if "pages" not in arguments:
            return None
        value = arguments["pages"]
        if not isinstance(value, list) or not value:
            raise PdfToolError("pages must be a non-empty array.")
        if len(value) > MAX_PAGES_PER_CALL:
            raise PdfToolError(
                f"pages may contain at most {MAX_PAGES_PER_CALL} page numbers."
            )
        pages: list[int] = []
        for page in value:
            if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                raise PdfToolError("pages must contain positive 1-based integers.")
            if page not in pages:
                pages.append(page)
        return pages

    @staticmethod
    def _integer(
        arguments: dict[str, Any],
        key: str,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        value = arguments.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise PdfToolError(f"{key} must be an integer.")
        if not minimum <= value <= maximum:
            raise PdfToolError(f"{key} must be between {minimum} and {maximum}.")
        return value
