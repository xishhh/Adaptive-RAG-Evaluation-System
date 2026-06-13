"""
Document loaders for the Adaptive RAG ingestion pipeline.

Responsibilities:
  - Accept a file path for any supported document type.
  - Extract full text content.
  - Extract available metadata (page count, sheet names, author, etc.).
  - Return a RawDocument ready for the chunker.

Supported formats:
  - PDF   → pdfplumber  (handles text-heavy and mixed-layout PDFs)
  - DOCX  → python-docx (preserves paragraph structure)
  - TXT   → built-in    (UTF-8 with Latin-1 fallback)
  - XLSX  → openpyxl    (row-level text extraction per sheet)

This module does NOT produce embeddings or interact with ChromaDB.
Those responsibilities belong to Phase 3.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pdfplumber
from docx import Document as DocxDocument
from openpyxl import load_workbook

from app.models.responses import RawDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Supported extensions mapped to their internal handler names.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({"pdf", "docx", "txt", "xlsx"})


class UnsupportedFileTypeError(Exception):
    """Raised when a file with an unsupported extension is submitted."""


class DocumentLoader:
    """
    Dispatches file loading to the appropriate format-specific method.

    Usage:
        loader = DocumentLoader()
        raw_doc = loader.load("/path/to/file.pdf")
    """

    def load(self, file_path: str | Path) -> RawDocument:
        """
        Load a document from disk and return a RawDocument.

        Args:
            file_path: Absolute or relative path to the document.

        Returns:
            RawDocument with extracted text and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedFileTypeError: If the extension is not supported.
            ValueError: If text extraction yields no content.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        extension = path.suffix.lstrip(".").lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '.{extension}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        logger.info("Loading document: %s (type=%s)", path.name, extension)

        dispatch: dict[str, Any] = {
            "pdf": self._load_pdf,
            "docx": self._load_docx,
            "txt": self._load_txt,
            "xlsx": self._load_xlsx,
        }

        raw_doc: RawDocument = dispatch[extension](path)

        if not raw_doc.full_text.strip():
            raise ValueError(
                f"No text could be extracted from '{path.name}'. "
                "The file may be empty, image-only, or corrupted."
            )

        logger.info(
            "Loaded '%s' — %d characters extracted.",
            path.name,
            len(raw_doc.full_text),
        )
        return raw_doc

    # ------------------------------------------------------------------ #
    # PDF                                                                  #
    # ------------------------------------------------------------------ #

    def _load_pdf(self, path: Path) -> RawDocument:
        """
        Extract text from a PDF using pdfplumber.

        pdfplumber is preferred over PyPDF2/pypdf because it handles
        complex layouts (multi-column, tables) more reliably and exposes
        clean per-page text without needing heuristic post-processing.

        Each page's text is joined with a newline. Page boundaries are
        preserved in the metadata so the chunker can annotate chunks
        with approximate page numbers.
        """
        pages_text: list[str] = []
        page_char_offsets: list[int] = []  # cumulative char offset per page
        cumulative = 0

        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
                page_char_offsets.append(cumulative)
                cumulative += len(text) + 1  # +1 for the joining newline

        full_text = "\n".join(pages_text)

        metadata: dict[str, Any] = {
            "total_pages": total_pages,
            # page_char_offsets maps each page index → start char position
            # in full_text. The chunker uses this to assign page numbers.
            "page_char_offsets": page_char_offsets,
        }

        return RawDocument(
            document_name=path.name,
            file_type="pdf",
            full_text=full_text,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # DOCX                                                                 #
    # ------------------------------------------------------------------ #

    def _load_docx(self, path: Path) -> RawDocument:
        """
        Extract text from a Word document using python-docx.

        Each paragraph is joined with a newline. Empty paragraphs
        (used as visual spacers in Word) are filtered to reduce noise.

        DOCX files do not have a reliable page-number concept at the
        paragraph level without rendering the document, so page_number
        will be None for all DOCX chunks.
        """
        doc = DocxDocument(str(path))

        paragraphs: list[str] = [
            para.text
            for para in doc.paragraphs
            if para.text.strip()
        ]

        # Also extract text from tables, which python-docx skips by default.
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    paragraphs.append(row_text)

        full_text = "\n".join(paragraphs)

        # Extract core document properties where available.
        props = doc.core_properties
        metadata: dict[str, Any] = {
            "author": props.author or "",
            "created": props.created.isoformat() if props.created else "",
            "modified": props.modified.isoformat() if props.modified else "",
            "title": props.title or "",
            "paragraph_count": len(paragraphs),
        }

        return RawDocument(
            document_name=path.name,
            file_type="docx",
            full_text=full_text,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # TXT                                                                  #
    # ------------------------------------------------------------------ #

    def _load_txt(self, path: Path) -> RawDocument:
        """
        Read a plain text file.

        Attempts UTF-8 first; falls back to Latin-1 to handle
        legacy documents without raising an encoding error.
        """
        try:
            full_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "'%s' is not valid UTF-8; falling back to Latin-1.", path.name
            )
            full_text = path.read_text(encoding="latin-1")

        metadata: dict[str, Any] = {
            "size_bytes": os.path.getsize(path),
            "line_count": full_text.count("\n"),
        }

        return RawDocument(
            document_name=path.name,
            file_type="txt",
            full_text=full_text,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # XLSX                                                                 #
    # ------------------------------------------------------------------ #

    def _load_xlsx(self, path: Path) -> RawDocument:
        """
        Extract text from an Excel workbook using openpyxl.

        Strategy:
          - Iterate over all sheets.
          - For each sheet, iterate over rows.
          - Convert each non-empty row to a pipe-delimited string.
          - Prefix each sheet's content with a header line.

        This produces structured, readable text that preserves the
        tabular nature of the data while remaining searchable as plain text.

        Excel does not have a page-number concept, so page_number will
        be None for all XLSX chunks.
        """
        wb = load_workbook(filename=str(path), read_only=True, data_only=True)
        sheet_texts: list[str] = []
        sheet_names: list[str] = wb.sheetnames

        for sheet_name in sheet_names:
            ws = wb[sheet_name]
            rows_text: list[str] = []

            for row in ws.iter_rows(values_only=True):
                # Skip entirely empty rows.
                non_empty = [str(cell) if cell is not None else "" for cell in row]
                if any(cell.strip() for cell in non_empty):
                    rows_text.append(" | ".join(non_empty))

            if rows_text:
                sheet_block = f"[Sheet: {sheet_name}]\n" + "\n".join(rows_text)
                sheet_texts.append(sheet_block)

        wb.close()

        full_text = "\n\n".join(sheet_texts)

        metadata: dict[str, Any] = {
            "sheet_names": sheet_names,
            "sheet_count": len(sheet_names),
        }

        return RawDocument(
            document_name=path.name,
            file_type="xlsx",
            full_text=full_text,
            metadata=metadata,
        )