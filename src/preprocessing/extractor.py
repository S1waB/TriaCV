"""Multi-format resume document text extractor.

Supports PDF (via pdfplumber), DOCX (python-docx), and TXT files.
"""
import io
import os
from typing import BinaryIO, Union

import pdfplumber

try:
    import docx
except ImportError:
    docx = None


class ResumeExtractor:
    """Extract raw text from PDF, DOCX, and TXT resume files."""

    @staticmethod
    def extract_from_pdf(file_source: Union[str, "os.PathLike[str]", BinaryIO, bytes]) -> str:
        """Extract text from a PDF using pdfplumber (layout-aware extraction).

        Args:
            file_source: File path, byte string, or file-like object.

        Returns:
            Concatenated text from all pages.
        """
        text_parts = []
        try:
            if isinstance(file_source, (str, os.PathLike)):
                with pdfplumber.open(file_source) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
            elif isinstance(file_source, bytes):
                with pdfplumber.open(io.BytesIO(file_source)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
            else:
                with pdfplumber.open(file_source) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
        except Exception as exc:
            raise ValueError(f"Failed to extract text from PDF: {exc}") from exc

        return "\n".join(text_parts).strip()

    @staticmethod
    def extract_from_docx(file_source: Union[str, "os.PathLike[str]", BinaryIO, bytes]) -> str:
        """Extract text from a DOCX document including tables.

        Args:
            file_source: File path, byte string, or file-like object.

        Returns:
            Full plain text of the document.
        """
        if docx is None:
            raise ImportError("python-docx is required for DOCX extraction. Install with: pip install python-docx")

        text_parts = []
        try:
            if isinstance(file_source, (str, os.PathLike)):
                doc = docx.Document(file_source)
            elif isinstance(file_source, bytes):
                doc = docx.Document(io.BytesIO(file_source))
            else:
                doc = docx.Document(file_source)

            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        text_parts.append(row_text)
        except Exception as exc:
            raise ValueError(f"Failed to extract text from DOCX: {exc}") from exc

        return "\n".join(text_parts).strip()

    @staticmethod
    def extract_from_txt(file_source: Union[str, "os.PathLike[str]", BinaryIO, bytes]) -> str:
        """Extract text from a plain text file with UTF-8 / Latin-1 fallback.

        Args:
            file_source: File path, byte string, or file-like object.

        Returns:
            Plain text content.
        """
        if isinstance(file_source, (str, os.PathLike)) and os.path.exists(str(file_source)):
            with open(file_source, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read().strip()
        elif isinstance(file_source, bytes):
            try:
                return file_source.decode("utf-8").strip()
            except UnicodeDecodeError:
                return file_source.decode("latin-1", errors="ignore").strip()
        elif hasattr(file_source, "read"):
            content = file_source.read()
            if isinstance(content, bytes):
                try:
                    return content.decode("utf-8").strip()
                except UnicodeDecodeError:
                    return content.decode("latin-1", errors="ignore").strip()
            return str(content).strip()
        return str(file_source).strip()

    @classmethod
    def extract(
        cls,
        file_source: Union[str, "os.PathLike[str]", BinaryIO, bytes],
        filename: str = "",
    ) -> str:
        """Auto-detect format by extension and dispatch to the right extractor.

        Args:
            file_source: File path, byte string, or file-like object.
            filename: Optional filename hint to determine format.

        Returns:
            Extracted plain text.
        """
        fname = filename.lower()
        if isinstance(file_source, (str, os.PathLike)) and not fname:
            fname = str(file_source).lower()

        if fname.endswith(".pdf"):
            return cls.extract_from_pdf(file_source)
        elif fname.endswith((".docx", ".doc")):
            return cls.extract_from_docx(file_source)
        elif fname.endswith((".txt", ".rtf", ".md")):
            return cls.extract_from_txt(file_source)
        else:
            # Fallback: try each format in order
            for method in (cls.extract_from_pdf, cls.extract_from_docx, cls.extract_from_txt):
                try:
                    result = method(file_source)
                    if result:
                        return result
                except Exception:
                    continue
            return ""


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <path_to_resume>")
        sys.exit(1)

    path = sys.argv[1]
    text = ResumeExtractor.extract(path)
    print(f"Extracted {len(text)} characters from '{path}'")
    print("--- Preview (first 500 chars) ---")
    print(text[:500])
