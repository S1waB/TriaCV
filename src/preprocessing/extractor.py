"""Multi-format resume document text extractor supporting PDF, DOCX, and TXT files."""
import io
import os
from typing import BinaryIO, Union

import pypdf
try:
    import docx
except ImportError:
    docx = None


class ResumeExtractor:
    """Extract raw text from PDF, DOCX, and TXT file uploads."""

    @staticmethod
    def extract_from_pdf(file_source: Union[str, BinaryIO, bytes]) -> str:
        """Extract text content from a PDF file path or stream."""
        text_parts = []
        try:
            if isinstance(file_source, (str, os.PathLike)):
                with open(file_source, "rb") as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
            elif isinstance(file_source, bytes):
                stream = io.BytesIO(file_source)
                reader = pypdf.PdfReader(stream)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            else:
                reader = pypdf.PdfReader(file_source)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}") from e

        return "\n".join(text_parts).strip()

    @staticmethod
    def extract_from_docx(file_source: Union[str, BinaryIO, bytes]) -> str:
        """Extract text content from a DOCX document."""
        if docx is None:
            raise ImportError("python-docx is required for DOCX extraction.")

        text_parts = []
        try:
            if isinstance(file_source, (str, os.PathLike)):
                doc = docx.Document(file_source)
            elif isinstance(file_source, bytes):
                doc = docx.Document(io.BytesIO(file_source))
            else:
                doc = docx.Document(file_source)

            for para in doc.paragraphs:
                if para.text:
                    text_parts.append(para.text)

            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        text_parts.append(row_text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {str(e)}") from e

        return "\n".join(text_parts).strip()

    @staticmethod
    def extract_from_txt(file_source: Union[str, BinaryIO, bytes]) -> str:
        """Extract text from plain text file or bytes with UTF-8/Latin fallback."""
        if isinstance(file_source, (str, os.PathLike)) and os.path.exists(str(file_source)):
            with open(file_source, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        elif isinstance(file_source, bytes):
            try:
                return file_source.decode("utf-8")
            except UnicodeDecodeError:
                return file_source.decode("latin-1", errors="ignore")
        elif hasattr(file_source, "read"):
            content = file_source.read()
            if isinstance(content, bytes):
                try:
                    return content.decode("utf-8")
                except UnicodeDecodeError:
                    return content.decode("latin-1", errors="ignore")
            return str(content)
        return str(file_source)

    @classmethod
    def extract(cls, file_source: Union[str, BinaryIO, bytes], filename: str = "") -> str:
        """Auto-detect format by extension or header and extract text."""
        fname = filename.lower()
        if isinstance(file_source, str) and not fname and os.path.exists(file_source):
            fname = file_source.lower()

        if fname.endswith(".pdf"):
            return cls.extract_from_pdf(file_source)
        elif fname.endswith(".docx") or fname.endswith(".doc"):
            return cls.extract_from_docx(file_source)
        elif fname.endswith(".txt") or fname.endswith(".rtf") or fname.endswith(".md"):
            return cls.extract_from_txt(file_source)
        else:
            # Fallback attempt
            try:
                return cls.extract_from_pdf(file_source)
            except Exception:
                try:
                    return cls.extract_from_docx(file_source)
                except Exception:
                    return cls.extract_from_txt(file_source)
