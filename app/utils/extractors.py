import logging
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def extract_text_from_file(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Extracts text and basic metadata from a given file path."""
    suffix = path.suffix.lower()
    extra_meta: Dict[str, Any] = {
        "original_filename": path.name,
        "file_extension": suffix,
    }
    text_content = ""

    try:
        if suffix == ".pdf":
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(path)
                text_content = "\n".join(page.get_text() for page in doc)
                extra_meta["page_count"] = doc.page_count
                extra_meta["document_type"] = "pdf"
                logger.info(
                    f"Successfully extracted text from PDF: {path.name} ({doc.page_count} pages)"
                )
            except ImportError:
                logger.error(
                    "PyMuPDF (fitz) is not installed. Please install it to process PDFs."
                )
                raise
            except Exception as e:
                logger.error(
                    f"Error processing PDF {path.name} with PyMuPDF: {e}", exc_info=True
                )
                # Attempt fallback with pdfplumber if PyMuPDF fails for some reason
                try:
                    logger.info(
                        f"Attempting fallback extraction for PDF {path.name} with pdfplumber."
                    )
                    import pdfplumber

                    with pdfplumber.open(path) as pdf:
                        text_content = "\n".join(
                            page.extract_text() or "" for page in pdf.pages
                        )
                        extra_meta["page_count"] = len(pdf.pages)
                        extra_meta["document_type"] = "pdf"
                    logger.info(
                        f"Successfully extracted text from PDF {path.name} using pdfplumber fallback."
                    )
                except ImportError:
                    logger.error(
                        "pdfplumber is not installed. Cannot use as fallback for PDF processing."
                    )
                    raise ValueError(
                        f"PDF processing failed for {path.name}. PyMuPDF and pdfplumber are unavailable or failed."
                    ) from e
                except Exception as e_fallback:
                    logger.error(
                        f"Fallback PDF extraction with pdfplumber also failed for {path.name}: {e_fallback}",
                        exc_info=True,
                    )
                    raise ValueError(
                        f"All PDF extraction attempts failed for {path.name}."
                    ) from e_fallback

        elif suffix in {".docx", ".doc"}:
            try:
                from docx import Document

                doc = Document(path)
                text_content = "\n".join(p.text for p in doc.paragraphs if p.text)
                # You could add more detailed metadata like table extraction here if needed
                extra_meta["document_type"] = "docx"
                logger.info(f"Successfully extracted text from DOCX: {path.name}")
            except ImportError:
                logger.error(
                    "python-docx is not installed. Please install it to process DOCX files."
                )
                raise
            except Exception as e:
                logger.error(f"Error processing DOCX {path.name}: {e}", exc_info=True)
                raise

        elif suffix in {".xls", ".xlsx"}:
            try:
                import pandas as pd

                # Read all sheets into a dictionary of DataFrames
                excel_file = pd.ExcelFile(path)
                text_parts = []
                sheet_names = excel_file.sheet_names
                for sheet_name in sheet_names:
                    df = excel_file.parse(sheet_name)
                    if not df.empty:
                        text_parts.append(f"### Sheet: {sheet_name}\n")
                        # Convert dataframe to markdown, try to preserve as much info as possible
                        # Using tabulate for a cleaner markdown table might be an option if installed
                        try:
                            md_table = df.to_markdown(index=False, na_rep="")
                        except TypeError:  # Older pandas might not have na_rep
                            md_table = df.to_markdown(index=False)
                        text_parts.append(md_table)
                text_content = "\n\n".join(text_parts)
                extra_meta["sheet_count"] = len(sheet_names)
                extra_meta["sheet_names"] = sheet_names
                extra_meta["document_type"] = "excel"
                logger.info(
                    f"Successfully extracted text from Excel: {path.name} ({len(sheet_names)} sheets)"
                )
            except ImportError:
                logger.error(
                    "pandas and openpyxl/xlrd are not installed. Please install them to process Excel files."
                )
                raise
            except Exception as e:
                logger.error(
                    f"Error processing Excel file {path.name}: {e}", exc_info=True
                )
                raise

        elif suffix == ".txt":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text_content = f.read()
                extra_meta["document_type"] = "text"
                logger.info(f"Successfully extracted text from TXT: {path.name}")
            except Exception as e:
                logger.error(f"Error reading TXT file {path.name}: {e}", exc_info=True)
                raise

        elif suffix == ".md":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text_content = f.read()
                extra_meta["document_type"] = "markdown"
                logger.info(f"Successfully extracted text from Markdown: {path.name}")
            except Exception as e:
                logger.error(
                    f"Error reading Markdown file {path.name}: {e}", exc_info=True
                )
                raise

        # Add more extractors here as needed (e.g., for .pptx, .json, .csv)
        elif suffix == ".csv":
            try:
                import csv

                text_parts = []
                with open(path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    for row_idx, row in enumerate(reader):
                        # Simple representation: join cells, add header if it's the first row and looks like one
                        # A more sophisticated approach might try to convert to markdown table
                        text_parts.append(", ".join(filter(None, row)))
                text_content = "\n".join(text_parts)
                extra_meta["document_type"] = "csv"
                logger.info(f"Successfully extracted text from CSV: {path.name}")
            except ImportError:
                logger.error("csv module (standard library) somehow unavailable.")
                raise
            except Exception as e:
                logger.error(
                    f"Error processing CSV file {path.name}: {e}", exc_info=True
                )
                raise

        else:
            logger.warning(
                f"Unsupported file format for text extraction: {path.suffix} for file {path.name}. Skipping text extraction."
            )
            # extra_meta is already populated with filename and extension
            # No text_content is set, will return empty string
            # Or, raise ValueError to indicate no extractor found:
            # raise ValueError(f"No extractor available for file type: {path.suffix}")

    except Exception as e_outer:
        logger.error(
            f"Critical error during text extraction for {path.name}: {e_outer}",
            exc_info=True,
        )
        # Ensure text_content is empty and provide a generic error in metadata perhaps
        text_content = ""
        extra_meta["extraction_error"] = str(e_outer)
        # Depending on policy, you might want to re-raise or just return empty text with error metadata

    if not text_content.strip() and "extraction_error" not in extra_meta:
        logger.warning(
            f"Extraction yielded no text for {path.name} (type: {extra_meta.get('document_type', 'unknown')}) but no explicit error was caught during extraction."
        )
        extra_meta["extraction_warning"] = "Extractor yielded no text content."

    return text_content, extra_meta
