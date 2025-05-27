# ingest_documents.py
import argparse
import asyncio  # Import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Add app directory to sys.path to allow imports from app.*
project_root = Path(__file__).parent.parent.resolve()  # Go up one level from scripts/
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv()

from app.core.config import get_settings  # To potentially get dir paths if needed
from app.core.llm_service import LLMService  # Import LLMService for summarization
from app.services.document_service import CX_GLOBAL_COLLECTION, DocumentService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ingest_global_knowledge")

# --- Configuration ---
# Path to the global knowledge documents.
# IMPORTANT: Adjust this path if your global_docs are located elsewhere.
# Provided path: /Users/az/CX Consulting Agent/data/global_docs/GlobalKnowledge
# We'll make it configurable via an environment variable or a default.
DEFAULT_GLOBAL_DOCS_PATH = (
    "/Users/az/CX Consulting Agent/data/global_docs/GlobalKnowledge"
)
GLOBAL_DOCS_DIR = os.getenv("GLOBAL_DOCS_PATH", DEFAULT_GLOBAL_DOCS_PATH)

# Configuration for DocumentService instantiation
# These could also come from environment variables or a shared config
# For simplicity, using paths relative to project_root or defaults from settings
APP_SETTINGS = get_settings()
DOC_SERVICE_CONFIG = {
    "documents_dir": os.getenv(
        "DOC_SERVICE_INPUT_DIR", str(project_root / "data" / "temp_processing_docs")
    ),
    "chunked_dir": os.getenv(
        "DOC_SERVICE_CHUNKED_DIR", str(project_root / "data" / "temp_chunked_output")
    ),  # May not be used if not saving chunks
    "vectorstore_dir": os.getenv(
        "VECTOR_DB_PATH",
        APP_SETTINGS.VECTOR_DB_PATH or str(project_root / "data" / "vector_db"),
    ),
    "embedding_model": os.getenv("EMBEDDING_MODEL", APP_SETTINGS.EMBEDDING_MODEL),
    "chunk_size": int(
        os.getenv("MAX_CHUNK_SIZE", str(APP_SETTINGS.MAX_CHUNK_SIZE or 512))
    ),
    "chunk_overlap": int(
        os.getenv("CHUNK_OVERLAP", str(APP_SETTINGS.CHUNK_OVERLAP or 50))
    ),
    "default_collection_name": "global_kb",  # This script specifically targets the global_kb
}

# Target collection for global knowledge
GLOBAL_COLLECTION_NAME = "global_kb"

# Summarization Prompt
SUMMARIZE_PROMPT_TEMPLATE = "Provide a concise 2-sentence summary of the following document excerpt, capturing its main topic and purpose:\n\n---\n{text_excerpt}\n---\n\nSummary (2 sentences):"
MAX_SUMMARY_TOKENS = 100  # Limit LLM output for summary
EXCERPT_LENGTH_FOR_SUMMARY = 4000  # Use first N chars for summary generation


# --- Helper Functions ---
def generate_summary(llm_service: LLMService, text: str) -> str:
    """Generates a 2-sentence summary using the LLMService."""
    if not text:
        return ""
    try:
        # Take the first N characters for efficiency
        excerpt = text[:EXCERPT_LENGTH_FOR_SUMMARY].strip()
        if not excerpt:
            return ""

        prompt = SUMMARIZE_PROMPT_TEMPLATE.format(text_excerpt=excerpt)
        summary = llm_service.generate_sync(
            prompt=prompt,
            max_tokens=MAX_SUMMARY_TOKENS,
            temperature=0.2,  # Lower temp for factual summary
        )
        # Basic cleaning of the summary
        summary = summary.strip().replace("\n", " ")
        # Ensure it's roughly 2 sentences (heuristic)
        sentences = summary.split(".")
        summary = ". ".join(s.strip() for s in sentences[:2] if s.strip()) + "."
        logger.info(f"Generated summary: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}", exc_info=True)
        return "Summary generation failed."


def get_document_type(file_path: Path) -> Optional[str]:
    """Determines the document type from file extension."""
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return "pdf"
    elif extension == ".docx":
        return "docx"
    elif (
        extension == ".doc"
    ):  # docx2txt might handle .doc, but often better to convert first
        return "doc"
    elif extension == ".txt":
        return "txt"
    elif extension == ".csv":
        return "csv"
    elif extension == ".xlsx":
        return "xlsx"
    # Add more types as needed (e.g., .md, .pptx if supported by DocumentService)
    logger.warning(f"Unsupported file extension: {extension} for file {file_path.name}")
    return None


# --- Main Ingestion Logic ---
async def ingest_global_documents(docs_path: str, collection_name: str):
    """Finds, processes, and ingests documents from a directory into a global collection."""
    settings = get_settings()  # Ensure settings are loaded if needed for service init
    docs_path_obj = Path(docs_path)
    if not docs_path_obj.is_dir():
        logger.error(f"Provided path is not a valid directory: {docs_path}")
        return

    logger.info(
        f"Starting ingestion from path: {docs_path} into collection: {collection_name}"
    )

    # Initialize DocumentService (adjust path settings as needed)
    # This assumes DocumentService can be initialized synchronously here
    # and its async methods are called below.
    try:
        doc_service = DocumentService(
            documents_dir=str(settings.DOCUMENTS_DIR),
            chunked_dir=str(settings.CHUNKED_DIR),
            vectorstore_dir=str(settings.VECTOR_DB_PATH),
            embedding_model=(
                settings.BGE_MODEL_NAME
                if settings.EMBEDDING_TYPE.lower() == "bge"
                else settings.EMBEDDING_MODEL
            ),
            default_collection_name=collection_name,  # Use specified collection as default for this run?
        )
        # Initialize async parts if needed (e.g., vector store connection)
        await doc_service._init_vector_store()
        # Ensure the target global collection is set
        await doc_service.set_collection(collection_name)

    except Exception as e:
        logger.error(f"Failed to initialize DocumentService: {e}", exc_info=True)
        return

    processed_files = 0
    failed_files = 0
    skipped_files = 0
    supported_extensions = {".pdf", ".txt", ".docx", ".doc", ".csv", ".xlsx"}

    for file_path in docs_path_obj.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            logger.info(f"Processing file: {file_path.name}")
            doc_type = file_path.suffix.lower().strip(".")
            doc_metadata = {
                "filename": file_path.name,
                "full_path": str(file_path.resolve()),
                # Add other relevant metadata if known
            }
            try:
                # Call the ASYNC add_document method directly
                success = await doc_service.add_document(
                    document_url=str(
                        file_path.resolve()
                    ),  # Use local path as URL for simplicity
                    document_type=doc_type,
                    metadata=doc_metadata,
                    is_global=True,
                    project_id=None,  # Global docs don't belong to a project
                    collection_name=collection_name,  # Explicitly target the global collection
                )
                if success:
                    processed_files += 1
                    logger.info(
                        f"Successfully processed and ingested: {file_path.name}"
                    )  # Log success per file
                else:
                    failed_files += 1
                    logger.error(f"Failed to process or ingest file: {file_path.name}")
            except Exception as e:
                logger.error(
                    f"Error processing file {file_path.name}: {e}", exc_info=True
                )
                failed_files += 1
        elif file_path.is_file():
            logger.warning(f"Skipping unsupported file type: {file_path.name}")
            skipped_files += 1

    logger.info(
        f"Ingestion complete. Processed: {processed_files}, Failed: {failed_files}, Skipped: {skipped_files}"
    )


if __name__ == "__main__":
    # ... (argparse logic remains the same) ...
    parser = argparse.ArgumentParser(description="Ingest global documents.")
    parser.add_argument(
        "--path", required=True, help="Directory path containing global documents."
    )
    parser.add_argument(
        "--collection",
        default=CX_GLOBAL_COLLECTION,
        help=f"Target collection name (default: {CX_GLOBAL_COLLECTION}).",
    )
    # The --is-global flag might be redundant if we always set is_global=True here,
    # but keep it if the script might be repurposed later.
    parser.add_argument(
        "--is-global",
        action="store_true",
        help="Flag documents as global (sets metadata).",
    )

    args = parser.parse_args()

    # Run the async function
    asyncio.run(ingest_global_documents(args.path, args.collection))
