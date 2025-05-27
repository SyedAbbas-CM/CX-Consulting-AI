#!/usr/bin/env python
"""
Global Knowledge Processing Script (v2)

This version adds:
• Environment‑driven paths (VECTOR_DIR, CHUNK_DIR, DATA_DIR)
• SHA‑256 duplicate detection (stored as metadata & checked before re‑ingest)
• Thread‑pooled parallel ingestion (CPU‑bound PDF parsing stays fast)
• Graceful fallback between RagEngine and DocumentService
• OCR fallback hook (placeholder) for scanned PDFs
• Structured logging & timing metrics

Dependencies (example):
    pip install langchain pdfminer.six unstructured sentence-transformers chromadb python-dotenv

Directory layout (override with env vars):
    ${DATA_DIR}/documents/          raw PDFs
    ${CHUNK_DIR}/                   JSON chunks (optional)
    ${VECTOR_DIR}/                  Chroma DB

Run:
    python global_knowledge_processor.py --workers 4 --ext pdf
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Set

# --- project imports ---------------------------------------------------------
# Add app directory to path (two levels up)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from app.services.document_service import DocumentService  # noqa: E402
from app.services.rag_engine import RagEngine  # noqa: E402

# --- env & logging -----------------------------------------------------------
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "app/data"))
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", DATA_DIR / "documents"))
CHUNK_DIR = Path(os.getenv("CHUNK_DIR", DATA_DIR / "chunked"))
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", DATA_DIR / "vectorstore"))
GLOBAL_KNOWLEDGE_DIR = Path(
    os.getenv("GLOBAL_KNOWLEDGE_DIR", PROJECT_ROOT / "app/GlobalKnowledge")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("global_knowledge_processor")

# -----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return SHA‑256 hexdigest for a file (streamed)."""
    hash_sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_sha.update(chunk)
    return hash_sha.hexdigest()


class KnowledgeIngestor:
    """Parallel ingestor that wraps DocumentService and RagEngine."""

    def __init__(self, workers: int = 4, file_ext: str = "pdf") -> None:
        self.workers = workers
        self.file_ext = file_ext
        self.doc_service = DocumentService(
            documents_dir=str(DOCUMENTS_DIR),
            chunked_dir=str(CHUNK_DIR),
            vectorstore_dir=str(VECTOR_DIR),
        )
        try:
            self.rag_engine = RagEngine()
            self.use_rag_engine = True
            logger.info("RagEngine initialised ✅")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not init RagEngine: %s — falling back to DocumentService", exc
            )
            self.use_rag_engine = False

        # Get existing document hashes from chunk files
        self._existing_hashes: Set[str] = self._get_existing_hashes()
        logger.info(f"Found {len(self._existing_hashes)} existing document hashes")

    def _get_existing_hashes(self) -> Set[str]:
        """Get hashes of all existing documents from chunk files."""
        existing_hashes = set()
        chunk_dir = Path(CHUNK_DIR)

        if not chunk_dir.exists():
            logger.warning(f"Chunk directory {chunk_dir} does not exist")
            return existing_hashes

        for chunk_file in chunk_dir.glob("*.json"):
            try:
                with open(chunk_file, "r") as f:
                    chunk_data = json.load(f)
                    if "metadata" in chunk_data and "hash" in chunk_data["metadata"]:
                        existing_hashes.add(chunk_data["metadata"]["hash"])
            except Exception as e:
                logger.warning(f"Error reading chunk file {chunk_file}: {e}")
                continue

        return existing_hashes

    # ---------------------------------------------------------------------
    def ingest_one(self, path: Path) -> None:
        file_hash = sha256_file(path)
        if file_hash in self._existing_hashes:
            logger.info("⏭  %s (duplicate, hash match)", path.name)
            return
        metadata = {
            "source": path.name,
            "hash": file_hash,
            "global_knowledge": True,
        }
        try:
            if self.use_rag_engine:
                with path.open("rb") as fh:
                    content = fh.read()
                res = self.rag_engine.process_document_sync(content, path.name, metadata=metadata)  # type: ignore[attr-defined]
                if res.get("status") == "success":
                    chunks = res.get("chunks_created", 0)
                    logger.info("✅ %s | %s chunks via RagEngine", path.name, chunks)
                else:
                    raise RuntimeError(res.get("error", "unknown error"))
            else:
                success = self.doc_service.add_document_sync(  # type: ignore[attr-defined]
                    document_url=str(path),
                    document_type="pdf",
                    metadata=metadata,
                    is_global=True,
                )
                if success:
                    logger.info("✅ %s | ingested via DocumentService", path.name)
                else:
                    raise RuntimeError("DocumentService returned False")
            # record hash to avoid future re‑processing
            self._existing_hashes.add(file_hash)
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ %s | %s", path.name, exc)

    # ---------------------------------------------------------------------
    def run(self) -> None:
        files: List[Path] = list(GLOBAL_KNOWLEDGE_DIR.glob(f"*.{self.file_ext}"))
        if not files:
            logger.warning(
                "No *.%s files found in %s", self.file_ext, GLOBAL_KNOWLEDGE_DIR
            )
            return
        logger.info("Found %d %s files", len(files), self.file_ext)
        start = time.time()
        with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(self.ingest_one, files))
        elapsed = time.time() - start
        logger.info(
            "Ingestion complete in %.2fs | Total docs in vector store: %d",
            elapsed,
            self.doc_service.get_document_count(),
        )


# -----------------------------------------------------------------------------


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest global knowledge PDFs into vector store"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "4")),
        help="Thread workers (default: 4)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=os.getenv("FILE_EXT", "pdf"),
        help="File extension to ingest (default: pdf)",
    )
    args = parser.parse_args()

    ingestor = KnowledgeIngestor(workers=args.workers, file_ext=args.ext)
    ingestor.run()


if __name__ == "__main__":  # pragma: no cover
    cli()
