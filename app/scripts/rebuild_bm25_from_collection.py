# app/scripts/rebuild_bm25_from_collection.py
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add app directory to sys.path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

load_dotenv()

from app.core.config import get_settings
from app.services.document_service import CX_GLOBAL_COLLECTION, DocumentService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rebuild_bm25_from_collection")


async def rebuild_bm25_for_existing_collection(collection_name: str):
    logger.info(
        f"Starting BM25 index rebuild for existing collection: {collection_name}"
    )
    settings = get_settings()

    try:
        doc_service = DocumentService(
            documents_dir=str(
                settings.DOCUMENTS_DIR
            ),  # May not be strictly needed if not reading new files
            chunked_dir=str(settings.CHUNKED_DIR),  # May not be strictly needed
            vectorstore_dir=str(settings.VECTOR_DB_PATH),
            embedding_model=(
                settings.BGE_MODEL_NAME
                if settings.EMBEDDING_TYPE.lower() == "bge"
                else settings.EMBEDDING_MODEL
            ),
            default_collection_name=collection_name,
        )

        await doc_service._init_vector_store()  # Initialize vector store connection
        await doc_service.set_collection(
            collection_name
        )  # Load the specified collection

        if doc_service.collection is None:
            logger.error(
                f"Collection '{collection_name}' could not be loaded or does not exist."
            )
            return

        logger.info(
            f"Fetching all documents from collection '{collection_name}' to rebuild BM25 index..."
        )

        # Fetch all documents from the Chroma collection
        # ChromaDB's get() method can fetch all items if no ids or where clause is provided.
        # We need documents (text content) and their IDs.
        # The include argument specifies what fields to return.
        collection_data = doc_service.collection.get(
            include=["documents", "metadatas"]
        )  # embeddings not needed for BM25

        ids = collection_data.get("ids", [])
        documents_content = collection_data.get("documents", [])
        metadatas = collection_data.get("metadatas", [])

        if not ids or not documents_content:
            logger.warning(
                f"No documents found in collection '{collection_name}'. Cannot build BM25 index."
            )
            return

        logger.info(
            f"Retrieved {len(ids)} document chunks from '{collection_name}'. Preparing for BM25 indexing."
        )

        # Prepare chunks in the format expected by _update_bm25_index
        # _update_bm25_index expects List[Dict[str, Any]], where each dict is a chunk
        # with at least 'id' and 'page_content'.
        chunks_for_bm25 = []
        for i, doc_id in enumerate(ids):
            content = documents_content[i]
            metadata = metadatas[i] if i < len(metadatas) else {}
            if content:  # Ensure content is not None or empty
                chunks_for_bm25.append(
                    {"id": doc_id, "text": content, "metadata": metadata}
                )
            else:
                logger.warning(
                    f"Chunk with id {doc_id} has no content. Skipping for BM25."
                )

        if not chunks_for_bm25:
            logger.warning(
                f"No valid content retrieved from '{collection_name}' to build BM25 index."
            )
            return

        # Call _update_bm25_index with all retrieved chunks
        # This method builds the BM25Okapi object in memory
        logger.info(
            f"Updating BM25 index in memory with {len(chunks_for_bm25)} chunks..."
        )
        await doc_service._update_bm25_index(chunks_for_bm25, collection_name)

        # Persist the BM25 data to disk
        logger.info(f"Saving BM25 index for collection '{collection_name}' to disk...")
        await doc_service._save_bm25_data(collection_name)

        logger.info(
            f"BM25 index for collection '{collection_name}' has been rebuilt and saved."
        )

    except Exception as e:
        logger.error(
            f"An error occurred during BM25 index rebuild for '{collection_name}': {e}",
            exc_info=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild BM25 index for an existing Chroma collection."
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Name of the existing Chroma collection to rebuild BM25 index for (e.g., cx_global).",
    )
    args = parser.parse_args()

    asyncio.run(rebuild_bm25_for_existing_collection(args.collection))
