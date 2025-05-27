# list_chroma_collections.py
import asyncio
import logging
import sys
from pathlib import Path

# Add app directory to sys.path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.services.document_service import DocumentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def list_and_count():
    settings = get_settings()
    ds = DocumentService(
        documents_dir=str(
            settings.DOCUMENTS_DIR
        ),  # Actual path less critical for just listing
        chunked_dir=str(
            settings.CHUNKED_DIR
        ),  # Actual path less critical for just listing
        vectorstore_dir=str(settings.VECTOR_DB_PATH),
        embedding_model=(
            settings.BGE_MODEL_NAME
            if settings.EMBEDDING_TYPE.lower() == "bge"
            else settings.EMBEDDING_MODEL
        ),
        default_collection_name="any_default_is_fine_here",  # Doesn't matter for listing
    )
    logger.info(f"Initializing vector store at: {settings.VECTOR_DB_PATH}")
    await ds._init_vector_store()  # Make sure this connects to your existing DB

    if ds.chroma_client:
        collections = ds.chroma_client.list_collections()
        logger.info(f"Available Chroma Collections: {[c.name for c in collections]}")

        target_collection_name = "global_kb"  # Based on your recollection
        if any(c.name == target_collection_name for c in collections):
            logger.info(f"Checking chunk count in '{target_collection_name}'...")
            await ds.set_collection(target_collection_name)
            if ds.collection:
                # Use count_documents or get with minimal include if possible
                # data = ds.collection.get(include=[]) # Just to get IDs for count
                count = ds.collection.count()
                logger.info(
                    f"Number of chunks/items in '{target_collection_name}': {count}"
                )
                if count == 0:
                    logger.warning(
                        f"Collection '{target_collection_name}' exists but is empty."
                    )
                else:
                    logger.info(
                        f"Collection '{target_collection_name}' has content. You can proceed to rebuild BM25 for this collection."
                    )
            else:
                logger.error(f"Failed to set collection to '{target_collection_name}'.")
        else:
            logger.warning(
                f"Collection '{target_collection_name}' not found in the list. Please verify the name."
            )
    else:
        logger.error(
            "Chroma client (ds.chroma_client) not initialized in DocumentService."
        )


if __name__ == "__main__":
    asyncio.run(list_and_count())
