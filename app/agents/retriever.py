import asyncio
import logging
import re  # Import re for sanitization
from typing import Any, Dict, List, Optional, Union

from app.api.models import DocumentGenerationConfig
from app.core.config import get_settings
from app.core.llm_service import LLMService

# Assuming HierarchicalSearcher lives here based on imports in main.py/dependencies.py
# If it moves, update the import.
# from app.services.retrieval_service import HierarchicalSearcher # No longer used
from app.services.document_service import (  # Use DocumentService as the searcher
    CX_GLOBAL_COLLECTION,
    DELIVERABLE_COLLECTION_PREFIX,
    USER_COLLECTION_PREFIX,
    DocumentService,
)
from app.services.project_manager import ProjectManager
from app.services.rag_engine import RagEngine

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """Agent responsible for retrieving relevant document chunks."""

    def __init__(
        self, document_service: DocumentService, project_manager: ProjectManager
    ):
        """
        Initializes the RetrieverAgent.

        Args:
            document_service: An instance of DocumentService, used for searching.
            project_manager: An instance of ProjectManager.
        """
        self.document_service = document_service  # Changed from self.searcher
        self.project_manager = project_manager
        logger.info(
            "RetrieverAgent initialized with DocumentService and ProjectManager."
        )

    async def _deduplicate_and_sort_results(
        self, results: List[Dict[str, Any]], limit: int
    ) -> List[Dict[str, Any]]:
        """Deduplicates search results based on text content and sorts by score."""
        seen_texts = set()
        unique_results = []
        # Sort by score first to keep the highest score for duplicates if text is the same
        # but other metadata might differ. Or, use a more robust ID if available.
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        for res in results:
            text_content = res.get("text", "")
            # A more robust deduplication would use a unique chunk ID from metadata if available
            # Example: chunk_id = res.get("metadata", {}).get("chunk_id")
            if text_content not in seen_texts:
                unique_results.append(res)
                seen_texts.add(text_content)

        # Already sorted by score due to initial sort if no unique ID is used
        # If a unique ID was used, re-sort after deduplication:
        # unique_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return unique_results[:limit]

    async def retrieve(
        self,
        query: str,
        project_id: Optional[str] = None,
        limit: int = 10,
        collection_name: Optional[str] = None,
        include_global: bool = True,
        doc_config: Optional[DocumentGenerationConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant chunks using DocumentService, potentially from multiple sources.

        Args:
            query: The search query.
            project_id: The project context, if any.
            limit: The maximum number of chunks to return in the final list.
            collection_name: Specific collection to search in (if provided, bypasses 4-way RAG).
            include_global: Whether to include results from the global collection (used in 4-way RAG).
            doc_config: Configuration for document generation, used for deliverable-specific RAG.

        Returns:
            A list of dictionaries, each containing chunk text, score, and metadata.
        """
        logger.info(
            f"RetrieverAgent retrieving documents for query: '{query[:50]}...' in project '{project_id}', doc_config: {doc_config is not None}"
        )

        all_retrieved_docs: List[Dict[str, Any]] = []

        try:
            if collection_name:
                # Direct search in a specific collection (existing behavior for override)
                logger.info(
                    f"Performing direct search in collection: {collection_name}"
                )
                retrieved_docs = await self.document_service.retrieve_documents(
                    query=query,
                    limit=limit,
                    project_id=project_id,
                    collection_name=collection_name,
                    include_global=False,
                )
                all_retrieved_docs.extend(retrieved_docs)
            else:
                # Implement 4-way RAG
                search_tasks = []

                # A) Global collection search (if include_global is True)
                # This is for general context related to the main query.
                if include_global:
                    logger.debug(
                        f"Queueing general global search in {CX_GLOBAL_COLLECTION} for query: '{query[:30]}...'"
                    )
                    search_tasks.append(
                        self.document_service.retrieve_documents(
                            query=query,
                            limit=limit,
                            project_id=None,
                            collection_name=CX_GLOBAL_COLLECTION,
                            include_global=True,
                        )
                    )

                # A.1) Global collection search for TEMPLATES if in document generation mode
                if doc_config and doc_config.deliverable_type and include_global:
                    template_query = f"{doc_config.deliverable_type.replace('_', ' ')} template example framework"
                    logger.debug(
                        f"Queueing template search in {CX_GLOBAL_COLLECTION} for query: '{template_query[:50]}...'"
                    )
                    search_tasks.append(
                        self.document_service.retrieve_documents(
                            query=template_query,
                            limit=max(
                                1, limit // 3
                            ),  # Fetch fewer template documents, e.g., 1-3
                            project_id=None,
                            collection_name=CX_GLOBAL_COLLECTION,
                            include_global=True,
                            # Add a metadata filter if templates are tagged, e.g., metadata_filter={"doc_category": "template"}
                        )
                    )

                # B) Project collection search (if project_id is provided)
                if project_id:
                    # Construct the project-specific user document collection name
                    sanitized_project_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", project_id)
                    project_user_collection = (
                        f"{USER_COLLECTION_PREFIX}{sanitized_project_id}"
                    )
                    project_user_collection = project_user_collection[
                        :60
                    ]  # Apply truncation
                    if len(project_user_collection) < 3:  # Apply padding
                        project_user_collection = f"{project_user_collection}___"

                    logger.debug(
                        f"Queueing search in PROJECT USER DOCS collection: {project_user_collection} for project {project_id} with query: '{query[:30]}...'"
                    )
                    search_tasks.append(
                        self.document_service.retrieve_documents(
                            query=query,
                            limit=limit,
                            collection_name=project_user_collection,  # Explicitly target project's user docs
                            project_id=project_id,  # Pass project_id for any metadata filtering within the collection if needed
                            include_global=False,  # Global search is handled by task (A)
                        )
                    )

                # C) Deliverable-type collection search (for existing similar deliverables within the project)
                if doc_config and doc_config.deliverable_type and project_id:
                    deliverable_collection = (
                        f"{DELIVERABLE_COLLECTION_PREFIX}{doc_config.deliverable_type}"
                    )
                    logger.debug(
                        f"Queueing search for existing project deliverables in {deliverable_collection} for project {project_id} with query: '{query[:30]}...'"
                    )
                    search_tasks.append(
                        self.document_service.retrieve_documents(
                            query=query,
                            limit=max(1, limit // 3),
                            project_id=project_id,
                            collection_name=deliverable_collection,
                            include_global=False,
                        )
                    )

                # D) User-uploaded documents for this project - This section is being removed.
                # It was attempting to use each document ID as a collection name, which is likely not
                # the intended way DocumentService handles project-specific user documents.
                # Project-specific documents should be covered by search (B) using project_id,
                # where DocumentService resolves project_id to the correct collection(s).
                # if project_id and self.project_manager:
                #    try:
                #        project_documents_metadata = await asyncio.to_thread(
                #            self.project_manager.get_project_documents, project_id
                #        )
                #        if project_documents_metadata:
                #            document_ids_to_search = [doc.get("id") for doc in project_documents_metadata if doc.get("id")]
                #            if document_ids_to_search:
                #                logger.debug(f"Queueing search for USER_UPLOADED docs in collections (derived from document IDs): {document_ids_to_search} for project {project_id}")
                #                for doc_id_as_collection_name in document_ids_to_search:
                #                    search_tasks.append(
                #                        self.document_service.retrieve_documents(
                #                            query=query, limit=limit, project_id=project_id,
                #                            collection_name=doc_id_as_collection_name, # Assuming doc ID is used as collection name
                #                            include_global=False
                #                        )
                #                    )
                #            else:
                #                logger.debug(f"No document IDs found in metadata for project {project_id}.")
                #        else:
                #            logger.debug(f"No document metadata returned by project_manager for project {project_id}.")
                #    except Exception as e_proj_docs:
                #        logger.error(f"Failed to get or search project document IDs for {project_id}: {e_proj_docs}", exc_info=True)

                if search_tasks:
                    logger.info(
                        f"Executing {len(search_tasks)} RAG searches in parallel."
                    )
                    results_from_tasks = await asyncio.gather(
                        *search_tasks, return_exceptions=True
                    )
                    for result_item in results_from_tasks:
                        if isinstance(result_item, list):
                            all_retrieved_docs.extend(result_item)
                        elif isinstance(result_item, Exception):
                            logger.error(
                                f"A RAG search task failed: {result_item}",
                                exc_info=result_item,
                            )
                else:
                    logger.warning("No RAG search tasks were queued.")

            # Format and deduplicate all collected documents
            # The document_service.retrieve_documents already returns List[Dict] with keys like
            # 'id', 'page_content', 'metadata', 'final_score' based on document_service
            # The old formatting step is less necessary if the document_service provides the target format.
            # For consistency with the plan, we ensure the format {"text": ..., "score": ..., "metadata": ...}
            formatted_results = []
            for doc in all_retrieved_docs:
                formatted_results.append(
                    {
                        "text": doc.get("page_content", doc.get("text")),
                        "score": doc.get("final_score", doc.get("score", 0.0)),
                        "metadata": doc.get("metadata", {}),
                    }
                )

            final_results = await self._deduplicate_and_sort_results(
                formatted_results, limit
            )
            logger.info(
                f"RetrieverAgent found {len(final_results)} relevant chunks after 4-way RAG & deduplication."
            )
            return final_results

        except Exception as e:
            logger.error(
                f"Error during retrieval in RetrieverAgent: {e}", exc_info=True
            )
            return []  # Return empty list on error
