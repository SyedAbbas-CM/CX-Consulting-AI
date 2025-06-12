import asyncio
import json  # Added for parsing doc_config from query if needed
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Placeholder for Answer type - define later based on actual output needed
class Answer:
    def __init__(self, content: str, sources: List[str] = None):
        self.content = content
        self.sources = sources or []


from app.api.models import (  # For doc_config typing in run method
    DocumentGenerationConfig,
)
from app.core.llm_service import LLMService
from app.services.chat_service import ChatService

# Placeholder imports for Services needed by agents
# from app.services.retrieval_service import HierarchicalSearcher # No longer HierarchicalSearcher
from app.services.document_service import (  # AgentRunner will use DocumentService for RetrieverAgent
    DocumentService,
)
from app.services.project_manager import ProjectManager  # Corrected import
from app.services.rag_engine import RagEngine  # For context retrieval
from app.template_wrappers.prompt_template import PromptTemplateManager

from .critiquer import CritiquerAgent
from .drafter import DraftingAgent
from .finalizer import FinalizerAgent

# Import implemented agents
from .retriever import RetrieverAgent

# from .finaliser import FinaliserAgent


logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(
        self,
        llm_service: LLMService,
        document_service: DocumentService,
        chat_service: ChatService,
        project_manager: ProjectManager,
        prompt_template_manager: PromptTemplateManager,
        settings: dict,
    ):
        """
        Initializes the Agent Runner with necessary services/components.

        Args:
            llm_service: Instance of the LLM Service.
            document_service: Instance of the Document Service (for RetrieverAgent).
            chat_service: Instance of the chat service.
            project_manager: Instance of the Project Manager.
            prompt_template_manager: Instance of the PromptTemplateManager.
            settings: Configuration settings for the agent runner.
        """
        self.llm_service = llm_service
        self.document_service = document_service
        self.chat_service = chat_service
        self.project_manager = project_manager
        self.prompt_template_manager = prompt_template_manager
        self.settings = settings

        # Initialize agents
        self.retriever_agent = RetrieverAgent(
            document_service, project_manager
        )  # Pass document_service
        self.drafting_agent = DraftingAgent(llm_service)
        self.critiquer_agent = CritiquerAgent(llm_service)
        self.finalizer_agent = FinalizerAgent(llm_service)
        # self.critic_agent = CriticAgent(llm_service)
        # self.finaliser_agent = FinaliserAgent(llm_service)
        logger.info("AgentRunner initialized with all agents and services.")

    async def generate_deliverable(
        self,
        deliverable_type: str,
        user_turn: str,
        project_id: Optional[str],
        conversation_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Generates a deliverable document based on conversational context.

        Args:
            deliverable_type: The key for the deliverable (e.g., "cx_strategy").
            user_turn: The latest user message/query.
            project_id: The current project ID.
            conversation_id: The current conversation ID.

        Returns:
            A dictionary representing the persisted document.
        """
        logger.info(
            f"Generating deliverable '{deliverable_type}' for project '{project_id}', convo '{conversation_id}'"
        )

        # 1. Grab raw deliverable template text
        try:
            # Ensure the template name matches how it's stored/loaded (e.g., with deliverable_ prefix)
            raw_template_name = (
                f"deliverable_{deliverable_type.lower().replace(' ', '_')}"
            )
            raw_tpl = self.prompt_template_manager.get_raw_template(raw_template_name)
            logger.debug(
                f"Successfully retrieved raw template for '{raw_template_name}'"
            )
        except ValueError as e:
            logger.error(f"Could not get raw template for {raw_template_name}: {e}")
            # Handle error appropriately, maybe raise HTTPException if called from route
            # For now, let's assume this would be caught by the route's error handling.
            raise Exception(f"Raw template for {raw_template_name} not found.") from e

        # 2. Collect context
        conversation_history_messages = []
        if conversation_id:
            try:
                conversation_history_messages = (
                    await self.chat_service.get_chat_history(conversation_id, limit=10)
                )  # Get last 10 messages
            except Exception as e:
                logger.warning(
                    f"Could not fetch chat history for {conversation_id}: {e}"
                )

        # Convert chat history to a simple string format for the prompt
        # Taking content from last 10 messages, simple join.
        simple_chat_history_str = "\n".join(
            [
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in conversation_history_messages
            ]
        )

        retrieved_chunks_data = []
        try:
            # Retrieve context based on the user_turn and project_id.
            # doc_config is None because we are not using specific parameters for retrieval in this flow.
            retrieved_chunks_data = await self.retriever_agent.retrieve(
                query=user_turn, project_id=project_id, doc_config=None
            )
        except Exception as e:
            logger.warning(f"Error retrieving context for deliverable generation: {e}")

        # Convert retrieved chunks to a simple string format for the prompt
        # Taking text from top 10 chunks.
        simple_retrieved_chunks_str = "\n---\n".join(
            [chunk.get("text", "") for chunk in retrieved_chunks_data[:10]]
        )

        # 3. Render meta-prompt
        try:
            meta_prompt = self.prompt_template_manager.render(
                "meta_deliverable",  # The new meta-prompt template
                raw_template=raw_tpl,
                conversation_history=simple_chat_history_str,
                retrieved_chunks=simple_retrieved_chunks_str,
                user_turn=user_turn,
            )
            logger.debug("Successfully rendered meta_deliverable prompt.")
        except Exception as e:
            logger.error(
                f"Failed to render meta_deliverable prompt: {e}", exc_info=True
            )
            raise Exception(
                "Failed to render meta-prompt for deliverable generation."
            ) from e

        # 4. LLM generate filled deliverable content
        try:
            # Consider adjusting max_tokens based on expected deliverable length
            # The plan suggested 2048, which might be small for some documents.
            # Making this configurable via settings would be good long-term.
            # For now, let's use a higher value, e.g., 4000, assuming LLM can handle it.
            MAX_TOKENS_DELIVERABLE = self.settings.get("MAX_TOKENS_DELIVERABLE", 4000)
            filled_content = await self.llm_service.generate(
                prompt=meta_prompt, max_tokens=MAX_TOKENS_DELIVERABLE
            )
            if not filled_content:
                logger.warning("LLM returned empty content for deliverable generation.")
                # Create a basic placeholder if LLM fails to generate anything
                filled_content = f"Failed to generate content for {deliverable_type}. Please try again or provide more context. Template:\n{raw_tpl}"
            logger.info(
                f"LLM successfully generated content for deliverable '{deliverable_type}'. Length: {len(filled_content)}"
            )
        except Exception as e:
            logger.error(
                f"LLM generation failed for deliverable '{deliverable_type}': {e}",
                exc_info=True,
            )
            raise Exception("LLM failed during deliverable generation.") from e

        # 5. Persist document
        try:
            # Generate a title for the document
            # Making title more descriptive with a timestamp to avoid collisions if user generates same type quickly
            timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            doc_title = (
                f"{deliverable_type.replace('_', ' ').title()} - {timestamp_str}"
            )

            # Using await asyncio.to_thread for project_manager synchronous methods
            persisted_doc_id = await asyncio.to_thread(
                self.project_manager.create_document,
                project_id=project_id,
                user_id="model",  # As per plan, indicating model generation
                title=doc_title,
                content=filled_content,
                document_type=deliverable_type,  # Store the original deliverable type key
                conversation_id=conversation_id,  # Associate with the conversation
                metadata={
                    "generated_by": "convo_deliverable_flow",  # New generation method
                    "source_user_turn": user_turn,
                    "retrieved_chunks_count": len(retrieved_chunks_data),
                    "chat_history_messages_count": len(conversation_history_messages),
                },
            )
            if not persisted_doc_id:
                logger.error(
                    f"Failed to persist generated deliverable '{deliverable_type}'. create_document returned no ID."
                )
                raise Exception("Failed to save the generated deliverable document.")

            logger.info(
                f"Persisted deliverable '{deliverable_type}' as document ID '{persisted_doc_id}'"
            )

            # Return the full document dictionary
            # Using await asyncio.to_thread for project_manager synchronous methods
            final_document = await asyncio.to_thread(
                self.project_manager.get_document, persisted_doc_id
            )
            if not final_document:
                logger.error(
                    f"Failed to retrieve document {persisted_doc_id} after creation."
                )
                # This is an issue, but we did create it. Maybe return a simpler dict?
                # For now, adhering to plan's expectation of returning the document dict.
                raise Exception("Failed to retrieve document immediately after saving.")
            return final_document
        except Exception as e:
            logger.error(
                f"Error persisting or retrieving deliverable document '{deliverable_type}': {e}",
                exc_info=True,
            )
            # If persistence fails, the user gets nothing. This is a critical error.
            raise Exception(
                "Critical error saving or retrieving the generated deliverable."
            ) from e

    async def refine_document(
        self, doc_id: str, user_prompt: str, replace_embeddings: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Refines a document using an LLM, saves it as a new revision, and re-indexes.

        Args:
            doc_id: The ID of the document to refine.
            user_prompt: The user's instruction for refinement.
            replace_embeddings: If True, delete old embeddings before adding new ones.

        Returns:
            The updated document dictionary if successful, None otherwise.
        """
        logger.info(
            f"Starting refinement for document {doc_id} with prompt: '{user_prompt[:50]}...'"
        )

        # 1. Get the document
        # Assuming project_manager methods are synchronous as per prior usage patterns.
        # If they become async, these calls need to be awaited (and AgentRunner methods might become async).
        # For now, wrapping sync calls in to_thread for safety if called from an async context.
        document = await asyncio.to_thread(self.project_manager.get_document, doc_id)
        if not document:
            logger.error(f"Refinement failed: Document {doc_id} not found.")
            # Consider raising HTTPException here if this is called directly from a route
            return None

        # 2. Build RAG context (current doc + project memos, etc.)
        # Assuming rag_engine.retrieve_for_refinement is async
        # context_chunks = await self.rag_engine.retrieve_for_refinement(document, project_id=document.get("project_id"))
        # For now, let's skip adding extra context chunks to the refinement prompt for simplicity,
        # as the prompt template provided focuses on the current_doc and user_prompt.
        # We can integrate context_chunks into the prompt template later if needed.
        logger.debug(
            f"Skipping explicit RAG context retrieval for refinement of doc {doc_id} for now."
        )

        # 3. Prepare and render the refinement prompt
        current_doc_content = document.get("content", "")
        # Add a guard for prompt length
        # MAX_REFINE_DOC_CHARS = self.settings.get("MAX_REFINE_DOC_CHARS", 20000) # Get from settings if available
        MAX_REFINE_DOC_CHARS = (
            20000  # Hardcode for now, make configurable later via settings
        )
        if len(current_doc_content) > MAX_REFINE_DOC_CHARS:
            logger.warning(
                f"Document content for doc {doc_id} is too long ({len(current_doc_content)} chars). Truncating to last {MAX_REFINE_DOC_CHARS} chars for refinement prompt."
            )
            # Truncate from the beginning, keeping the end of the document, which might be more relevant
            current_doc_content = current_doc_content[-MAX_REFINE_DOC_CHARS:]

        try:
            # Assuming prompt_manager.render is synchronous
            refine_prompt_template_name = "deliverable_refine"  # As per blueprint
            rendered_prompt = await asyncio.to_thread(
                self.prompt_template_manager.render,
                refine_prompt_template_name,
                CURRENT_DOC=current_doc_content,  # Use potentially truncated content
                USER_PROMPT=user_prompt,
            )
        except Exception as e:
            logger.error(
                f"Failed to render refinement prompt for doc {doc_id}: {e}",
                exc_info=True,
            )
            return None

        # 4. Generate refined content with LLM
        logger.debug(
            f"Sending refinement prompt to LLM for doc {doc_id} (first 100 chars): {rendered_prompt[:100]}..."
        )
        try:
            # Assuming llm_service.generate is async
            refined_content = await self.llm_service.generate(
                prompt=rendered_prompt,
                # Consider adding other LLM params like max_tokens if necessary
            )
            if not refined_content:
                logger.error(f"LLM returned empty content for refining doc {doc_id}.")
                return None
            logger.info(f"LLM successfully generated refined content for doc {doc_id}.")
        except Exception as e:
            logger.error(
                f"LLM generation failed during refinement for doc {doc_id}: {e}",
                exc_info=True,
            )
            return None

        # 5. Persist as new revision
        revision_id = str(uuid.uuid4())
        author = "model"  # As per blueprint

        try:
            # Assuming project_manager.save_revision is synchronous
            updated_document = await asyncio.to_thread(
                self.project_manager.save_revision,
                doc_id,
                revision_id,
                refined_content,
                author,
                refinement_prompt=user_prompt,  # Log the user prompt used for this revision
            )
            if not updated_document:
                logger.error(
                    f"Failed to save revision {revision_id} for document {doc_id}."
                )
                return None
            logger.info(f"Revision {revision_id} saved for document {doc_id}.")
        except Exception as e:
            logger.error(
                f"Error saving revision {revision_id} for document {doc_id}: {e}",
                exc_info=True,
            )
            return None

        # 6. Re-embed (handle `replace_embeddings`)
        # The `save_revision` method in ProjectManager already schedules re-indexing via add_documents.
        # `add_documents` typically updates if the ID exists.
        # If `replace_embeddings` is true, we need to explicitly delete old embeddings first.
        if replace_embeddings:
            logger.info(
                f"Explicitly deleting existing embeddings for document {doc_id} before re-indexing revision {revision_id}."
            )
            try:
                # Assuming document_service.delete_document is async and deletes embeddings
                delete_success = await self.document_service.delete_document(doc_id)
                if delete_success:
                    logger.info(
                        f"Successfully deleted old embeddings for document {doc_id}."
                    )
                else:
                    logger.warning(
                        f"Could not delete old embeddings for document {doc_id} (may not have existed or other issue)."
                    )

                # After deletion, re-add the new content (save_revision already calls add_documents which will now be an insert)
                # For clarity and to ensure it happens *after* potential deletion if replace_embeddings=True,
                # we could re-trigger add_documents here. However, save_revision *already* does this.
                # To make it explicit, we might call a re-index method that knows if it's an update or new add.
                # For now, save_revision's call to add_documents should suffice, as it uses the latest content.
                # The main concern is the *order* if replace_embeddings is true.
                # The current save_revision might call add_documents *before* we get to delete here.
                # Re-thinking: save_revision should NOT call add_documents. AgentRunner should control indexing for refinement.

                # CORRECTED LOGIC: save_revision should not re-index. This method handles it.
                # The following is the intended re-indexing after potential deletion.
                doc_metadata_for_indexing = {
                    "project_id": updated_document.get("project_id"),
                    "document_type": updated_document.get("document_type"),
                    "source": doc_id,  # Original document ID is the source
                    "title": updated_document.get("title", ""),
                    "created_at": updated_document.get(
                        "created_at"
                    ),  # or original doc's created_at?
                    "updated_at": updated_document.get(
                        "updated_at"
                    ),  # This will be the new revision's timestamp
                    "last_revision_id": revision_id,
                    "last_revised_by": author,
                    "is_global": updated_document.get(
                        "is_global", False
                    ),  # Pass is_global if available
                    # Add any other relevant metadata expected by index_document_content or collection policy
                }
                # Ensure document_type is present, default if necessary, or log error
                if not doc_metadata_for_indexing.get("document_type"):
                    logger.warning(
                        f"Document type missing for doc_id {doc_id} during refine re-indexing. Attempting to retrieve or default."
                    )
                    # Attempt to get it from the original document if not in updated_document (e.g. if it's a pure content update)
                    original_doc_details = (
                        await self.project_manager.get_document_details(doc_id)
                    )
                    if original_doc_details and original_doc_details.get(
                        "document_type"
                    ):
                        doc_metadata_for_indexing["document_type"] = (
                            original_doc_details.get("document_type")
                        )
                        logger.info(
                            f"Using document_type '{doc_metadata_for_indexing['document_type']}' from original document for {doc_id}"
                        )
                    else:
                        # Fallback if still not found - this might be an issue for collection routing
                        # Consider a more robust way to ensure document_type consistency
                        doc_metadata_for_indexing["document_type"] = (
                            "deliverable"  # Defaulting, but this might be risky
                        )
                        logger.warning(
                            f"Defaulting document_type to 'deliverable' for {doc_id}. This might affect collection routing."
                        )

                index_success = await self.document_service.index_document_content(
                    doc_id=doc_id,  # or perhaps revision_id if chunks should be associated with revision? For now, doc_id.
                    content=updated_document["content"],
                    doc_metadata=doc_metadata_for_indexing,
                )
                if index_success:
                    logger.info(
                        f"Successfully re-indexed content for document {doc_id} (revision {revision_id}) using index_document_content after explicit deletion (replace_embeddings=True)."
                    )
                else:
                    logger.error(
                        f"Failed to re-index content for document {doc_id} (revision {revision_id}) using index_document_content after explicit deletion."
                    )

            except Exception as e:
                logger.error(
                    f"Error during explicit deletion/re-indexing for document {doc_id} (replace_embeddings=True): {e}",
                    exc_info=True,
                )
                # updated_document is still valid, but re-indexing might be compromised.
        else:
            # If not replacing embeddings, just re-index the new content.
            doc_metadata_for_indexing = {
                "project_id": updated_document.get("project_id"),
                "document_type": updated_document.get("document_type"),
                "source": doc_id,  # Original document ID is the source
                "title": updated_document.get("title", ""),
                "created_at": updated_document.get("created_at"),
                "updated_at": updated_document.get("updated_at"),
                "last_revision_id": revision_id,
                "last_revised_by": author,
                "is_global": updated_document.get("is_global", False),
            }
            if not doc_metadata_for_indexing.get("document_type"):
                logger.warning(
                    f"Document type missing for doc_id {doc_id} during refine re-indexing (no replace). Attempting to retrieve or default."
                )
                original_doc_details = await self.project_manager.get_document_details(
                    doc_id
                )
                if original_doc_details and original_doc_details.get("document_type"):
                    doc_metadata_for_indexing["document_type"] = (
                        original_doc_details.get("document_type")
                    )
                    logger.info(
                        f"Using document_type '{doc_metadata_for_indexing['document_type']}' from original document for {doc_id} (no replace)"
                    )
                else:
                    doc_metadata_for_indexing["document_type"] = "deliverable"
                    logger.warning(
                        f"Defaulting document_type to 'deliverable' for {doc_id} (no replace). This might affect collection routing."
                    )

            index_success = await self.document_service.index_document_content(
                doc_id=doc_id,
                content=updated_document["content"],
                doc_metadata=doc_metadata_for_indexing,
            )
            if index_success:
                logger.info(
                    f"Successfully re-indexed content for document {doc_id} (revision {revision_id}) using index_document_content."
                )
            else:
                logger.error(
                    f"Failed to re-index content for document {doc_id} (revision {revision_id}) using index_document_content."
                )

        return updated_document

    async def run(
        self,
        task_type: str,
        query: str,
        project_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        doc_config: Optional[DocumentGenerationConfig] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrates the agent pipeline to answer a query or generate a document.

        Args:
            task_type: The type of task (e.g., 'question', 'roi_analysis').
            query: The user's query or main input string (e.g. for QA, or JSON string of params for docs).
            project_id: The associated project ID, if any.
            conversation_id: The associated conversation ID (chat_id), if any.
            doc_config: Document generation configuration, if task_type is for a document.

        Returns:
            A dictionary containing the final response and potentially intermediate results.
        """
        logger.info(
            f"AgentRunner starting task '{task_type}' for project '{project_id}', conversation '{conversation_id}' with query: {query[:50]}..., doc_config present: {doc_config is not None}"
        )

        pipeline_results = {}
        actual_query_for_retrieval = query
        
        # If task_type is document generation, the query might be structured parameters.
        # The retriever might need a more natural language query derived from these.
        # For now, we pass the raw query string to retriever. 
        # If doc_config is present, it's passed to retriever for collection scoping.

        # 1. Retrieve Context
        logger.debug("Agent Pipeline Step 1: Retrieval")
        retrieved_context = await self.retriever_agent.retrieve(
            query=actual_query_for_retrieval, 
            project_id=project_id, 
            doc_config=doc_config,  # Pass doc_config here
        )
        pipeline_results["retrieved_context"] = retrieved_context
        if not retrieved_context:
            logger.warning(
                "Retrieval agent returned no context. Proceeding without context."
            )
            # Decide fallback behavior - maybe skip drafting/critique? Or try LLM directly?
            # For now, we proceed but Drafting agent should handle no context
        else:
             logger.info(f"Retrieved {len(retrieved_context)} context chunks.")

        # 2. Get Chat History (Optional)
        chat_history = None
        if conversation_id:  # Use conversation_id if provided
            logger.debug(f"Fetching chat history for conversation: {conversation_id}")
            try:
                # Use ChatService to get history for the specific conversation_id
                chat_history = await self.chat_service.get_chat_history(conversation_id)
                if chat_history:
                    logger.info(
                        f"Fetched {len(chat_history)} messages from history for conversation {conversation_id}."
                    )
            except Exception as e:
                logger.warning(
                    f"Could not fetch chat history for conversation {conversation_id}: {e}"
                )
        elif project_id:
            logger.info(
                f"No conversation_id provided, but project_id {project_id} is present. Chat history not fetched for agent run."
            )
        else:
            logger.info(
                "No conversation_id or project_id provided. Chat history not fetched."
            )
        
        # 3. Generate Drafts
        logger.debug("Agent Pipeline Step 2: Drafting")
        drafts = await self.drafting_agent.generate_drafts(
            query=query,
            retrieved_context=retrieved_context,
            project_id=project_id,
            chat_history=chat_history,  # Pass history
        )
        pipeline_results["drafts"] = drafts
        if not drafts:
             logger.error("Drafting agent failed to generate drafts.")
             # Handle failure - maybe return an error or try a simpler response?
            return {"error": "Failed to generate response drafts."}
        else:
             logger.info(f"Generated {len(drafts)} drafts.")

        # 4. Critique Drafts
        logger.debug("Agent Pipeline Step 3: Critiquing")
        critiques, best_draft_index = await self.critiquer_agent.critique_and_select(
            query=query,
            drafts=drafts,
            retrieved_context=retrieved_context,
            chat_history=chat_history,
        )
        pipeline_results["critiques"] = critiques
        pipeline_results["best_draft_index"] = best_draft_index
        logger.info(
            f"Critique step completed. Selected best draft index: {best_draft_index}"
        )

        # 5. Finalize Answer
        logger.debug("Agent Pipeline Step 4: Finalizing")
        if not (0 <= best_draft_index < len(drafts)):
            logger.warning(
                f"Best draft index {best_draft_index} out of range. Using index 0."
            )
            best_draft_index = 0  # Default to first draft if index is invalid
        
        best_draft = drafts[best_draft_index]
        final_answer, sources = await self.finalizer_agent.finalize_answer(
             query=query,
             best_draft=best_draft,
             critiques=critiques,
             retrieved_context=retrieved_context,
            chat_history=chat_history,
        )
        pipeline_results["final_answer"] = final_answer
        pipeline_results["sources"] = sources
        logger.info("Finalizing step completed.")

        logger.info(f"AgentRunner finished task '{task_type}'.")
        return pipeline_results 
