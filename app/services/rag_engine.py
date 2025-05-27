import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from dotenv import load_dotenv

# --- Add imports for Hybrid Retriever and CrossEncoder ---
from sentence_transformers import CrossEncoder  # For re-ranking
from tenacity import (  # Import tenacity for retries
    retry,
    stop_after_attempt,
    wait_fixed,
)

from app.api.models import OriginType  # Added import for OriginType
from app.core.config import get_settings  # Local import OK here
from app.core.llm_service import LLMService
from app.services.chat_service import ChatService
from app.services.context_optimizer import ContextOptimizer
from app.services.document_service import (
    CX_GLOBAL_COLLECTION,
    DELIVERABLE_COLLECTION_PREFIX,
    USER_COLLECTION_PREFIX,
    DocumentService,
)
from app.services.retrieval import (  # Assuming Retreival.py is renamed/moved
    HybridRetriever,
)
from app.template_wrappers.prompt_template import PromptTemplateManager

# --- End new imports ---

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger("cx_consulting_ai.rag_engine")


# --- Custom Exception for G3 ---
class TemplateRenderingError(Exception):
    """Custom exception for errors during prompt template rendering."""

    def __init__(self, missing_key: str, template_name: str):
        self.missing_key = missing_key
        self.template_name = template_name
        super().__init__(f"Missing key '{missing_key}' in template '{template_name}'")


# --- End Custom Exception ---

# Define constants for RAG classification - REFINED PROMPT
NEEDS_RAG_PROMPT = """Does the following query require specific information found ONLY in internal documents (like client details, project plans, past proposals, specific consulting frameworks) to be answered accurately?
Focus ONLY on whether *internal documents* are strictly necessary. Do not consider general knowledge or conversation history.
Answer only YES or NO.

Query: {query}"""
CLASSIFICATION_MAX_TOKENS = 10
CLASSIFICATION_TEMP = 0.0


class RagEngine:
    """Main RAG engine implementation for CX consulting AI."""

    SOURCE_SNIPPET_MAX_LENGTH = 200

    def __init__(
        self,
        llm_service: LLMService,
        document_service: DocumentService,
        template_manager: PromptTemplateManager,
        context_optimizer: ContextOptimizer,
        chat_service: ChatService,
    ):
        """
        Initialize the RAG engine.
        Args:
            llm_service: LLM service for generation
            document_service: Document service for retrieval and corpus data
            template_manager: Prompt template manager
            context_optimizer: Context optimizer for refinement
            chat_service: Chat service for persistent chat history
        """
        self.llm_service = llm_service
        self.document_service = document_service
        self.template_manager = template_manager
        self.context_optimizer = context_optimizer
        self.chat_service = chat_service

        # --- Initialize CrossEncoder ---
        # Use a small, fast model for re-ranking. Can be configured via ENV.
        cross_encoder_model_name = os.getenv(
            "CROSS_ENCODER_MODEL_RAG", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        try:
            self.cross_encoder = CrossEncoder(
                cross_encoder_model_name, device=get_settings().EMBEDDING_DEVICE
            )
            logger.info(
                f"CrossEncoder model '{cross_encoder_model_name}' loaded successfully on device {get_settings().EMBEDDING_DEVICE}."
            )
        except Exception as e:
            logger.error(
                f"Failed to load CrossEncoder model '{cross_encoder_model_name}': {e}. Re-ranking will be disabled.",
                exc_info=True,
            )
            self.cross_encoder = None  # Disable re-ranking if model fails to load
        # --- End CrossEncoder Init ---

        # Load max documents per query from environment
        # These might be used by HybridRetriever (M and K parameters)
        self.hybrid_retriever_top_m = int(os.getenv("HYBRID_RETRIEVER_TOP_M", "50"))
        self.hybrid_retriever_top_k = int(os.getenv("HYBRID_RETRIEVER_TOP_K", "10"))

        logger.info(
            f"RAG Engine initialized to use HybridRetriever. "
            f"Search Params: top_m (candidates)={self.hybrid_retriever_top_m}, top_k (final)={self.hybrid_retriever_top_k}"
        )

    # --- Intent Classification ---
    def classify(self, user_msg: str) -> Literal["qa", "deliverable", "chit_chat"]:
        """Classifies user intent based on keywords and basic heuristics."""
        user_msg_lower = user_msg.lower().strip()

        # 1. Check for simple greetings / short phrases
        greetings = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        if (
            user_msg_lower in greetings or len(user_msg_lower.split()) <= 2
        ):  # Treat very short inputs as chit-chat
            logger.debug(
                f"Classified intent as 'chit_chat' for query: '{user_msg_lower[:50]}...'"
            )
            return "chit_chat"

        # 2. Check for deliverable keywords
        deliverable_keywords = [
            "roi deck",
            "proposal",
            "journey map",
            "intake",
            "cx strategy",
            "generate roi",
            "create proposal",
            "build journey map",
            "start intake",
            "deliverable",
            "report",
            # Add variations as needed
        ]
        if any(keyword in user_msg_lower for keyword in deliverable_keywords):
            logger.debug(
                f"Classified intent as 'deliverable' for query: '{user_msg_lower[:50]}...'"
            )
            return "deliverable"

        # 3. Default to QA
        logger.debug(f"Classified intent as 'qa' for query: '{user_msg_lower[:50]}...'")
        return "qa"

    async def _build_rag_prompt_with_history_truncation(
        self,
        question: str,
        context: str,
        chat_history_messages: List[Dict[str, str]],
        project_id: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Builds the RAG prompt, managing conversation history and handling template errors.
        Returns the formatted prompt string and the actual history used.
        Raises TemplateRenderingError on KeyError during formatting.
        """
        # --- System Prompt ---
        try:
            system_prompt_template = self.template_manager.get_template("system")
            # Assuming system template might need context - adjust if needed
            system_prompt = system_prompt_template.format()
        except KeyError as e:
            logger.error(
                f"KeyError rendering 'system' template: Missing key '{e}'",
                exc_info=True,
            )
            raise TemplateRenderingError(
                missing_key=str(e), template_name="system"
            ) from e
        except Exception as e_sys:
            logger.error(
                f"Error getting/formatting 'system' template: {e_sys}", exc_info=True
            )
            # Fallback or re-raise depending on desired behavior
            system_prompt = "You are a helpful AI assistant."  # Generic fallback

        # Get model's max context window
        # Ensure max_model_len is an int; provide a fallback if None
        model_max_tokens = (
            self.llm_service.max_model_len or 4096
        )  # Default to 4096 if not set

        # Buffer for the LLM's response
        # TODO: Make this configurable or use max_tokens from generation call
        response_buffer_tokens = int(os.getenv("LLM_RESPONSE_BUFFER_TOKENS", "512"))

        # Calculate tokens for fixed parts of the prompt
        system_prompt_tokens = self.llm_service.count_tokens(system_prompt)
        question_tokens = self.llm_service.count_tokens(question)
        context_tokens = self.llm_service.count_tokens(context)

        # Available tokens for conversation history
        available_for_history_tokens = (
            model_max_tokens
            - system_prompt_tokens
            - question_tokens
            - context_tokens
            - response_buffer_tokens
        )

        if available_for_history_tokens < 0:
            logger.warning(
                f"Not enough token space for even basic RAG prompt parts. "
                f"Model max: {model_max_tokens}, System: {system_prompt_tokens}, "
                f"Question: {question_tokens}, Context: {context_tokens}, Buffer: {response_buffer_tokens}. "
                f"History will be empty."
            )
            available_for_history_tokens = 0

        truncated_history_messages: List[Dict[str, str]] = []
        current_history_tokens = 0

        # Iterate history from oldest to newest (already in this order from ChatService)
        for message in reversed(
            chat_history_messages
        ):  # Iterate from newest to oldest to fill budget, then reverse for final prompt
            message_content = message.get("content", "")
            message_tokens = self.llm_service.count_tokens(
                f"{message.get('role', '')}: {message_content}"
            )  # Approximate token count with role

            if (
                current_history_tokens + message_tokens
            ) <= available_for_history_tokens:
                truncated_history_messages.append(message)
                current_history_tokens += message_tokens
            else:
                # Not enough space for this older message
                break

        # Reverse again to get oldest to newest for the prompt
        final_history_messages = list(reversed(truncated_history_messages))

        # Format conversation history for the prompt
        history_lines = [
            f"{msg['role']}: {msg['content']}" for msg in final_history_messages
        ]
        conversation_history_block = (
            "### Recent conversation\n" + "\n".join(history_lines) + "\n"
            if history_lines
            else ""
        )

        # --- Final RAG Prompt ---
        try:
            rag_prompt_template = self.template_manager.get_template("rag")
            final_prompt = rag_prompt_template.format(
                system_prompt=system_prompt,
                context=context,
                conversation_history_block=conversation_history_block,
                query=question,
            )
        except KeyError as e:
            logger.error(
                f"KeyError rendering 'rag' template: Missing key '{e}'", exc_info=True
            )
            raise TemplateRenderingError(missing_key=str(e), template_name="rag") from e
        except Exception as e_rag:
            logger.error(
                f"Error getting/formatting 'rag' template: {e_rag}", exc_info=True
            )
            # Fallback or re-raise? For now, create a minimal prompt
            final_prompt = f"Context: {context}\n\nUser: {question}\nAssistant:"

        final_prompt_tokens = self.llm_service.count_tokens(final_prompt)
        logger.info(
            f"Final RAG prompt tokens: {final_prompt_tokens}/{model_max_tokens}. "
            f"History messages used: {len(final_history_messages)}/{len(chat_history_messages)}. "
            f"History tokens: {current_history_tokens}/{available_for_history_tokens} budget."
        )

        if final_prompt_tokens > model_max_tokens:
            logger.warning(
                f"Final prompt ({final_prompt_tokens} tokens) still exceeds model max tokens ({model_max_tokens}) "
                f"despite history truncation. Consider reducing context or query length."
            )
            # Potentially further truncate context or raise an error if critical

        return final_prompt, final_history_messages

    async def process_document(
        self,
        file_bytes_or_path: Union[bytes, str],
        filename: str,
        project_id: Optional[str] = None,
        is_global: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a document (from bytes or file path) for the knowledge base.

        Args:
            file_bytes_or_path: Document content as bytes or path to the document file.
            filename: Original filename.
            project_id: Optional project ID to associate the document with.
            is_global: Flag indicating if the document is global.

        Returns:
            Dictionary with processing results.
        """
        actual_file_path = None
        temp_file_created = False
        try:
            if isinstance(file_bytes_or_path, bytes):
                # Create a temporary file to store the bytes
                # Ensure the suffix matches the original filename for type detection
                file_suffix = os.path.splitext(filename)[1]
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_suffix
                ) as tmp_file:
                    tmp_file.write(file_bytes_or_path)
                    actual_file_path = tmp_file.name
                temp_file_created = True
                logger.info(
                    f"Created temporary file {actual_file_path} for uploaded content of {filename}"
                )
            elif isinstance(file_bytes_or_path, str):
                actual_file_path = file_bytes_or_path
            else:
                err_msg = "file_bytes_or_path must be bytes or a string path."
                logger.error(err_msg)
                # Return an error structure consistent with other error returns in this method
                return {
                    "filename": filename,
                    "document_id": None,
                    "chunks_created": 0,
                    "status": "error",
                    "error": err_msg,
                }

            if not actual_file_path:  # Should not happen if logic above is correct
                err_msg_path = (
                    "Could not determine actual file path for document processing."
                )
                logger.error(err_msg_path)
                return {
                    "filename": filename,
                    "document_id": None,
                    "chunks_created": 0,
                    "status": "error",
                    "error": err_msg_path,
                }

            logger.info(
                f"Processing document: {filename} from path: {actual_file_path}"
            )

            # Determine document type from extension (using original filename for consistency)
            document_type = filename.split(".")[-1].lower()
            # Ensure allowed_types matches DocumentService or is more general
            allowed_types = [
                "pdf",
                "txt",
                "docx",
                "doc",
                "csv",
                "xlsx",
                "md",
            ]  # This list can be managed by DocumentService ideally

            if document_type not in allowed_types:
                logger.warning(
                    f"File type '{document_type}' for {filename} might not be directly processed by RAG engine's old logic, DocumentService will handle it."
                )

            doc_metadata = {
                "source": filename,
                "project_id": project_id,
                "is_global": is_global,
            }

            if not self.document_service:
                logger.error("DocumentService is not initialized in RagEngine.")
                raise RuntimeError(
                    "Document processing service not available."
                )  # Or return error dict

            success = await self.document_service.add_document(
                document_url=actual_file_path,
                document_type=document_type,  # Pass determined type
                metadata=doc_metadata,
                project_id=project_id,
                is_global=is_global,
            )

            if success:
                logger.info(
                    f"Successfully initiated processing for document {filename} via DocumentService."
                )
                return {
                    "filename": filename,
                    "document_id": f"processed_by_doc_service_{filename}",  # This ID might come from DocumentService
                    "chunks_created": -1,  # This count might come from DocumentService
                    "status": "success",
                }
            else:
                logger.error(
                    f"DocumentService.add_document reported failure for {filename}"
                )
                return {
                    "filename": filename,
                    "document_id": None,
                    "chunks_created": 0,
                    "status": "error",
                    "error": f"DocumentService failed to process {filename}",
                }

        except Exception as e:
            logger.error(
                f"Error in RagEngine.process_document for {filename}: {str(e)}",
                exc_info=True,
            )
            return {
                "filename": filename,
                "document_id": None,
                "chunks_created": 0,
                "status": "error",
                "error": str(e),
            }
        finally:
            if (
                temp_file_created
                and actual_file_path
                and os.path.exists(actual_file_path)
            ):
                try:
                    os.remove(actual_file_path)
                    logger.info(f"Removed temporary file {actual_file_path}")
                except Exception as e_clean:
                    logger.error(
                        f"Error cleaning up temporary file {actual_file_path}: {e_clean}"
                    )

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def _needs_rag(
        self, question: str, conversation_history: List[Dict[str, Any]]
    ) -> bool:
        """Determine if RAG is needed using LLM classification."""
        try:
            normalized_question = question.lower().strip()

            if len(normalized_question.split()) < 3 or normalized_question in [
                "hi",
                "hello",
                "thanks",
                "thank you",
                "ok",
                "okay",
                "bye",
                "goodbye",
            ]:
                logger.info(
                    f"Query '{question}' deemed too short or conversational, skipping RAG."
                )
                return False

            meta_question_triggers = [
                "who are you",
                "what are you",
                "what can you do",
                "what is your purpose",
                "what are you made for",
                "what are you supposed to do",
                "tell me about yourself",
            ]
            if any(
                trigger in normalized_question for trigger in meta_question_triggers
            ):
                logger.info(
                    f"Query '{question}' identified as meta-question, skipping RAG."
                )
                return False

            follow_up_triggers = ["tell me more", "explain that", "why did you say"]
            if (
                conversation_history
                and len(normalized_question.split()) < 10
                and any(term in normalized_question for term in follow_up_triggers)
            ):
                logger.info(
                    f"Query '{question}' seems like a simple follow-up, skipping RAG."
                )
                return False

            prompt = NEEDS_RAG_PROMPT.format(query=question)
            logger.info(f"Classifying query relevance for RAG using LLM: '{question}'")

            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=CLASSIFICATION_TEMP,
                max_tokens=CLASSIFICATION_MAX_TOKENS,
            )

            response_text = response.strip().upper()
            logger.info(f"RAG classification result: {response_text}")

            if response_text.startswith("YES"):
                logger.info("RAG classification = YES")
                return True
            elif response_text.startswith("NO"):
                logger.info("RAG classification = NO")
                return False
            else:
                logger.warning(
                    f"Unclear RAG classification response: '{response_text}'. Defaulting to NO."
                )
                return False

        except Exception as e:
            logger.error(
                f"Error during RAG classification: {str(e)}. Defaulting to NO.",
                exc_info=True,
            )
            return False

    async def ask(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        retrieval_active: bool = True,
        retrieval_mode: str = "semantic",
        top_meta_k: Optional[int] = None,
        top_chunks_k: Optional[int] = None,
        include_global: bool = False,
    ) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(
            f"RAG Engine processing question: '{question[:50]}...' for project '{project_id}', chat '{conversation_id}'"
        )

        intent = self.classify(question)
        logger.info(f"Classified intent as: {intent}")

        chat_history_messages = []

        if user_id:
            logger.debug(
                f"[ask] user={user_id} retrieval_active={retrieval_active} mode={retrieval_mode}"
            )

        if not retrieval_active:
            logger.info("retrieval_active=False → skipping document search.")

        if conversation_id:
            try:
                history_data = await self.chat_service.get_chat_history(
                    conversation_id, limit=50
                )
                chat_history_messages = (
                    history_data if isinstance(history_data, list) else []
                )
                logger.debug(
                    f"Retrieved {len(chat_history_messages)} messages for chat {conversation_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to retrieve chat history for {conversation_id}: {e}",
                    exc_info=True,
                )

        if intent == "chit_chat" or not retrieval_active:
            log_reason = (
                "intent is chit_chat"
                if intent == "chit_chat"
                else "retrieval_active is False"
            )
            logger.info(f"Generating direct LLM response because {log_reason}.")

            try:
                simple_system_prompt = "You are a helpful and friendly assistant."
                history_to_format = chat_history_messages[-5:]
                history_lines = [
                    f"{msg['role']}: {msg['content']}" for msg in history_to_format
                ]
                conversation_history_block = (
                    "\n### Recent conversation\n" + "\n".join(history_lines) + "\n"
                    if history_lines
                    else ""
                )
                prompt = f"{simple_system_prompt}{conversation_history_block}\n\nUser: {question}\nAssistant:"
                logger.debug(f"Built direct response prompt (len={len(prompt)} chars)")

            except Exception as e_tmpl:
                logger.error(
                    f"Error preparing direct response prompt: {e_tmpl}", exc_info=True
                )
                history_lines = [
                    f"{msg['role']}: {msg['content']}"
                    for msg in chat_history_messages[-5:]
                ]
                conversation_history_block = (
                    "\n### Recent conversation\n" + "\n".join(history_lines) + "\n"
                    if history_lines
                    else ""
                )
                simple_system_prompt = "You are a helpful assistant."
                prompt = f"{simple_system_prompt}{conversation_history_block}\n\nUser: {question}\nAssistant:"

            try:
                answer = await self.llm_service.generate(prompt=prompt, max_tokens=150)
                processing_time = time.time() - start_time
                logger.info(f"Generated direct response in {processing_time:.2f}s")
                return {"answer": answer, "sources": []}
            # Catch TimeoutError specifically (G1)
            except asyncio.TimeoutError:
                processing_time = time.time() - start_time
                logger.error(
                    f"Direct LLM generation timed out after {processing_time:.2f}s."
                )
                return {
                    "answer": "Sorry, the request timed out while generating the response.",
                    "sources": [],
                    "error": "timeout",
                }
            except Exception as e:
                logger.error(f"Error during direct LLM generation: {e}", exc_info=True)
                return {
                    "answer": "Sorry, I encountered an error trying to respond.",
                    "sources": [],
                }

        logger.info(f"Proceeding with RAG pipeline for intent: {intent}")

        retrieved_sources = []
        final_context = "No relevant context found using Hybrid Retriever."
        active_collections_data = []

        collections_to_search = []
        global_collection_name_hr = CX_GLOBAL_COLLECTION
        collections_to_search.append(global_collection_name_hr)

        project_collection_name_hr = None
        if project_id:
            sanitized_project_id_hr = re.sub(r"[^a-zA-Z0-9_-]+", "_", project_id)[:50]
            project_collection_name_hr = (
                f"{USER_COLLECTION_PREFIX}{sanitized_project_id_hr}"
            )
            if project_collection_name_hr != global_collection_name_hr:
                collections_to_search.append(project_collection_name_hr)
            else:
                logger.debug(
                    f"Project collection '{project_collection_name_hr}' is same as global. Will search once."
                )

        for coll_name in collections_to_search:
            await self.document_service.set_collection(coll_name)
            retrieval_data = (
                self.document_service.get_bm25_retrieval_data_for_collection(coll_name)
            )
            if retrieval_data:
                doc_ids, corpus_texts, tokenized_corpus, bm25_instance = retrieval_data
                active_collections_data.append(
                    (coll_name, doc_ids, corpus_texts, tokenized_corpus, bm25_instance)
                )
            else:
                logger.warning(
                    f"Could not get retrieval data for collection '{coll_name}'. It will be skipped."
                )

        if not active_collections_data:
            logger.warning("No collections found with data for hybrid retrieval.")
        else:
            all_hybrid_results = []
            for (
                coll_name,
                doc_ids,
                corpus_texts,
                tokenized_corpus,
                _,
            ) in active_collections_data:
                if not corpus_texts or not doc_ids or not tokenized_corpus:
                    logger.warning(
                        f"Skipping collection '{coll_name}' due to missing corpus/ids/tokenized_corpus for HybridRetriever."
                    )
                    continue

                logger.info(
                    f"Instantiating HybridRetriever for collection: '{coll_name}' with {len(corpus_texts)} documents."
                )

                async def dense_search_for_collection(
                    query_str: str, k_dense: int
                ) -> List[Dict]:
                    raw_dense_results = await self.document_service.retrieve_documents(
                        query=query_str,
                        collection_name=coll_name,
                        limit=k_dense,
                        retrieval_mode="semantic",
                    )
                    adapted_results = []
                    for res_item in raw_dense_results:
                        adapted_results.append(
                            {
                                "id": res_item.get("id"),
                                "score": res_item.get("score", 0.0),
                                "text": res_item.get("page_content", ""),
                                "metadata": res_item.get("metadata", {}),
                            }
                        )
                    return adapted_results

                try:
                    hybrid_retriever = HybridRetriever(
                        corpus=corpus_texts,
                        doc_ids=doc_ids,
                        tokenized_corpus=tokenized_corpus,
                        dense_search_fn=dense_search_for_collection,
                        cross_encoder=(
                            self.cross_encoder
                            if self.cross_encoder
                            and self.context_optimizer.use_reranking
                            else None
                        ),
                    )

                    if self.cross_encoder and self.context_optimizer.use_reranking:
                        logger.info(
                            f"Performing hybrid search WITH CrossEncoder re-ranking for '{coll_name}'."
                        )
                        collection_results = (
                            await hybrid_retriever.hybrid_cross_encoder(
                                query=question,
                                M=self.hybrid_retriever_top_m,
                                K=self.hybrid_retriever_top_k,
                            )
                        )
                    else:
                        logger.info(
                            f"Performing flat hybrid search (RRF) for '{coll_name}'. CrossEncoder re-ranking is OFF."
                        )
                        collection_results = await hybrid_retriever.flat_hybrid_ranking(
                            query=question, k=self.hybrid_retriever_top_k
                        )
                    all_hybrid_results.extend(collection_results)
                    logger.info(
                        f"Retrieved {len(collection_results)} results from '{coll_name}' using HybridRetriever."
                    )
                except Exception as e_hr:
                    logger.error(
                        f"Error using HybridRetriever for collection '{coll_name}': {e_hr}",
                        exc_info=True,
                    )

            if len(active_collections_data) > 1:
                final_unique_results = {}
                for res in all_hybrid_results:
                    if (
                        res["id"] not in final_unique_results
                        or res["score"] > final_unique_results[res["id"]]["score"]
                    ):
                        final_unique_results[res["id"]] = res
                sorted_hybrid_results = sorted(
                    final_unique_results.values(),
                    key=lambda x: x["score"],
                    reverse=True,
                )
            else:
                sorted_hybrid_results = sorted(
                    all_hybrid_results, key=lambda x: x["score"], reverse=True
                )

            chunks_for_optimizer = []
            for item in sorted_hybrid_results[: self.hybrid_retriever_top_k]:
                chunks_for_optimizer.append(
                    {
                        "id": item.get("id"),
                        "page_content": item.get("text", ""),
                        "metadata": item.get("metadata", {}),
                        "score": item.get("score", 0.0),
                    }
                )

            retrieved_sources = chunks_for_optimizer

            if chunks_for_optimizer:
                logger.info(
                    f"Optimizing/reranking {len(chunks_for_optimizer)} combined chunks from Hybrid Retriever..."
                )
                final_context = await self.context_optimizer.optimize(
                    question=question,
                    documents=chunks_for_optimizer,
                    conversation_history=chat_history_messages,
                )
                logger.info(
                    f"Context optimized. Final context length: {len(final_context.split())} words."
                )
            else:
                logger.warning(
                    "Hybrid Retriever returned no results. Proceeding with empty context."
                )
                final_context = "No relevant context found by Hybrid Retriever."

        try:
            # Check template manager availability
            if not self.template_manager:
                logger.error("TemplateManager is not initialized in RagEngine.")
                raise RuntimeError("Template service not available.")

            # Build prompt (this can now raise TemplateRenderingError)
            prompt, history_used = await self._build_rag_prompt_with_history_truncation(
                question=question,
                context=final_context,
                chat_history_messages=chat_history_messages,
                project_id=project_id,
            )

            # Check LLM service availability
            if not self.llm_service:
                logger.error("LLMService is not initialized in RagEngine.")
                raise RuntimeError("LLM service not available.")

            # Generate response (this can raise TimeoutError)
            try:
                answer = await self.llm_service.generate(prompt=prompt)
                logger.debug("LLM response generated.")
            except asyncio.TimeoutError:
                processing_time = time.time() - start_time
                logger.error(
                    f"RAG LLM generation timed out after {processing_time:.2f}s."
                )
                # Return specific timeout error structure (as implemented before)
                return {
                    "answer": "Sorry, the request timed out while generating the response.",
                    "sources": [],
                    "error": "timeout",
                }

        # Catch template rendering errors specifically (G3)
        except TemplateRenderingError as e:
            logger.error(f"Template rendering failed: {e}", exc_info=True)
            return {
                "answer": f"Sorry, there was an issue preparing the request. Missing template variable: '{e.missing_key}' in template '{e.template_name}'.",
                "sources": [],
                "error": "template_render",
                "detail": f"Missing template key: {e.missing_key}",  # Pass detail for API layer
            }
        # Catch other general exceptions during prompt/generation
        except Exception as e_prompt_gen:
            logger.error(
                f"Error building prompt or generating response: {e_prompt_gen}",
                exc_info=True,
            )
            answer = "Sorry, I encountered an error while processing your request."
            retrieved_sources = []  # Ensure sources are empty
            # If we are here, set answer and return generic error structure at the end

        if hasattr(self, "_save_interaction_for_improvement"):
            self._save_interaction_for_improvement(question, answer, final_context)
        else:
            logger.warning("_save_interaction_for_improvement method not found.")

        processing_time = time.time() - start_time
        logger.info(f"RAG response generated in {processing_time:.2f}s")

        # Format sources for the response
        formatted_sources = []
        if retrieved_sources:
            logger.debug(
                f"Formatting {len(retrieved_sources)} retrieved sources for API response."
            )
            for src in retrieved_sources:
                metadata = src.get("metadata", {})
                # Use chunk_id if available, otherwise fall back to a generic "source_doc"
                chunk_id = metadata.get(
                    "chunk_id", "source_doc"
                )  # Should always be there from our chunker

                # Align with app.api.models.SearchResult
                formatted_sources.append(
                    {
                        "source": metadata.get(
                            "filename", metadata.get("source", chunk_id)
                        ),  # Pydantic 'source'
                        "score": src.get(
                            "score", src.get("final_score", 0.0)
                        ),  # Pydantic 'score'
                        "text_snippet": (
                            (src.get("page_content", "") or "")[
                                : self.SOURCE_SNIPPET_MAX_LENGTH
                            ]
                            + "..."
                            if src.get("page_content")
                            else ""
                        ),  # Pydantic 'text_snippet'
                        # Removed: "id": chunk_id,
                        # Removed: "filename": metadata.get("filename", metadata.get("source")), (now mapped to "source")
                        # Removed: "text_preview": (src.get("page_content", "") or "")[:200] + "...", (now mapped to "text_snippet")
                        # Removed: "doc_id": metadata.get("doc_id"),
                        # Removed: "chunk_id": chunk_id,
                        # Removed: "page_number": metadata.get("page_number")
                    }
                )
        logger.debug(f"Formatted sources: {formatted_sources}")

        # This return is now only reached if no TimeoutError or TemplateRenderingError occurred
        return {"answer": answer, "sources": formatted_sources}

    def _save_interaction_for_improvement(
        self, question: str, response: str, context: str
    ):
        try:
            improvement_dir = os.path.join("app", "data", "improvement")
            os.makedirs(improvement_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"interaction-{timestamp}.json"
            filepath = os.path.join(improvement_dir, filename)

            interaction_data = {
                "timestamp": time.time(),
                "question": question,
                "response": response,
                "context": context,
                "metadata": {
                    "context_length": len(context),
                    "response_length": len(response),
                },
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(interaction_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved interaction for improvement to {filepath}")
        except Exception as e:
            logger.warning(f"Error saving interaction for improvement: {str(e)}")

    async def retrieve_for_refinement(
        self, document: Dict[str, Any], project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant context for refining a given document.
        For now, this is a simple wrapper around the standard retriever.
        """
        if not document or not document.get("content"):
            logger.warning(
                "Cannot retrieve for refinement: Document content is missing."
            )
            return []

        doc_content = document["content"]
        doc_project_id = document.get(
            "project_id", project_id
        )  # Prefer doc's own project_id

        if not doc_project_id:
            logger.warning(
                f"Cannot retrieve for refinement of document {document.get('id')}: Project ID is missing."
            )
            return []

        logger.debug(
            f"Retrieving context for refinement of document {document.get('id')} in project {doc_project_id} using its content as query."
        )

        # Utilize the RetrieverAgent instance within RagEngine
        # Assuming RetrieverAgent is available as self.retriever_agent
        # or that RagEngine itself has a retrieve method directly calling document_service or similar.
        # Based on previous context, RagEngine has a RetrieverAgent.

        if not hasattr(self, "retriever_agent") or not self.retriever_agent:
            logger.error(
                "RetrieverAgent not initialized in RagEngine. Cannot retrieve for refinement."
            )
            return []

        try:
            # Pass the document content as the query.
            # We might want to limit the number of retrieved chunks or use specific settings.
            retrieved_chunks = await self.retriever_agent.retrieve(
                query=doc_content,  # Use full document content as query
                project_id=doc_project_id,
                limit=5,  # Arbitrary limit, can be configured
                collection_name=None,  # Use project default or global as per retriever logic
                include_global=True,  # Consider if global context is useful for refinement
            )
            logger.info(
                f"Retrieved {len(retrieved_chunks)} chunks for refining document {document.get('id')}"
            )
            return retrieved_chunks
        except Exception as e:
            logger.error(
                f"Error during retrieve_for_refinement for document {document.get('id')}: {e}",
                exc_info=True,
            )
            return []
