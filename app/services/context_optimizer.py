import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

from app.core.config import get_settings

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger("cx_consulting_ai.context_optimizer")


class ContextOptimizer:
    """Optimizer for refining retrieved context before sending to LLM."""

    def __init__(
        self, max_tokens: Optional[int] = None, rerank_model: Optional[str] = None
    ):
        """
        Initialize the context optimizer.

        Args:
            max_tokens: Maximum tokens for context. If None, derived from settings.
            rerank_model: Optional cross-encoder model for reranking. If None, from settings.
        """
        settings = get_settings()

        # Max tokens for context optimizer (derived if not provided)
        if max_tokens is None:
            llm_max_model_len = settings.MAX_MODEL_LEN
            prompt_buffer_tokens = settings.CONTEXT_OPTIMIZER_PROMPT_BUFFER
            derived_max_tokens = llm_max_model_len - prompt_buffer_tokens
            self.max_tokens = (
                derived_max_tokens if derived_max_tokens > 0 else 1024
            )  # Fallback if buffer is too large
        else:
            self.max_tokens = max_tokens

        self.rerank_model = rerank_model or settings.CROSS_ENCODER_MODEL
        self.use_reranking = settings.USE_RERANKING
        self.rerank_min_score_threshold = settings.RERANK_MIN_SCORE_THRESHOLD

        if self.use_reranking:
            self._init_cross_encoder()

        logger.info(
            f"Context Optimizer initialized with max_tokens={self.max_tokens}, "
            f"reranking={'enabled' if self.use_reranking else 'disabled'}, "
            f"rerank_threshold={self.rerank_min_score_threshold if self.use_reranking else 'N/A'}"
        )

    def _init_cross_encoder(self):
        """Initialize the cross-encoder for reranking."""
        try:
            from sentence_transformers import CrossEncoder

            # O3 Fix: Determine device for cross-encoder (same logic as EmbeddingManager)
            device = self._detect_device()
            logger.info(
                f"Loading cross-encoder model: {self.rerank_model} onto device: {device}"
            )
            # O3 Fix: Pass device to CrossEncoder
            self.cross_encoder = CrossEncoder(self.rerank_model, device=device)
            logger.info("Cross-encoder loaded successfully")

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Reranking will be disabled."
            )
            self.use_reranking = False

        except Exception as e:
            logger.error(f"Error loading cross-encoder: {str(e)}")
            self.use_reranking = False

    # Need device detection logic here, copied from EmbeddingManager for now
    # Consider moving to a shared utility if used in multiple places
    def _detect_device(self) -> str:
        """Detect the best available device."""
        try:
            import torch

            if torch.backends.mps.is_available():
                logger.info("MPS is available, using Apple Silicon accelerator")
                return "mps"
            elif torch.cuda.is_available():
                logger.info("CUDA is available, using GPU")
                return "cuda"
            else:
                logger.info("Using CPU for cross-encoder")
                return "cpu"
        except:
            logger.info("Could not detect PyTorch devices, using CPU for cross-encoder")
            return "cpu"

    def optimize_sync(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]] = None,
    ) -> str:
        """
        Optimize context for the given question (synchronous version).

        Args:
            question: The query or question
            documents: Retrieved documents
            conversation_history: Optional conversation history

        Returns:
            Optimized context string
        """
        logger.info(f"Optimizing context SYNC for question: {question}")

        # If no documents, return empty context
        if not documents:
            logger.warning("No documents provided for context optimization")
            return ""

        # Rerank documents if enabled
        # NOTE: Reranking is disabled in optimize_sync because _rerank_documents is now async.
        # Use the async optimize() method for reranking capabilities.
        if self.use_reranking and len(documents) > 1:
            logger.warning(
                "Reranking in optimize_sync is disabled as _rerank_documents is now async. Use async optimize() for reranking."
            )
            # documents = self._rerank_documents(question, documents) # This would block/fail

        # Extract relevance scores for logging
        scores = [doc.get("score", doc.get("distance", 0)) for doc in documents]
        logger.debug(f"Document scores after sync optimization: {scores}")

        # Create context string
        context = self._create_context_string(documents)

        # Truncate if needed
        if len(context) > self.max_tokens * 4:  # Rough character estimate
            logger.warning(f"Context too long ({len(context)} chars), truncating")
            context = context[: self.max_tokens * 4]

        logger.info(
            f"SYNC Context optimized: {len(context)} chars, {len(documents)} documents"
        )
        return context

    async def _rerank_documents(
        self, question: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder. Now async.

        Args:
            question: The query
            documents: Retrieved documents

        Returns:
            Reranked documents
        """
        if not self.cross_encoder:
            logger.warning("Cross-encoder not available for reranking. Skipping.")
            return documents

        logger.info(f"Reranking {len(documents)} documents asynchronously")

        try:
            # Prepare document pairs for reranking
            doc_pairs = []
            valid_docs_indices = []  # Keep track of original indices of docs with text
            for i, doc in enumerate(documents):
                text_to_rerank = doc.get("text", doc.get("page_content"))
                if text_to_rerank is not None:
                    doc_pairs.append([question, text_to_rerank])
                    valid_docs_indices.append(i)
                else:
                    logger.warning(
                        f"Document missing 'text' and 'page_content' for reranking: {doc.get('id', 'Unknown ID')}"
                    )

            if not doc_pairs:
                logger.warning(
                    "No valid document pairs to rerank after checking for text content."
                )
                return documents  # Return original list if no pairs could be formed

            # Get scores from cross-encoder
            loop = asyncio.get_event_loop()

            # O3 Fix B-5: Only use executor if model is on CPU
            if self.cross_encoder.device.type == "cpu":
                logger.debug("Running CPU cross-encoder prediction in executor.")
                scores = await loop.run_in_executor(
                    None, self.cross_encoder.predict, doc_pairs, show_progress_bar=False
                )
            else:
                # If on GPU/MPS, assume predict is efficient enough to call directly
                # (Sentence Transformers handles internal batching)
                logger.debug(
                    f"Running cross-encoder prediction directly on device: {self.cross_encoder.device.type}"
                )
                # Note: self.cross_encoder.predict is sync, but we await a potentially sync call
                # This is generally okay in asyncio if the sync call is fast (like GPU inference often is).
                # If it *blocks* significantly, further optimization might be needed,
                # but this avoids the unnecessary thread hop for GPU.
                scores = self.cross_encoder.predict(doc_pairs, show_progress_bar=False)

            # Normalize scores (e.g., to 0-1 range or handle negative scores appropriately)
            # This is a simple normalization; more sophisticated methods might be needed.
            # Check if scores is not None and has elements before processing
            if scores is not None and scores.size > 0:
                min_score = np.min(scores)  # Use np.min for numpy arrays
                max_score = np.max(scores)  # Use np.max for numpy arrays
                if max_score == min_score:
                    normalized_scores = (
                        np.ones_like(scores) if max_score > 0 else np.zeros_like(scores)
                    )
                else:
                    normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                # Handle empty or None scores: assign empty array or default scores
                normalized_scores = np.array([])

            # Add scores back to the *original* documents using the valid_docs_indices
            reranked_documents = []
            # Ensure we only iterate up to the number of available scores
            num_scores_available = len(normalized_scores)
            for i in range(len(valid_docs_indices)):
                original_doc_idx = valid_docs_indices[i]
                # Construct the document dictionary with guaranteed keys from original document
                doc_to_append = {
                    "page_content": documents[original_doc_idx].get(
                        "page_content", ""
                    ),  # Ensure page_content exists
                    "metadata": documents[original_doc_idx].get(
                        "metadata", {}
                    ),  # Ensure metadata exists
                }
                if i < num_scores_available:
                    doc_to_append["score"] = float(normalized_scores[i])
                else:
                    # This case indicates that cross_encoder.predict returned fewer scores than input pairs.
                    logger.warning(
                        f"Mismatch between number of scores ({num_scores_available}) and valid documents ({len(valid_docs_indices)}). "
                        f"Assigning default score to document originally at index {original_doc_idx} (valid_docs_indices[{i}])."
                    )
                    doc_to_append["score"] = 0.0  # Assign a default score
                reranked_documents.append(doc_to_append)

            # Sort by score (descending)
            reranked_documents = sorted(
                reranked_documents,
                key=lambda x: x.get("score", -float("inf")),
                reverse=True,
            )

            logger.info(f"Documents reranked successfully asynchronously")
            return reranked_documents

        except Exception as e:
            logger.error(
                f"Error reranking documents asynchronously: {str(e)}", exc_info=True
            )
            return documents  # Return original documents on error

    def _create_context_string(self, documents: List[Dict[str, Any]]) -> str:
        """
        Create a context string from documents, respecting max token limits.
        Each part includes source information.

        Args:
            documents: List of documents (should be sorted by relevance)

        Returns:
            Formatted context string within approximate token limits.
        """
        context_parts = []
        current_length = 0
        char_limit = self.max_tokens * 4

        logger.debug(
            f"Creating context string with char limit: {char_limit} (max_tokens: {self.max_tokens})"
        )

        for i, doc in enumerate(documents):
            text = doc.get("page_content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", f"doc_{i+1}")  # Use index as fallback ID
            # Use a more stable ID if available, e.g., document_id
            source_id = metadata.get("document_id", source)

            # --- SOURCE INFO EMBEDDED ---
            # Include source identifier directly in the text for the LLM
            doc_part = f'<source id="{source_id}" name="{source}">\n{text}\n</source>\n'
            part_length = len(doc_part)

            if current_length + part_length > char_limit:
                remaining_chars = (
                    char_limit
                    - current_length
                    - len(
                        f'<source id="{source_id}" name="{source}">\n...\n</source>\n'
                    )
                )
                if remaining_chars > 100:
                    partial_text = text[:remaining_chars]
                    doc_part = f'<source id="{source_id}" name="{source}">\n{partial_text}...\n</source>\n'
                    context_parts.append(doc_part)
                    current_length += len(doc_part)
                    logger.debug(
                        f"Adding partial document {i+1} (id: {source_id}), length: {len(doc_part)}"
                    )
                else:
                    logger.debug(
                        f"Skipping document {i+1} (id: {source_id}) as it exceeds char limit ({part_length} chars). Current total: {current_length}"
                    )
                break

            context_parts.append(doc_part)
            current_length += part_length
            logger.debug(
                f"Added document {i+1} (id: {source_id}), length: {part_length}. Cumulative length: {current_length}"
            )

        final_context = "\n\n".join(context_parts)
        logger.debug(f"Final context string length: {len(final_context)}")
        return final_context

    def enhance_with_conversation(
        self, context: str, conversation_history: List[Dict[str, Any]]
    ) -> str:
        """
        Enhance context with conversation history.

        Args:
            context: Base context string
            conversation_history: Conversation history

        Returns:
            Enhanced context
        """
        if not conversation_history:
            return context

        # Format conversation history (use last 3 messages)
        history_str = "Conversation History:\n"

        for interaction in conversation_history[-3:]:
            # --- FIX: Check role and use content ---
            if isinstance(interaction, dict):
                role = interaction.get("role")
                content = interaction.get("content")
                if role and content:
                    if role == "user":
                        history_str += f"User: {content}\n"
                    elif role == "assistant":
                        history_str += f"Assistant: {content}\n\n"
            # --- END FIX ---

        # Combine with context
        return f"{history_str}\n\n{context}"

    async def optimize(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]] = None,
        followup_context: str = None,
    ) -> str:
        """
        Optimize context asynchronously, including reranking and threshold filtering.

        Args:
            question: The query or question
            documents: Retrieved documents (expected to have 'text' or 'page_content')
            conversation_history: Optional conversation history
            followup_context: Optional context extracted from history

        Returns:
            Optimized context string
        """
        logger.info(f"Optimizing context ASYNC for question: {question[:50]}...")

        if not documents:
            logger.warning("No documents provided for async context optimization")
            return ""

        # Rerank documents if enabled
        if self.use_reranking and len(documents) > 1:
            try:
                reranked_documents = await self._rerank_documents(question, documents)
                logger.info(
                    f"Reranking completed, {len(reranked_documents)} docs remain."
                )
            except Exception as e_rerank:
                logger.error(f"Error during reranking: {e_rerank}", exc_info=True)
                # O3 Fix: Fallback to original documents
                reranked_documents = documents
                logger.warning("Rerank failed - falling back to original ranking.")
        else:
            # If reranking is disabled or not enough docs, use original list
            reranked_documents = documents

        # Extract relevance scores for logging
        scores = [doc.get("score", 0) for doc in reranked_documents]
        logger.debug(f"Document scores after optimization (rerank+filter): {scores}")

        # Create context string from filtered documents
        context = self._create_context_string(reranked_documents)

        # Optional: Enhance with conversation history context
        if conversation_history and followup_context:
            # We might need to decide how to prioritize/merge this with document context
            # For now, let's just append it if there's space, or maybe prepend?
            # Prepending might be better for recency.
            logger.debug("Adding followup context from conversation history.")
            context = followup_context + "\n\n---\n\n" + context
            # Re-check length after adding history? Or assume history context is small?

        # Note: Truncation now happens inside _create_context_string
        # if len(context) > self.max_tokens * 4: # Rough estimate
        #     logger.warning(f"Context potentially too long after adding history ({len(context)} chars), check truncation logic.")
        # context = context[: self.max_tokens * 4]

        logger.info(
            f"ASYNC Context optimized: {len(context)} chars, used {len(reranked_documents)} documents"
        )
        return context

    def extract_followup_context(
        self, question: str, conversation_history: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Extract context from previous conversations that may be relevant for follow-up questions.

        Args:
            question: Current user question
            conversation_history: Previous conversation history

        Returns:
            Extracted context or None
        """
        if not conversation_history or len(conversation_history) < 2:
            return None

        # Get the last assistant response
        last_response = conversation_history[-1].get("assistant", "")

        # For simplicity, just return the last response
        # This could be enhanced with more sophisticated logic in the future
        return last_response
