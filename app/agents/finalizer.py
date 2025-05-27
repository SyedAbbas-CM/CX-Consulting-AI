import logging
from typing import Any, Dict, List, Optional, Tuple

from app.core.llm_service import LLMService

logger = logging.getLogger(__name__)


class FinalizerAgent:
    """
    Agent responsible for generating the final answer based on the selected draft,
    critiques, and context, and extracting source information.
    """

    def __init__(self, llm_service: LLMService, temperature: float = 0.1):
        """
        Initializes the FinalizerAgent.

        Args:
            llm_service: The LLM service instance for generation.
            temperature: Very low temperature for focused final output.
        """
        self.llm_service = llm_service
        self.temperature = temperature
        logger.info(f"FinalizerAgent initialized with LLM Temp: {self.temperature}")

    async def finalize_answer(
        self,
        query: str,
        best_draft: str,
        critiques: List[str],  # All critiques for context
        retrieved_context: List[Dict[str, Any]],
        chat_history: Optional[List[Dict]] = None,  # Optional context
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generates the final answer by potentially refining the best draft and extracts sources.

        Args:
            query: The user's query.
            best_draft: The selected best draft answer.
            critiques: The critiques generated for all drafts.
            retrieved_context: The context chunks used.
            chat_history: Optional recent chat history.

        Returns:
            A tuple containing:
            - The final answer string.
            - A list of source dictionaries (e.g., {"filename": ..., "page": ...}).
            Returns ("Error generating final answer.", []) on failure.
        """
        logger.info(f"FinalizerAgent finalizing answer for query: '{query[:50]}...'")

        if not self.llm_service:
            logger.error("LLM Service not available to FinalizerAgent.")
            return "Error: LLM Service unavailable.", []

        # For now, we will directly use the best draft without LLM refinement based on critiques.
        # TODO: Implement LLM call to refine the best_draft using critiques.
        # Example refinement prompt structure (if implemented later):
        # prompt = f"""
        # User Query: {query}
        # Context: {context_str}
        # Selected Draft Answer:
        # {best_draft}
        # Critiques Provided:
        # {critique_str}
        # Task: Refine the Selected Draft Answer based on the critiques to produce the best possible final answer.
        # Ensure the final answer is accurate, complete according to context, clear, and directly addresses the query.
        # Final Answer:
        # """
        final_answer_content = best_draft
        logger.info(
            "FinalizerAgent using selected best draft directly (refinement step skipped)."
        )

        # --- Extract Sources ---
        # Simple source extraction: list metadata from all retrieved context chunks.
        # More sophisticated logic could involve checking which specific sources were cited
        # in the final_answer_content (if LLM refinement step included citation logic).
        sources = []
        if retrieved_context:
            for ctx in retrieved_context:
                metadata = ctx.get("metadata", {})
                source_info = {
                    "document_id": metadata.get("document_id"),
                    "filename": metadata.get("filename"),
                    "page_number": metadata.get("page_number"),
                    # Add other relevant metadata like chunk ID, score if needed
                    "score": ctx.get("score"),
                }
                # Avoid adding duplicate sources based on filename/doc_id (simple check)
                if not any(
                    s["document_id"] == source_info["document_id"]
                    for s in sources
                    if s.get("document_id")
                ):
                    sources.append(source_info)

        logger.info(f"Extracted {len(sources)} unique sources from context.")

        return final_answer_content, sources
