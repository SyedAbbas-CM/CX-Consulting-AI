import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.core.llm_service import LLMService

logger = logging.getLogger(__name__)


class DraftingAgent:
    """
    Agent responsible for generating multiple initial drafts based on a query
    and retrieved context, using a Chain-of-Thought approach.
    """

    def __init__(self, llm_service: LLMService, temperature: float = 0.7):
        """
        Initializes the DraftingAgent.

        Args:
            llm_service: The LLM service instance for generation.
            temperature: The temperature setting for the LLM generation.
        """
        self.llm_service = llm_service
        self.temperature = temperature
        logger.info(f"DraftingAgent initialized with LLM Temp: {self.temperature}")

    async def generate_drafts(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        project_id: Optional[str] = None,  # Optional context
        chat_history: Optional[List[Dict]] = None,  # Optional context
    ) -> List[str]:
        """
        Generates three distinct drafts for an answer using a CoT prompt.

        Args:
            query: The user's query.
            retrieved_context: A list of context chunks (dictionaries with 'text', 'metadata').
            project_id: Optional project ID for context.
            chat_history: Optional recent chat history.

        Returns:
            A list containing three generated draft answers. Returns empty list on failure.
        """
        logger.info(
            f"DraftingAgent generating drafts for query: '{query[:50]}...' for project: {project_id}"
        )

        if not self.llm_service:
            logger.error("LLM Service not available to DraftingAgent.")
            return []

        # --- Prepare Context String ---
        context_str = self.build_context_string(retrieved_context)
        if not context_str:
            logger.warning("No context provided to DraftingAgent.")
            context_str = "No context provided."  # Explicitly state no context

        # --- Prepare History String (Optional) ---
        history_str = ""
        if chat_history:
            history_str = "\n".join(
                [
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    for msg in chat_history[-4:]
                ]
            )  # Last 4 turns
            history_str = f"\n\nRecent Conversation History:\n{history_str}"

        # --- Construct CoT Prompt ---
        # This is a basic example, enhance as needed
        prompt = f"""
User Query: {query}
{history_str}

Available Context Documents:
---
{context_str}
---

Task: Generate 3 distinct draft answers to the user query based *only* on the provided context documents and conversation history. Use a Chain-of-Thought approach for each draft.

Chain of Thought Process (for each draft):
1.  Identify the key information required by the user query.
2.  Scan the provided context documents and history for relevant sentences or paragraphs.
3.  Synthesize the relevant information into a coherent answer.
4.  Ensure the answer directly addresses the user query.
5.  Format the answer clearly.
6.  Critically evaluate if the answer relies *only* on the provided context and history. If not, revise.

Generate the 3 drafts below. Each draft should start with "Draft X:" on a new line.

Draft 1:
[Provide draft 1 here]

Draft 2:
[Provide draft 2 here]

Draft 3:
[Provide draft 3 here]
"""

        # --- Generate Drafts using LLM ---
        generated_text = ""
        try:
            logger.debug(f"Sending prompt to LLM (length: {len(prompt)} chars)")
            # Assuming llm_service.generate handles async correctly
            generated_text = await self.llm_service.generate(
                prompt=prompt,
                temperature=self.temperature,
                # Add other relevant parameters like max_tokens if needed, based on LLMService
                # max_tokens=1500 # Example: limit output length
            )
            logger.debug(f"LLM generated text length: {len(generated_text)}")

        except Exception as e:
            logger.error(f"Error generating drafts from LLM: {e}", exc_info=True)
            return []

        # --- Parse Drafts ---
        drafts = []
        try:
            # Simple parsing based on "Draft X:" delimiter
            parts = generated_text.split("\nDraft ")
            if len(parts) >= 4:  # Expecting initial split + 3 drafts
                # parts[0] might be empty or contain reasoning before Draft 1
                drafts.append(parts[1].split(":", 1)[1].strip())  # Draft 1 content
                drafts.append(parts[2].split(":", 1)[1].strip())  # Draft 2 content
                drafts.append(parts[3].split(":", 1)[1].strip())  # Draft 3 content
            else:
                logger.warning(
                    f"Could not parse 3 drafts from LLM output. Found {len(parts)-1} parts. Output: {generated_text[:500]}..."
                )
                # Fallback: maybe the whole output is one draft? Or split by double newline?
                # For now, return empty if parsing fails strict expectation.
                return []

        except Exception as e:
            logger.error(f"Error parsing generated drafts: {e}", exc_info=True)
            logger.debug(f"Failed parsing LLM Raw Output:\n{generated_text}")
            return []

        # --- Validate and Return ---
        if len(drafts) == 3:
            logger.info(f"Successfully generated {len(drafts)} drafts.")
            return drafts
        else:
            logger.warning(f"Expected 3 drafts, but parsed {len(drafts)}.")
            return []  # Return empty list if parsing didn't yield exactly 3

    def build_context_string(self, contexts: List[Dict[str, Any]]) -> str:
        context_strings = []
        # Ensure each context has 'text', 'score', and 'metadata' (with 'filename')
        # Default to empty strings or N/A if keys are missing to prevent errors.
        for i, ctx in enumerate(contexts):
            context_strings.append(
                f"Source {i+1} (Score: {ctx.get('score', 'N/A'):.2f}, "
                f"File: {ctx.get('metadata', {}).get('filename', 'Unknown')}): "
                f"{ctx.get('text', '')}"
            )
        return "\n---\n".join(context_strings)
