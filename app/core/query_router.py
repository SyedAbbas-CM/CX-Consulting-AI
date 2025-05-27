"""
Query Router to classify user queries and determine the appropriate processing path.
"""

import logging
import re
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)

# Simple keyword-based classification
GREETING_TRIGGERS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "greetings",
    "sup",
    "what's up",
    "howdy",
}
FAREWELL_TRIGGERS = {
    "bye",
    "goodbye",
    "see you",
    "later",
    "farewell",
    "ttyl",
    "take care",
}
# More could be added, e.g., for thanks, apologies, etc.

# Could also use a small model or more sophisticated heuristics here in the future
# For now, simple length and keyword based.


# Define QueryIntent as a proper Enum
class QueryIntent(str, Enum):
    RAG = "rag"
    CHIT_CHAT = "chit_chat"
    DIRECT = "direct"  # Added based on routes.py usage
    RAG_KEYWORD_SEARCH = "rag_keyword_search"  # Added based on routes.py usage
    DELIVERABLE = (
        "deliverable"  # Kept for potential future use, though routes handle it
    )
    META_DOC_CHECK = "meta_doc_check"  # New intent
    UNKNOWN = "unknown"


def classify_query_intent(query: str) -> QueryIntent:
    """
    Classifies the user's query intent.

    Args:
        query: The user's query string.

    Returns:
        The classified intent as a QueryIntent enum member.
    """
    normalized_query = query.lower().strip()

    if not normalized_query:
        return QueryIntent.CHIT_CHAT  # Return Enum member

    # Check for meta document check query
    # Regex to match phrases like "did you read the document I uploaded?"
    meta_doc_check_pattern = r".*(did|have).* (you )?read .*document .*upload(ed)?[?]?"
    if re.fullmatch(meta_doc_check_pattern, normalized_query, re.IGNORECASE):
        logger.debug(f"Query classified as META_DOC_CHECK: '{query}'")
        return QueryIntent.META_DOC_CHECK

    # Check for greetings
    if normalized_query in GREETING_TRIGGERS:
        logger.debug(f"Query classified as GREETING (chit_chat): '{query}'")
        return QueryIntent.CHIT_CHAT  # Return Enum member

    # Check for farewells
    if normalized_query in FAREWELL_TRIGGERS:
        logger.debug(f"Query classified as FAREWELL (chit_chat): '{query}'")
        return QueryIntent.CHIT_CHAT  # Return Enum member

    # Example for deliverable (can be expanded with keywords from RagEngine.classify)
    # This is a simplified version. RagEngine's classify is more comprehensive for deliverables.
    # Note: The /ask route currently uses RAG for deliverables, this might need adjustment if DELIVERABLE intent should be handled differently.
    deliverable_keywords_simple = [
        "generate",
        "create",
        "build",
        "proposal",
        "report",
        "deck",
    ]
    if (
        any(keyword in normalized_query for keyword in deliverable_keywords_simple)
        and len(normalized_query.split()) > 2
    ):  # Avoid single word triggers
        logger.debug(f"Query classified as DELIVERABLE (potential): '{query}'")
        # Returning RAG for now as per current route logic for /ask
        # return QueryIntent.DELIVERABLE
        return QueryIntent.RAG

    # If query is very short and not a greeting/farewell, could be simple chit-chat or quick question
    # This threshold can be adjusted
    if len(normalized_query.split()) <= 3:
        logger.debug(f"Query classified as SHORT (chit_chat): '{query}'")
        return QueryIntent.CHIT_CHAT  # Return Enum member

    # Default to RAG for more complex queries
    # Future: Could add logic here for DIRECT or RAG_KEYWORD_SEARCH based on query patterns
    logger.debug(f"Query classified as RAG (default): '{query}'")
    return QueryIntent.RAG  # Return Enum member
