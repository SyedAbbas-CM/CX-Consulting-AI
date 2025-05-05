from typing import Dict, List, Any, Optional
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger("cx_consulting_ai.context_optimizer")

class ContextOptimizer:
    """Optimizer for refining retrieved context before sending to LLM."""
    
    def __init__(
        self,
        max_tokens: int = None,
        rerank_model: str = None
    ):
        """
        Initialize the context optimizer.
        
        Args:
            max_tokens: Maximum tokens for context
            rerank_model: Optional cross-encoder model for reranking
        """
        self.max_tokens = max_tokens or int(os.getenv("MAX_CHUNK_LENGTH_TOKENS", "4096"))
        self.rerank_model = rerank_model or os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Flag to enable/disable reranking
        self.use_reranking = os.getenv("USE_RERANKING", "true").lower() in ("true", "1", "t")
        
        # Initialize cross-encoder if needed
        if self.use_reranking:
            self._init_cross_encoder()
        
        logger.info(f"Context Optimizer initialized with max_tokens={self.max_tokens}, reranking={'enabled' if self.use_reranking else 'disabled'}")
    
    def _init_cross_encoder(self):
        """Initialize the cross-encoder for reranking."""
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading cross-encoder model: {self.rerank_model}")
            self.cross_encoder = CrossEncoder(self.rerank_model)
            logger.info("Cross-encoder loaded successfully")
        
        except ImportError:
            logger.warning("sentence-transformers not installed. Reranking will be disabled.")
            self.use_reranking = False
        
        except Exception as e:
            logger.error(f"Error loading cross-encoder: {str(e)}")
            self.use_reranking = False
    
    def optimize_sync(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]] = None
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
        logger.info(f"Optimizing context for question: {question}")
        
        # If no documents, return empty context
        if not documents:
            logger.warning("No documents provided for context optimization")
            return ""
        
        # Rerank documents if enabled
        if self.use_reranking and len(documents) > 1:
            documents = self._rerank_documents(question, documents)
        
        # Extract relevance scores for logging
        scores = [doc.get('score', doc.get('distance', 0)) for doc in documents]
        logger.debug(f"Document scores after optimization: {scores}")
        
        # Create context string
        context = self._create_context_string(documents)
        
        # Truncate if needed
        if len(context) > self.max_tokens * 4:  # Rough character estimate
            logger.warning(f"Context too long ({len(context)} chars), truncating")
            context = context[:self.max_tokens * 4]
        
        logger.info(f"Context optimized: {len(context)} chars, {len(documents)} documents")
        return context
    
    def _rerank_documents(
        self,
        question: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            question: The query
            documents: Retrieved documents
            
        Returns:
            Reranked documents
        """
        logger.info(f"Reranking {len(documents)} documents")
        
        try:
            # Prepare document pairs for reranking
            doc_pairs = [[question, doc['text']] for doc in documents]
            
            # Get scores from cross-encoder
            scores = self.cross_encoder.predict(doc_pairs)
            
            # Add scores to documents
            for i, score in enumerate(scores):
                documents[i]['score'] = float(score)
            
            # Sort by score (descending)
            documents = sorted(documents, key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Documents reranked successfully")
            return documents
        
        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}")
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
        
        logger.debug(f"Creating context string with char limit: {char_limit} (max_tokens: {self.max_tokens})")

        for i, doc in enumerate(documents):
            text = doc.get('page_content', '') 
            metadata = doc.get('metadata', {})
            source = metadata.get('source', f'doc_{i+1}') # Use index as fallback ID
            # Use a more stable ID if available, e.g., document_id
            source_id = metadata.get('document_id', source)
            
            # --- SOURCE INFO EMBEDDED --- 
            # Include source identifier directly in the text for the LLM
            doc_part = f"<source id=\"{source_id}\" name=\"{source}\">\n{text}\n</source>\n"
            part_length = len(doc_part)
            
            if current_length + part_length > char_limit:
                remaining_chars = char_limit - current_length - len(f"<source id=\"{source_id}\" name=\"{source}\">\n...\n</source>\n")
                if remaining_chars > 100: 
                    partial_text = text[:remaining_chars]
                    doc_part = f"<source id=\"{source_id}\" name=\"{source}\">\n{partial_text}...\n</source>\n"
                    context_parts.append(doc_part)
                    current_length += len(doc_part)
                    logger.debug(f"Adding partial document {i+1} (id: {source_id}), length: {len(doc_part)}")
                else:
                    logger.debug(f"Skipping document {i+1} (id: {source_id}) as it exceeds char limit ({part_length} chars). Current total: {current_length}")
                break 
            
            context_parts.append(doc_part)
            current_length += part_length
            logger.debug(f"Added document {i+1} (id: {source_id}), length: {part_length}. Cumulative length: {current_length}")
        
        final_context = "\n\n".join(context_parts)
        logger.debug(f"Final context string length: {len(final_context)}")
        return final_context
    
    def enhance_with_conversation(
        self,
        context: str,
        conversation_history: List[Dict[str, Any]]
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
                role = interaction.get('role')
                content = interaction.get('content')
                if role and content:
                    if role == 'user':
                        history_str += f"User: {content}\n"
                    elif role == 'assistant':
                        history_str += f"Assistant: {content}\n\n"
            # --- END FIX --- 
        
        # Combine with context
        return f"{history_str}\n\n{context}"
    
    async def optimize(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]] = None,
        followup_context: str = None
    ) -> str:
        """
        Async optimize context for the given question.
        
        Args:
            question: The query or question
            documents: Retrieved documents
            conversation_history: Optional conversation history
            followup_context: Optional context from follow-up questions
            
        Returns:
            Optimized context string
        """
        logger.info(f"Optimizing context for question: {question}")
        
        # If no documents, return empty context
        if not documents:
            logger.warning("No documents provided for context optimization")
            return ""
        
        # Rerank documents if enabled
        if self.use_reranking and len(documents) > 1:
            documents = self._rerank_documents(question, documents)
        
        # Extract relevance scores for logging
        scores = [doc.get('score', doc.get('distance', 0)) for doc in documents]
        logger.debug(f"Document scores after optimization: {scores}")
        
        # Create context string
        context = self._create_context_string(documents)
        
        # Enhance with conversation history if available
        if conversation_history:
            context = self.enhance_with_conversation(context, conversation_history)
            
        # Add followup context if available
        if followup_context:
            context += f"\n\nAdditional context from previous conversation: {followup_context}"
        
        # Truncate if needed
        if len(context) > self.max_tokens * 4:  # Rough character estimate
            logger.warning(f"Context too long ({len(context)} chars), truncating")
            context = context[:self.max_tokens * 4]
        
        logger.info(f"Context optimized: {len(context)} chars, {len(documents)} documents")
        return context
    
    def extract_followup_context(
        self,
        question: str,
        conversation_history: List[Dict[str, Any]]
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
        last_response = conversation_history[-1].get('assistant', '')
        
        # For simplicity, just return the last response
        # This could be enhanced with more sophisticated logic in the future
        return last_response 