from typing import Dict, Any, List, Optional, Tuple
import os
import time
import uuid
import logging
from dotenv import load_dotenv
import json
from tenacity import retry, stop_after_attempt, wait_fixed # Import tenacity for retries
from app.core.llm_service import LLMService
from app.services.memory_manager import MemoryManager
from app.services.document_service import DocumentService

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger("cx_consulting_ai.rag_engine")

# Define constants for RAG classification - REFINED PROMPT
NEEDS_RAG_PROMPT = """Does the following query require specific information found ONLY in internal documents (like client details, project plans, past proposals, specific consulting frameworks) to be answered accurately? 
Focus ONLY on whether *internal documents* are strictly necessary. Do not consider general knowledge or conversation history.
Answer only YES or NO.

Query: {query}"""
CLASSIFICATION_MAX_TOKENS = 10
CLASSIFICATION_TEMP = 0.0

class RagEngine:
    """Main RAG engine implementation for CX consulting AI."""
    
    def __init__(
        self,
        llm_service,
        document_service,
        template_manager,
        context_optimizer,
        memory_manager
    ):
        """
        Initialize the RAG engine.
        
        Args:
            llm_service: LLM service for generation
            document_service: Document service for retrieval
            template_manager: Prompt template manager
            context_optimizer: Context optimizer for refinement
            memory_manager: Memory manager for conversation history
        """
        self.llm_service = llm_service
        self.document_service = document_service
        self.template_manager = template_manager
        self.context_optimizer = context_optimizer
        self.memory_manager = memory_manager
        
        # Load max documents per query from environment
        self.max_documents = int(os.getenv("MAX_DOCUMENTS_PER_QUERY", "5"))
        logger.info(f"RAG Engine initialized with max_documents={self.max_documents}")
    
    async def process_document(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process a document for the knowledge base.
        
        Args:
            content: Document content as bytes
            filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing document: {filename}")
            
            # Determine document type from extension
            document_type = filename.split('.')[-1].lower()
            allowed_types = ['pdf', 'txt', 'docx', 'doc', 'csv', 'xlsx']
            
            if document_type not in allowed_types:
                raise ValueError(f"Unsupported file type: {document_type}. Supported types: {', '.join(allowed_types)}")
            
            # Safety check for content size
            content_size_mb = len(content) / (1024 * 1024)
            max_size_mb = 50  # 50 MB max
            
            if content_size_mb > max_size_mb:
                logger.warning(f"Document size ({content_size_mb:.2f} MB) exceeds recommended limit of {max_size_mb} MB")
            
            # Save content to a temporary file
            temp_path = os.path.join("/tmp", filename)
            with open(temp_path, "wb") as f:
                f.write(content)
            
            # Process the file
            document_id = str(uuid.uuid4())
            metadata = {
                "source": filename,
                "document_id": document_id,
                "upload_time": time.time(),
                "size_bytes": len(content)
            }
            
            try:
                # Document processing
                document_text = await self.document_service._extract_text(temp_path, document_type)
                chunks = self.document_service._create_chunks(document_text)
                chunk_count = len(chunks)
                
                # Enforce maximum number of chunks
                MAX_CHUNKS = 1000
                if chunk_count > MAX_CHUNKS:
                    logger.warning(f"Document produced {chunk_count} chunks, limiting to {MAX_CHUNKS}")
                    chunks = chunks[:MAX_CHUNKS]
                    chunk_count = MAX_CHUNKS
                
                # Add to vectorstore
                await self.document_service._add_chunks_to_vectorstore(
                    chunks=chunks,
                    metadata=metadata
                )
                
                logger.info(f"Successfully processed document {filename} into {chunk_count} chunks")
                
                # Return processing results
                return {
                    "filename": filename,
                    "document_id": document_id,
                    "chunks_created": chunk_count,
                    "status": "success"
                }
            except Exception as processing_error:
                logger.error(f"Error during document processing: {str(processing_error)}")
                return {
                    "filename": filename,
                    "document_id": document_id,
                    "chunks_created": 0,
                    "status": "error",
                    "error": str(processing_error)
                }
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temp file: {str(cleanup_error)}")
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            # Return a structured error response
            return {
                "filename": filename,
                "document_id": None,
                "chunks_created": 0,
                "status": "error",
                "error": str(e)
            }
    
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1)) # Add retry logic
    async def _needs_rag(self, question: str, conversation_history: List[Dict[str, Any]]) -> bool:
        """Determine if RAG is needed using LLM classification."""
        try:
            normalized_question = question.lower().strip()
            
            # === HEURISTICS FIRST ===
            # Simple heuristic: very short messages or greetings
            if len(normalized_question.split()) < 3 or normalized_question in ["hi", "hello", "thanks", "thank you", "ok", "okay", "bye", "goodbye"]:
                logger.info(f"Query '{question}' deemed too short or conversational, skipping RAG.")
                return False

            # Heuristic: Meta-questions about the AI itself
            meta_question_triggers = [
                "who are you", "what are you", "what can you do", "what is your purpose", 
                "what are you made for", "what are you supposed to do", "tell me about yourself"
            ]
            if any(trigger in normalized_question for trigger in meta_question_triggers):
                 logger.info(f"Query '{question}' identified as meta-question, skipping RAG.")
                 return False

            # Heuristic: Simple follow-up questions (can be expanded)
            follow_up_triggers = ["tell me more", "explain that", "why did you say"]
            if conversation_history and len(normalized_question.split()) < 10 and any(term in normalized_question for term in follow_up_triggers):
                 logger.info(f"Query '{question}' seems like a simple follow-up, skipping RAG.")
                 return False
            # === END HEURISTICS ===

            # If no heuristic matched, use LLM to classify
            prompt = NEEDS_RAG_PROMPT.format(query=question)
            logger.info(f"Classifying query relevance for RAG using LLM: '{question}'")
            
            # Use generate with low temp and max tokens for YES/NO
            response = await self.llm_service.generate(
                prompt=prompt, 
                temperature=CLASSIFICATION_TEMP, 
                max_tokens=CLASSIFICATION_MAX_TOKENS
            )
            
            response_text = response.strip().upper()
            logger.info(f"RAG classification result: {response_text}")
            
            # Check for YES/NO, stricter matching
            if response_text.startswith("YES"):
                logger.info("RAG classification = YES")
                return True
            elif response_text.startswith("NO"):
                logger.info("RAG classification = NO")
                return False
            else:
                # If classification is unclear, default to NOT using RAG for safety (avoids unnecessary lookups)
                # You could change this default back to True if preferred
                logger.warning(f"Unclear RAG classification response: '{response_text}'. Defaulting to NO.")
                return False
                
        except Exception as e:
            logger.error(f"Error during RAG classification: {str(e)}. Defaulting to NO.", exc_info=True)
            # Default to NOT using RAG if classification fails
            return False

    async def ask(self, question: str, conversation_id: Optional[str] = None, project_id: Optional[str] = None) -> str:
        """Process a question, deciding whether to use RAG."""
        try:
            # Start logging interaction
            logger.info(f"Processing question: {question}")
            if conversation_id:
                logger.info(f"Using conversation ID: {conversation_id}")
            if project_id:
                logger.info(f"Using project ID: {project_id}")

            # Get conversation history (needed for both paths)
            conversation_history = []
            if conversation_id:
                conversation_history = await self.memory_manager.get_conversation(conversation_id)
                logger.info(f"Found conversation history with {len(conversation_history)} interactions")
            
            # === CONDITIONAL RAG CHECK ===
            use_rag = await self._needs_rag(question, conversation_history)
            # ==============================

            if use_rag:
                logger.info("RAG required for this query.")
                # --- RAG PATH --- 
                followup_context = None
                if conversation_history: # Check history again for context extraction
                    followup_context = self.context_optimizer.extract_followup_context(question, conversation_history)
                    if followup_context:
                        logger.info(f"Extracted followup context: {followup_context[:100]}...")
                
                # Retrieve relevant documents
                retrieved_docs = await self.document_service.retrieve_documents(
                    query=question,
                    limit=self.max_documents,
                    project_id=project_id
                )
                logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {question}")
                for i, doc in enumerate(retrieved_docs):
                    metadata = doc.get("metadata", {})
                    page_content = doc.get("page_content", "")
                    score = doc.get("score", 0.0)
                    doc_source = metadata.get("source", "unknown")
                    doc_id = metadata.get("document_id", "unknown")
                    doc_score = score
                    logger.info(f"Document {i+1}: source={doc_source}, id={doc_id}, score={doc_score:.4f}")
                    logger.debug(f"Document {i+1} content preview: {page_content[:100]}...")
                
                # Optimize the context 
                optimized_context = await self.context_optimizer.optimize(
                    question=question,
                    documents=retrieved_docs,
                    conversation_history=conversation_history, # Still useful for optimizer
                    followup_context=followup_context
                )
                logger.info(f"Optimized RAG context length: {len(optimized_context)} characters")
                
                # Prepare message list for RAG generation
                chat_history_messages = []
                if conversation_history:
                    for entry in conversation_history:
                        if isinstance(entry, dict):
                            role = entry.get('role')
                            content = entry.get('content')
                            if role and content:
                                chat_history_messages.append({"role": role, "content": content})
                
                if chat_history_messages and chat_history_messages[-1]['role'] == 'user':
                     chat_history_messages = chat_history_messages[:-1]
                
                # Construct the final user message including context - SIMPLIFIED & WITH CITATION INSTRUCTION
                final_user_prompt = f"""Context with source information:
{optimized_context}

Question: {question}

Answer the question based *only* on the provided context. 
Cite the source ID for each piece of information you use by referencing the `id` attribute in the `<source>` tags (e.g., "According to source [doc-abc], the NPS is...")."""
                # logger.info(f"Final user prompt for chat: {final_user_prompt}") # Optional: Log the simplified prompt

                messages_for_llm = []
                # Optional: Add a system prompt specific to RAG tasks if needed
                # messages_for_llm.append({\"role\": \"system\", \"content\": \"You are a helpful assistant answering based on provided documents.\"})
                messages_for_llm.extend(chat_history_messages)
                messages_for_llm.append({"role": "user", "content": final_user_prompt})
                
                logger.info(f"Sending {len(messages_for_llm)} messages to LLM (RAG path).")
                # --- END RAG PATH --- 

            else:
                logger.info("Skipping RAG, using direct generation.")
                # --- NO RAG PATH --- 
                # Prepare message list using only history and current question
                chat_history_messages = []
                if conversation_history:
                    for entry in conversation_history[-10:]: # Limit history for direct chat
                        if isinstance(entry, dict):
                            role = entry.get('role')
                            content = entry.get('content')
                            if role and content:
                                chat_history_messages.append({"role": role, "content": content})
                
                # Ensure alternation for direct chat too
                if chat_history_messages and chat_history_messages[-1]['role'] == 'user':
                     chat_history_messages = chat_history_messages[:-1]
                
                messages_for_llm = []
                # System prompt is removed here; handled by LLMService massage if needed
                messages_for_llm.extend(chat_history_messages)
                # Add the user question directly
                messages_for_llm.append({"role": "user", "content": question})
                
                logger.info(f"Sending {len(messages_for_llm)} messages to LLM (No RAG path).")
                # --- END NO RAG PATH ---

            # Generate response using the prepared message list
            # Pass temperature explicitly, maybe slightly higher for non-RAG?
            generation_temp = 0.1 if use_rag else 0.3 # Example: slightly more creative if not RAG
            response = await self.llm_service.generate(
                messages=messages_for_llm, 
                temperature=generation_temp
            )
            
            # Log the LLM response
            logger.info(f"LLM response generated with length: {len(response)} characters using temp={generation_temp}")
            logger.info(f"LLM response preview: {response[:200]}...")
            
            # Save interaction (maybe only save RAG interactions? TBD)
            try:
                # Decide whether to save based on use_rag flag or context presence
                if use_rag and optimized_context:
                     self._save_interaction_for_improvement(question, response, optimized_context)
            except Exception as e:
                logger.warning(f"Failed to save interaction for improvement: {str(e)}")
                
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG processing: {str(e)}", exc_info=True)
            return "I encountered an error while processing your question. Please try again or contact support if the issue persists."
            
    def _save_interaction_for_improvement(self, question: str, response: str, context: str):
        """
        Save the interaction to a file for model improvement.
        
        Args:
            question: The user's question
            response: The LLM's response
            context: The context used for generation
        """
        try:
            # Create directory if it doesn't exist
            improvement_dir = os.path.join("app", "data", "improvement")
            os.makedirs(improvement_dir, exist_ok=True)
            
            # Generate a timestamped filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"interaction-{timestamp}.json"
            filepath = os.path.join(improvement_dir, filename)
            
            # Create data structure
            interaction_data = {
                "timestamp": time.time(),
                "question": question,
                "response": response,
                "context": context,
                "metadata": {
                    "context_length": len(context),
                    "response_length": len(response)
                }
            }
            
            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(interaction_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved interaction for improvement to {filepath}")
        except Exception as e:
            logger.warning(f"Error saving interaction for improvement: {str(e)}")
    
    async def create_proposal(self, client_info: str, requirements: str, conversation_id: str = None) -> Tuple[str, str, int]:
        """DEPRECATED: Use create_cx_strategy_from_template instead."""
        logger.warning("Deprecated method create_proposal called. Use create_cx_strategy_from_template.")
        # Basic placeholder or redirect to new method logic if possible
        # For now, return a simple deprecation message
        response = "This method is deprecated. Use the CX Strategy generator." 
        tokens_used = 0 # Placeholder
        # Ensure conversation_id is returned correctly
        if not conversation_id:
            conversation_id = self.memory_manager.create_conversation()
        self.memory_manager.add_message(conversation_id, "user", f"Generate deprecated proposal for {client_info}")
        self.memory_manager.add_message(conversation_id, "assistant", response)
        return response, conversation_id, tokens_used

    async def create_roi_analysis(self, client_info: str, project_details: str, conversation_id: str = None) -> Tuple[str, str, int]:
        """DEPRECATED: Use create_roi_analysis_from_template instead."""
        logger.warning("Deprecated method create_roi_analysis called. Use create_roi_analysis_from_template.")
        response = "This method is deprecated. Use the ROI Analysis generator."
        tokens_used = 0
        if not conversation_id:
            conversation_id = self.memory_manager.create_conversation()
        self.memory_manager.add_message(conversation_id, "user", f"Generate deprecated ROI for {client_info}")
        self.memory_manager.add_message(conversation_id, "assistant", response)
        return response, conversation_id, tokens_used
        
    async def create_journey_map(self, persona: str, scenario: str, conversation_id: str = None) -> Tuple[str, str, int]:
        """DEPRECATED: Use create_journey_map_from_template instead."""
        logger.warning("Deprecated method create_journey_map called. Use create_journey_map_from_template.")
        response = "This method is deprecated. Use the Journey Map generator."
        tokens_used = 0
        if not conversation_id:
            conversation_id = self.memory_manager.create_conversation()
        self.memory_manager.add_message(conversation_id, "user", f"Generate deprecated Journey Map for {persona}")
        self.memory_manager.add_message(conversation_id, "assistant", response)
        return response, conversation_id, tokens_used

    # --- NEW TEMPLATE-BASED METHODS --- 

    async def create_cx_strategy_from_template(
        self,
        request_data: Dict[str, Any], # Corresponds to CXStrategyRequest model fields
        conversation_id: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """Generates a CX Strategy document based on the v2 template."""
        logger.info(f"Generating CX Strategy for client: {request_data.get('client_name')}")
        # TODO: Implement LLM call using the request_data and the CX Strategy template structure
        
        # Placeholder response using the template structure
        placeholder_content = f"""
# 🚀 Customer Experience Strategy: {request_data.get('client_name', 'N/A')}

## 1. Vision & Principles

**CX Vision:** ‹Inspiring one‑liner›

**Guiding Principles:**

*   ‹Principle 1› – definition & decision rule
*   ‹Principle 2› – definition & decision rule

## 2. Strategic Context

*   **Business Goals:** {request_data.get('business_goals', '‹Growth, Efficiency, ESG›')}
*   **Market Trends:** {request_data.get('market_trends', '‹AI support, Self‑service›')}
*   **Challenges Addressed:** {request_data.get('challenges', 'N/A')}
*   _(Other fields from template omitted for brevity)_

## 3. Segments & Priority Journeys

*   _(Placeholder: Add segmentation & journey matrix based on input/analysis)_

## 4. Initiative Portfolio

*   _(Placeholder: Add initiatives based on input/analysis)_

## 5. Roadmap (0‑24m)

*   _(Placeholder: Gantt or table)_

## 6. Governance & Operating Model

*   _(Placeholder: CX Council details & RACI)_

## 7. Measurement Framework

*   _(Placeholder: Strategic, Financial, Operational KPIs)_

## 8. Business Case Snapshot

*   _(Placeholder: Investment, Benefit, ROI)_

## 9. Risks & Mitigations

*   _(Placeholder: Top 5 risks)_

*Last updated {datetime.now().strftime('%Y-%m-%d')} | Template v2*
"""

        generated_content = placeholder_content # Replace with actual LLM response later
        tokens_used = 0 # Replace with actual token count later
        
        # Ensure conversation exists and add messages
        if not conversation_id:
            # Need project_id if creating a new conversation - assume it's in request_data
            project_id = request_data.get('project_id')
            conversation_id = self.memory_manager.create_conversation(project_id=project_id)
            logger.info(f"Created new conversation {conversation_id} for CX Strategy generation.")

        user_prompt = f"Generate CX Strategy for {request_data.get('client_name')}. Challenges: {request_data.get('challenges')}"
        self.memory_manager.add_message(conversation_id, "user", user_prompt)
        self.memory_manager.add_message(conversation_id, "assistant", generated_content)

        return generated_content, conversation_id, tokens_used

    async def create_roi_analysis_from_template(
        self,
        request_data: Dict[str, Any], # Corresponds to ROIAnalysisRequest model fields
        conversation_id: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """Generates an ROI Analysis document based on the v2 template."""
        logger.info(f"Generating ROI Analysis for client: {request_data.get('client_name')}")
        # TODO: Implement LLM call using request_data and the ROI template

        placeholder_content = f"""
# 💰 CX ROI Analysis: {request_data.get('client_name', 'N/A')} - {request_data.get('project_description', 'Project')}

## 1. Executive Snapshot

Initiative: {request_data.get('project_description', 'N/A')} | Period: ‹3 yrs› | ROI: ‹X %› | Payback: ‹Z months›

*_(Placeholder: One-sentence business rationale)_

## 2. Baseline & Cost of Inaction

*   **Current Metrics:** {request_data.get('current_metrics', 'N/A')}
*   _(Placeholder: Add Cost of Inaction table)_

## 3. Proposed Solution & Benefits

*   _(Placeholder: Improvement Levers & Non-Financial Benefits)_

## 4. Investment Summary

*   _(Placeholder: CapEx/OpEx table)_

## 5. Financial Model

*   NPV @ ‹WACC%›: ‹€›
*   IRR: ‹%›
*   Payback: ‹Months›

## 6. Sensitivity & Risks

*   _(Placeholder: Scenarios & mitigations)_

## 7. Recommendation

*   _(Placeholder: Go / No-go)_

*Generated {datetime.now().strftime('%Y-%m-%d')} | Template v2*
"""
        generated_content = placeholder_content
        tokens_used = 0
        
        if not conversation_id:
            project_id = request_data.get('project_id')
            conversation_id = self.memory_manager.create_conversation(project_id=project_id)
            logger.info(f"Created new conversation {conversation_id} for ROI Analysis generation.")
            
        user_prompt = f"Generate ROI Analysis for {request_data.get('client_name')}, Project: {request_data.get('project_description')}. Current Metrics: {request_data.get('current_metrics')}"
        self.memory_manager.add_message(conversation_id, "user", user_prompt)
        self.memory_manager.add_message(conversation_id, "assistant", generated_content)
        
        return generated_content, conversation_id, tokens_used

    async def create_journey_map_from_template(
        self,
        request_data: Dict[str, Any], # Corresponds to JourneyMapRequest model fields
        conversation_id: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """Generates a Customer Journey Map document based on the v2 template."""
        logger.info(f"Generating Journey Map for persona: {request_data.get('persona')}")
        # TODO: Implement LLM call using request_data and the Journey Map template
        
        placeholder_content = f"""
# ✨ Customer Journey Map: {request_data.get('persona', 'N/A')}

## 1. Journey Header

*   **Journey Name:** {request_data.get('journey_name', '‹Default Journey Name›')}
*   **Persona:** {request_data.get('persona', 'N/A')}
*   **Scenario / Trigger:** {request_data.get('scenario', 'N/A')}
*   **Desired Outcome:** ‹What success looks like›
*   _(Other fields omitted)_

## 2. Pre‑Journey Context

*   _(Placeholder: Customer & Business Context)_

## 3. Stage Block (repeat as needed)

**Stage 1 – ‹Discovery›**

*   **Customer Actions:** 1. ‹Action› 2. ‹Action›
*   **Touchpoints:** ‹Website - Web - CMS›
*   **Customer Thoughts:** "‹I need to solve X...›"
*   **Emotions (1‑5):** 😐 3
*   **Pain Points:** • ‹Confusing navigation›
*   **Opportunities:** • ‹Clearer value prop›
*   **Backstage / Enablers:** • People: Marketing • Tech: CMS
*   **Success Metrics:** Bounce Rate: 60% → 40%
*   **Moment of Truth?** ☐ No

_(Placeholder: Add more stage blocks based on analysis)_

## 4. Emotional Journey

*   _(Placeholder: Plot or list emotions across stages)_

## 5. Insights & Roadmap

*   _(Placeholder: Critical Pain Points, MoTs, Unmet Needs)_
*   _(Placeholder: Improvement Roadmap Table)_

*Created {datetime.now().strftime('%Y-%m-%d')} | Template v2*
"""
        generated_content = placeholder_content
        tokens_used = 0
        
        if not conversation_id:
            project_id = request_data.get('project_id')
            conversation_id = self.memory_manager.create_conversation(project_id=project_id)
            logger.info(f"Created new conversation {conversation_id} for Journey Map generation.")
        
        user_prompt = f"Generate Journey Map for Persona: {request_data.get('persona')}, Scenario: {request_data.get('scenario')}"
        self.memory_manager.add_message(conversation_id, "user", user_prompt)
        self.memory_manager.add_message(conversation_id, "assistant", generated_content)
        
        return generated_content, conversation_id, tokens_used 