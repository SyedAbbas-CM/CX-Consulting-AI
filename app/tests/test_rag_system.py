#!/usr/bin/env python3
"""
Test script for the complete RAG system in the CX Consulting Agent.
This script tests the integration of document ingestion, retrieval, and LLM response generation.
"""

import os
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_rag_system")

# Import necessary components with correct paths
from app.services.document_service import DocumentService
from app.services.memory_manager import MemoryManager
from app.services.rag_engine import RagEngine
from app.core.llm_service import LLMService
from app.templates.prompt_template import PromptTemplateManager as TemplateManager
from app.services.context_optimizer import ContextOptimizer

async def test_rag_system():
    """Test the complete RAG system including document ingestion, retrieval and LLM response generation."""
    
    # Create temporary directories for testing
    temp_base = tempfile.mkdtemp()
    docs_dir = os.path.join(temp_base, "documents")
    chunks_dir = os.path.join(temp_base, "chunks")
    vector_dir = os.path.join(temp_base, "vector_db")
    
    # Create test project
    project_id = "test-project-001"
    
    # Ensure directories exist
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)
    
    try:
        # Step 1: Initialize services
        logger.info("Step 1: Initializing services")
        
        # Document service
        doc_service = DocumentService(
            documents_dir=docs_dir,
            chunked_dir=chunks_dir,
            vectorstore_dir=vector_dir,
            chunk_size=512,
            chunk_overlap=50
        )
        
        # Memory manager
        memory_manager = MemoryManager(max_items=10)
        
        # Template manager
        template_manager = TemplateManager()
        
        # LLM service (use default model path from env vars)
        llm_service = LLMService()
        
        # Context optimizer
        context_optimizer = ContextOptimizer(max_tokens=384)
        
        # RAG engine
        rag_engine = RagEngine(
            llm_service=llm_service,
            document_service=doc_service,
            memory_manager=memory_manager,
            template_manager=template_manager,
            context_optimizer=context_optimizer
        )
        
        logger.info("All services initialized successfully")
        
        # Step 2: Create test documents
        logger.info("Step 2: Creating test documents")
        
        # Create a test document about customer experience
        test_cx_doc_path = os.path.join(docs_dir, "cx_strategy.txt")
        with open(test_cx_doc_path, "w") as f:
            f.write("""
            # Customer Experience Strategy Guide
            
            Customer experience (CX) refers to how a business engages with its customers at every point of their 
            journey—from marketing to sales to customer service and everywhere in between. It's the sum of all 
            experiences a customer has with your brand.
            
            ## Key Components of CX Strategy
            
            1. **Customer Understanding**: Develop deep insights into your customers' needs, preferences, and pain points
            2. **Journey Mapping**: Visualize the entire customer journey to identify touchpoints and opportunities
            3. **Technology Integration**: Implement technologies that enhance the customer experience
            4. **Employee Engagement**: Ensure employees are trained and motivated to deliver exceptional experiences
            5. **Continuous Improvement**: Regularly measure and refine the customer experience
            
            ## Measuring CX Success
            
            The most common metrics for measuring customer experience include:
            
            - Net Promoter Score (NPS): Measures customer loyalty and likelihood to recommend
            - Customer Satisfaction Score (CSAT): Measures satisfaction with a specific interaction
            - Customer Effort Score (CES): Measures ease of service experience
            - Churn Rate: Rate at which customers stop doing business with a company
            """)
        
        # Create a test document about ROI analysis
        test_roi_doc_path = os.path.join(docs_dir, "roi_analysis.txt")
        with open(test_roi_doc_path, "w") as f:
            f.write("""
            # CX ROI Analysis Framework
            
            Return on Investment (ROI) analysis for Customer Experience initiatives helps organizations 
            quantify the business value of improving customer experiences.
            
            ## Key ROI Metrics for CX
            
            1. **Revenue Impact**: Increased sales from improved customer satisfaction and loyalty
            2. **Cost Reduction**: Lower service costs due to fewer complaints and support requests
            3. **Customer Retention**: Value of reduced churn and extended customer lifetime
            4. **Word-of-Mouth Value**: Customer acquisition through referrals and positive reviews
            5. **Price Premium**: Ability to command higher prices due to superior experiences
            
            ## Calculating CX ROI
            
            ROI = (Net Benefits from CX / Cost of CX Initiatives) × 100%
            
            Net Benefits include both revenue increases and cost savings attributable to CX improvements.
            Costs include technology, training, research, and operational changes needed for CX initiatives.
            
            ## Example ROI Calculation
            
            A company invests $500,000 in CX improvements and sees:
            - Additional revenue: $800,000
            - Cost savings: $300,000
            
            ROI = ($1,100,000 - $500,000) / $500,000 × 100% = 120%
            """)
        
        # Step 3: Add documents to the system
        logger.info("Step 3: Adding documents to the system")
        
        # Add CX strategy document as global
        success = await doc_service.add_document(
            document_url=test_cx_doc_path,
            document_type="txt",
            metadata={"title": "CX Strategy Guide", "author": "CX Consulting Team"},
            is_global=True,
            project_id=None
        )
        assert success, "Failed to add global CX strategy document"
        
        # Add ROI analysis document to specific project
        success = await doc_service.add_document(
            document_url=test_roi_doc_path,
            document_type="txt",
            metadata={"title": "ROI Analysis Framework", "author": "CX Consulting Team"},
            is_global=False,
            project_id=project_id
        )
        assert success, "Failed to add project-specific ROI document"
        
        logger.info("✅ Documents added successfully")
        
        # Step 4: Test document retrieval
        logger.info("Step 4: Testing document retrieval")
        
        # Query for CX information (should retrieve from global document)
        cx_docs = await doc_service.retrieve_documents(
            query="What are the key components of a customer experience strategy?",
            limit=3
        )
        assert len(cx_docs) > 0, "Failed to retrieve CX strategy documents"
        logger.info(f"Retrieved {len(cx_docs)} documents for CX strategy query")
        
        # Query for ROI information with project filter
        roi_docs = await doc_service.retrieve_documents(
            query="How do you calculate ROI for customer experience initiatives?",
            limit=3,
            project_id=project_id
        )
        assert len(roi_docs) > 0, "Failed to retrieve ROI documents with project filter"
        logger.info(f"Retrieved {len(roi_docs)} documents for ROI query with project filter")
        
        # Step 5: Test RAG engine end-to-end
        logger.info("Step 5: Testing RAG engine end-to-end")
        
        # Set up conversation
        conversation_id = "test-conversation-001"
        await memory_manager.add_interaction(
            conversation_id=conversation_id,
            user_message="I need help with customer experience strategy.",
            assistant_message="I'd be happy to help with customer experience strategy. What would you like to know?"
        )
        
        # Test query 1: CX Strategy components
        response1 = await rag_engine.ask(
            question="What are the key components of a customer experience strategy?",
            conversation_id=conversation_id,
            project_id=project_id
        )
        
        assert response1 and len(response1) > 100, "Failed to generate meaningful response for CX strategy question"
        logger.info("✅ Generated response for CX strategy question")
        logger.info(f"Response preview: {response1[:150]}...")
        
        # Add follow-up question to conversation
        await memory_manager.add_interaction(
            conversation_id=conversation_id,
            user_message="What are the key components of a customer experience strategy?",
            assistant_message=response1
        )
        
        # Test query 2: ROI calculation (should retrieve from project-specific document)
        response2 = await rag_engine.ask(
            question="How do I calculate ROI for our CX initiatives?",
            conversation_id=conversation_id,
            project_id=project_id
        )
        
        assert response2 and len(response2) > 100, "Failed to generate meaningful response for ROI question"
        logger.info("✅ Generated response for ROI calculation question")
        logger.info(f"Response preview: {response2[:150]}...")
        
        # Test query 3: Following up without specific document content
        await memory_manager.add_interaction(
            conversation_id=conversation_id,
            user_message="How do I calculate ROI for our CX initiatives?",
            assistant_message=response2
        )
        
        response3 = await rag_engine.ask(
            question="Can you give me an example calculation?",
            conversation_id=conversation_id,
            project_id=project_id
        )
        
        assert response3 and len(response3) > 100, "Failed to generate meaningful response for follow-up question"
        logger.info("✅ Generated response for follow-up question")
        logger.info(f"Response preview: {response3[:150]}...")
        
        # Print conversation history
        conv_history = await memory_manager.get_conversation(conversation_id)
        logger.info(f"Conversation had {len(conv_history)} interactions")
        
        logger.info("All RAG system tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
        
    finally:
        # Clean up
        logger.info("Cleaning up test resources")
        await memory_manager.clear_conversation(conversation_id)
        shutil.rmtree(temp_base)

if __name__ == "__main__":
    asyncio.run(test_rag_system()) 