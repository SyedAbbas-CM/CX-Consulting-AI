#!/usr/bin/env python3
"""
Test script for document ingestion and retrieval in the CX Consulting Agent.
This script tests the document service's ability to add and retrieve documents.
"""

import os
import asyncio
import logging
import tempfile
import shutil
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_document_ingestion")

# Import document service
from app.services.document_service import DocumentService

async def test_document_ingestion():
    """Test document ingestion and retrieval functionality."""
    
    # Create temporary directories for testing
    temp_base = tempfile.mkdtemp()
    docs_dir = os.path.join(temp_base, "documents")
    chunks_dir = os.path.join(temp_base, "chunks")
    vector_dir = os.path.join(temp_base, "vector_db")
    
    # Ensure directories exist
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)
    
    try:
        # Create document service
        doc_service = DocumentService(
            documents_dir=docs_dir,
            chunked_dir=chunks_dir,
            vectorstore_dir=vector_dir,
            chunk_size=512,
            chunk_overlap=50
        )
        
        logger.info("Document service initialized for testing")
        
        # Create test document with sample content
        test_doc_path = os.path.join(docs_dir, "test_document.txt")
        with open(test_doc_path, "w") as f:
            f.write("""
            # Customer Experience Strategy
            
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
            
            ## Benefits of Strong CX
            
            - Increased customer loyalty and retention
            - Higher customer lifetime value
            - Positive word-of-mouth and referrals
            - Competitive differentiation
            - Reduced service costs
            """)
        
        # Test 1: Add document with project ID set to None
        logger.info("TEST 1: Adding document with project_id=None")
        success = await doc_service.add_document(
            document_url=test_doc_path,
            document_type="txt",
            metadata={"title": "CX Strategy Guide", "author": "CX Consulting Team"},
            is_global=True,
            project_id=None
        )
        
        assert success, "Failed to add document with project_id=None"
        logger.info("✅ Successfully added document with project_id=None")
        
        # Test 2: Add document with a specific project ID
        logger.info("TEST 2: Adding document with specific project_id")
        success = await doc_service.add_document(
            document_url=test_doc_path,
            document_type="txt",
            metadata={"title": "Project CX Strategy Guide", "author": "CX Project Team"},
            is_global=False,
            project_id="test-project-123"
        )
        
        assert success, "Failed to add document with specific project_id"
        logger.info("✅ Successfully added document with specific project_id")
        
        # Test 3: Retrieve documents with a simple query
        logger.info("TEST 3: Retrieving documents with query")
        docs = await doc_service.retrieve_documents(
            query="What is customer experience?",
            limit=3
        )
        
        assert len(docs) > 0, "No documents retrieved for query"
        logger.info(f"✅ Retrieved {len(docs)} documents for query")
        
        # Test 4: Retrieve documents with project filter
        logger.info("TEST 4: Retrieving documents with project filter")
        project_docs = await doc_service.retrieve_documents(
            query="customer experience strategy",
            limit=3,
            project_id="test-project-123"
        )
        
        assert len(project_docs) > 0, "No documents retrieved with project filter"
        logger.info(f"✅ Retrieved {len(project_docs)} documents with project filter")
        
        # Test 5: Check document count
        logger.info("TEST 5: Checking document count")
        doc_count = doc_service.get_document_count()
        
        # We expect at least 2 documents (or more if chunks were created)
        assert doc_count >= 2, f"Unexpected document count: {doc_count}"
        logger.info(f"✅ Document count is {doc_count} (expected 2 or more due to chunking)")
        
        # Print details of first document from each query
        if docs:
            logger.info("\nSample retrieved document:")
            logger.info(f"ID: {docs[0]['id']}")
            logger.info(f"Metadata: {docs[0]['metadata']}")
            logger.info(f"Text snippet: {docs[0]['text'][:100]}...")
        
        logger.info("\nAll tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
        
    finally:
        # Clean up temporary directories
        shutil.rmtree(temp_base)
        logger.info("Cleaned up test directories")

if __name__ == "__main__":
    asyncio.run(test_document_ingestion()) 