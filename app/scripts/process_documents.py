#!/usr/bin/env python
"""
Document Processing Script

This script processes documents in the documents directory and adds them to the vector store.
"""
import os
import sys
import glob
import asyncio
import logging
from pathlib import Path

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.services.document_service import DocumentService
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("document_processor")

async def process_documents():
    """Process all documents in the documents directory."""
    # Initialize document service
    doc_service = DocumentService(
        documents_dir=os.path.join("app", "data", "documents"),
        chunked_dir=os.path.join("app", "data", "chunked"),
        vectorstore_dir=os.path.join("app", "data", "vectorstore")
    )
    
    logger.info("Document service initialized")
    
    # Get document files
    documents_dir = os.path.join("app", "data", "documents")
    document_files = []
    
    for ext in ["txt", "pdf", "docx"]:
        document_files.extend(glob.glob(os.path.join(documents_dir, f"*.{ext}")))
    
    logger.info(f"Found {len(document_files)} documents to process")
    
    # Process each document
    for doc_path in document_files:
        file_name = os.path.basename(doc_path)
        file_type = file_name.split(".")[-1]
        
        logger.info(f"Processing document: {file_name}")
        
        # Check if the document has already been processed
        doc_id = Path(doc_path).stem
        chunk_path = os.path.join("app", "data", "chunked", f"{doc_id}.json")
        
        if os.path.exists(chunk_path):
            logger.info(f"Document {file_name} already processed, skipping")
            continue
        
        # Add document to vector store
        try:
            success = await doc_service.add_document(
                document_url=doc_path,
                document_type=file_type,
                metadata={"source": file_name}
            )
            
            if success:
                logger.info(f"Document {file_name} processed successfully")
            else:
                logger.error(f"Failed to process document {file_name}")
        
        except Exception as e:
            logger.error(f"Error processing document {file_name}: {str(e)}")
    
    # Print summary
    doc_count = doc_service.get_document_count()
    logger.info(f"Document processing complete. Total documents in vector store: {doc_count}")

if __name__ == "__main__":
    asyncio.run(process_documents()) 