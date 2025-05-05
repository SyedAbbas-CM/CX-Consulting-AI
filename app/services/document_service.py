from typing import Dict, List, Any, Optional, Tuple
import os
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import json
import aiohttp
import shutil
import time
import numpy as np
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger("cx_consulting_ai.document_service")

class DocumentService:
    """Service for document processing and retrieval."""
    
    def __init__(
        self,
        documents_dir: str,
        chunked_dir: str,
        vectorstore_dir: str,
        embedding_model: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the document service.
        
        Args:
            documents_dir: Directory containing source documents
            chunked_dir: Directory for storing chunked documents
            vectorstore_dir: Directory for the vector database
            embedding_model: Embedding model name
            chunk_size: Maximum chunk size
            chunk_overlap: Chunk overlap size
        """
        # Set directories
        self.documents_dir = documents_dir
        self.chunked_dir = chunked_dir
        self.vectorstore_dir = vectorstore_dir
        
        # Create directories if they don't exist
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.chunked_dir, exist_ok=True)
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        
        # Set embedding model and chunking parameters
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.chunk_size = chunk_size or int(os.getenv("MAX_CHUNK_SIZE", "512"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "50"))
        
        # Vector database type (Chroma or FAISS)
        self.vector_db_type = os.getenv("VECTOR_DB_TYPE", "chroma").lower()
        
        # Initialize embedding function *first*
        self._init_embeddings()
        # Then initialize vector store (which might need the dimension)
        self._init_vector_store()
        
        logger.info(f"Document Service initialized with {self.vector_db_type} vector store")
    
    def _init_embeddings(self):
        """Initialize the embedding function with support for multiple backends."""
        try:
            # Import the embedding manager
            from app.utils.embedding_manager import create_embedding_manager
            
            # Get embedding type and model name from environment
            embedding_type = os.getenv("EMBEDDING_TYPE", "bge").lower()
            model_name = self.embedding_model
            
            if embedding_type == "bge":
                # Override with BGE-specific model if needed
                model_name = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-en-v1.5")
            
            # Create embedding manager - REMOVED dimension argument
            self.embedding_manager = create_embedding_manager(
                model_type=embedding_type,
                model_name=model_name,
                use_offline_fallback=True
            )
            
            # Get embedding functions
            self.embed_documents = self.embedding_manager.get_embedding_function(for_queries=False)
            self.embed_query = self.embedding_manager.get_embedding_function(for_queries=True)
            
            logger.info(f"Embedding initialized: Type={embedding_type}, Model={model_name}, Dim={self.embedding_manager.dimension_size}")
                
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            logger.warning("Falling back to local embedding function")
            self._init_local_fallback()
    
    def _init_local_fallback(self):
        """Initialize with a simple local embedding function as fallback."""
        import hashlib
        
        logger.info("Using simple local embedding function (offline mode)")
        
        # Define a simple embedding function using hash values
        def simple_embedding(text):
            if isinstance(text, list):
                return [simple_embedding(t) for t in text]
            
            # Create a hash of the text
            hash_object = hashlib.md5(text.encode())
            hash_hex = hash_object.hexdigest()
            
            # Convert hash to a vector of floats
            vector = []
            for i in range(0, len(hash_hex), 2):
                if i+2 <= len(hash_hex):
                    hex_pair = hash_hex[i:i+2]
                    value = int(hex_pair, 16) / 255.0  # Normalize to [0, 1]
                    vector.append(value)
            
            # Pad to 384 dimensions for a reasonably sized vector
            vector = vector + [0.0] * (384 - len(vector))
            return vector
        
        # Set embedding functions
        self.embed_documents = lambda texts: simple_embedding(texts)
        self.embed_query = lambda text: simple_embedding(text)
        
        logger.info("Local embedding function initialized")
    
    def _init_vector_store(self):
        """Initialize the vector store."""
        if self.vector_db_type == "chroma":
            try:
                import chromadb
                
                # Create a custom embedding function for Chroma
                class CustomEmbeddingFunction:
                    def __init__(self, embed_fn):
                        self.embed_fn = embed_fn
                    
                    def __call__(self, input):
                        # Convert the input to the format expected by our embed function
                        return self.embed_fn(input)
                
                # Create persistent client
                self.chroma_client = chromadb.PersistentClient(path=self.vectorstore_dir)
                
                # Get or create collection with our custom embedding function
                self.collection = self.chroma_client.get_or_create_collection(
                    name="cx_documents",
                    embedding_function=CustomEmbeddingFunction(self.embed_documents)
                )
                
                logger.info(f"Chroma DB initialized with collection: cx_documents")
            
            except ImportError:
                logger.error("chromadb not installed. Please install it: pip install chromadb")
                raise
        
        elif self.vector_db_type == "faiss":
            try:
                import faiss
                
                # Path to save the FAISS index
                self.faiss_index_path = os.path.join(self.vectorstore_dir, "faiss_index.bin")
                self.faiss_docstore_path = os.path.join(self.vectorstore_dir, "faiss_docstore.json")
                
                # <-- Get dimension from manager -->
                embedding_dim = self.embedding_manager.dimension_size
                if not embedding_dim:
                     # This should only happen if fallback occurred and failed somehow
                     logger.error("Could not determine embedding dimension for FAISS index creation!")
                     raise ValueError("Embedding dimension is unknown")
                # <-- End Get dimension -->

                # Check if index exists
                if os.path.exists(self.faiss_index_path) and os.path.exists(self.faiss_docstore_path):
                    # Load existing index
                    self.index = faiss.read_index(self.faiss_index_path)
                    
                    # Load document store
                    with open(self.faiss_docstore_path, 'r') as f:
                        self.docstore = json.load(f)
                    
                    # <-- Check loaded index dimension -->
                    if self.index.d != embedding_dim:
                         logger.warning(f"Loaded FAISS index dimension ({self.index.d}) does not match model dimension ({embedding_dim}). Re-initializing index.")
                         self.index = faiss.IndexFlatL2(embedding_dim)
                         self.docstore = {"documents": [], "ids": []}
                    else:
                         logger.info(f"Loaded existing FAISS index (Dim: {self.index.d}) with {self.index.ntotal} vectors")
                    # <-- End Check dimension -->
                
                else:
                    # Create new index - using L2 distance
                    self.index = faiss.IndexFlatL2(embedding_dim)
                    
                    # Initialize document store
                    self.docstore = {"documents": [], "ids": []}
                    
                    logger.info(f"Created new FAISS index with dimension {embedding_dim}")
                
                logger.info("FAISS initialized")
            
            except ImportError:
                logger.error("faiss not installed. Please install it: pip install faiss-cpu")
                raise
        
        else:
            logger.error(f"Unsupported vector database type: {self.vector_db_type}")
            raise ValueError(f"Unsupported vector database type: {self.vector_db_type}")
    
    async def add_document(
        self,
        document_url: str,
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        is_global: bool = False,
        project_id: Optional[str] = None
    ) -> bool:
        """
        Add a document to the knowledge base. Includes text extraction, cleaning, and chunking.
        
        Args:
            document_url: URL or path to the document
            document_type: Type of document (pdf, txt, docx, etc.)
            metadata: Additional metadata for the document
            is_global: Whether the document is available globally across all projects
            project_id: ID of the project this document belongs to (None for global documents)
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Adding document: {document_url}, type: {document_type}, global: {is_global}, project: {project_id}")
        
        try:
            # Download document if it's a URL
            if document_url.startswith(("http://", "https://")):
                local_path = await self._download_document(document_url, document_type)
            else:
                # It's a local path
                local_path = document_url
            
            # 1. Extract text
            document_text = await self._extract_text(local_path, document_type)
            
            # 2. Clean text (moved before chunking)
            cleaned_text = self._clean_text(document_text)
            if not cleaned_text:
                logger.warning(f"Document {document_url} resulted in empty text after cleaning.")
                # Decide if this should be an error or just skip adding
                return False # Or raise error

            # 3. Create chunks using RecursiveCharacterTextSplitter
            chunks = self._create_chunks(cleaned_text) # Pass cleaned text
            logger.info(f"Created {len(chunks)} chunks from document {document_url}")
            
            if not chunks:
                logger.warning(f"Document {document_url} resulted in zero chunks.")
                return False # Or raise error

            # 4. Prepare metadata
            document_id = Path(local_path).stem # Use stem for cleaner ID
            doc_metadata = metadata or {}
            doc_metadata.update({
                "source": document_url, # Keep original URL/path as source
                "filename": Path(local_path).name, # Add filename
                "type": document_type,
                "id": document_id,
                "chunk_count": len(chunks),
                "is_global": is_global,
                "project_id": project_id or "_global_" if is_global else "_orphan_", # Store project/global status clearly
                "added_at": time.time()
            })

            # 5. Add chunks to vectorstore
            await self._add_chunks_to_vectorstore(
                chunks=chunks,
                metadata=doc_metadata # Pass combined metadata
            )
            
            logger.info(f"Successfully added document {document_url} to vector store.")
            return True

        except Exception as e:
            logger.error(f"Failed to add document {document_url}: {str(e)}", exc_info=True)
            return False
        finally:
            # Clean up downloaded file if it was downloaded
            if document_url.startswith(("http://", "https://")) and 'local_path' in locals() and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    logger.debug(f"Cleaned up temporary file: {local_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file {local_path}: {str(cleanup_error)}")
    
    def add_document_sync(
        self,
        document_url: str,
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        is_global: bool = False,
        project_id: Optional[str] = None
    ) -> bool:
        """
        Synchronous version of add_document for use in non-async contexts.
        
        Args:
            document_url: URL or path to the document
            document_type: Type of document (pdf, txt, docx, etc.)
            metadata: Additional metadata for the document
            is_global: Whether the document is available globally across all projects
            project_id: ID of the project this document belongs to (None for global documents)
            
        Returns:
            True if successful, False otherwise
        """
        import asyncio
        
        # Create a new event loop for this thread if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            self.add_document(
                document_url=document_url,
                document_type=document_type,
                metadata=metadata,
                is_global=is_global,
                project_id=project_id
            )
        )
    
    async def _download_document(self, url: str, document_type: str) -> str:
        """
        Download a document from a URL.
        
        Args:
            url: URL to download from
            document_type: Type of document
            
        Returns:
            Path to the downloaded file
        """
        logger.info(f"Downloading document from: {url}")
        
        # Create a filename
        filename = url.split("/")[-1]
        if not filename:
            filename = f"document_{hash(url)}.{document_type}"
        
        # Download path
        download_path = os.path.join(self.documents_dir, filename)
        
        # Download the file
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(download_path, 'wb') as f:
                        f.write(await response.read())
                    
                    logger.info(f"Document downloaded to: {download_path}")
                    return download_path
                else:
                    logger.error(f"Failed to download document: {response.status}")
                    raise Exception(f"Failed to download document: {response.status}")
    
    async def _extract_text(self, file_path: str, document_type: str) -> str:
        """
        Extract text content from various document types.
        
        Args:
            file_path: Path to the document
            document_type: Type of document
            
        Returns:
            Extracted text
        """
        logger.info(f"Extracting text from {file_path} (type: {document_type})")
        text = ""
        try:
            if document_type == "pdf":
                try:
                    import pypdf
                    with open(file_path, 'rb') as f:
                        reader = pypdf.PdfReader(f)
                        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                except ImportError:
                    logger.error("pypdf not installed. Please install it: pip install pypdf")
                    raise
            elif document_type in ["docx", "doc"]:
                try:
                    import docx2txt
                    text = docx2txt.process(file_path)
                except ImportError:
                    logger.error("docx2txt not installed. Please install it: pip install docx2txt")
                    raise
            elif document_type == "txt":
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    logger.warning(f"UTF-8 decoding failed for {file_path}, trying latin-1")
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            text = f.read()
                    except Exception as e_enc:
                        logger.error(f"Could not decode text file {file_path}: {e_enc}")
                        raise
            elif document_type in ["csv", "xlsx"]:
                try:
                    import pandas as pd
                    if document_type == "csv":
                        df = pd.read_csv(file_path)
                    else: # xlsx
                        df = pd.read_excel(file_path)
                    # Basic conversion: concatenate all cells as strings
                    text = " ".join(df.astype(str).stack().tolist())
                    logger.info(f"Extracted text from {document_type} using pandas, shape: {df.shape}")
                except ImportError:
                    logger.error("pandas not installed. Please install it: pip install pandas openpyxl")
                    raise
            else:
                logger.warning(f"Extraction not supported for type: {document_type}")
                raise ValueError(f"Unsupported document type for text extraction: {document_type}")

            logger.info(f"Successfully extracted text from {file_path}, length: {len(text)}")
            return text

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}", exc_info=True)
            raise # Re-raise the exception to be caught by add_document
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning before chunking."""
        if not text:
            return ""
            
        # Remove excessive whitespace (multiple spaces, tabs, newlines)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Optional: Remove simple headers/footers (example patterns, needs refinement)
        # text = re.sub(r'^Page \d+ of \d+$', '', text, flags=re.MULTILINE) # Example footer
        # text = re.sub(r'^Confidential$', '', text, flags=re.MULTILINE) # Example header

        # Optional: Add more cleaning steps (e.g., remove special characters, normalize unicode)
        
        logger.debug(f"Cleaned text length: {len(text)}")
        return text
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create text chunks using RecursiveCharacterTextSplitter."""
        if not text:
            logger.warning("Attempted to chunk empty text.")
            return []
            
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False, # Use default separators like \n\n, \n, space, etc.
            )
            
            chunks = text_splitter.split_text(text)
            
            # Filter out very small chunks that might just be whitespace or remnants
            min_chunk_size_threshold = 20 # Example threshold
            chunks = [chunk for chunk in chunks if len(chunk.strip()) >= min_chunk_size_threshold]
            
            logger.info(f"Split text into {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
            return chunks
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}", exc_info=True)
            # Depending on severity, either return empty list or raise
            return [] # Return empty list on error for now
    
    async def _add_chunks_to_vectorstore(
        self,
        chunks: List[str],
        metadata: Dict[str, Any]
    ) -> None:
        """Add document chunks to the vector store with metadata."""
        if not chunks:
            logger.warning("No chunks provided to add to vector store.")
            return
            
        logger.info(f"Adding {len(chunks)} chunks to {self.vector_db_type} vector store for doc id {metadata.get('id')}")
        start_time = time.time()
        
        # Generate unique IDs for each chunk
        base_doc_id = metadata.get('id', 'unknown_doc')
        chunk_ids = [f"{base_doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Prepare metadata for each chunk (copy base metadata)
        metadatas = [metadata.copy() for _ in chunks]
        for i, meta in enumerate(metadatas):
             meta['chunk_index'] = i
             # Store chunk text length?
             meta['chunk_length'] = len(chunks[i])

        try:
            if self.vector_db_type == "chroma":
                 # Embeddings are handled by Chroma's custom function here
                 self.collection.add(
                     ids=chunk_ids,
                     documents=chunks, # Chroma expects 'documents' field for text
                     metadatas=metadatas
                 )
            elif self.vector_db_type == "faiss":
                 # Embed chunks first
                 embeddings = self.embed_documents(chunks)
                 
                 # Add embeddings to FAISS index
                 self.index.add(np.array(embeddings).astype('float32'))
                 
                 # Store document text and metadata in docstore
                 for i, chunk_id in enumerate(chunk_ids):
                      self.docstore["ids"].append(chunk_id)
                      self.docstore["documents"].append({
                          "page_content": chunks[i],
                          "metadata": metadatas[i]
                      })
                 
                 # Persist FAISS index and docstore
                 faiss.write_index(self.index, self.faiss_index_path)
                 with open(self.faiss_docstore_path, 'w') as f:
                      json.dump(self.docstore, f)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully added {len(chunks)} chunks in {elapsed_time:.2f}s")

        except Exception as e:
            logger.error(f"Error adding chunks to {self.vector_db_type}: {str(e)}", exc_info=True)
            # Should we attempt rollback or just log? Logging for now.
            raise # Re-raise to indicate failure in add_document
    
    async def retrieve_documents(
        self,
        query: str,
        limit: int = 5,
        project_id: Optional[str] = None,
        include_global: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks based on a query, with filtering.
        Includes conceptual placeholder for Hybrid Search.
        
        Args:
            query: The search query
            limit: Maximum number of documents to return
            project_id: Optional project ID to filter by
            include_global: Whether to include globally available documents (if project_id is set)
            filters: Additional metadata filters
            
        Returns:
            List of relevant document chunks with metadata and score
        """
        logger.info(f"Retrieving documents for query: '{query[:100]}...' (limit={limit}, project_id={project_id}, include_global={include_global})")
        start_time = time.time()
        
        # --- Semantic Search ---
        try:
            query_embedding = self.embed_query(query)
            # Increase retrieval limit slightly before filtering/ranking
            retrieve_limit = limit * 3 # Retrieve more initially for potential filtering/reranking
            
            semantic_results = []
            if self.vector_db_type == "chroma":
                 # Build Chroma WHERE filter
                 where_filter = {}
                 project_filters = []
                 if project_id:
                      project_filters.append({"project_id": project_id})
                 if include_global:
                      project_filters.append({"project_id": "_global_"})
                 
                 if project_filters:
                      # If multiple project criteria, use $or
                      if len(project_filters) > 1:
                           where_filter["$or"] = project_filters
                      else:
                           where_filter.update(project_filters[0])
                 
                 # Add any additional user-provided filters (AND logic)
                 if filters:
                     if where_filter:
                         # Combine existing project filters with user filters using $and
                         where_filter = {"$and": [where_filter, filters]}
                     else:
                         # Only user filters are present
                         where_filter = filters
                 
                 logger.debug(f"Chroma where filter: {where_filter}")
                 
                 results = self.collection.query(
                     query_embeddings=[query_embedding],
                     n_results=retrieve_limit,
                     where=where_filter if where_filter else None, # Pass None if empty
                     include=["metadatas", "documents", "distances"] 
                 )
                 
                 # Process Chroma results
                 if results and results.get('ids') and results['ids'][0]:
                     for i, doc_id in enumerate(results['ids'][0]):
                         semantic_results.append({
                             "id": doc_id,
                             "page_content": results['documents'][0][i],
                             "metadata": results['metadatas'][0][i],
                             "score": 1.0 - results['distances'][0][i] # Convert distance to similarity score (0-1)
                         })
                         
            elif self.vector_db_type == "faiss":
                 if self.index.ntotal == 0:
                      logger.warning("FAISS index is empty. Cannot retrieve.")
                      return []

                 # Retrieve more results initially for FAISS filtering
                 faiss_retrieve_limit = max(retrieve_limit, 50) # Retrieve at least 50 for filtering
                 if faiss_retrieve_limit > self.index.ntotal:
                     faiss_retrieve_limit = self.index.ntotal # Don't ask for more than exists
                     
                 logger.debug(f"Querying FAISS index (ntotal={self.index.ntotal}) for {faiss_retrieve_limit} results.")
                 distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), faiss_retrieve_limit)
                 
                 # Process FAISS results and filter *afterwards*
                 potential_matches = []
                 if len(indices) > 0:
                      for i, idx in enumerate(indices[0]):
                          if idx < 0 or idx >= len(self.docstore["ids"]): continue # Invalid index protection
                          doc_id = self.docstore["ids"][idx]
                          doc_data = self.docstore["documents"][idx]
                          potential_matches.append({
                              "id": doc_id,
                              "page_content": doc_data.get("page_content"),
                              "metadata": doc_data.get("metadata"),
                              "score": 1.0 / (1.0 + distances[0][i]) # Convert L2 distance to similarity score (approx)
                          })
                 
                 # Filter potential matches by project_id and is_global
                 filtered_results = []
                 for doc in potential_matches:
                      meta = doc.get("metadata", {})
                      doc_project_id = meta.get("project_id")
                      doc_is_global = meta.get("is_global", False)
                      
                      passes_filter = False
                      if project_id and doc_project_id == project_id:
                          passes_filter = True
                      if include_global and (doc_is_global or doc_project_id == "_global_"):
                           passes_filter = True
                      # Add check for additional filters if provided
                      if filters:
                          # Simple AND logic for filters for now
                          if all(meta.get(k) == v for k, v in filters.items()):
                               pass # Already passes basic filter or doesn't matter
                          else:
                               passes_filter = False # Fails additional filter

                      if passes_filter:
                           filtered_results.append(doc)
                           if len(filtered_results) >= limit: # Stop once we have enough post-filtering
                                break 
                 semantic_results = filtered_results # Assign filtered results

        except Exception as e:
            logger.error(f"Error during semantic search: {str(e)}", exc_info=True)
            semantic_results = [] # Return empty on error

        # --- Keyword Search (Placeholder) ---
        keyword_results = []
        # TODO: Implement keyword search (e.g., using BM25 on stored chunks or a separate keyword index)
        # Example:
        # try:
        #     keyword_results = self.keyword_search_service.search(query, limit=limit, project_id=project_id, ...)
        # except Exception as e:
        #     logger.error(f"Keyword search failed: {e}")

        # --- Hybrid Ranking (Placeholder using RRF) ---
        final_results = []
        if keyword_results: # If hybrid search was implemented
            logger.info(f"Combining {len(semantic_results)} semantic and {len(keyword_results)} keyword results.")
            # Example using Reciprocal Rank Fusion (RRF)
            combined_scores = {}
            k = 60 # RRF constant, often set to 60
            
            # Process semantic results
            for rank, doc in enumerate(semantic_results):
                doc_id = doc.get("id")
                if doc_id:
                    combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
                    doc["_semantic_rank"] = rank + 1
            
            # Process keyword results
            for rank, doc in enumerate(keyword_results):
                 doc_id = doc.get("id")
                 if doc_id:
                      combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
                      doc["_keyword_rank"] = rank + 1 # Store rank for potential use

            # Create a unified list of all unique documents
            all_docs_dict = {doc.get("id"): doc for doc in semantic_results + keyword_results if doc.get("id")}
            
            # Sort documents by combined RRF score
            sorted_doc_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)
            
            # Build final list up to the limit
            final_results = [all_docs_dict[doc_id] for doc_id in sorted_doc_ids if doc_id in all_docs_dict][:limit]
            
        else: # If only semantic search is available
            final_results = semantic_results[:limit] # Already sorted by score (descending similarity)

        elapsed_time = time.time() - start_time
        logger.info(f"Retrieved {len(final_results)} final documents in {elapsed_time:.2f}s")
        
        # Log final retrieved doc sources and scores for debugging
        for i, doc in enumerate(final_results):
            meta = doc.get("metadata", {})
            logger.debug(f"Final Doc {i+1}: ID={doc.get('id')}, Source={meta.get('filename', meta.get('source', 'N/A'))}, Score={doc.get('score'):.4f}")
            
        return final_results
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        if self.vector_db_type == "chroma":
            return self.collection.count()
        elif self.vector_db_type == "faiss":
            return self.index.ntotal
        return 0
    
    def clear_vectorstore(self) -> bool:
        """Clear the vector store."""
        try:
            if self.vector_db_type == "chroma":
                # Delete collection
                self.chroma_client.delete_collection("cx_documents")
                
                # Recreate collection
                self.collection = self.chroma_client.create_collection(
                    name="cx_documents",
                    embedding_function=None  # Will be set when needed
                )
            
            elif self.vector_db_type == "faiss":
                # Create new index
                embedding_dim = len(self.embed_query("test"))
                self.index = faiss.IndexFlatL2(embedding_dim)
                
                # Initialize document store
                self.docstore = {"documents": [], "ids": []}
                
                # Save empty index and store
                faiss.write_index(self.index, self.faiss_index_path)
                with open(self.faiss_docstore_path, 'w') as f:
                    json.dump(self.docstore, f)
            
            # Clear chunked documents
            for file_path in Path(self.chunked_dir).glob("*.json"):
                os.remove(file_path)
            
            logger.info("Vector store cleared")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False 