"""
Local Embedding Utilities

Simple vector database implementation for local use without external dependencies.
"""
import os
import json
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import hashlib

logger = logging.getLogger(__name__)

class LocalVectorDB:
    """Simple vector database implementation for local use."""
    
    def __init__(self, storage_path: str):
        """
        Initialize the local vector database.
        
        Args:
            storage_path: Path to store the vector database
        """
        self.storage_path = storage_path
        self.vectors_path = os.path.join(storage_path, "vectors.json")
        self.metadata_path = os.path.join(storage_path, "metadata.json")
        
        # Create directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize or load vectors and metadata
        if os.path.exists(self.vectors_path) and os.path.exists(self.metadata_path):
            self.load()
        else:
            self.vectors = {}  # id -> vector
            self.metadata = {}  # id -> metadata
            self.save()
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """
        Add vectors to the database.
        
        Args:
            ids: List of document IDs
            embeddings: List of embedding vectors
            documents: List of document texts
            metadatas: List of document metadata
        """
        for i, doc_id in enumerate(ids):
            self.vectors[doc_id] = embeddings[i]
            
            metadata = metadatas[i].copy()
            metadata["text"] = documents[i]
            self.metadata[doc_id] = metadata
        
        self.save()
        return ids
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the database.
        
        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return
            where: Optional filters
            
        Returns:
            Dictionary with query results
        """
        results = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }
        
        for query_embedding in query_embeddings:
            # Calculate distances
            distances = {}
            for doc_id, vector in self.vectors.items():
                # Apply filters if provided
                if where:
                    match = True
                    for key, value in where.items():
                        if key not in self.metadata[doc_id] or self.metadata[doc_id][key] != value:
                            match = False
                            break
                    
                    if not match:
                        continue
                
                # Calculate euclidean distance
                distance = np.sum(np.square(np.array(query_embedding) - np.array(vector)))
                distances[doc_id] = distance
            
            # Sort by distance
            sorted_items = sorted(distances.items(), key=lambda x: x[1])[:n_results]
            
            # Prepare results
            ids = []
            documents = []
            metadatas = []
            result_distances = []
            
            for doc_id, distance in sorted_items:
                ids.append(doc_id)
                documents.append(self.metadata[doc_id]["text"])
                
                # Create a copy of metadata without the text field
                metadata_copy = self.metadata[doc_id].copy()
                if "text" in metadata_copy:
                    del metadata_copy["text"]
                
                metadatas.append(metadata_copy)
                result_distances.append(distance)
            
            results["ids"].append(ids)
            results["documents"].append(documents)
            results["metadatas"].append(metadatas)
            results["distances"].append(result_distances)
        
        return results
    
    def count(self) -> int:
        """Get the number of documents in the database."""
        return len(self.vectors)
    
    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None):
        """
        Delete documents from the database.
        
        Args:
            ids: Optional list of document IDs to delete
            where: Optional filters
        """
        if ids:
            for doc_id in ids:
                if doc_id in self.vectors:
                    del self.vectors[doc_id]
                if doc_id in self.metadata:
                    del self.metadata[doc_id]
        
        elif where:
            to_delete = []
            for doc_id, metadata in self.metadata.items():
                match = True
                for key, value in where.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                
                if match:
                    to_delete.append(doc_id)
            
            for doc_id in to_delete:
                del self.vectors[doc_id]
                del self.metadata[doc_id]
        
        self.save()
    
    def save(self):
        """Save the database to disk."""
        with open(self.vectors_path, 'w') as f:
            json.dump(self.vectors, f)
        
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def load(self):
        """Load the database from disk."""
        with open(self.vectors_path, 'r') as f:
            self.vectors = json.load(f)
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)


def get_local_embedding_fn(dimensions: int = 384, normalize: bool = True):
    """
    Get a simple local embedding function.
    
    Args:
        dimensions: Embedding dimensions
        normalize: Whether to normalize the embeddings
        
    Returns:
        Embedding function
    """
    def embed_fn(texts):
        """Embed text using MD5 hash."""
        if isinstance(texts, str):
            texts = [texts]
        
        result = []
        for text in texts:
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
            
            # Pad to desired dimensions
            vector = vector + [0.0] * (dimensions - len(vector))
            
            # Normalize if requested
            if normalize:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = [v / norm for v in vector]
            
            result.append(vector)
        
        return result
    
    return embed_fn 