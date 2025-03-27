"""
SemanticChunker.py
A module for semantic-aware text chunking using embeddings and similarity metrics.

This module provides functionality to:
- Split text into semantically coherent chunks
- Merge similar chunks based on cosine similarity
- Maintain chunk size constraints
- Calculate semantic similarity between text segments
"""

import logging
from typing import List, Optional, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
from core.BaseChunker import BaseChunker

logger = logging.getLogger(__name__)

class SemanticChunker(BaseChunker):
    """Chunks text based on semantic similarity and size constraints"""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        embedding_model: Optional[Any] = None,
        chunk_size: int = 200, 
        chunk_overlap: int = 0, 
        similarity_threshold: float = 0.9, 
        separator: str = " "
    ):
        """
        Initialize the semantic chunker with configurable parameters
        
        Args:
            model_name: Name of the model for tokenization
            embedding_model: Model for generating embeddings
            chunk_size: Maximum size of chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            similarity_threshold: Threshold for considering chunks similar (0-1)
            separator: Default separator for splitting text
        """
        # Validate parameters
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not (0 <= similarity_threshold <= 1):
            raise ValueError("similarity_threshold must be between 0 and 1.")
            
        # Initialize BaseChunker first
        super().__init__(model_name, embedding_model)
        
        # Set semantic chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.separator = separator
        
        # Use provided embedding model or initialize sentence transformer
        # Check if embedding_model is a dummy model (has encode method but returns fixed values)
        is_dummy = False
        if embedding_model is not None:
            try:
                # Test if it's a dummy model by checking a fixed output length
                test_output = embedding_model.encode("test")
                if isinstance(test_output, list) and len(test_output) == 384 and all(x == 0.0 for x in test_output):
                    is_dummy = True
            except:
                pass

        # Use provided embedding model or initialize sentence transformer
        if embedding_model is None or is_dummy:
            try:
                self.sentence_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
                self.embedding_model = self.sentence_model
                logger.info("Initialized SentenceTransformer for semantic chunking")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer: {e}")
                # Provide a fallback
                class DummyEmbedder:
                    def encode(self, text, **kwargs):
                        return [0.0] * 384  # Return dummy vector
                self.sentence_model = DummyEmbedder()
                self.embedding_model = self.sentence_model
        else:
            self.sentence_model = embedding_model
            logger.info("Using provided embedding model for semantic chunking")
        
        # Initialize text splitter for initial chunking
        self.text_splitter = SpacyTextSplitter(
            chunk_size=self.chunk_size - self.chunk_overlap,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator 
        )

    def _enforce_size_immediately(self, text: str) -> List[str]:
        """
        Split text into chunks while strictly enforcing size limits
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks respecting size limits
        """
        if not text.strip():
            return []
            
        chunks = []
        current_chunk = []
        words = text.split()
        
        for word in words:
            # Check if adding word would exceed size limit (including spaces)
            if sum(len(w) for w in current_chunk) + len(word) + len(current_chunk) <= self.chunk_size:
                current_chunk.append(word)
            else:
                # Save current chunk and start a new one
                if current_chunk:  # Avoid empty chunks
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                
        # Add final chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def get_semantic_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Process documents into semantically coherent chunks
        
        Args:
            documents: List of documents to process
            
        Returns:
            Semantically coherent document chunks
        """
        if not documents:
            logger.warning("No documents provided for semantic chunking")
            return []
            
        try:
            # Initial document splitting
            base_chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Initial splitting created {len(base_chunks)} base chunks")
            
            if not base_chunks:
                return []
                
            # Generate embeddings for semantic comparison
            chunk_contents = [doc.page_content for doc in base_chunks]
            chunk_embeddings = self.sentence_model.encode(chunk_contents)
            
            grouped_chunks = []
            current_group = []
            current_embedding = None

            for i, base_chunk in enumerate(base_chunks):
                if not current_group:
                    current_group.append(base_chunk)
                    current_embedding = chunk_embeddings[i].reshape(1, -1)
                    continue
                    
                # Calculate similarity and combine if appropriate
                similarity = cosine_similarity(current_embedding, chunk_embeddings[i].reshape(1, -1))[0][0]
                combined_content = " ".join([doc.page_content for doc in current_group] + [base_chunk.page_content])

                if similarity >= self.similarity_threshold and len(combined_content) <= self.chunk_size:
                    current_group.append(base_chunk)
                else:
                    # Process current group and start a new one
                    grouped_chunks.extend(self._finalize_chunk_group(current_group))
                    current_group = [base_chunk]
                    current_embedding = chunk_embeddings[i].reshape(1, -1)
                    
            # Finalize any remaining chunks
            if current_group:
                grouped_chunks.extend(self._finalize_chunk_group(current_group))
                
            logger.info(f"Created {len(grouped_chunks)} semantic chunks")
            return grouped_chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            # Fall back to original documents
            return documents

    def _finalize_chunk_group(self, group: List[Document]) -> List[Document]:
        """
        Process a group of related chunks into final documents.
        
        Args:
            group: List of related document chunks
            
        Returns:
            Finalized document chunks
        """
        if not group:
            return []
            
        processed_chunks = []
        content = " ".join([doc.page_content for doc in group])
        size_limited_chunks = self._enforce_size_immediately(content)
        
        base_metadata = group[0].metadata.copy()
        
        for i, chunk in enumerate(size_limited_chunks):
            # Enhanced metadata with stats
            stats = self.analyze_text(chunk)
            metadata = base_metadata.copy()
            metadata.update({
                "chunk_index": i + 1,
                "chunk_count": len(size_limited_chunks),
                "char_count": stats["char_count"],
                "token_count": stats["token_count"],
                "sentence_count": stats["sentence_count"],
                "word_count": stats["word_count"],
                "chunk_type": "semantic"
            })
            
            processed_chunks.append(Document(page_content=chunk, metadata=metadata))
            
        return processed_chunks

    def semantic_process_document(self, file_path: str, preprocess: bool = False) -> List[Document]:
        """
        Process document using semantic chunking strategy.
        
        Args:
            file_path: Path to the document file
            preprocess: Whether to preprocess text
            
        Returns:
            List of semantically chunked Document objects
        """
        try:
            logger.info(f"Processing document with semantic chunking: {file_path}")
            
            # Load document using BaseChunker's method
            raw_documents = self.load_document(file_path)
            
            # Optionally preprocess documents
            processed_documents = []
            for doc in raw_documents:
                content = doc.page_content
                if preprocess:
                    content = self.preprocess_text(content)
                processed_documents.append(Document(
                    page_content=content,
                    metadata=doc.metadata
                ))
            
            # Perform semantic chunking
            documents = self.get_semantic_chunks(processed_documents)
            logger.info(f"Created {len(documents)} semantic chunks")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in semantic_process_document: {e}")
            raise
    
    def process_document(self, file_path: str, preprocess: bool = True) -> List[Document]:
        """
        Process document using semantic chunking strategy (implements abstract method).
        
        Args:
            file_path: Path to the document file
            preprocess: Whether to preprocess text
            
        Returns:
            List of semantically chunked Document objects
        """
        return self.semantic_process_document(file_path, preprocess)