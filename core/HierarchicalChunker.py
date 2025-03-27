"""
HierarchicalChunker.py

A module for hierarchical document chunking that combines page-level and semantic chunking.

Features:
- Multi-level document representation (pages and chunks)
- Semantic chunking with sentence boundaries
- Size and overlap controls
- Hierarchical metadata
"""

import logging
import spacy
from typing import Dict, List, Optional, Any
from langchain_core.documents import Document
from core.PageChunker import PageChunker

logger = logging.getLogger(__name__)

class HierarchicalChunker(PageChunker):
    """Handles document chunking at multiple hierarchical levels."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_model: Optional[Any] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize hierarchical chunker with specified models and parameters.
        
        Args:
            model_name: Name of the model for tokenization
            embedding_model: Model for generating embeddings
            chunk_size: Maximum size of semantic chunks
            chunk_overlap: Overlap between chunks
            similarity_threshold: Similarity threshold for merging chunks
        """
        super().__init__(model_name, embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        
        # Initialize spaCy for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Installing spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                         capture_output=True)
            self.nlp = spacy.load("en_core_web_sm")

    def _create_semantic_chunks(self, content: str, page_number: int) -> List[Document]:
        """
        Create semantic chunks with detailed metadata.
        
        Args:
            content: The page content to chunk
            page_number: The page number
            
        Returns:
            List of Document objects representing semantic chunks
        """
        if not content.strip():
            return []
            
        sentences = list(self.nlp(content).sents)
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            sent_text = sent.text.strip()
            sent_length = len(sent_text)

            if current_length + sent_length > self.chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    stats = self.analyze_text(chunk_text)
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "level": "chunk",
                            "page_num": page_number,
                            "chunk_num": len(chunks) + 1,
                            "parent_page": page_number,
                            "char_count": stats["char_count"],
                            "token_count": stats["token_count"],
                            "sentence_count": stats["sentence_count"],
                            "word_count": stats["word_count"],
                            "has_ocr": stats.get("has_content", "true")
                        }
                    ))
                current_chunk = [sent_text]
                current_length = sent_length
            else:
                current_chunk.append(sent_text)
                current_length += sent_length

        # Handle final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            stats = self.analyze_text(chunk_text)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "level": "chunk",
                    "page_num": page_number,
                    "chunk_num": len(chunks) + 1,
                    "parent_page": page_number,
                    "char_count": stats["char_count"],
                    "token_count": stats["token_count"],
                    "sentence_count": stats["sentence_count"],
                    "word_count": stats["word_count"],
                    "has_ocr": stats.get("has_content", "true")
                }
            ))
        
        self.page_stats.append(f"Created {len(chunks)} chunks for page {page_number}")
        return chunks

    def hierarchical_process_document(self, file_path: str, preprocess: bool = True) -> Dict[str, List[Document]]:
        """
        Process document with hierarchical chunking strategy.
        
        Args:
            file_path: Path to the PDF file
            preprocess: Whether to preprocess text
            
        Returns:
            Dictionary with 'pages' and 'chunks' lists of Documents
        """
        self.page_stats = []  # Reset stats
        
        # First get the page-level documents using PageChunker
        page_docs = super().page_process_document(file_path, preprocess)
        
        # Now create chunk-level documents
        chunk_docs = []
        total_chunks = 0
        
        for page_doc in page_docs:
            page_num = page_doc.metadata["page"]
            
            # Mark this as a page-level document
            page_doc.metadata["level"] = "page"
            
            # Create chunks for this page
            page_chunks = self._create_semantic_chunks(
                page_doc.page_content, 
                page_num
            )
            
            chunk_docs.extend(page_chunks)
            total_chunks += len(page_chunks)
        
        # Log summary information
        logger.info(f"\nHierarchical Processing Summary:")
        logger.info(f"Total Pages: {len(page_docs)}")
        logger.info(f"Total Chunks: {total_chunks}")
        logger.info("\n".join(self.page_stats))
        
        return {
            "pages": page_docs,
            "chunks": chunk_docs
        }
        
    def process_document(self, file_path: str, preprocess: bool = True) -> Dict[str, List[Document]]:
        """
        Process document using hierarchical chunking strategy (implements abstract method).
        
        Args:
            file_path: Path to the PDF file
            preprocess: Whether to preprocess text
            
        Returns:
            Dictionary with 'pages' and 'chunks' lists of Documents
        """
        return self.hierarchical_process_document(file_path, preprocess)