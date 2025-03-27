"""
PageChunker.py

A module for page-level document chunking with token counting and preprocessing.

Features:
- Page-based document splitting
- Content validation
- Blank page detection
- Document metadata enrichment
"""

from typing import List, Optional
import logging
from langchain_core.documents import Document
from core.BaseChunker import BaseChunker

logger = logging.getLogger(__name__)

class PageChunker(BaseChunker):
    """Handles document chunking at the page level."""
    
    def __init__(self, model_name=None, embedding_model=None):
        """
        Initialize page chunker with specified models.
        
        Args:
            model_name: Name of the model for tokenization
            embedding_model: Model for generating embeddings
        """
        super().__init__(model_name, embedding_model)
        self.page_stats = []

    def _is_blank_page(self, text: str) -> bool:
        """Check if page is blank or contains only whitespace/special characters."""
        cleaned_text = text.strip().replace('\n', '').replace('\r', '').replace('\t', '')
        return len(cleaned_text) < self.BLANK_THRESHOLD

    def _process_single_page(self, content: str, page_number: int, preprocess: bool) -> Optional[Document]:
        """
        Process a single page with optional preprocessing and analysis.
        
        Args:
            content: The page content
            page_number: The page number
            preprocess: Whether to preprocess the text
            
        Returns:
            Document object with processed content and metadata, or None if page is blank
        """
        if self._is_blank_page(content):
            self.page_stats.append(f"Page {page_number} is blank.")
            return None
            
        # Optionally preprocess the text
        if preprocess:
            content = self.preprocess_text(content)
            
        # Analyze the page and generate metadata
        stats = self.analyze_text(content)
        
        metadata = {
            "page": page_number,
            "char_count": stats["char_count"],
            "token_count": stats["token_count"],
            "sentence_count": stats["sentence_count"],
            "word_count": stats["word_count"],
            "has_ocr": str(stats.get("has_content", True)),
            "is_blank": "false"
        }
        
        return Document(page_content=content, metadata=metadata)

    def page_process_document(self, file_path: str, preprocess: bool = False) -> List[Document]:
        """
        Process PDF document page by page with analysis and optional preprocessing.
        
        Args:
            file_path: Path to the PDF file
            preprocess: Whether to preprocess page text
            
        Returns:
            List of Document objects, one per non-blank page
        """
        try:
            self.page_stats = []  # Reset stats for this document
            raw_pages = self.load_document(file_path)
            processed_pages = []
            
            logger.info(f"Processing document with {len(raw_pages)} pages")
            
            for idx, page in enumerate(raw_pages):
                processed_page = self._process_single_page(page.page_content, idx + 1, preprocess)
                if processed_page:
                    processed_pages.append(processed_page)
            
            # Output skipped pages for transparency
            if self.page_stats:
                logger.info("\n".join(self.page_stats))
                
            logger.info(f"Processed {len(processed_pages)} non-blank pages")
            return processed_pages
            
        except Exception as e:
            logger.error(f"Error in page_process_document: {e}")
            raise
    
    def process_document(self, file_path: str, preprocess: bool = True) -> List[Document]:
        """
        Process document using page chunking strategy (implements abstract method).
        
        Args:
            file_path: Path to the PDF file
            preprocess: Whether to preprocess page text
            
        Returns:
            List of Document objects, one per non-blank page
        """
        return self.page_process_document(file_path, preprocess)