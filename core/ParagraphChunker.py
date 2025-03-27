"""
ParagraphChunker.py

A module for paragraph-level document chunking with token counting and preprocessing.

Features:
- Paragraph-based document splitting
- Content validation
- Multi-level delimiter detection
- Smart paragraph boundary detection
"""

import logging
import spacy
from typing import List, Optional
from langchain_core.documents import Document
from core.BaseChunker import BaseChunker

logger = logging.getLogger(__name__)

class ParagraphChunker(BaseChunker):
    """Handles document chunking at the paragraph level with token counting."""
    
    PARAGRAPH_MIN_LENGTH = 50  # Minimum characters for a valid paragraph
    
    def __init__(self, model_name=None, embedding_model=None):
        """
        Initialize paragraph chunker with specified models.
        
        Args:
            model_name: Name of the model for tokenization
            embedding_model: Model for generating embeddings
        """
        super().__init__(model_name, embedding_model)
        self.page_stats = []
        
        # Initialize spaCy for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            import subprocess
            logger.info("Installing spaCy model...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                         capture_output=True)
            self.nlp = spacy.load("en_core_web_sm")
        
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs using NLP and multiple delimiter patterns.
        
        Args:
            text: The text content to split
            
        Returns:
            List of paragraphs
        """
        # Pre-clean the text
        text = text.replace('\r', '\n')
        
        # Common paragraph delimiters
        delimiters = [
            '\n\n',           # Double line breaks
            '\n    ',         # Indented new lines
            '\n\t',           # Tab-indented new lines
            '\n•',            # Bullet points
            '\n-',            # Dashed lists
            r'\n\d+\.',        # Numbered lists
        ]
        
        # Initial split using spaCy for sentence boundaries
        doc = self.nlp(text)
        potential_paragraphs = []
        current_paragraph = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Check if sentence starts a new paragraph
            starts_new = any(sent_text.startswith(d.strip()) for d in delimiters)
            if starts_new and current_paragraph:
                potential_paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sent_text]
            else:
                current_paragraph.append(sent_text)
        
        # Add the last paragraph
        if current_paragraph:
            potential_paragraphs.append(' '.join(current_paragraph))
        
        # Filter and clean paragraphs
        cleaned_paragraphs = []
        for para in potential_paragraphs:
            clean_para = ' '.join(para.split())
            if len(clean_para) >= self.PARAGRAPH_MIN_LENGTH:
                cleaned_paragraphs.append(clean_para)
        
        return cleaned_paragraphs

    def _process_single_paragraph(self, content: str, page_number: int, 
                                 para_number: int, preprocess: bool) -> Optional[Document]:
        """
        Process a single paragraph with analysis and metadata.
        
        Args:
            content: The paragraph content
            page_number: The page number
            para_number: The paragraph number
            preprocess: Whether to preprocess the text
            
        Returns:
            Document object with processed content and metadata, or None if paragraph is invalid
        """
        # First check character length
        if len(content.strip()) < self.PARAGRAPH_MIN_LENGTH:
            self.page_stats.append(f"Paragraph {para_number} on page {page_number} is too short.")
            return None
            
        # Optionally preprocess the text
        if preprocess:
            content = self.preprocess_text(content)
            
        # Analyze the paragraph and generate metadata
        stats = self.analyze_text(content)
        
        # Check token threshold
        if stats["token_count"] < self.TOKEN_THRESHOLD:
            self.page_stats.append(
                f"Paragraph {para_number} on page {page_number} dropped: "
                f"only {stats['token_count']} tokens"
            )
            return None
            
        metadata = {
            "page": page_number,
            "paragraph": para_number,
            "char_count": stats["char_count"],
            "token_count": stats["token_count"],
            "sentence_count": stats["sentence_count"],
            "word_count": stats["word_count"],
            "has_ocr": str(stats.get("has_content", True))
        }
        
        return Document(page_content=content, metadata=metadata)

    def paragraph_process_document(self, file_path: str, preprocess: bool = False) -> List[Document]:
        """
        Process PDF document paragraph by paragraph with analysis.
        
        Args:
            file_path: Path to the PDF file
            preprocess: Whether to preprocess paragraph text
            
        Returns:
            List of Document objects, one per valid paragraph
        """
        try:
            self.page_stats = []  # Reset stats for this document
            raw_pages = self.load_document(file_path)
            processed_paragraphs = []
            
            logger.info(f"Processing document with {len(raw_pages)} pages")
            
            for page_idx, page in enumerate(raw_pages):
                paragraphs = self._split_into_paragraphs(page.page_content)
                logger.info(f"Page {page_idx+1}: Found {len(paragraphs)} paragraphs")
                
                for para_idx, paragraph in enumerate(paragraphs):
                    processed_para = self._process_single_paragraph(
                        paragraph, 
                        page_idx + 1, 
                        para_idx + 1, 
                        preprocess
                    )
                    if processed_para:
                        processed_paragraphs.append(processed_para)
                        
            # Output skipped paragraphs for transparency
            if self.page_stats:
                logger.info("\n".join(self.page_stats))
                
            logger.info(f"Processed {len(processed_paragraphs)} valid paragraphs")
            return processed_paragraphs
            
        except Exception as e:
            logger.error(f"Error in paragraph_process_document: {e}")
            raise
    
    def process_document(self, file_path: str, preprocess: bool = True) -> List[Document]:
        """
        Process document using paragraph chunking strategy (implements abstract method).
        
        Args:
            file_path: Path to the PDF file
            preprocess: Whether to preprocess paragraph text
            
        Returns:
            List of Document objects, one per valid paragraph
        """
        return self.paragraph_process_document(file_path, preprocess)