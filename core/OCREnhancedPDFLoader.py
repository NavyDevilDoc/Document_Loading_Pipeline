"""
OCREnhancedPDFLoader.py

A module for loading PDFs with OCR (Optical Character Recognition) support.

Features:
- PDF loading with OCR fallback
- Image text extraction
- Multiple page handling
- Text quality optimization
"""

import os
import pytesseract
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from pdf2image import convert_from_path

class OCREnhancedPDFLoader:
    """Loads PDFs with OCR support for text extraction"""
    
    BLANK_THRESHOLD = 10  # Minimum character count to consider a page non-blank

    def __init__(self, file_path: str, tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        """Initialize the OCR-enhanced PDF loader."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")
        
        self.file_path = file_path
        self.skipped_pages = []
        
        # Set Tesseract path, validating existence
        if not os.path.isfile(tesseract_path):
            raise ValueError(f"Tesseract executable not found at path: {tesseract_path}")
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


    def _is_blank_page(self, text: str) -> bool:
        """Check if page is blank or contains only whitespace/special characters"""
        if not text or not text.strip():
            return True
        cleaned_text = text.strip().replace('\n', '').replace('\r', '').replace('\t', '')
        return len(cleaned_text) < self.BLANK_THRESHOLD


    def _process_page(self, doc, img, page_number: int):
        """Process a single page by choosing the best text source"""
        existing_text = doc.page_content
        
        # If existing text is substantial, use it
        if len(existing_text.strip()) > self.BLANK_THRESHOLD * 5:  # Using a higher threshold
            combined_text = existing_text
            ocr_used = False
        else:
            # Otherwise fall back to OCR
            try:
                ocr_text = pytesseract.image_to_string(img)
                combined_text = ocr_text
                ocr_used = True
            except Exception as e:
                print(f"Error applying OCR to page {page_number}: {e}")
                combined_text = existing_text
                ocr_used = False
        
        # Check if the page is blank after combining
        if self._is_blank_page(combined_text):
            self.skipped_pages.append(page_number)
            return None
        
        # Return the Document with the selected content and metadata
        return Document(
            page_content=combined_text,
            metadata={
                **doc.metadata,
                "source": "ocr" if ocr_used else "text_extraction",
                "page": page_number,
                "is_blank": "false",
                "has_ocr": str(ocr_used)
            }
        )


    def load(self):
        """Load and process PDF file with OCR enhancement"""
        try:
            # Standard PDF text extraction using PyMuPDF
            loader = PyMuPDFLoader(self.file_path)
            text_documents = loader.load()
            
            # Convert PDF pages to high-resolution images for OCR
            images = convert_from_path(self.file_path, dpi=300)
            
            # Process each page and combine standard extraction with OCR
            enhanced_documents = []
            for idx, (doc, img) in enumerate(zip(text_documents, images)):
                page_number = idx + 1
                enhanced_doc = self._process_page(doc, img, page_number)
                
                # Append only non-blank pages
                if enhanced_doc:
                    enhanced_documents.append(enhanced_doc)
            
            # Report skipped blank pages, if any
            if self.skipped_pages:
                print(f"Skipped {len(self.skipped_pages)} blank pages: {self.skipped_pages}")
            
            return enhanced_documents
            
        except Exception as e:
            print(f"Error in OCR-enhanced loading: {e}")
            raise
