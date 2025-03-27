"""
BaseChunker.py

An abstract base class defining the interface for document chunking strategies.
"""

import logging
from core.OCREnhancedPDFLoader import OCREnhancedPDFLoader
from core.TextPreprocessor import TextPreprocessor
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

import spacy
from langchain_core.documents import Document

# Import tiktoken at the module level
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not installed. Some tokenization features will be limited. "
                  "Install with: pip install tiktoken")

logger = logging.getLogger(__name__)

class BaseChunker(ABC):
    """Abstract base class for document chunking strategies."""
    
    # Common constants
    BLANK_THRESHOLD = 20  # Minimum characters for non-blank text
    TOKEN_THRESHOLD = 10  # Minimum tokens for valid content
    
    # Model type indicators
    TIKTOKEN_MODELS = ["gpt", "davinci", "curie", "babbage", "ada"]
    BASIC_TOKENIZER_MODELS = ["llama", "mistral", "granite"]
    
    def __init__(self, model_name: Optional[str] = None, embedding_model: Optional[Any] = None):
        """
        Initialize base chunker with model settings.
        
        Args:
            model_name: Name of the model for tokenization
            embedding_model: Model for generating embeddings
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.uses_tiktoken = False
        self.uses_basic_tokenizer = False
        self.tokenizer = None
        
        self._initialize_tokenizer()
        
        # Initialize NLP pipeline for text analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Installing spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                         capture_output=True)
            self.nlp = spacy.load("en_core_web_sm")


    def _initialize_tokenizer(self):
        """Initialize the appropriate tokenizer based on model name."""
        if not self.model_name:
            logger.warning("No model name provided. Using basic tokenization.")
            self.uses_basic_tokenizer = True
            return
        
        # Check if model is supported by tiktoken
        if TIKTOKEN_AVAILABLE and self.model_name in ["cl100k_base", "p50k_base", "r50k_base", "gpt2"]:
            try:
                encoding = tiktoken.get_encoding(self.model_name)
                
                # Create a tokenizer-like interface for tiktoken
                class TiktokenWrapper:
                    def __init__(self, encoding):
                        self.encoding = encoding
                        
                    def tokenize(self, text):
                        return self.encoding.encode(text)
                
                self.tokenizer = TiktokenWrapper(encoding)
                self.uses_tiktoken = True
                logger.info(f"Initialized tiktoken tokenizer for model: {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"Error with specified tiktoken model: {e}")
                # Fall back to a standard encoding
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    
                    class TiktokenWrapper:
                        def __init__(self, encoding):
                            self.encoding = encoding
                            
                        def tokenize(self, text):
                            return self.encoding.encode(text)
                    
                    self.tokenizer = TiktokenWrapper(encoding)
                    self.uses_tiktoken = True
                    logger.info("Initialized tiktoken with cl100k_base encoding")
                except Exception as e:
                    logger.warning(f"Error initializing tiktoken: {e}")
                    self.uses_basic_tokenizer = True

        if TIKTOKEN_AVAILABLE and (
            any(model in self.model_name.lower() for model in self.TIKTOKEN_MODELS) or 
            self.model_name.startswith("gpt-") or 
            self.model_name.endswith("-base")
        ):
            try:
                encoding = tiktoken.get_encoding(self.model_name)
                
                # Create a tokenizer-like interface for tiktoken
                class TiktokenWrapper:
                    def __init__(self, encoding):
                        self.encoding = encoding
                        
                    def tokenize(self, text):
                        return self.encoding.encode(text)
                
                self.tokenizer = TiktokenWrapper(encoding)
                self.uses_tiktoken = True
                logger.info(f"Initialized tiktoken tokenizer for model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Error with specified tiktoken model: {e}")
                # Fall back to a standard encoding
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    
                    class TiktokenWrapper:
                        def __init__(self, encoding):
                            self.encoding = encoding
                            
                        def tokenize(self, text):
                            return self.encoding.encode(text)
                    
                    self.tokenizer = TiktokenWrapper(encoding)
                    self.uses_tiktoken = True
                    logger.info("Initialized tiktoken with cl100k_base encoding")
                except Exception as e:
                    logger.warning(f"Error initializing tiktoken: {e}")
                    self.uses_basic_tokenizer = True

        # Check if model uses basic tokenization
        elif any(model in self.model_name.lower() for model in self.BASIC_TOKENIZER_MODELS):
            self.uses_basic_tokenizer = True
            logger.info("Using basic tokenization for model")
            
        # Fall back to transformers tokenizer
        else:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.info(f"Initialized transformers tokenizer for model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Error initializing transformer tokenizer: {e}")
                logger.warning("Falling back to basic tokenization")
                self.uses_basic_tokenizer = True
                
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string using the available tokenizer."""
        if not text:
            return 0
            
        try:
            # Try with the standard tokenizer
            if self.tokenizer:
                if self.uses_tiktoken:
                    # For tiktoken wrapper
                    return len(self.tokenizer.tokenize(text))
                else:
                    # For transformers tokenizer
                    tokens = self.tokenizer.tokenize(text)
                    return len(tokens)
        except Exception as e:
            logger.warning(f"Primary tokenization failed: {e}")
        
        # Basic tokenization fallback
        if self.uses_basic_tokenizer or not self.tokenizer:
            # Simple approximation (word count)
            return len(text.split())
            
        # If we somehow got here, return a reasonable approximation
        return len(text) // 4  # Rough character-to-token ratio
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding vector for text."""
        if not text.strip() or not self.embedding_model:
            return None
            
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Perform detailed analysis of text content."""
        if not text.strip():
            return {
                "char_count": 0,
                "token_count": 0,
                "sentence_count": 0,
                "word_count": 0,
                "embedding_dim": 0,
                "has_content": False
            }
            
        try:
            embedding = self.get_embedding(text)
            doc = self.nlp(text)
            
            return {
                "char_count": len(text),
                "token_count": self.count_tokens(text),
                "sentence_count": len(list(doc.sents)),
                "word_count": len(text.split()),
                "embedding_dim": len(embedding) if embedding is not None else 0,
                "has_content": bool(text.strip())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                "char_count": len(text),
                "token_count": 0,
                "sentence_count": 0,
                "word_count": len(text.split()),
                "embedding_dim": 0,
                "has_content": bool(text.strip())
            }
    
    def is_content_valid(self, text: str, min_chars: int = None, min_tokens: int = None) -> bool:
        """Check if content meets minimum requirements."""
        if not text.strip():
            return False
            
        min_chars = min_chars or self.BLANK_THRESHOLD
        min_tokens = min_tokens or self.TOKEN_THRESHOLD
        
        if len(text.strip()) < min_chars:
            return False
            
        token_count = self.count_tokens(text)
        return token_count >= min_tokens
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document using OCREnhancedPDFLoader."""
        try:
            loader = OCREnhancedPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            raise
    
    def preprocess_text(self, text: str, remove_headers_footers: bool = True) -> str:
        """Preprocess text using TextPreprocessor."""
        try:
            preprocessor = TextPreprocessor()
            return preprocessor.preprocess(text, remove_headers_footers)
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    @abstractmethod
    def process_document(self, file_path: str, preprocess: bool = True) -> Union[List[Document], Dict[str, List[Document]]]:
        """Process document using specific chunking strategy."""
        pass