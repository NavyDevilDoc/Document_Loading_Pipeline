"""
EmbeddingManager.py

A module for managing text embeddings from different providers.

Supported embedding providers:
- Sentence Transformers
- OpenAI
- Hugging Face Transformers
- Ollama
"""

import os
import logging
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Enumeration of available embedding providers."""
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

class EmbeddingManager:
    """Manager class for text embeddings from different providers."""
    
    # Default models for each provider
    DEFAULT_MODELS = {
        EmbeddingProvider.SENTENCE_TRANSFORMER: "ibm-granite/granite-embedding-125m-english",
        EmbeddingProvider.OPENAI: "text-embedding-3-small",
        EmbeddingProvider.HUGGINGFACE: "sentence-transformers/all-mpnet-base-v2",
        EmbeddingProvider.OLLAMA: "nomic-embed-text"
    }
    
    # Required environment variables for each provider
    REQUIRED_ENV_VARS = {
        EmbeddingProvider.OPENAI: ["OPENAI_API_KEY"],
        EmbeddingProvider.HUGGINGFACE: ["HUGGINGFACE_API_KEY"],
    }
    
    def __init__(
        self, 
        provider: str = EmbeddingProvider.SENTENCE_TRANSFORMER,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = False,
        **kwargs
    ):
        """
        Initialize the embedding manager.
        
        Args:
            provider: Name of the embedding provider
            model_name: Name of the embedding model (defaults to provider-specific default)
            cache_dir: Directory for caching models and embeddings
            batch_size: Number of texts to embed in each batch
            show_progress: Whether to show a progress bar during embedding
            kwargs: Additional provider-specific parameters
        """
        self.provider = provider.lower()
        self.model_name = self._validate_model_name(model_name, provider)
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.kwargs = kwargs
        self._model = None
        self._embed_function = None
        
        # Create cache directory if specified
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Check required environment variables
        self._check_environment()

    def _validate_model_name(self, model_name: Optional[str], provider: str) -> str:
        """
        Validate that the model name is appropriate for the provider or return a default.
        """
        # If no model name provided, use the default for the provider
        if not model_name:
            default_model = self.DEFAULT_MODELS.get(provider)
            logger.info(f"No model name provided. Using default for {provider}: {default_model}")
            return default_model
            
        # Provider-specific validation
        if provider == EmbeddingProvider.OPENAI:
            if not (model_name.startswith("text-embedding")):
                default_model = self.DEFAULT_MODELS[provider]
                logger.warning(f"'{model_name}' may not be a valid OpenAI embedding model. "
                            f"Using default: {default_model}")
                return default_model
                
        elif provider == EmbeddingProvider.SENTENCE_TRANSFORMER:
            # Don't try to use OpenAI models with SentenceTransformer
            if model_name.startswith("text-embedding") or "openai" in model_name.lower():
                logger.warning(f"'{model_name}' appears to be an OpenAI model but provider is "
                            f"SentenceTransformer. Using default: {self.DEFAULT_MODELS[provider]}")
                return self.DEFAULT_MODELS[provider]
                
        return model_name

    def _check_environment(self) -> None:
        """Check that required environment variables are set."""
        if self.provider in self.REQUIRED_ENV_VARS:
            missing_vars = []
            for env_var in self.REQUIRED_ENV_VARS[self.provider]:
                if not os.getenv(env_var):
                    missing_vars.append(env_var)
            
            if missing_vars:
                provider_name = {
                    EmbeddingProvider.OPENAI: "OpenAI",
                    EmbeddingProvider.HUGGINGFACE: "HuggingFace",
                    EmbeddingProvider.OLLAMA: "Ollama"
                }.get(self.provider, self.provider)
                
                logger.warning(f"Required environment variables for {provider_name} not found: {', '.join(missing_vars)}")
                logger.warning(f"Falling back to Sentence Transformer provider")
                self.provider = EmbeddingProvider.SENTENCE_TRANSFORMER
                self.model_name = self.DEFAULT_MODELS[EmbeddingProvider.SENTENCE_TRANSFORMER]
                return False
        
        return True
    
    @property
    def embedding_function(self) -> Callable:
        """Get or create the embedding function."""
        if self._embed_function is None:
            self._embed_function = self._create_embedding_function()
        return self._embed_function
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings produced by this model."""
        # Create a sample embedding to determine dimension
        sample_text = "This is a sample text to determine embedding dimension."
        sample_embedding = self.embed_texts([sample_text])[0]
        return len(sample_embedding)
    
    def _create_embedding_function(self) -> Callable:
        """Create an embedding function based on the selected provider."""
        if self.provider == EmbeddingProvider.SENTENCE_TRANSFORMER:
            return self._create_sentence_transformer_embedder()
        elif self.provider == EmbeddingProvider.OPENAI:
            return self._create_openai_embedder()
        elif self.provider == EmbeddingProvider.HUGGINGFACE:
            return self._create_huggingface_embedder()
        elif self.provider == EmbeddingProvider.OLLAMA:
            return self._create_ollama_embedder()
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")
    
   
    def _create_sentence_transformer_embedder(self) -> Callable:
        """Create an embedding function using Sentence Transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading Sentence Transformer model: {self.model_name}")
            cache_folder = self.cache_dir if self.cache_dir else None
            
            model = SentenceTransformer(
                self.model_name, 
                cache_folder=cache_folder,
                **self.kwargs
            )
            self._model = model
            
            # Log successful loading and model details
            logger.info(f"Successfully loaded Sentence Transformer model: {self.model_name}")
            logger.info(f"Model embedding dimension: {model.get_sentence_embedding_dimension()}")

            def embed_function(texts: List[str]) -> List[List[float]]:
                if not texts:
                    return []
                    
                # Handle batching if needed
                all_embeddings = []
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    batch_embeddings = model.encode(
                        batch, 
                        show_progress_bar=self.show_progress,
                        convert_to_numpy=True
                    )
                    all_embeddings.extend(batch_embeddings.tolist())
                return all_embeddings
                
            return embed_function
            
        except ImportError:
            logger.error("Sentence Transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error creating Sentence Transformer embedder: {e}")
            raise
    
    def _create_openai_embedder(self) -> Callable:
        """Create an embedding function using OpenAI API."""
        try:
            from openai import OpenAI

            logger.info(f"Initializing OpenAI embedding model: {self.model_name}")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

            client = OpenAI(api_key=api_key)
            self._model = client
            model = self.model_name
            
            logger.info(f"Using OpenAI embedding model: {model}")
            
            def embed_function(texts: List[str]) -> List[List[float]]:
                if not texts:
                    return []
                
                # Handle batching for OpenAI rate limits
                all_embeddings = []
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    try:
                        response = client.embeddings.create(
                            model=model,
                            input=batch,
                            **self.kwargs
                        )
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                    except Exception as e:
                        logger.error(f"OpenAI embedding error at batch {i}: {e}")
                        # Add empty vectors for failed items to maintain alignment
                        all_embeddings.extend([[0.0] * 1536] * len(batch))  # Typical OpenAI dimension
                return all_embeddings
                
            return embed_function
            
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Error creating OpenAI embedder: {e}")
            raise
    
    def _create_huggingface_embedder(self) -> Callable:
        """Create an embedding function using Hugging Face Transformers."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Optional API key for gated models
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if api_key:
                from huggingface_hub import login
                login(api_key)
            
            logger.info(f"Loading Hugging Face model: {self.model_name}")
            
            # Load model with caching if specified
            cache_dir = self.cache_dir if self.cache_dir else None
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
            model = AutoModel.from_pretrained(self.model_name, cache_dir=cache_dir)
            
            self._model = model
            
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            def embed_function(texts: List[str]) -> List[List[float]]:
                if not texts:
                    return []
                    
                all_embeddings = []
                # Process in batches
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    
                    # Tokenize and prepare inputs
                    encoded_input = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        model_output = model(**encoded_input)
                    
                    # Use CLS token as sentence embedding
                    sentence_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
                    all_embeddings.extend(sentence_embeddings.tolist())
                
                return all_embeddings
                
            return embed_function
            
        except ImportError:
            logger.error("Transformers not installed. Run: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Error creating Hugging Face embedder: {e}")
            raise
    
    def _create_ollama_embedder(self) -> Callable:
        """Create an embedding function using Ollama API."""
        try:
            from langchain_ollama import OllamaEmbeddings
            
            # Set a default base URL if none was specified
            ollama_base_url = "http://localhost:11434"  # Default local Ollama address
            
            # Create Ollama embeddings
            logger.info(f"Initializing Ollama embeddings with model: {self.model_name} at {ollama_base_url}")
            
            # Create the base embedder
            embedder = OllamaEmbeddings(
                model=self.model_name or "nomic-embed-text",
                base_url=ollama_base_url
            )
            
            # We'll need to create a custom embedding function that ensures float values
            def embed_function(texts: List[str]) -> List[List[float]]:
                if not texts:
                    return []
                
                try:
                    # Use the embedder's method to get embeddings
                    if len(texts) == 1:
                        # Single text
                        result = embedder.embed_query(texts[0])
                        # Ensure all values are floats
                        return [[float(x) for x in result]]
                    else:
                        # Multiple texts
                        results = []
                        for text in texts:
                            embedding = embedder.embed_query(text)
                            # Ensure all values are floats
                            results.append([float(x) for x in embedding])
                        return results
                except Exception as e:
                    logger.error(f"Ollama embedding error: {e}")
                    raise
            
            return embed_function
            
        except ImportError:
            logger.error("LangChain and Ollama dependencies not installed. Run: pip install langchain-ollama")
            raise
        except Exception as e:
            logger.error(f"Error creating Ollama embedder: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Basic input validation
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
            
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if len(valid_texts) < len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
        
        if not valid_texts:
            return []
            
        try:
            return self.embedding_function(valid_texts)
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            # Return zero vectors in case of failure to maintain data alignment
            return [[0.0] * self.embedding_dimension] * len(valid_texts)
    

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of document dictionaries with text and metadata
            
        Returns:
            Documents with embeddings added
        """
        if not documents:
            return []
        
        # Extract text content
        texts = [doc["page_content"] for doc in documents]
        
        # Generate embeddings
        if not self._embed_function:
            logger.info(f"Initializing embedding function with provider: {self.provider}")
            self._embed_function = self._create_embedding_function()
        
        # This is the critical line to add - log the actual embedding generation
        if self.provider == EmbeddingProvider.OPENAI:
            logger.info(f"Generating embeddings using OpenAI model: {self.model_name}")
        elif self.provider == EmbeddingProvider.SENTENCE_TRANSFORMER:
            logger.info(f"Generating embeddings using Sentence Transformer model: {self.model_name}")
        elif self.provider == EmbeddingProvider.HUGGINGFACE:
            logger.info(f"Generating embeddings using HuggingFace model: {self.model_name}")
        elif self.provider == EmbeddingProvider.OLLAMA:
            logger.info(f"Generating embeddings using Ollama model: {self.model_name}")
            
        start_time = time.time()
        embeddings = self._embed_function(texts)
        end_time = time.time()
        
        logger.info(f"Generated {len(embeddings)} embeddings in {end_time - start_time:.2f}s")
        
        # Add embeddings to documents
        result = []
        for i, doc in enumerate(documents):
            result.append({
                **doc,
                "embedding": embeddings[i]
            })
        
        return result