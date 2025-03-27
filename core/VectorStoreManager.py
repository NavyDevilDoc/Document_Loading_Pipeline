"""
VectorStorageManager.py

A unified interface for vector database storage operations with support
for multiple backends including Chroma, Pinecone, and in-memory storage.
"""

import os
import logging
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional
import chromadb
from datetime import datetime
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class StorageType(str, Enum):
    """Storage provider options."""
    CHROMA = "chroma"
    PINECONE = "pinecone" 
    MEMORY = "memory"

class VectorStorageManager:
    
    def __init__(
        self,
        storage_type: str = StorageType.CHROMA,
        collection_name: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
        storage_path: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        pinecone_environment: str = "gcp-starter",
        pinecone_create_new: bool = False,
        pinecone_namespace: Optional[str] = None,
        chroma_delete: bool = False,
    ):
        
        if isinstance(storage_type, StorageType):
            self.storage_type = storage_type.value
        else:
            self.storage_type = str(storage_type).lower()
    
        self.collection_name = collection_name or "document_collection"
        self.embedding_function = embedding_function
        self.storage_path = storage_path
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment
        self.pinecone_create_new = pinecone_create_new
        self.pinecone_namespace = pinecone_namespace 
        self.chroma_delete = chroma_delete
        
        # Initialize storage client
        self.client = None
        self.store = None
        
        # Validate storage type
        self._validate_storage_type()
    

    def _validate_storage_type(self):
        if self.storage_type == "pinecone" and not PINECONE_AVAILABLE:
            raise ImportError(
                "Pinecone storage selected but dependencies not installed. "
                "Run: pip install pinecone-client langchain-pinecone"
            )
        elif self.storage_type == "chroma" and not CHROMA_AVAILABLE:
            raise ImportError(
                "Chroma storage selected but dependencies not installed. "
                "Run: pip install chromadb langchain-community"
            )
    

    def initialize_store(self):
        """Initialize the vector store based on selected type."""
        logger.info(f"Initializing {self.storage_type} vector store")
        # Handle different storage types
        if self.storage_type == "chroma":
            return self._initialize_chroma()
        elif self.storage_type == "pinecone":
            return self._initialize_pinecone()
        elif self.storage_type == "memory":
            return self._initialize_memory()
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    

    def _initialize_chroma(self):
        """Initialize Chroma vector database."""
        if not self.embedding_function:
            raise ValueError("Embedding function required for Chroma")
            
        # Set up storage path
        persist_dir = self.storage_path or os.path.join(os.getcwd(), "vector_db")
        logger.info(f"Initializing Chroma at {persist_dir}")
        
        # Create directory if it doesn't exist
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get embedding dimension
        embedding_dim = self._get_embedding_dimension()
        
        # Build comprehensive embedding metadata
        embedding_metadata = {
            "embedding_dimension": embedding_dim,
            "created_at": datetime.now().isoformat()
        }
        
        # Add embedding model details
        if hasattr(self.embedding_function, "embedding_manager"):
            # This is our custom wrapper class, extract info directly
            manager = self.embedding_function.embedding_manager
            embedding_metadata["embedding_provider"] = manager.provider
            embedding_metadata["embedding_model"] = manager.model_name or "default"
        elif hasattr(self.embedding_function, "model_name"):
            # Direct access to model_name
            embedding_metadata["embedding_model"] = self.embedding_function.model_name
            
        # Try to identify the provider from common attributes
        if "embedding_provider" not in embedding_metadata:
            for provider_name in ["openai", "ollama", "huggingface", "sentence_transformers"]:
                if hasattr(self.embedding_function, provider_name) or provider_name in str(self.embedding_function.__class__):
                    embedding_metadata["embedding_provider"] = provider_name
                    break
        
        logger.info(f"Creating Chroma collection with metadata: {embedding_metadata}")
        
        # Create Chroma collection with metadata
        try:
            try:
                existing_collection = self.client.get_collection(name=self.collection_name)
                has_collection = True
            except:
                has_collection = False
                
            if has_collection and not self.chroma_delete:
                # Collection exists and we're not deleting it
                logger.info(f"Using existing Chroma collection: {self.collection_name}")
                # Don't update metadata on existing collection
            else:
                # Create new collection with metadata
                if has_collection:
                    logger.info(f"Deleting existing Chroma collection: {self.collection_name}")
                    self.client.delete_collection(name=self.collection_name)
                    
                logger.info(f"Creating new Chroma collection: {self.collection_name}")
                self.client.create_collection(
                    name=self.collection_name,
                    metadata=embedding_metadata
                )
                logger.info(f"Chroma collection created with metadata: {embedding_metadata}")
        except Exception as e:
            logger.error(f"Error creating/managing Chroma collection: {e}")
            # Continue anyway to create the store object
        
        # Create LangChain Chroma store 
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=persist_dir
        )
        return self.store
    

    def _initialize_pinecone(self):
        """Initialize Pinecone vector store."""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone dependencies not installed")
            
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key required")
            
        if not self.embedding_function:
            raise ValueError("Embedding function required for Pinecone")
        
        # Clean collection name for Pinecone compatibility
        index_name = self._clean_pinecone_name(self.collection_name)
        logger.info(f"Initializing Pinecone with index: {index_name}")
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Check if we need to create a new index
        if self.pinecone_create_new:
            # Delete existing index if it exists
            if index_name in pc.list_indexes().names():
                logger.info(f"Deleting existing Pinecone index: {index_name}")
                pc.delete_index(index_name)
                
            # Get embedding dimensions from embedding function
            embed_dim = self._get_embedding_dimension()
            logger.info(f"Creating new Pinecone index with dimension {embed_dim}")
            
            # Create new index
            spec = ServerlessSpec(cloud='aws', region='us-east-1')
            pc.create_index(
                name=index_name,
                dimension=embed_dim,
                metric='cosine',
                spec=spec
            )
            
            # Wait for index to be ready
            start_time = time.time()
            while not pc.describe_index(index_name).status['ready']:
                if time.time() - start_time > 300:  # 5 minute timeout
                    raise TimeoutError("Pinecone index creation timed out")
                time.sleep(1)
                
            logger.info(f"Pinecone index {index_name} is ready")
        
            self.pinecone_create_new = False  # Avoid recreating the index for each document

        # Create Pinecone store
        self.store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=self.embedding_function,
            namespace=self.pinecone_namespace
        )
        
        # If we just created a new index, store metadata
        if self.pinecone_create_new:
            self._store_pinecone_metadata()

        return self.store
    

    def _initialize_memory(self):
        """Initialize in-memory vector store."""
        if not self.embedding_function:
            raise ValueError("Embedding function required for in-memory store")
            
        logger.info("Initializing in-memory vector store")
        self.store = InMemoryVectorStore(embedding=self.embedding_function)
        return self.store
    

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not self.store:
            self.initialize_store()
                
        logger.info(f"Adding {len(documents)} documents to {self.storage_type} store")
        try:
            # Check embedding compatibility for existing collections
            if not self._check_embedding_compatibility(self.collection_name):
                if self.storage_type == "chroma":
                    flag = "--chroma-delete"
                elif self.storage_type == "pinecone":
                    flag = "--pinecone-create-new"
                else:
                    flag = "(appropriate flag)"
                        
                raise ValueError(
                    f"CRITICAL: Embedding model incompatibility detected for collection '{self.collection_name}'. "
                    f"Adding documents with a different model would corrupt the vector store! "
                    f"Use {flag} flag to recreate the collection with the new model."
                )

            # If using Pinecone, add embedding metadata to each document
            if self.storage_type == "pinecone":
                # Get current model info
                current_model = "Unknown"
                current_provider = "Unknown"
                
                if hasattr(self.embedding_function, "embedding_manager"):
                    current_model = self.embedding_function.embedding_manager.model_name
                    current_provider = self.embedding_function.embedding_manager.provider
                elif hasattr(self.embedding_function, "model_name"):
                    current_model = self.embedding_function.model_name
                    
                # Add model metadata to each document
                for doc in documents:
                    # Only add if not already present
                    if "embedding_model" not in doc.metadata:
                        doc.metadata["embedding_model"] = current_model
                    if "embedding_provider" not in doc.metadata:
                        doc.metadata["embedding_provider"] = current_provider

            # Set up additional parameters for specific storage types
            kwargs = {}
            if self.storage_type == "pinecone" and self.pinecone_namespace:
                kwargs["namespace"] = self.pinecone_namespace
                logger.info(f"Using Pinecone namespace: {self.pinecone_namespace}")
            
            # Add documents with any additional parameters
            ids = self.store.add_documents(documents, **kwargs)
            
            # Provide detailed success logging based on storage type
            if self.storage_type == "pinecone":
                if self.pinecone_namespace:
                    logger.info(f"Successfully added {len(documents)} documents to Pinecone index '{self.collection_name}' in namespace '{self.pinecone_namespace}'")
                else:
                    logger.info(f"Successfully added {len(documents)} documents to Pinecone index '{self.collection_name}'")
            elif self.storage_type == "chroma":
                logger.info(f"Successfully added {len(documents)} documents to Chroma collection '{self.collection_name}'")
            else:
                logger.info(f"Successfully added {len(documents)} documents to {self.storage_type} store")
            
            # Persist if needed
            if hasattr(self.store, "persist") and callable(getattr(self.store, "persist")):
                self.store.persist()
                
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    

    def search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """
        Search for documents similar to query.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of similar documents
        """
        if not self.store:
            raise RuntimeError("Vector store not initialized")
            
        logger.info(f"Searching with query: '{query[:50]}...' (k={k})")
        
        try:
            results = self.store.similarity_search_with_score(query, k=k, **kwargs)
            logger.info(f"Search returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise


    def create_new_pinecone_index(self):
        """Create a new Pinecone index without adding documents."""
        if self.storage_type != "pinecone":
            raise ValueError("This method is only for Pinecone storage")
            
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone dependencies not installed")
                
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key required")
        
        # Clean collection name for Pinecone compatibility
        index_name = self._clean_pinecone_name(self.collection_name)
        logger.info(f"Creating new Pinecone index: {index_name}")
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Delete existing index if it exists
        if index_name in pc.list_indexes().names():
            logger.info(f"Deleting existing Pinecone index: {index_name}")
            pc.delete_index(index_name)
                
        # Get embedding dimensions from embedding function
        embed_dim = self._get_embedding_dimension()
        logger.info(f"Creating new Pinecone index with dimension {embed_dim}")
        
        # Create new index
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
        pc.create_index(
            name=index_name,
            dimension=embed_dim,
            metric='cosine',
            spec=spec
        )
        
        # Wait for index to be ready
        start_time = time.time()
        while not pc.describe_index(index_name).status['ready']:
            if time.time() - start_time > 300:  # 5 minute timeout
                raise TimeoutError("Pinecone index creation timed out")
            time.sleep(1)
                
        logger.info(f"Pinecone index {index_name} is ready")
        
        # Set the flag to false to avoid recreating the index for each document
        self.pinecone_create_new = False
        
        return True


    def delete_pinecone_index(self):
        """
        Delete a Pinecone index if it exists.
        
        Returns:
            bool: True if index was deleted, False if it didn't exist
        """
        if self.storage_type != "pinecone":
            raise ValueError("This method is only for Pinecone storage")
        
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone dependencies not installed")
                
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key required")
        
        try:
            # Clean collection name for Pinecone compatibility
            index_name = self._clean_pinecone_name(self.collection_name)
            
            # Initialize Pinecone client
            pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists
            index_exists = index_name in pc.list_indexes().names()
            
            if not index_exists:
                logger.info(f"Pinecone index '{index_name}' does not exist, nothing to delete")
                return False
            
            # Delete the index
            logger.info(f"Deleting Pinecone index: {index_name}")
            pc.delete_index(index_name)
            logger.info(f"Successfully deleted Pinecone index: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting Pinecone index: {e}")
            raise


    def delete_chroma_collection(self):
        """
        Delete a Chroma collection if it exists.
        
        Returns:
            bool: True if collection was deleted, False if it didn't exist
        """
        if self.storage_type != "chroma":
            raise ValueError("This method is only for Chroma storage")
        
        # Create client if it doesn't exist
        if not self.client:
            persist_dir = self.storage_path or os.path.join(os.getcwd(), "vector_db")
            self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Try to get and delete the collection
        try:
            collection_exists = False
            try:
                # Check if collection exists first
                self.client.get_collection(self.collection_name)
                collection_exists = True
            except:
                logger.info(f"Collection '{self.collection_name}' does not exist, nothing to delete")
                return False
        
            # Delete the collection if it exists
            if collection_exists:
                logger.info(f"Deleting Chroma collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                logger.info(f"Successfully deleted Chroma collection: {self.collection_name}")
                return True
        
        except Exception as e:
            logger.error(f"Error deleting Chroma collection: {e}")
            raise


    def _clean_pinecone_name(self, name: str) -> str:
        """Clean a name to be Pinecone-compatible (lowercase alphanumeric and hyphens only)."""
        import re
        # Replace any non-alphanumeric character with a hyphen (including underscores)
        clean_name = re.sub(r'[^a-zA-Z0-9-]', '-', name)
        # Ensure lowercase
        clean_name = clean_name.lower()
        # Remove consecutive hyphens
        clean_name = re.sub(r'-+', '-', clean_name)
        # Remove leading and trailing hyphens
        clean_name = clean_name.strip('-')
        # Ensure name starts with letter/number
        if not clean_name or not clean_name[0].isalnum():
            clean_name = 'idx-' + clean_name
        # Truncate if too long (Pinecone limit is 45 chars)
        return clean_name[:45]
    

    def _get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model by testing it."""
        if not self.embedding_function:
            raise ValueError("No embedding function available")
            
        # Test the embedding function with a simple text
        test_embedding = self.embedding_function.embed_query("test")
        return len(test_embedding)
    

    def _check_embedding_compatibility(self, collection_name: str) -> bool:
        """
        Check if the current embedding model is compatible with the existing collection.
        
        Returns:
            bool: True if compatible or new collection, False if incompatible
        """
        try:
            # Get current model info
            current_model = "Unknown"
            current_provider = "Unknown"
            
            if hasattr(self.embedding_function, "embedding_manager"):
                # Our custom wrapper class
                current_model = self.embedding_function.embedding_manager.model_name
                current_provider = self.embedding_function.embedding_manager.provider
            elif hasattr(self.embedding_function, "model_name"):
                current_model = self.embedding_function.model_name
                
            # Get current embedding dimension
            try:
                current_dim = self._get_embedding_dimension()
            except Exception as e:
                logger.warning(f"Couldn't determine current embedding dimension: {e}")
                current_dim = 0
            
            # Check compatibility based on storage type
            if self.storage_type == "chroma":
                return self._check_chroma_compatibility(collection_name, current_model, current_provider, current_dim)
            elif self.storage_type == "pinecone":
                return self._check_pinecone_compatibility(collection_name, current_model, current_provider, current_dim)
            else:
                # For other storage types, assume compatible
                return True
                    
        except Exception as e:
            logger.error(f"Error checking embedding compatibility: {e}")
            # Default to DISALLOWING the operation in case of errors
            return False
        
    def _check_chroma_compatibility(self, collection_name: str, current_model: str, current_provider: str, current_dim: int) -> bool:
        """Check compatibility with Chroma collection."""
        # Get client if not already initialized
        if not self.client:
            persist_dir = self.storage_path or os.path.join(os.getcwd(), "vector_db")
            self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Check if collection exists
        try:
            collection = self.client.get_collection(name=collection_name)
        except Exception:
            # Collection doesn't exist yet, so compatibility is fine
            logger.debug(f"Chroma collection '{collection_name}' doesn't exist yet, compatibility check passed")
            return True
            
        # Collection exists, get its metadata
        metadata = collection.metadata or {}
        
        collection_model = metadata.get("embedding_model", "Unknown")
        collection_provider = metadata.get("embedding_provider", "Unknown")
        collection_dim = metadata.get("embedding_dimension", 0)
        
        # Check for dimension mismatch (non-negotiable - different dimensions will break storage)
        if collection_dim and current_dim and collection_dim != current_dim:
            logger.error(f"Embedding dimension mismatch: collection={collection_dim}, current={current_dim}")
            return False
            
        # Check for model mismatch (required for consistency)
        if current_model != "Unknown" and collection_model != "Unknown":
            if current_model != collection_model:
                logger.error(f"Embedding model mismatch: collection='{collection_model}', current='{current_model}'")
                logger.error(f"Adding documents with a different embedding model would corrupt the vector store!")
                logger.error(f"Use --chroma-delete to recreate the collection with the new model")
                return False
                
        # Check for provider mismatch (enforced)
        if current_provider != "Unknown" and collection_provider != "Unknown":
            if current_provider != collection_provider:
                logger.error(f"Embedding provider mismatch: collection='{collection_provider}', current='{current_provider}'")
                logger.error(f"Adding documents with a different embedding provider would corrupt the vector store!")
                logger.error(f"Use --chroma-delete to recreate the collection with the new provider")
                return False
        
        # All checks passed
        logger.info(f"Embedding compatibility check passed for Chroma collection '{collection_name}'")
        return True
    

    def _check_pinecone_compatibility(self, collection_name: str, current_model: str, current_provider: str, current_dim: int) -> bool:
        """Check compatibility with Pinecone index."""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone dependencies not installed")
            
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key required for compatibility check")
        
        # Clean collection name for Pinecone compatibility
        index_name = self._clean_pinecone_name(collection_name)
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Check if index exists
        try:
            index_list = pc.list_indexes().names()
            if index_name not in index_list:
                # Index doesn't exist yet, so compatibility is fine
                logger.debug(f"Pinecone index '{index_name}' doesn't exist yet, compatibility check passed")
                return True
                
            # Index exists, get its dimension from the API
            index_info = pc.describe_index(index_name)
            collection_dim = index_info.dimension
            collection_model = "Unknown"
            collection_provider = "Unknown"
            
            try:
                # Create a temporary connection to the index
                index = pc.Index(index_name)
                fetch_response = index.fetch(ids=["metadata_vector"], namespace=self.pinecone_namespace or "")
                
                # Check if we got a response and if it contains metadata
                if fetch_response.vectors and "metadata_vector" in fetch_response.vectors:
                    metadata = fetch_response.vectors["metadata_vector"].metadata or {}
                    collection_model = metadata.get("embedding_model", "Unknown")
                    collection_provider = metadata.get("embedding_provider", "Unknown")
                    logger.info(f"Found metadata in Pinecone index: model={collection_model}, provider={collection_provider}")
                else:
                    logger.warning(f"No metadata vector found in Pinecone index '{index_name}'")

                    try:
                        stats = index.describe_index_stats()
                        if stats.total_vector_count > 0:
                            # Try to query for a sample vector with metadata
                            query_response = index.query(
                                vector=[0.0] * collection_dim,
                                top_k=1,
                                include_metadata=True,
                                namespace=self.pinecone_namespace or ""
                            )
                            
                            if query_response.matches and query_response.matches[0].metadata:
                                metadata = query_response.matches[0].metadata
                                if "embedding_model" in metadata:
                                    collection_model = metadata.get("embedding_model")
                                    collection_provider = metadata.get("embedding_provider", "Unknown")
                                    logger.info(f"Found metadata in sample vector: model={collection_model}, provider={collection_provider}")
                    except Exception as e:
                        logger.warning(f"Could not query Pinecone for sample metadata: {e}")
            except Exception as e:
                logger.warning(f"Error retrieving metadata from Pinecone: {e}")
                    
                # Check for dimension mismatch (critical)
                if collection_dim != current_dim:
                    logger.error(f"Embedding dimension mismatch: index={collection_dim}, current={current_dim}")
                    logger.error(f"Cannot add documents with different dimensions to Pinecone index")
                    logger.error(f"Use --pinecone-create-new to recreate the index with the new model")
                    return False
                
            # Log warnings for model/provider mismatches
            # For Pinecone, we can't enforce these without reliable metadata, so we just warn
            if collection_model != "Unknown" and current_model != "Unknown" and collection_model != current_model:
                logger.error(f"Embedding model mismatch: index='{collection_model}', current='{current_model}'")
                logger.error(f"Adding documents with a different embedding model may corrupt search results!")
                logger.error(f"Use --pinecone-create-new to recreate the index with the new model")
                # Continue anyway since dimension matches
            
            if collection_provider != "Unknown" and current_provider != "Unknown" and collection_provider != current_provider:
                logger.error(f"Embedding provider mismatch: index='{collection_provider}', current='{current_provider}'")
                logger.error(f"Adding documents with a different embedding provider may corrupt search results!")
                logger.error(f"Use --pinecone-create-new to recreate the index with the new model")
                return False
            
            # If we got here but don't have model/provider info, store it now
            if collection_model == "Unknown" or collection_provider == "Unknown":
                logger.info(f"Storing embedding metadata in Pinecone index '{index_name}'")
                self._store_pinecone_metadata()
                
            # Dimensions match, so technically compatible even if model differs
            logger.info(f"Embedding dimension compatibility check passed for Pinecone index '{index_name}'")
            return True
                
        except Exception as e:
            logger.error(f"Error checking Pinecone compatibility: {e}")
            return False
        

    def _store_pinecone_metadata(self):
        """Store embedding metadata in Pinecone index for future compatibility checks."""
        if self.storage_type != "pinecone":
            return
                
        try:
            # Get current model info
            current_model = "Unknown"
            current_provider = "Unknown"
            
            if hasattr(self.embedding_function, "embedding_manager"):
                current_model = self.embedding_function.embedding_manager.model_name
                current_provider = self.embedding_function.embedding_manager.provider
            elif hasattr(self.embedding_function, "model_name"):
                current_model = self.embedding_function.model_name
                    
            current_dim = self._get_embedding_dimension()
            
            # Create metadata document
            metadata = {
                "embedding_model": current_model,
                "embedding_provider": current_provider,
                "embedding_dimension": current_dim,
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"Storing metadata in Pinecone: {metadata}")
            
            # Get index for direct upsert
            index_name = self._clean_pinecone_name(self.collection_name)
            pc = Pinecone(api_key=self.pinecone_api_key)
            index = pc.Index(index_name)
            
            # Create a dummy vector full of zeros
            dummy_vector = [0.0] * current_dim
            
            # Store metadata in a special vector
            try:
                # Upsert the metadata vector with a special ID
                index.upsert(
                    vectors=[{
                        "id": "metadata_vector",
                        "values": dummy_vector,
                        "metadata": metadata
                    }],
                    namespace=self.pinecone_namespace or ""
                )
                
                logger.info(f"Successfully stored metadata in special vector")
            except Exception as e:
                logger.warning(f"Error storing metadata vector: {e}")
                
                # Fallback: Add metadata to each document as we insert them
                # We'll modify add_documents to include embedding metadata in every document
                
                logger.info(f"Will attach metadata to individual documents instead")
                
            logger.info(f"Stored embedding metadata in Pinecone index '{index_name}'")
                
        except Exception as e:
            logger.warning(f"Could not store metadata in Pinecone index: {e}")