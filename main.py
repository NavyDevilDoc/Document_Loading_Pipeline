"""
Document Processing Pipeline

A comprehensive pipeline for document processing:
1. Load and OCR documents
2. Preprocess text
3. Split into chunks
4. Generate embeddings
5. Store in vector database
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from core.ChunkingManager import ChunkingManager, ChunkingStrategy
from core.EmbeddingManager import EmbeddingManager, EmbeddingProvider
from core.VectorStoreManager import VectorStorageManager, StorageType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose HTTP logs
for logger_name in ["httpx", "urllib3", "httpcore.connection", "httpcore.http11"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def load_environment_variables(env_file_path):
    """Load environment variables from file and check for critical API keys."""
    path = Path(env_file_path)
    
    if not path.exists():
        logger.warning(f"Environment file not found: {path}")
        return False
        
    success = load_dotenv(dotenv_path=str(path), override=True)
    if not success:
        logger.warning(f"Failed to load environment variables from: {path}")
        return False
        
    logger.info(f"Environment variables loaded from: {path}")
    
    # Check for critical API keys
    api_keys = ["OPENAI_API_KEY", "PINECONE_API_KEY", "HUGGINGFACE_API_KEY"]
    missing_keys = []
    
    for key in api_keys:
        if os.getenv(key):
            logger.info(f"{key} found in environment variables")
        else:
            logger.warning(f"{key} not found in environment variables")
            missing_keys.append(key)
    
    return True


class DocumentProcessingPipeline:
    """
    Complete pipeline for document processing, from loading to vector storage.
    """
    
    def __init__(
        self,
        chunking_strategy: str = ChunkingStrategy.PARAGRAPH,
        embedding_provider: str = EmbeddingProvider.SENTENCE_TRANSFORMER,
        embedding_model: Optional[str] = None,
        vector_db_path: Optional[str] = None,
        preprocess: bool = True,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        storage_type: str = StorageType.CHROMA,
        pinecone_api_key: Optional[str] = None,
        pinecone_create_new: bool = False,
        pinecone_namespace: Optional[str] = None,
        chroma_delete: bool = False
    ):
        self.chunking_strategy = chunking_strategy
        self.embedding_provider = embedding_provider
        self.preprocess = preprocess
        storage_type_value = storage_type.value if isinstance(storage_type, StorageType) else storage_type
        self.storage_type = storage_type_value
        self.pinecone_namespace = pinecone_namespace
        
        # Initialize embedding manager first
        logger.info(f"Initializing embeddings with provider: {embedding_provider}")
        self.embedding_manager = EmbeddingManager(
            provider=embedding_provider,
            model_name=embedding_model,
            cache_dir=cache_dir,
            batch_size=batch_size,
            show_progress=True
        )

        # Determine appropriate tokenizer model
        token_model = None
        if embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMER:
            token_model = embedding_model or "all-MiniLM-L6-v2"
        else:
            token_model = "gpt-4"
        
        # Initialize chunking manager
        logger.info(f"Initializing chunking with strategy: {chunking_strategy}")
        self.chunking_manager = ChunkingManager(
            embedding_model_name=token_model,
            token_model_name="cl100k_base"
        )
        
        # Initialize vector storage manager
        logger.info(f"Initializing vector storage with provider: {storage_type_value}")
        storage_type_value = storage_type.value if isinstance(storage_type, StorageType) else storage_type
        logger.info(f"Using storage type: {storage_type_value}")
        self.vector_store_manager = VectorStorageManager(
            storage_type=storage_type_value,
            collection_name=None,
            embedding_function=self._get_langchain_embeddings(),
            storage_path=vector_db_path,
            pinecone_api_key=pinecone_api_key,
            pinecone_create_new=pinecone_create_new,
            pinecone_namespace=pinecone_namespace,
            chroma_delete=chroma_delete
        )
        
        # Initialize vector database path
        self.vector_db_path = vector_db_path or os.path.join(os.getcwd(), "vector_db")
        logger.info(f"Vector database location: {self.vector_db_path}")
    

    def _normalize_document_list(
        self, 
        chunks: Union[List[Document], Dict[str, List[Document]]]
    ) -> List[Document]:
        """Normalize document output from different chunkers to a flat list."""
        if isinstance(chunks, dict):
            # For hierarchical chunkers, use the chunks, not the pages
            if "chunks" in chunks:
                return chunks["chunks"]
            # Flatten any dictionary of lists
            flat_list = []
            for doc_list in chunks.values():
                if isinstance(doc_list, list):
                    flat_list.extend(doc_list)
            return flat_list
        return chunks
    

    def _get_langchain_embeddings(self) -> Embeddings:
        """Get LangChain-compatible embeddings interface."""
        from langchain_core.embeddings import Embeddings
        
        class LangChainEmbeddings(Embeddings):
            def __init__(self, embedding_manager):
                self.embedding_manager = embedding_manager
                # Store provider and model info directly for easier metadata access
                self.provider = embedding_manager.provider
                self.model_name = embedding_manager.model_name
                
            def embed_documents(self, texts):
                docs = [{"page_content": text, "metadata": {}} for text in texts]
                embedded_docs = self.embedding_manager.embed_documents(docs)
                return [doc["embedding"] for doc in embedded_docs]
                
            def embed_query(self, text):
                docs = [{"page_content": text, "metadata": {}}]
                embedded_docs = self.embedding_manager.embed_documents(docs)
                return embedded_docs[0]["embedding"]
        
        return LangChainEmbeddings(self.embedding_manager)
    

    def process_document(
        self, 
        file_path: str, 
        collection_name: Optional[str] = None,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            collection_name: Name for the vector store collection
            document_metadata: Additional metadata for the document
                
        Returns:
            Dictionary with processing results and statistics
        """
        path = Path(file_path)

        # Create document-level metadata
        doc_metadata = {
            "document_name": path.name,
            "document_path": str(path),
            "document_type": path.suffix.lower().replace('.', ''),
            "document_size_kb": round(path.stat().st_size / 1024, 2),
            "processing_timestamp": datetime.now().isoformat(),
            "chunking_strategy": self.chunking_strategy,
            "preprocessing_applied": self.preprocess,
        }
        
        # Add any custom metadata provided
        if document_metadata:
            doc_metadata.update(document_metadata)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Default collection name is the filename without extension
        if not collection_name:
            collection_name = path.stem
        
        logger.info(f"Processing document: {path.name}")
        
        # Step 1 & 2: Load, OCR, and chunk document
        chunks = self.chunking_manager.process_document(
            file_path=str(path),
            strategy=self.chunking_strategy,
            preprocess=self.preprocess
        )
        
        # Normalize chunks to flat list
        document_chunks = self._normalize_document_list(chunks)
        logger.info(f"Created {len(document_chunks)} document chunks")
        
        # Step 3: Generate embeddings
        chunk_dicts = [
            {
                "id": f"{path.stem}_{i}",  # Include document name in ID
                "page_content": doc.page_content,
                "metadata": {
                    **doc.metadata,  # Original chunk metadata
                    **doc_metadata,  # Document-level metadata
                }
            }
            for i, doc in enumerate(document_chunks)
        ]
        
        embedded_chunks = self.embedding_manager.embed_documents(chunk_dicts)

        if embedded_chunks:
            logger.info(f"Generated embeddings with dimension: {len(embedded_chunks[0]['embedding'])}")
        else:
            logger.warning("No chunks were embedded")        

        # Step 4: Store in vector database (only if collection is specified)
        if collection_name:
            logger.info(f"Adding document to collection: {collection_name}")

            # Set collection name
            self.vector_store_manager.collection_name = collection_name
            
            # Check if we need to delete existing collection first
            if self.storage_type == "chroma" and self.vector_store_manager.chroma_delete:
                try:
                    logger.info(f"Attempting to delete existing Chroma collection '{collection_name}' due to --chroma-delete flag")
                    deleted = self.vector_store_manager.delete_chroma_collection()
                    if deleted:
                        logger.info(f"Successfully deleted existing Chroma collection: {collection_name}")
                    else:
                        logger.info(f"No existing Chroma collection '{collection_name}' to delete")
                    # Reset the flag so we don't try to delete again
                    self.vector_store_manager.chroma_delete = False
                except Exception as e:
                    logger.warning(f"Error deleting Chroma collection: {e}")

            # Convert embedded chunks back to Document objects
            doc_objects = [
                Document(
                    page_content=chunk["page_content"],
                    metadata=chunk["metadata"]
                ) for chunk in embedded_chunks
            ]
            
            # Initialize store and add documents
            try:
                self.vector_store_manager.initialize_store()
                ids = self.vector_store_manager.add_documents(doc_objects)
            
                result = {
                    "file_path": file_path,
                    "status": "success",
                    "chunks": len(doc_objects),
                    "collection": collection_name,
                    "ids": ids,
                    "storage_type": self.storage_type,
                    "embedding_provider": self.embedding_provider,
                    "embedding_model": self.embedding_manager.model_name,
                }
                # Add Pinecone-specific details if applicable
                if self.storage_type == "pinecone" and self.pinecone_namespace:
                    result["pinecone_namespace"] = self.pinecone_namespace
                    logger.info(f"Document processed and stored in Pinecone namespace: {self.pinecone_namespace}")
                    
                return result
            
            except Exception as e:
                if "Embedding dimension" in str(e) and "does not match collection dimensionality" in str(e):
                    # Extract dimensions from error message
                    error_msg = str(e)
                    import re
                    dims = re.findall(r'dimension (\d+).+?dimensionality (\d+)', error_msg)
                    if dims:
                        new_dim, old_dim = dims[0]
                        raise ValueError(
                            f"Dimension mismatch: Current embeddings ({new_dim}) don't match existing collection ({old_dim}). "
                            f"Use --chroma-delete to remove the existing collection first."
                        )
                raise  # Re-raise other exceptions
                
        else:
            logger.info("Document processing complete (no collection specified for storage)")
            return {
                "file_path": file_path,
                "status": "success",
                "chunks": len(document_chunks)
            }    


    def process_directory(
        self, 
        dir_path: str,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process all PDF documents in a directory.
        
        Args:
            dir_path: Directory containing PDF files
            collection_prefix: Prefix for collection names
            
        Returns:
            List of processing results for each document
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
            
        pdf_files = list(path.glob("**/*.pdf"))
        csv_files = list(path.glob("**/*.csv"))
        logger.info(f"Found {len(pdf_files)} PDF files and {len(csv_files)} CSV files in directory")
        
        # If using Pinecone with create_new, create the index once before processing
        if self.storage_type == "pinecone" and hasattr(self.vector_store_manager, 'pinecone_create_new') and self.vector_store_manager.pinecone_create_new:
            # Set the collection name
            self.vector_store_manager.collection_name = collection_name
            
            # Create the index once
            logger.info(f"Creating new Pinecone index '{collection_name}' for all documents")
            self.vector_store_manager.create_new_pinecone_index()
            
            # After creating the index, set flag to false to prevent recreation
            self.vector_store_manager.pinecone_create_new = False

        # If using Chroma with recreate, set the flag once before processing
        elif self.storage_type == "chroma" and hasattr(self.vector_store_manager, 'chroma_delete') and self.vector_store_manager.chroma_delete:
            # Set the collection name
            self.vector_store_manager.collection_name = collection_name
            
            try:
                deleted = self.vector_store_manager.delete_chroma_collection()
                if deleted:
                    logger.info(f"Deleted existing Chroma collection '{collection_name}' before processing")
                else:
                    logger.info(f"No existing Chroma collection '{collection_name}' to delete")
            except Exception as e:
                logger.warning(f"Failed to delete Chroma collection: {e}")
            
            self.vector_store_manager.chroma_delete = False

        results = []
        for pdf_file in pdf_files:
            try:
                result = self.process_document(
                    file_path=str(pdf_file),
                    collection_name=collection_name
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                results.append({
                    "file_path": str(pdf_file),
                    "status": "error",
                    "error": str(e)
                })        
        return results


    def _sanitize_pinecone_name(name: str) -> str:
        """Sanitize a name for use with Pinecone (lowercase, alphanumeric, and hyphens only)."""
        import re
        # Replace any non-alphanumeric character with a hyphen
        sanitized = re.sub(r'[^a-zA-Z0-9-]', '-', name)
        # Ensure lowercase
        sanitized = sanitized.lower()
        # Remove consecutive hyphens
        sanitized = re.sub(r'-+', '-', sanitized)
        # Remove leading and trailing hyphens
        sanitized = sanitized.strip('-')
        return sanitized


    def visualize_chunks(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        max_chunks: int = 5,
        show_metadata: bool = True
    ) -> str:
        """
        Process a document and save visualization of chunks for inspection.
        
        Args:
            file_path: Path to the document file
            output_dir: Directory to save visualization files (default: current dir)
            max_chunks: Maximum number of chunks to show in detail
            show_metadata: Whether to include metadata in visualization
            
        Returns:
            Path to the generated visualization file
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            project_dir = Path(__file__).parent
            output_path = project_dir / "test_results"

        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualization will be saved to: {output_path}")

        # Process document with current settings
        logger.info(f"Visualizing chunking for: {path.name}")
        chunks = self.chunking_manager.process_document(
            file_path=str(path),
            strategy=self.chunking_strategy,
            preprocess=self.preprocess
        )
        
        # Normalize chunks to flat list
        document_chunks = self._normalize_document_list(chunks)
        
        # Create HTML visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = output_path / f"chunks_visualization_{path.stem}_{timestamp}.html"
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Document Chunking Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            .chunk {{ border: 1px solid #ddd; margin-bottom: 15px; padding: 10px; border-radius: 5px; }}
            .chunk-content {{ white-space: pre-wrap; background-color: #f8f8f8; padding: 10px; border-radius: 3px; }}
            .metadata {{ font-size: 0.9em; color: #555; margin-top: 10px; }}
            .metadata-item {{ margin-bottom: 3px; }}
            .statistics {{ margin-top: 30px; }}
            .more-indicator {{ text-align: center; padding: 15px; background-color: #eeeeff; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Document Chunking Visualization</h1>
            <p><strong>File:</strong> {path.name}</p>
            <p><strong>Chunking Strategy:</strong> {self.chunking_strategy}</p>
            <p><strong>Embedding Provider:</strong> {self.embedding_provider}</p>
            <p><strong>Preprocessing:</strong> {'Enabled' if self.preprocess else 'Disabled'}</p>
            <p><strong>Total Chunks:</strong> {len(document_chunks)}</p>
        </div>

        <h2>Sample Chunks</h2>
        <p>Showing up to {max_chunks} chunks from the document:</p>
    """)

            # Show sample chunks
            for i, chunk in enumerate(document_chunks[:max_chunks]):
                f.write(f"""
        <div class="chunk">
            <h3>Chunk {i+1}</h3>
            <div class="chunk-content">{chunk.page_content}</div>
    """)
                
                if show_metadata:
                    f.write("""
            <div class="metadata">
                <h4>Metadata:</h4>
    """)
                    
                    for key, value in chunk.metadata.items():
                        f.write(f"""
                <div class="metadata-item"><strong>{key}:</strong> {value}</div>""")
                    
                    f.write("""
            </div>""")
                    
                f.write("""
        </div>""")
                
            # Show indicator if there are more chunks
            if len(document_chunks) > max_chunks:
                f.write(f"""
        <div class="more-indicator">
            <p>{len(document_chunks) - max_chunks} more chunks not shown</p>
        </div>
    """)
                
            # Add statistics section
            f.write("""
        <div class="statistics">
            <h2>Chunk Statistics</h2>
            <table>
                <tr>
                    <th>Statistic</th>
                    <th>Value</th>
                </tr>
    """)

            # Calculate statistics
            total_tokens = sum(chunk.metadata.get("token_count", 0) for chunk in document_chunks)
            avg_tokens = total_tokens / len(document_chunks) if document_chunks else 0
            min_tokens = min((chunk.metadata.get("token_count", 0) for chunk in document_chunks), default=0)
            max_tokens = max((chunk.metadata.get("token_count", 0) for chunk in document_chunks), default=0)
            
            total_chars = sum(len(chunk.page_content) for chunk in document_chunks)
            avg_chars = total_chars / len(document_chunks) if document_chunks else 0
            min_chars = min((len(chunk.page_content) for chunk in document_chunks), default=0)
            max_chars = max((len(chunk.page_content) for chunk in document_chunks), default=0)
            
            # Write statistics
            stats = [
                ("Total chunks", len(document_chunks)),
                ("Total tokens", total_tokens),
                ("Average tokens per chunk", f"{avg_tokens:.1f}"),
                ("Min tokens in a chunk", min_tokens),
                ("Max tokens in a chunk", max_tokens),
                ("Total characters", total_chars),
                ("Average characters per chunk", f"{avg_chars:.1f}"),
                ("Min characters in a chunk", min_chars),
                ("Max characters in a chunk", max_chars)
            ]
            
            for key, value in stats:
                f.write(f"""
                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>""")
                
            # Close HTML tags
            f.write("""
            </table>
        </div>
    </body>
    </html>
    """)
        
        # Test and log which embedding model would be used (without making API calls)
        if self.embedding_provider != EmbeddingProvider.SENTENCE_TRANSFORMER:
            if self.embedding_provider == EmbeddingProvider.OPENAI:
                logger.info(f"Note: Full processing would use OpenAI embedding model: {self.embedding_manager.model_name}")
            elif self.embedding_provider == EmbeddingProvider.OLLAMA:
                logger.info(f"Note: Full processing would use Ollama embedding model: {self.embedding_manager.model_name}")
            elif self.embedding_provider == EmbeddingProvider.HUGGINGFACE:
                logger.info(f"Note: Full processing would use HuggingFace embedding model: {self.embedding_manager.model_name}")

        logger.info(f"Visualization saved to: {html_path}")
        return str(html_path)


    def visualize_directory(
        self,
        dir_path: str,
        output_dir: Optional[str] = None,
        max_chunks_per_doc: int = 1,
        show_metadata: bool = True
    ) -> str:
        """
        Visualize sample chunks from all documents in a directory.
        
        Args:
            dir_path: Path to directory containing documents
            output_dir: Directory to save visualization files
            max_chunks_per_doc: Maximum chunks to show per document
            show_metadata: Whether to include metadata in visualization
            
        Returns:
            Path to the generated summary HTML file
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        # Create output directory - default to "test_results" in the project directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            # Get the project directory
            project_dir = Path(__file__).parent
            output_path = project_dir / "test_results"
        
        # Create the directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created or confirmed output directory at: {output_path.absolute()}")
        
        # Find all PDF files in the directory (recursively)
        pdf_files = list(path.glob("**/*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {dir_path}")
            return None
        
        logger.info(f"Found {len(pdf_files)} PDF files in {dir_path}")
        
        # Process each PDF and track results
        results = []
        for file_path in pdf_files:
            try:
                # Generate a visualization for this document with limited chunks
                vis_path = self.visualize_chunks(
                    file_path=str(file_path),
                    output_dir=str(output_path),
                    max_chunks=max_chunks_per_doc,
                    show_metadata=show_metadata
                )
                results.append({
                    "file_path": str(file_path),
                    "visualization_path": vis_path,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error visualizing {file_path}: {e}")
                results.append({
                    "file_path": str(file_path),
                    "error": str(e),
                    "status": "error"
                })
        
        # Generate summary HTML page
        summary_path = output_path / "directory_summary.html"
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Document Processing Summary</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                max-width: 1100px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .summary {
                margin-bottom: 30px;
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
            }
            .document-card {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .document-card h3 {
                margin-top: 0;
                margin-bottom: 10px;
            }
            .success {
                color: #28a745;
            }
            .error {
                color: #dc3545;
            }
            .document-meta {
                font-size: 14px;
                color: #666;
                margin-bottom: 10px;
            }
            a {
                color: #007bff;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <h1>Document Processing Summary</h1>
        
        <div class="summary">
            <p><strong>Directory:</strong> {dir_path}</p>
            <p><strong>Documents Processed:</strong> {total_docs}</p>
            <p><strong>Successful:</strong> {success_count}</p>
            <p><strong>Failed:</strong> {error_count}</p>
            <p><strong>Chunking Strategy:</strong> {chunking_strategy}</p>
            <p><strong>Date:</strong> {timestamp}</p>
        </div>
        
        <h2>Documents</h2>
        """.format(
                dir_path=dir_path,
                total_docs=len(results),
                success_count=sum(1 for r in results if r["status"] == "success"),
                error_count=sum(1 for r in results if r["status"] == "error"),
                chunking_strategy=self.chunking_strategy,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            
            # Add document cards
            for result in results:
                file_path = Path(result["file_path"])
                status_class = "success" if result["status"] == "success" else "error"
                status_text = "Success" if result["status"] == "success" else f"Error: {result.get('error', 'Unknown error')}"
                
                f.write(f"""
        <div class="document-card">
            <h3>{file_path.name}</h3>
            <div class="document-meta">
                Size: {round(file_path.stat().st_size / 1024, 2)} KB
            </div>
            <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
        """)
                
                if result["status"] == "success":
                    vis_path = Path(result["visualization_path"])
                    rel_path = vis_path.name
                    f.write(f"""
            <p><a href="{rel_path}" target="_blank">View Visualization</a></p>
        """)
                
                f.write("""
        </div>
        """)
            
            # Close HTML
            f.write("""
    </body>
    </html>
        """)
        
        logger.info(f"Directory visualization summary saved to: {summary_path}")
        return str(summary_path)


def main():
    # Load environment variables from the same directory as the script
    env_file_path = Path(__file__).parent / "env_variables.env"
    load_environment_variables(env_file_path)

    """Main entry point for the document processing pipeline."""
    parser = argparse.ArgumentParser(description="Document Processing Pipeline")
    parser.add_argument("input_path", type=str, help="Path to document file or directory")
    
    # Chunking options
    parser.add_argument("--chunking", type=str, default=ChunkingStrategy.PARAGRAPH,
                      choices=["page", "paragraph", "semantic", "hierarchical"],
                      help="Chunking strategy")
    # Embedding options
    parser.add_argument("--embedding", type=str, default=EmbeddingProvider.SENTENCE_TRANSFORMER,
                      choices=["sentence_transformer", "openai", "huggingface", "ollama"],
                      help="Embedding provider")
    parser.add_argument("--model", type=str, default=None,
                      help="Specific model name for embeddings")
    # Processing options
    parser.add_argument("--no-preprocess", action="store_true", 
                      help="Skip text preprocessing")
    parser.add_argument("--collection", type=str, default=None,
                      help="Collection name for vector storage")
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path for vector database storage")
    parser.add_argument("--cache-dir", type=str, default=None,
                      help="Directory for caching models")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Batch size for embedding generation")
    # Visualization options
    parser.add_argument("--visualize", action="store_true", 
                      help="Generate visualization of document chunks")
    parser.add_argument("--visualize-output", type=str, default=None,
                      help="Directory to save visualization files")
    parser.add_argument("--metadata", type=str, default=None,
                      help="JSON string with custom metadata for the document")
    parser.add_argument("--visualize-directory", action="store_true",
                      help="Generate visualization for all documents in a directory")
    parser.add_argument("--max-chunks-per-doc", type=int, default=1,
                      help="Maximum chunks to show per document when visualizing directory")
    # Storage options
    parser.add_argument("--storage", type=str, default="chroma",
                    choices=["chroma", "pinecone", "memory"],
                    help="Vector storage provider")
    parser.add_argument("--pinecone-api-key", type=str, default=None,
                    help="Pinecone API key (required for Pinecone storage)")
    parser.add_argument("--pinecone-create-new", action="store_true",
                    help="Create a new Pinecone index")
    parser.add_argument("--pinecone-namespace", type=str, default=None,
                    help="Namespace within Pinecone index (optional)")
    parser.add_argument("--chroma-delete", action="store_true",
                    help="Delete existing Chroma collection if it exists before processing")
    parser.add_argument("--delete-collection-only", action="store_true",
                    help="Only delete the specified collection without processing documents")

    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    try:
        pinecone_api_key = args.pinecone_api_key
        if args.storage == "pinecone" and not pinecone_api_key:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if pinecone_api_key:
                logger.info("Using Pinecone API key from environment variables")
            else:
                logger.error("No Pinecone API key provided. Check your environment variables or use --pinecone-api-key flag.")
                return 1

        # Initialize the pipeline
        pipeline = DocumentProcessingPipeline(
            chunking_strategy=args.chunking,
            embedding_provider=args.embedding,
            embedding_model=args.model,
            vector_db_path=args.db_path,
            preprocess=not args.no_preprocess,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            storage_type=args.storage,
            pinecone_api_key=pinecone_api_key,
            pinecone_create_new=args.pinecone_create_new,
            pinecone_namespace=args.pinecone_namespace,
            chroma_delete=args.chroma_delete
        )
        
        if args.delete_collection_only and args.collection:
            try:
                # Set the collection name
                pipeline.vector_store_manager.collection_name = args.collection
                
                # Delete the collection based on storage type
                if args.storage == "chroma":
                    result = pipeline.vector_store_manager.delete_chroma_collection()
                    if result:
                        logger.info(f"Successfully deleted Chroma collection: {args.collection}")
                    else:
                        logger.info(f"Chroma collection '{args.collection}' did not exist or was already deleted")
                elif args.storage == "pinecone":
                    # We need to create a method for Pinecone similar to delete_chroma_collection
                    result = pipeline.vector_store_manager.delete_pinecone_index()
                    if result:
                        logger.info(f"Successfully deleted Pinecone index: {args.collection}")
                    else:
                        logger.info(f"Pinecone index '{args.collection}' did not exist or was already deleted")
                else:
                    logger.warning(f"Collection deletion not supported for storage type: {args.storage}")
                    
                return 0
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")
                return 1

        # Handle visualization if requested
        if args.visualize:
            # Case 1: Directory with --visualize-directory flag
            if input_path.is_dir() and args.visualize_directory:
                result = pipeline.visualize_directory(
                    dir_path=str(input_path),
                    output_dir=args.visualize_output,
                    max_chunks_per_doc=args.max_chunks_per_doc,
                    show_metadata=True
                )
                logger.info(f"Directory visualization complete: {result}")
                
                # Exit after visualization if not continuing to full processing
                if not args.collection:
                    return 0
                    
            # Case 2: Directory without --visualize-directory flag
            elif input_path.is_dir():
                logger.error("For directory visualization, use --visualize-directory flag")
                return 1
                
            # Case 3: Single file
            elif input_path.is_file():
                result = pipeline.visualize_chunks(
                    file_path=str(input_path),
                    output_dir=args.visualize_output,
                    max_chunks=5,  # Consistent value
                    show_metadata=True
                )
                logger.info(f"Visualization complete: {result}")
                
                # Exit after visualization if not continuing to full processing
                if not args.collection:
                    return 0
                    
            # Case 4: Invalid path
            else:
                logger.error(f"Input path does not exist: {input_path}")
                return 1

        # Process file or directory
        if input_path.is_file():
            custom_metadata = json.loads(args.metadata) if args.metadata else None
            result = pipeline.process_document(
                file_path=str(input_path),
                collection_name=args.collection,
                document_metadata=custom_metadata
            )
            logger.info(f"Processing complete: {result}")
            
        elif input_path.is_dir():
            results = pipeline.process_directory(
                dir_path=str(input_path),
                collection_name=args.collection
            )
            success_count = sum(1 for r in results if r["status"] == "success")
            logger.info(f"Processing complete: {success_count}/{len(results)} documents successful")
            
        else:
            logger.error(f"Input path does not exist: {input_path}")
            return 1
            
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())