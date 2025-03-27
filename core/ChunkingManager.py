"""
ChunkingManager.py

A manager class that orchestrates document chunking using different strategies.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# Import chunker strategies
from core.BaseChunker import BaseChunker
from core.PageChunker import PageChunker
from core.ParagraphChunker import ParagraphChunker
from core.SemanticChunker import SemanticChunker
from core.HierarchicalChunker import HierarchicalChunker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkingStrategy:
    """Enumeration of available chunking strategies."""
    PAGE = "page"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"

class ChunkingManager:
    """Manager class for document chunking strategies."""
    
    def __init__(
        self, 
        embedding_model_name: str = "all-mpnet-base-v2",
        token_model_name: Optional[str] = None
    ):
        """
        Initialize chunking manager.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            token_model_name: Name of the token counting model
        """
        self.token_model_name = token_model_name
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None
        self._chunkers = {}
        
    @property
    def embedding_model(self):
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            try:
                # Only try to load as SentenceTransformer if it's a known SentenceTransformer model
                if self.embedding_model_name and not any(x in self.embedding_model_name.lower() for x in ["gpt", "text-embedding", "openai"]):
                    logger.info(f"Loading embedding model: {self.embedding_model_name}")
                    self._embedding_model = SentenceTransformer(self.embedding_model_name)
                else:
                    # Return a dummy embedding model that returns None
                    logger.info("Using dummy embedding model for tokenization only")
                    class DummyEmbedder:
                        def encode(self, text, **kwargs):
                            return [0.0] * 384  # Return dummy vector
                    self._embedding_model = DummyEmbedder()
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                # Return a dummy embedding model that returns None
                class DummyEmbedder:
                    def encode(self, text, **kwargs):
                        return [0.0] * 384  # Return dummy vector
                self._embedding_model = DummyEmbedder()
        return self._embedding_model
    
    def _get_chunker(self, strategy: str) -> BaseChunker:
        """Get or create chunker for the specified strategy."""
        strategy = strategy.lower()
        
        if strategy not in self._chunkers:
            if strategy == ChunkingStrategy.PAGE:
                self._chunkers[strategy] = PageChunker(
                    model_name=self.token_model_name,
                    embedding_model=self.embedding_model
                )
            elif strategy == ChunkingStrategy.PARAGRAPH:
                self._chunkers[strategy] = ParagraphChunker(
                    model_name=self.token_model_name,
                    embedding_model=self.embedding_model
                )
            elif strategy == ChunkingStrategy.SEMANTIC:
                self._chunkers[strategy] = SemanticChunker(
                    embedding_model=self.embedding_model,
                    model_name=self.token_model_name
                )
            elif strategy == ChunkingStrategy.HIERARCHICAL:
                self._chunkers[strategy] = HierarchicalChunker(
                    model_name=self.token_model_name,
                    embedding_model=self.embedding_model
                )
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")
                
        return self._chunkers[strategy]
    
    def process_document(
        self, 
        file_path: str, 
        strategy: str = ChunkingStrategy.PARAGRAPH,
        preprocess: bool = True
    ) -> Union[List[Document], Dict[str, List[Document]]]:
        """
        Process document using specified chunking strategy.
        
        Args:
            file_path: Path to document file
            strategy: Chunking strategy to use
            preprocess: Whether to preprocess text
            
        Returns:
            Chunked document(s) according to strategy
        """
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type
        file_extension = path.suffix.lower()
        
        # Process based on file type
        if file_extension == '.csv':
            return self._process_csv(file_path, strategy)
        elif file_extension == '.pdf':
            # Get appropriate chunker and process document
            chunker = self._get_chunker(strategy)
            
            logger.info(f"Processing document using {strategy} chunking strategy")
            
            if strategy == ChunkingStrategy.PAGE:
                return chunker.page_process_document(file_path, preprocess)
            elif strategy == ChunkingStrategy.PARAGRAPH:
                return chunker.paragraph_process_document(file_path, preprocess)
            elif strategy == ChunkingStrategy.SEMANTIC:
                return chunker.semantic_process_document(file_path, preprocess)
            elif strategy == ChunkingStrategy.HIERARCHICAL:
                return chunker.hierarchical_process_document(file_path, preprocess)
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types: .pdf, .csv")
    
    def process_directory(
        self, 
        dir_path: str, 
        strategy: str = ChunkingStrategy.PARAGRAPH,
        preprocess: bool = True
    ) -> Dict[str, Union[List[Document], Dict[str, List[Document]]]]:
        """
        Process all PDF documents in a directory.
        
        Args:
            dir_path: Directory containing PDF files
            strategy: Chunking strategy to use
            preprocess: Whether to preprocess text
            
        Returns:
            Dictionary mapping filenames to their processed documents
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
            
        results = {}

        # Find supported files (PDFs and CSVs)
        pdf_files = list(path.glob("**/*.pdf"))
        csv_files = list(path.glob("**/*.csv"))
        all_files = pdf_files + csv_files
        
        logger.info(f"Found {len(pdf_files)} PDF files and {len(csv_files)} CSV files in {dir_path}")
    
        pdf_files = list(path.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {dir_path}")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}")
                result = self.process_document(
                    str(pdf_file),
                    strategy=strategy,
                    preprocess=preprocess
                )
                results[pdf_file.name] = result
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                results[pdf_file.name] = {"error": str(e)}
                
        return results
    
    def _process_csv(self, file_path: str, strategy: str) -> List[Document]:
        """Process a CSV file into document chunks."""
        import pandas as pd
        
        logger.info(f"Loading CSV file: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Determine the chunking approach based on strategy
        if strategy == ChunkingStrategy.PARAGRAPH:
            # For these strategies, we treat each row as a separate document
            # with columns combined into a structured text format
            return self._chunk_csv_by_row(df, file_path)
        elif strategy == ChunkingStrategy.PAGE:
            # For page strategy, we create larger chunks with multiple rows
            return self._chunk_csv_by_page(df, file_path)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            # For hierarchical, create documents with metadata structure
            return {"chunks": self._chunk_csv_by_row(df, file_path)}
        else:
            raise ValueError(f"Unsupported chunking strategy for CSV: {strategy}")
        
    def _chunk_csv_by_row(self, df, file_path: str) -> List[Document]:
        """Convert each CSV row to a document chunk."""
        chunks = []
        file_name = Path(file_path).name
        
        # Get column names
        columns = df.columns.tolist()
        
        # Process each row
        for i, row in df.iterrows():
            # Convert row to formatted text
            content = "\n".join([f"{col}: {row[col]}" for col in columns])
            
            # Create metadata
            metadata = {
                "source": file_path,
                "file_name": file_name,
                "file_type": "csv",
                "row_index": i,
                "chunk_type": "csv_row",
            }
            
            # Add columns as additional metadata
            for col in columns:
                # Convert to string to ensure compatibility
                metadata[f"csv_{col}"] = str(row[col])
            
            # Create document
            doc = Document(page_content=content, metadata=metadata)
            chunks.append(doc)
        
        logger.info(f"Created {len(chunks)} chunks from CSV (row-based)")
        return chunks

    def _chunk_csv_by_page(self, df, file_path: str, rows_per_chunk: int = 20) -> List[Document]:
        """Convert CSV into larger chunks with multiple rows per chunk."""
        chunks = []
        file_name = Path(file_path).name
        columns = df.columns.tolist()
        
        # Calculate number of chunks
        total_rows = len(df)
        chunk_count = (total_rows + rows_per_chunk - 1) // rows_per_chunk  # Ceiling division
        
        # Generate chunks
        for chunk_idx in range(chunk_count):
            start_row = chunk_idx * rows_per_chunk
            end_row = min(start_row + rows_per_chunk, total_rows)
            
            chunk_df = df.iloc[start_row:end_row]
            
            # Format the chunk content
            content = f"CSV Data (Rows {start_row+1}-{end_row}):\n\n"
            
            # Add header row
            content += " | ".join(columns) + "\n"
            content += "-" * (sum(len(col) for col in columns) + 3 * (len(columns) - 1)) + "\n"
            
            # Add data rows
            for _, row in chunk_df.iterrows():
                content += " | ".join(str(row[col]) for col in columns) + "\n"
            
            # Create metadata
            metadata = {
                "source": file_path,
                "file_name": file_name,
                "file_type": "csv",
                "chunk_type": "csv_page",
                "start_row": start_row,
                "end_row": end_row - 1,
                "row_count": end_row - start_row,
            }
            
            # Create document
            doc = Document(page_content=content, metadata=metadata)
            chunks.append(doc)
        
        logger.info(f"Created {len(chunks)} chunks from CSV (page-based)")
        return chunks