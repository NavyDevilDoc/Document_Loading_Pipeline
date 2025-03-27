Document Loading Pipeline Usage Guide

Table of Contents
    1. Basic Usage
    2. Document Visualization
    3. Chunking Strategies
    4. Embedding Options
    5. Vector Storage Options
    6. Vector Store Administration
    7. Chroma Vector Store Inspection
    8. Processing Options
    9. Custom Metadata
    10. Batch Processing
    11. Common Workflows
    12. Best Practices
    13. Troubleshooting


1. Basic Usage
    Process a document with default settings:
        Paragraph-based chunking
        Sentence Transformer embeddings (all-MiniLM-L6-v2)
        Text preprocessing enabled
        Local Chroma vector storage
            python main.py "path/to/document.pdf"


2. Document Visualization
    a. Generate an HTML report to visualize how documents are chunked:
            python main.py "path/to/document.pdf" --visualize

    b. To specify an output directory for visualization files:
            python main.py "path/to/document.pdf" --visualize --visualize-output reports/

    The visualization shows:
        Sample document chunks
        Text highlighting
        Chunk statistics (tokens, characters)
        Embedding information


3. Chunking Strategies
    a. Paragraph-level Chunking (Default)
        Splits by paragraphs - balances context and granularity:
            python main.py "path/to/document.pdf" --chunking paragraph

    b. Page-level Chunking
        One chunk per page - best for preserving document structure:
            python main.py "path/to/document.pdf" --chunking page

    c. Hierarchical Chunking
        Combines page and semantic approaches:
            python main.py "path/to/document.pdf" --chunking hierarchical

    d. Semantic Chunking (deprecated and will be removed in a future version)
        Creates chunks based on content similarity:
            python main.py "path/to/document.pdf" --chunking semantic


4. Embedding Options
    All embedding options are listed with their default model. To change the embedding model, enter
    a different one in the --model parameter

    a. Sentence Transformers (Default)
        Fast, local embedding generation:
            python main.py "path/to/document.pdf" --embedding sentence_transformer --model all-MiniLM-L6-v2

    b. OpenAI Embeddings
        High-quality cloud embeddings (requires API key):
            python main.py "path/to/document.pdf" --embedding openai --model text-embedding-3-small

    c. Hugging Face
        Alternative models from Hugging Face (requires API key):
            python main.py "path/to/document.pdf" --embedding huggingface --model sentence-transformers/all-mpnet-base-v2

    d. Ollama (Local LLM)
        Run embeddings locally with Ollama:
            python main.py "path/to/document.pdf" --embedding ollama --model nomic-embed-text


5. Vector Storage Options
    a. Chroma DB (Default - Local Storage)
        1. Store vectors in a local database:
            python main.py "path/to/document.pdf" --storage chroma --collection my_collection

        2. Specify a custom storage directory:
            python main.py "path/to/document.pdf" --storage chroma --db-path /path/to/vector_db --collection my_collection

    b. Pinecone (Cloud Storage)
        1. Create a new Pinecone index:
            python main.py "path/to/document.pdf" --storage pinecone --pinecone-create-new --collection my_index

        2. Store vectors in Pinecone cloud service (requires API key):
            python main.py "path/to/document.pdf" --storage pinecone --collection my_index

        3. Use a specific namespace:
            python main.py "path/to/document.pdf" --storage pinecone --collection my_index --pinecone-namespace research_docs

    c. In-Memory Storage
        Temporary storage that's not persisted:
            python main.py "path/to/document.pdf" --storage memory --collection temp_docs


6. Vector Store Administration
    a. Delete a ChromaDB Collection
        Remove an existing Chroma collection without processing any documents:
            python main.py dummy_file.txt --delete-collection-only --collection my_collection --storage chroma

    b. Delete a Pinecone Index
        Remove an existing Pinecone index without processing any documents:
            python main.py dummy_file.txt --delete-collection-only --collection my_index --storage pinecone

    c. Replace an Existing ChromaDB Collection
        Delete and recreate a Chroma collection when processing documents:
            python main.py "path/to/document.pdf" --chunking page --storage chroma --collection my_collection --chroma-delete

    d. Replace an Existing Pinecone Index
        Delete and recreate a Pinecone index when processing documents:
            python main.py "path/to/document.pdf" --chunking page --storage pinecone --collection my_index --pinecone-create-new

    e. Process Directory with Replacing Existing Collection
        Process all documents in a directory and replace an existing collection:
            python main.py documents_folder/ --chunking paragraph --storage chroma --collection medical_docs --chroma-delete

    f. Change Embedding Dimensions
        When switching embedding models with different dimensions, you need to recreate the collection:
            python main.py "path/to/document.pdf" --embedding openai --storage chroma --collection research_docs --chroma-delete


7. Chroma Vector Store Inspection
    a. List Collections
       View all available ChromaDB collections:
            python chroma_inspector.py list --db-path ./vector_db

    b. Collection Information
       Get detailed information about a specific collection, including embedding model and dimensions:
            python chroma_inspector.py info my_collection --db-path ./vector_db

       Metadata:
       - Document count
       - Embedding dimension
       - Embedding model used
       - Embedding provider
       - Creation date

    c. Preview Documents
       View sample documents from a collection:
            python chroma_inspector.py peek my_collection --limit 3 --db-path ./vector_db

    d. Export Collection
       Export a collection's documents and metadata to CSV:
            python chroma_inspector.py export my_collection --output my_export.csv --db-path ./vector_db

    e. Custom Database Path
       Specify a non-default ChromaDB path:
            python chroma_inspector.py list --db-path /path/to/custom/vector_db


8. Processing Options
    a. Skip text preprocessing:
            python main.py "path/to/document.pdf" --no-preprocess

    b. Use a specific collection name:
            python main.py "path/to/document.pdf" --collection medical_documents

    c. Custom cache directory for models:
            python main.py "path/to/document.pdf" --cache-dir /path/to/cache

    d. Adjust batch size for embedding generation:
            python main.py "path/to/document.pdf" --batch-size 64


9. Custom Metadata
    Add structured metadata to your documents (stored with each chunk and is searchable):
            python main.py "path/to/document.pdf" --metadata '{"author":"John Doe", "subject":"Medical Research", "department":"Oncology"}'


10. Batch Processing
    a. Process all PDF documents in a directory:
            python main.py documents_folder/

    b. Process directory with a collection prefix:
            python main.py documents_folder/ --collection medical_docs


11. Common Workflows
    Quick Inspection
        Check how a document will be chunked without storing it:
            python main.py "path/to/document.pdf" --visualize

        Standard Processing
        Process with default settings and store for retrieval:
            python main.py "path/to/document.pdf" --collection my_docs

    High-Quality Embeddings
        Use OpenAI for superior semantic search:
            python main.py "path/to/document.pdf" --embedding openai --model text-embedding-3-small --collection research_docs

    Optimized for Research Papers
        Settings tuned for academic content:
            python main.py "path/to/document.pdf" --chunking semantic --embedding openai --model text-embedding-3-small --collection academic_papers

    Production Pipeline
        Cloud storage with advanced chunking:
            python main.py "path/to/document.pdf" --chunking paragraph --embedding openai --model text-embedding-3-small --storage pinecone --collection prod_docs --pinecone-namespace client_data

    Combining Multiple Options
        Advanced configuration with multiple parameters:
            python main.py "path/to/document.pdf" --chunking semantic --embedding openai --model text-embedding-3-small --no-preprocess --metadata '{"confidentiality":"high"}' --collection research_docs

    Visualize First, Then Process
        Two-step workflow for verification:

        First visualize to check chunking
            python main.py "path/to/document.pdf" --visualize --chunking semantic

        Then process with the same settings
            python main.py "path/to/document.pdf" --chunking semantic --collection research_docs


12. Best Practices:
    Test First: Always use --visualize before committing to a chunking strategy
    Match Content to Strategy: Use semantic chunking for varied content; paragraph for structured text
    API Key Management: Store API keys in environment variables, not command line
    Consistent Settings: Use the same settings for documents that will be searched together
    Metadata Organization: Add detailed metadata to improve search and filtering
    Namespace Organization: Use Pinecone namespaces to group related documents
    Collection Management: Use descriptive collection names and delete unused collections
    Embedding Consistency: Use the same embedding model for all documents in a collection
    Recreate Collections: When changing embedding models, recreate collections with --chroma-delete or --pinecone-create-new


13. Troubleshooting:
    Missing API Keys: Check environment variables or use explicit command line parameters
    Out of Memory: Reduce batch size with --batch-size
    OCR Quality Issues: Try disabling preprocessing with --no-preprocess
    Slow Processing: Consider switching to a lighter embedding model
    Dimension Mismatch Errors: Use --chroma-delete when changing embedding models
    Recreating Collections: Use --chroma-delete for ChromaDB or --pinecone-create-new for Pinecone
    Managing Storage: Use --delete-collection-only to clean up unused collections
    Multiple Collections: Create separate collections for documents with different embedding dimensions