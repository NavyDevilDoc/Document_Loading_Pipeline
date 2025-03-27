"""
chroma_inspector.py - A utility script to inspect ChromaDB collections
"""
import argparse
import logging
import os
import chromadb
import pandas as pd
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_collections(db_path: str) -> List[str]:
    """List all collections in the ChromaDB database."""
    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    return collections


def get_collection_info(db_path: str, collection_name: str) -> Dict[str, Any]:
    """Get information about a specific collection."""
    client = chromadb.PersistentClient(path=db_path)
    try:
        # Get the collection
        collection = client.get_collection(collection_name)
        count = collection.count()
        
        # Get metadata directly from the collection's db
        try:
            metadata = collection.metadata
            logger.info(f"Retrieved collection metadata: {metadata}")
        except Exception as e:
            logger.warning(f"Could not retrieve collection metadata: {e}")
            metadata = {}
        
        # Extract embedding information from metadata
        embedding_dim = metadata.get("embedding_dimension", "Unknown")
        embedding_model = metadata.get("embedding_model", "Unknown")
        embedding_provider = metadata.get("embedding_provider", "Unknown")
        created_at = metadata.get("created_at", "Unknown")
        
        # If dimension is still unknown, try to detect it from error
        if embedding_dim == "Unknown":
            try:
                # This will fail with a message that includes the dimension
                collection.query(query_texts=["test"], n_results=1)
            except Exception as e:
                error_str = str(e)
                import re
                match = re.search(r'dimensionality (\d+)', error_str)
                if match:
                    embedding_dim = int(match.group(1))
                    logger.info(f"Detected dimension from error: {embedding_dim}")
        
        return {
            "name": collection_name,
            "document_count": count,
            "embedding_dimension": embedding_dim,
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
            "created_at": created_at,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return {"name": collection_name, "error": str(e)}


def peek_collection(db_path: str, collection_name: str, limit: int = 5) -> Dict[str, Any]:
    """Get a sample of documents from a collection."""
    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection(collection_name)
        results = collection.get(limit=limit, include=["documents", "metadatas", "embeddings"])
        return results
    except Exception as e:
        logger.error(f"Error peeking collection: {e}")
        return {"error": str(e)}


def display_results(results: Dict[str, Any]) -> None:
    """Display search or peek results in a readable format."""
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # For peek operation
    if "ids" in results:
        print(f"Found {len(results['ids'])} documents:")
        for i, (doc_id, doc, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
            print(f"\n--- Document {i+1} ({doc_id}) ---")
            print(f"Content (first 200 chars): {doc[:200]}...")
            print("Metadata:")
            for k, v in metadata.items():
                print(f"  {k}: {v}")
    
    # For search operation
    elif "ids" in results["ids"]:
        print(f"Search results ({len(results['ids'][0])} matches):")
        for i, (doc_id, doc, metadata, distance) in enumerate(
            zip(results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0])
        ):
            print(f"\n--- Match {i+1} ({doc_id}) - Similarity: {1-distance:.4f} ---")
            print(f"Content (first 200 chars): {doc[:200]}...")
            print("Metadata:")
            for k, v in metadata.items():
                print(f"  {k}: {v}")


def export_collection(db_path: str, collection_name: str, output_path: str) -> None:
    """Export a collection to a CSV file."""
    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection(collection_name)
        results = collection.get(include=["documents", "metadatas"])
        
        # Prepare data for export
        data = []
        for i, (doc_id, doc, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
            row = {"id": doc_id, "content": doc}
            # Add metadata fields
            for k, v in metadata.items():
                row[f"metadata_{k}"] = v
            data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Collection exported to {output_path}")
    except Exception as e:
        logger.error(f"Error exporting collection: {e}")


def main():
    parser = argparse.ArgumentParser(description="ChromaDB Collection Inspector")
    parser.add_argument("--db-path", type=str, default="./vector_db", 
                      help="Path to ChromaDB database directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List collections command
    list_parser = subparsers.add_parser("list", help="List all collections")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get collection information")
    info_parser.add_argument("collection", type=str, help="Collection name")
    
    # Peek command
    peek_parser = subparsers.add_parser("peek", help="View sample documents")
    peek_parser.add_argument("collection", type=str, help="Collection name")
    peek_parser.add_argument("--limit", type=int, default=5, help="Number of documents to retrieve")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export collection to CSV")
    export_parser.add_argument("collection", type=str, help="Collection name")
    export_parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    # Ensure the db_path exists
    if not os.path.exists(args.db_path):
        logger.error(f"Database path not found: {args.db_path}")
        return 1
    
    try:
        if args.command == "list":
            collections = list_collections(args.db_path)
            print(f"Found {len(collections)} collections:")
            for coll in collections:
                print(f"- {coll}")
                
        elif args.command == "info":
            info = get_collection_info(args.db_path, args.collection)
            if "error" in info:
                print(f"Error: {info['error']}")
            else:
                print(f"Collection: {info['name']}")
                print(f"Document count: {info['document_count']}")
                print(f"Embedding dimension: {info['embedding_dimension']}")
                print(f"Likely embedding model: {info['embedding_model']}")
                print(f"Embedding provider: {info['embedding_provider']}")
                if info.get('metadata') and info['metadata'] != {'embedding_model', 'embedding_provider', 'embedding_dimension'}:
                    print("Additional metadata:")
                    for k, v in info['metadata'].items():
                        if k not in ['embedding_model', 'embedding_provider', 'embedding_dimension']:
                            print(f"  {k}: {v}")
                
        elif args.command == "peek":
            results = peek_collection(args.db_path, args.collection, args.limit)
            display_results(results)
            
        elif args.command == "export":
            output_path = args.output or f"{args.collection}_export.csv"
            export_collection(args.db_path, args.collection, output_path)
            
        else:
            print("Please specify a command. Use --help for options.")
            
        return 0
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())