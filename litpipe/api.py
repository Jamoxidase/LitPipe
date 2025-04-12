#!/usr/bin/env python3
"""
Python API for using LitPipe as a library
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

# Import from litpipe
from litpipe import (
    ContextualChunker,
    process_pdf_file,
    process_pdfs_from_directory,
    process_pdf_from_doi,
    process_pdf_from_url,
    process_pdfs_from_query,
    DocumentChunk
)

# Import configuration
from litpipe.config import (
    CHUNKING, CONTEXT, EMBEDDING, OUTPUT, SCIPDF, SURREAL,
    INPUT_PDF_DIR, PARSED_OUTPUT_DIR, CHUNKS_OUTPUT_PATH, TEMP_DIR
)

class LitPipe:
    """
    Python API for using LitPipe as a library
    """
    
    def __init__(
        self,
        output_path: Optional[Union[str, Path]] = None,
        enable_haiku: bool = False,
        enable_embedding: bool = False,
        anthropic_api_key: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        chunking_config: Optional[Dict[str, Any]] = None,
        context_config: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        scipdf_config: Optional[Dict[str, Any]] = None,
        surreal_config: Optional[Dict[str, Any]] = None,
        group: str = "LitPipe"
    ):
        """
        Initialize the LitPipe API
        
        Args:
            output_path: Path to save processed chunks
            enable_haiku: Whether to use Haiku for context generation
            enable_embedding: Whether to generate embeddings
            anthropic_api_key: API key for Anthropic (Claude)
            voyage_api_key: API key for Voyage AI
            chunking_config: Configuration for chunking
            context_config: Configuration for context generation
            embedding_config: Configuration for embeddings
            output_config: Configuration for output
            scipdf_config: Configuration for ScipdfParser
            surreal_config: Configuration for SurrealDB
            group: Group name for metadata
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        PARSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set output path
        self.output_path = Path(output_path) if output_path else CHUNKS_OUTPUT_PATH
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set configurations
        self.chunking_config = chunking_config or CHUNKING.copy()
        self.context_config = context_config or CONTEXT.copy()
        self.embedding_config = embedding_config or EMBEDDING.copy()
        self.output_config = output_config or OUTPUT.copy()
        self.scipdf_config = scipdf_config or SCIPDF.copy()
        self.surreal_config = surreal_config or SURREAL.copy()
        
        # Override context and embedding settings
        self.context_config["enabled"] = enable_haiku
        self.embedding_config["enabled"] = enable_embedding
        
        # Set API keys
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        if voyage_api_key:
            os.environ["VOYAGE_API_KEY"] = voyage_api_key
        
        # Initialize chunker
        self.chunker = ContextualChunker(
            enable_haiku=enable_haiku,
            enable_embedding=enable_embedding,
            save_path=self.output_path,
            chunking_config=self.chunking_config,
            context_config=self.context_config,
            embedding_config=self.embedding_config,
            output_config=self.output_config,
            group=group
        )
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> Tuple[bool, List[DocumentChunk]]:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (success, chunks)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return False, []
        
        # Create a file:// URL as the source identifier
        file_url = f"file://{pdf_path.absolute()}"
        
        # Process the PDF
        success = process_pdf_file(pdf_path, self.chunker, PARSED_OUTPUT_DIR, self.scipdf_config, file_url)
        
        return success, self.chunker.chunks
    
    def process_pdf_directory(self, directory: Union[str, Path]) -> Tuple[int, List[DocumentChunk]]:
        """
        Process all PDFs in a directory
        
        Args:
            directory: Directory containing PDFs
            
        Returns:
            Tuple of (success_count, chunks)
        """
        directory = Path(directory)
        if not directory.exists():
            self.logger.error(f"Directory not found: {directory}")
            return 0, []
        
        # Process the PDFs
        success_count = process_pdfs_from_directory(directory, self.chunker, PARSED_OUTPUT_DIR, self.scipdf_config)
        
        return success_count, self.chunker.chunks
    
    def process_pdf_from_doi(self, doi: str) -> Tuple[bool, List[DocumentChunk]]:
        """
        Process a PDF from a DOI
        
        Args:
            doi: DOI to process
            
        Returns:
            Tuple of (success, chunks)
        """
        # Process the PDF
        success = process_pdf_from_doi(doi, self.chunker, PARSED_OUTPUT_DIR, self.scipdf_config)
        
        return success, self.chunker.chunks
    
    def process_pdf_from_url(self, url: str) -> Tuple[bool, List[DocumentChunk]]:
        """
        Process a PDF from a URL
        
        Args:
            url: URL to process
            
        Returns:
            Tuple of (success, chunks)
        """
        # Process the PDF
        success = process_pdf_from_url(url, self.chunker, PARSED_OUTPUT_DIR, self.scipdf_config)
        
        return success, self.chunker.chunks
    
    def process_pdfs_from_query(self, query: str, max_results: int = 5) -> Tuple[int, List[DocumentChunk]]:
        """
        Process PDFs from a search query
        
        Args:
            query: Search query
            max_results: Maximum number of results to process
            
        Returns:
            Tuple of (success_count, chunks)
        """
        # Process the PDFs
        success_count = process_pdfs_from_query(query, self.chunker, PARSED_OUTPUT_DIR, self.scipdf_config)
        
        return success_count, self.chunker.chunks
    
    def get_chunks(self) -> List[DocumentChunk]:
        """
        Get the processed chunks
        
        Returns:
            List of DocumentChunk objects
        """
        return self.chunker.chunks
    
    def save_chunks(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the processed chunks to a file
        
        Args:
            output_path: Path to save the chunks
        """
        if output_path:
            # Create a new chunker with the same configuration but a different output path
            temp_chunker = ContextualChunker(
                enable_haiku=self.context_config["enabled"],
                enable_embedding=self.embedding_config["enabled"],
                save_path=Path(output_path),
                chunking_config=self.chunking_config,
                context_config=self.context_config,
                embedding_config=self.embedding_config,
                output_config=self.output_config,
                group=self.chunker.group
            )
            
            # Save each chunk
            for chunk in self.chunker.chunks:
                temp_chunker.save_chunk(chunk)
        else:
            # Chunks are already saved to self.output_path during processing
            pass
    
    def store_in_surrealdb(self) -> bool:
        """
        Store the processed chunks in SurrealDB
        
        Returns:
            True if successful, False otherwise
        """
        if not self.surreal_config["enabled"]:
            self.logger.warning("SurrealDB is not enabled")
            return False
        
        try:
            # Use the WebSocket API
            import asyncio
            
            # Try to import the surrealdb module
            try:
                from litpipe import store_in_surrealdb
                
                # Store the chunks in SurrealDB
                self.logger.info(f"Storing chunks in SurrealDB from {self.output_path}")
                result = asyncio.run(store_in_surrealdb(self.output_path, self.surreal_config))
                
                if result:
                    self.logger.info("Successfully stored chunks in SurrealDB")
                    return True
                else:
                    self.logger.error("Failed to store chunks in SurrealDB")
                    return False
                    
            except ImportError:
                self.logger.error("surrealdb module not installed. Please install it with 'pip install surrealdb'")
                return False
                
        except Exception as e:
            self.logger.error(f"Error storing chunks in SurrealDB: {str(e)}")
            return False
    
    def clear_chunks(self) -> None:
        """
        Clear the processed chunks
        """
        self.chunker.chunks = []
    
    def __len__(self) -> int:
        """
        Get the number of processed chunks
        
        Returns:
            Number of chunks
        """
        return len(self.chunker.chunks)
    
    def __getitem__(self, index: int) -> DocumentChunk:
        """
        Get a chunk by index
        
        Args:
            index: Index of the chunk
            
        Returns:
            DocumentChunk object
        """
        return self.chunker.chunks[index]
    
    def __iter__(self):
        """
        Iterate over the processed chunks
        
        Returns:
            Iterator over DocumentChunk objects
        """
        return iter(self.chunker.chunks)


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Initialize LitPipe
    litpipe = LitPipe(
        output_path="/workspace/litpipe/api_test.jsonl",
        enable_haiku=False,
        enable_embedding=False
    )
    
    # Process a PDF
    success, chunks = litpipe.process_pdf("/workspace/papers/litPipe copy/testPDFs/ijms-22-11454.pdf")
    
    if success:
        print(f"Successfully processed PDF, generated {len(chunks)} chunks")
        print(f"First chunk: {chunks[0].chunk_text[:100]}...")
    else:
        print("Failed to process PDF")