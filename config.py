"""
Configuration for LitPipe

This file contains all configurable parameters for LitPipe.
"""

import os
from pathlib import Path

# Directory Structure
PROJECT_ROOT = Path(__file__).parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"  # Main workspace directory
PDF_DIR = WORKSPACE_DIR / "pdfs"            # Directory for PDF files
OUTPUT_DIR = WORKSPACE_DIR / "output"       # Directory for output files
TEMP_DIR = WORKSPACE_DIR / "temp"           # Directory for temporary files

# For compatibility with litpipe module
# These are used by the original litpipe package
INPUT_PDF_DIR = PDF_DIR                     # Alias for PDF_DIR
PARSED_OUTPUT_DIR = OUTPUT_DIR / "parsed"   # Directory for parsed output
CHUNKS_OUTPUT_PATH = OUTPUT_DIR / "processed_chunks.jsonl"  # Default path for chunks

# Ensure directories exist
WORKSPACE_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PARSED_OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# API Keys - Set these to your actual API keys
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")  # Set your Voyage API key here
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # Set your Anthropic API key here

# Chunking Parameters
CHUNKING = {
    "target_tokens": 400,       # Target size for each chunk (~1600 chars)
    "min_tokens": 100,          # Minimum size for a chunk
    "max_tokens": 800,          # Maximum size for a chunk
    "overlap_tokens": 50,       # Number of tokens to overlap between chunks
    "chars_per_token": 4,       # Approximate number of characters per token
    "preserve_sentences": True, # Whether to preserve sentence boundaries
    "preserve_paragraphs": False, # Whether to preserve paragraph boundaries
    "section_aware": True,      # Whether to be aware of document sections
}

# Context Generation Parameters
CONTEXT = {
    "enabled": False,           # Whether to generate context by default
    "target_tokens": 75,        # Target size for generated context
    "max_tokens": 150,          # Maximum size for generated context
    "temperature": 0.0,         # Temperature for generation
    "model": "claude-3-5-haiku-latest", # Model to use
    "include_title": True,      # Whether to include the title in context
    "include_abstract": True,   # Whether to include the abstract in context
    "include_section_heading": True, # Whether to include section heading in context
    "full_text": False,        # Whether to include the full text in context- default is section context
}

# Embedding Parameters
EMBEDDING = {
    "enabled": True,            # Whether to generate embeddings by default
    "model": "voyage-3-large",        # Model to use
    "dimensions": 1024,         # Dimensions of the embedding
    "include_context": True,    # Whether to include context in embedding input
    "normalize": False,         # Whether to normalize embeddings
}

# API Retry Parameters
API = {
    "max_retries": 3,           # Number of API retry attempts
    "retry_delay": 2,           # Seconds to wait between retries
    "timeout": 30,              # Timeout for API requests in seconds
    "batch_size": 10,           # Batch size for API requests
}

# ScipdfParser Parameters
SCIPDF = {
    "extract_references": True, # Whether to extract references
    "extract_figures": True,   # Whether to extract figures
    "extract_tables": True,    # Whether to extract tables
    "extract_equations": True, # Whether to extract equations
    "extract_metadata": True,   # Whether to extract metadata
    "extract_sections": True,   # Whether to extract sections
    "extract_abstract": True,   # Whether to extract abstract
    "extract_authors": True,    # Whether to extract authors
    "extract_doi": True,        # Whether to extract DOI
    "whole_sections": False,    # Whether to return whole sections instead of chunking
    "section_filters": [],      # List of section names to include (empty = all)
    "exclude_sections": [],     # List of section names to exclude
    "extract_images": False,     # Whether to extract images
}

# Output Parameters
OUTPUT = {
    "save_jsonl": True,         # Whether to save JSONL output by default
    "format": "jsonl",          # Output format (jsonl, json, csv)
    "include_embedding": True,  # Whether to include embeddings in output
    "include_metadata": True,   # Whether to include metadata in output
    "include_context": True,    # Whether to include context in output
    "include_chunk_text": True, # Whether to include chunk text in output
    "include_section_info": True, # Whether to include section info in output
    "include_uuid": True,       # Whether to include UUID in output
    "include_group": True,      # Whether to include group in output
    "include_url": True,        # Whether to include URL in output
    "include_timestamp": True,  # Whether to include timestamp in output
    "include_doi": True,        # Whether to include DOI in output
    "include_title": True,      # Whether to include title in output
    "include_temp_path": False, # Whether to include temporary file path in output
    "include_section_index": True,  # Note: This is an index of CHUNKS within a section
    "include_abstract": True,   # Whether to include abstract in output
    "fields": [
        "uuid", "paper_title", "doi", "url", "chunk_text", "context",
        "embedding", "group", "timestamp", "section_heading", "section_index"  # Add section_index here
    ]
}

# SurrealDB Configuration
SURREAL = {
    "enabled": False,           # Whether to store data in SurrealDB by default
    "url": "wss://whole-rook-06akll27k5qvbdmbg17hgaosk8.aws-use1.surreal.cloud/rpc",
    "namespace": "LinThesis",
    "database": "test1",
    "table": "context2",
    "credentials": {"username": "James", "password": "Password!"},
    "create_mtree_index": True, # Whether to create an mtree index
    "mtree_dimension": 1024,    # Dimension of the mtree index
    "mtree_distance": "EUCLIDEAN", # Distance metric for mtree index
    "mtree_type": "F32",        # Type for mtree index
}


# GROBID Configuration
GROBID = {
    "url": "https://kermitt2-grobid.hf.space", # Default GROBID URL
}

# Web Interface Configuration
WEB = {
    "port": int(os.environ.get("PORT", 7860)),  # Default port
    "host": "0.0.0.0",                          # Default host
    "share": False,                             # Whether to create a public link
    "allowed_paths": ["*"],                     # Allowed paths for file access
    "show_api": True,                           # Whether to show API documentation
}

# Default metadata
DEFAULT_GROUP = "LitPipe"

# Simplified access to common parameters
TARGET_CHUNK_TOKENS = CHUNKING["target_tokens"]
TARGET_CONTEXT_TOKENS = CONTEXT["target_tokens"]
MAX_RETRIES = API["max_retries"]
RETRY_DELAY = API["retry_delay"]
ENABLE_HAIKU = CONTEXT["enabled"]
ENABLE_EMBEDDING = EMBEDDING["enabled"]
SAVE_JSONL = OUTPUT["save_jsonl"]
HAIKU_MODEL = CONTEXT["model"]
EMBEDDING_MODEL = EMBEDDING["model"]
SURREAL_ENABLED = SURREAL["enabled"]
SURREAL_URL = SURREAL["url"]
SURREAL_CREDENTIALS = SURREAL["credentials"]
SURREAL_NAMESPACE = SURREAL["namespace"]
SURREAL_DATABASE = SURREAL["database"]
SURREAL_TABLE = SURREAL["table"]
GROBID_URL = GROBID["url"]
WEB_PORT = WEB["port"]
WEB_HOST = WEB["host"]