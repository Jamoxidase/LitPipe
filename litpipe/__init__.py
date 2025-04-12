"""
LitPipe: PDF Paper Processing Pipeline

This module processes academic PDFs into chunked, contextualized, and embedded format.
"""

from litpipe.litpipe import (
    DocumentChunk, 
    ContextualChunker, 
    process_pdf_file,
    process_pdfs_from_directory,
    process_pdf_from_doi,
    process_pdf_from_url,
    process_pdfs_from_query,
    store_in_surrealdb
)

from litpipe.api import LitPipe
from litpipe.paper_search import search_papers

# Import the litpipe function
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Import directly from modules
from litpipe.paper_search import search_papers, search_unpaywall

__all__ = [
    'DocumentChunk',
    'ContextualChunker',
    'process_pdf_file',
    'process_pdfs_from_directory',
    'process_pdf_from_doi',
    'process_pdf_from_url',
    'process_pdfs_from_query',
    'store_in_surrealdb',
    'LitPipe',
    'search_papers',
    'parse_pdf_to_dict',
    'search_unpaywall',
    'download_pdf'
]