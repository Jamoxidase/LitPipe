# LitPipe - Clean Implementation

This is a clean implementation of LitPipe with a direct Python API and web interface.

## Directory Structure

- `core/` - Core Python API implementation
  - `litpipe_core.py` - Clean, direct Python API for LitPipe

- `web/` - Web interface implementation
  - `litpipe_web.py` - Web interface using the core API

- `litpipe/` - Original LitPipe package (used by the core API)
  - `scipdf_parser/` - PDF parsing module using GROBID

- `workspace/` - Working directory for LitPipe
  - `pdfs/` - Directory for downloaded and uploaded PDFs
  - `output/` - Directory for output files (JSONL)
  - `temp/` - Directory for temporary files

## Usage

1. Run the web interface:
   ```
   python run.py
   ```
   
   Or directly:
   ```
   python web/litpipe_web.py
   ```

2. Use the core API directly:
   ```python
   from core.litpipe_core import LitPipeCore
   
   # Initialize LitPipeCore
   litpipe = LitPipeCore(workspace_dir="workspace")
   
   # Process a PDF from a DOI
   metadata = litpipe.process_pdf_from_doi(doi="10.1038/s41586-021-03819-2")
   
   # Get chunks
   chunks = litpipe.get_chunks()
   
   # Save chunks
   output_path = litpipe.save_chunks()
   ```

## Features

- Search for papers by query or DOI
- Process PDFs from URLs, DOIs, or local files
- Extract metadata (title, authors, abstract, sections, references, figures)
- Create and save chunks
- Generate embeddings with Voyage AI
- Generate context with Claude Haiku
- Store in SurrealDB (optional)

## Configuration

Configuration settings are defined in `config.py`, including:
- Workspace directories
- API keys for Voyage AI and Claude
- Chunking parameters
- SurrealDB connection settings
- GROBID URL
