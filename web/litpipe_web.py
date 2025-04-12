#!/usr/bin/env python3
"""
LitPipe Web Interface

A web interface for LitPipe that directly uses the Python API without CLI dependencies.
"""

import os
import sys
import logging
import json
import shutil
from pathlib import Path
import threading
import time
from typing import Dict, Any, List, Optional, Union

import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration and LitPipe API
sys.path.append(str(Path(__file__).parent.parent))
import config
from litpipe.api import LitPipe

# Use workspace directory from config
WORKSPACE_DIR = config.WORKSPACE_DIR

# Global variables to store job status
jobs = {}

def get_surrealdb_config(enabled: bool = False) -> Dict[str, Any]:
    """Get SurrealDB configuration from config file"""
    return {
        "enabled": enabled,
        "url": config.SURREAL["url"],
        "namespace": config.SURREAL["namespace"],
        "database": config.SURREAL["database"],
        "table": config.SURREAL["table"],
        "credentials": config.SURREAL["credentials"],
        "create_mtree_index": config.SURREAL["create_mtree_index"],
        "mtree_dimension": config.SURREAL["mtree_dimension"],
        "mtree_distance": config.SURREAL["mtree_distance"],
        "mtree_type": config.SURREAL["mtree_type"]
    }

def run_doi_job(
    job_id: str,
    doi: str,
    fulltext: bool,
    parse_figures: bool,
    as_list: bool,
    grobid_url: str,
    enable_embedding: bool,
    enable_haiku: bool,
    surrealdb_enabled: bool,
    target_chunk_size: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> None:
    """Run a job to process a PDF from a DOI"""
    try:
        # Update job status
        jobs[job_id]["status"] = "searching"
        
        # Get SurrealDB config
        surrealdb_config = get_surrealdb_config(surrealdb_enabled)
        
        # Debug log the SurrealDB config
        logger.info(f"SurrealDB config: enabled={surrealdb_enabled}, url={surrealdb_config['url']}")
        
        # Initialize LitPipe
        output_path = WORKSPACE_DIR / "output" / f"chunks_{int(time.time())}.jsonl"
        
        # Create chunking, context, and embedding configs based on parameters
        chunking_config = {
            "target_tokens": target_chunk_size if target_chunk_size else config.CHUNKING["target_tokens"],
            "min_tokens": min_chunk_size if min_chunk_size else config.CHUNKING["min_tokens"],
            "max_tokens": max_chunk_size if max_chunk_size else config.CHUNKING["max_tokens"],
            "overlap_tokens": chunk_overlap if chunk_overlap else config.CHUNKING["overlap_tokens"],
            "chars_per_token": config.CHUNKING["chars_per_token"],
            "preserve_sentences": config.CHUNKING["preserve_sentences"],
            "preserve_paragraphs": config.CHUNKING["preserve_paragraphs"],
            "section_aware": config.CHUNKING["section_aware"],
        }
        
        # Create scipdf config based on parameters
        scipdf_config = config.SCIPDF.copy()
        scipdf_config.update({
            "extract_figures": parse_figures,
            "whole_sections": not fulltext,
        })
        
        litpipe = LitPipe(
            output_path=output_path,
            enable_embedding=enable_embedding,
            enable_haiku=enable_haiku,
            chunking_config=chunking_config,
            scipdf_config=scipdf_config,
            surreal_config=surrealdb_config
        )
        
        # Process the PDF from DOI
        success, chunks = litpipe.process_pdf_from_doi(doi)
        
        if success:
            # Get metadata from the first chunk if available
            metadata = {}
            if chunks and len(chunks) > 0:
                metadata = {
                    "title": chunks[0].paper_title,
                    "doi": chunks[0].doi,
                    "url": chunks[0].url
                }
            
            # Save chunks
            litpipe.save_chunks()
            
            # Store in SurrealDB if enabled
            surrealdb_success = False
            if surrealdb_enabled:
                surrealdb_success = litpipe.store_in_surrealdb()
            
            # Update job status
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {
                "success": True,
                "message": f"Successfully processed PDF from DOI {doi}",
                "metadata": metadata,
                "chunks": [
                    {
                        "paper_title": chunk.paper_title,
                        "chunk_text": chunk.chunk_text,
                        "context": chunk.context,
                        "uuid": chunk.uuid,
                        "doi": chunk.doi,
                        "url": chunk.url,
                        "section_heading": chunk.section_heading
                    } for chunk in chunks
                ],
                "output_path": str(output_path),
                "papers_found": 1,
                "papers_processed": 1,
                "chunks_generated": len(chunks),
                "surrealdb_success": surrealdb_success if surrealdb_enabled else "Not enabled",
                "embeddings_generated": enable_embedding,
                "context_generated": enable_haiku,
                "jsonl_output": f"JSONL output saved to: {output_path}"
            }
        else:
            # Update job status
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"Failed to process PDF from DOI {doi}"
    
    except Exception as e:
        # Update job status
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.exception(f"Error in DOI job {job_id}: {str(e)}")

def run_pdf_job(
    job_id: str,
    pdf_path: Union[str, Path],
    fulltext: bool,
    parse_figures: bool,
    as_list: bool,
    grobid_url: str,
    enable_embedding: bool,
    enable_haiku: bool,
    surrealdb_enabled: bool,
    target_chunk_size: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> None:
    """Run a job to process an uploaded PDF file"""
    try:
        # Update job status
        jobs[job_id]["status"] = "processing"
        
        # Ensure the PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file does not exist: {pdf_path}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"PDF file does not exist: {pdf_path}"
            return
        
        # Get SurrealDB config
        surrealdb_config = get_surrealdb_config(surrealdb_enabled)
        
        # Debug log the SurrealDB config
        logger.info(f"SurrealDB config: enabled={surrealdb_enabled}, url={surrealdb_config['url']}")
        
        # Initialize LitPipe
        output_path = WORKSPACE_DIR / "output" / f"chunks_{int(time.time())}.jsonl"
        
        # Create chunking, context, and embedding configs based on parameters
        chunking_config = {
            "target_tokens": target_chunk_size if target_chunk_size else config.CHUNKING["target_tokens"],
            "min_tokens": min_chunk_size if min_chunk_size else config.CHUNKING["min_tokens"],
            "max_tokens": max_chunk_size if max_chunk_size else config.CHUNKING["max_tokens"],
            "overlap_tokens": chunk_overlap if chunk_overlap else config.CHUNKING["overlap_tokens"],
            "chars_per_token": config.CHUNKING["chars_per_token"],
            "preserve_sentences": config.CHUNKING["preserve_sentences"],
            "preserve_paragraphs": config.CHUNKING["preserve_paragraphs"],
            "section_aware": config.CHUNKING["section_aware"],
        }
        
        # Create scipdf config based on parameters
        scipdf_config = config.SCIPDF.copy()
        scipdf_config.update({
            "extract_figures": parse_figures,
            "whole_sections": not fulltext,
        })
        
        litpipe = LitPipe(
            output_path=output_path,
            enable_embedding=enable_embedding,
            enable_haiku=enable_haiku,
            chunking_config=chunking_config,
            scipdf_config=scipdf_config,
            surreal_config=surrealdb_config
        )
        
        # Process the PDF
        logger.info(f"Processing PDF file: {pdf_path}")
        success, chunks = litpipe.process_pdf(pdf_path)
        
        if success:
            # Get metadata from the first chunk if available
            metadata = {}
            if chunks and len(chunks) > 0:
                metadata = {
                    "title": chunks[0].paper_title,
                    "doi": chunks[0].doi,
                    "url": chunks[0].url
                }
            
            # Check if chunks is None or empty
            if not chunks:
                logger.warning(f"No chunks generated for PDF: {pdf_path}")
                chunks = []
            
            # Save chunks
            try:
                litpipe.save_chunks()
            except Exception as e:
                logger.error(f"Error saving chunks: {str(e)}")
                output_path = "Failed to save chunks"
            
            # Store in SurrealDB if enabled
            surrealdb_success = False
            if surrealdb_enabled:
                try:
                    surrealdb_success = litpipe.store_in_surrealdb()
                except Exception as e:
                    logger.error(f"Error storing in SurrealDB: {str(e)}")
            
            # Update job status
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {
                "success": True,
                "message": f"Successfully processed PDF file: {pdf_path}",
                "metadata": metadata,
                "chunks": [
                    {
                        "paper_title": chunk.paper_title,
                        "chunk_text": chunk.chunk_text,
                        "context": chunk.context,
                        "uuid": chunk.uuid,
                        "doi": chunk.doi,
                        "url": chunk.url,
                        "section_heading": chunk.section_heading
                    } for chunk in chunks
                ],
                "output_path": str(output_path),
                "papers_found": 1,
                "papers_processed": 1,
                "chunks_generated": len(chunks),
                "surrealdb_success": surrealdb_success if surrealdb_enabled else "Not enabled",
                "embeddings_generated": enable_embedding,
                "context_generated": enable_haiku,
                "jsonl_output": f"JSONL output saved to: {output_path}"
            }
        else:
            # Update job status
            logger.error(f"Failed to process PDF file: {pdf_path}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"Failed to process PDF file: {pdf_path}"
    
    except Exception as e:
        # Update job status
        logger.error(f"Error in PDF job {job_id}: {str(e)}")
        logger.exception(e)  # Log the full stack trace
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

def run_query_job(
    job_id: str,
    query: str,
    sources: List[str],
    max_results: int,
    fulltext: bool,
    parse_figures: bool,
    as_list: bool,
    grobid_url: str,
    enable_embedding: bool,
    enable_haiku: bool,
    surrealdb_enabled: bool,
    target_chunk_size: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> None:
    """Run a job to process PDFs from a search query"""
    try:
        # Update job status
        jobs[job_id]["status"] = "searching"
        
        # Get SurrealDB config
        surrealdb_config = get_surrealdb_config(surrealdb_enabled)
        
        # Debug log the SurrealDB config
        logger.info(f"SurrealDB config: enabled={surrealdb_enabled}, url={surrealdb_config['url']}")
        
        # Initialize LitPipe
        output_path = WORKSPACE_DIR / "output" / f"chunks_{int(time.time())}.jsonl"
        
        # Create chunking, context, and embedding configs based on parameters
        chunking_config = {
            "target_tokens": target_chunk_size if target_chunk_size else config.CHUNKING["target_tokens"],
            "min_tokens": min_chunk_size if min_chunk_size else config.CHUNKING["min_tokens"],
            "max_tokens": max_chunk_size if max_chunk_size else config.CHUNKING["max_tokens"],
            "overlap_tokens": chunk_overlap if chunk_overlap else config.CHUNKING["overlap_tokens"],
            "chars_per_token": config.CHUNKING["chars_per_token"],
            "preserve_sentences": config.CHUNKING["preserve_sentences"],
            "preserve_paragraphs": config.CHUNKING["preserve_paragraphs"],
            "section_aware": config.CHUNKING["section_aware"],
        }
        
        # Create scipdf config based on parameters
        scipdf_config = config.SCIPDF.copy()
        scipdf_config.update({
            "extract_figures": parse_figures,
            "whole_sections": not fulltext,
        })
        
        litpipe = LitPipe(
            output_path=output_path,
            enable_embedding=enable_embedding,
            enable_haiku=enable_haiku,
            chunking_config=chunking_config,
            scipdf_config=scipdf_config,
            surreal_config=surrealdb_config
        )
        
        logger.info(f"Starting search for query: {query}")
        logger.info(f"Sources: {sources}, max_results: {max_results}")
        
        # Process PDFs from query - note that LitPipe doesn't support sources parameter
        # We'll need to modify the litpipe module later if this is important
        success_count, chunks = litpipe.process_pdfs_from_query(
            query=query,
            max_results=max_results
        )
        
        logger.info(f"Search completed, found {success_count} papers, generated {len(chunks)} chunks")
        
        if success_count > 0 and chunks:
            # Extract metadata from chunks
            metadata_list = []
            paper_titles = set()
            
            for chunk in chunks:
                if chunk.paper_title not in paper_titles:
                    paper_titles.add(chunk.paper_title)
                    metadata_list.append({
                        "title": chunk.paper_title,
                        "doi": chunk.doi,
                        "url": chunk.url
                    })
            
            # Save chunks
            litpipe.save_chunks()
            
            # Store in SurrealDB if enabled
            surrealdb_success = False
            if surrealdb_enabled:
                surrealdb_success = litpipe.store_in_surrealdb()
            
            # Update job status
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {
                "success": True,
                "message": f"Successfully processed {success_count} papers from query: {query}",
                "metadata_list": metadata_list,
                "chunks": [
                    {
                        "paper_title": chunk.paper_title,
                        "chunk_text": chunk.chunk_text,
                        "context": chunk.context,
                        "uuid": chunk.uuid,
                        "doi": chunk.doi,
                        "url": chunk.url,
                        "section_heading": chunk.section_heading
                    } for chunk in chunks
                ],
                "output_path": str(output_path),
                "papers_found": len(metadata_list),
                "papers_processed": success_count,
                "chunks_generated": len(chunks),
                "surrealdb_success": surrealdb_success if surrealdb_enabled else "Not enabled",
                "embeddings_generated": enable_embedding,
                "context_generated": enable_haiku,
                "jsonl_output": f"JSONL output saved to: {output_path}"
            }
        else:
            # Update job status
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"No papers found or processed for query: {query}"
    
    except Exception as e:
        # Update job status
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.exception(f"Error in query job {job_id}: {str(e)}")

def submit_doi_job(
    doi: str,
    fulltext: bool = True,
    parse_figures: bool = True,
    as_list: bool = False,
    grobid_url: str = "https://kermitt2-grobid.hf.space",
    enable_embedding: bool = False,
    enable_haiku: bool = False,
    surrealdb_enabled: bool = False,
    target_chunk_size: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> Dict[str, Any]:
    """Submit a job to process a PDF from a DOI"""
    logger.info(f"Submit DOI job called with DOI: {doi}")
    
    if not doi:
        logger.error("No DOI provided")
        return {"error": "No DOI provided"}
    
    try:
        # Generate a job ID
        job_id = f"job_{int(time.time())}"
        
        # Create a job entry
        jobs[job_id] = {
            "id": job_id,
            "type": "doi",
            "doi": doi,
            "fulltext": fulltext,
            "parse_figures": parse_figures,
            "as_list": as_list,
            "grobid_url": grobid_url,
            "enable_embedding": enable_embedding,
            "enable_haiku": enable_haiku,
            "surrealdb_enabled": surrealdb_enabled,
            "target_chunk_size": target_chunk_size,
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size,
            "chunk_overlap": chunk_overlap,
            "status": "pending",
            "result": None,
            "error": None,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Start a thread to run the job
        thread = threading.Thread(
            target=run_doi_job,
            args=(job_id, doi, fulltext, parse_figures, as_list, grobid_url, 
                  enable_embedding, enable_haiku, surrealdb_enabled, 
                  target_chunk_size, min_chunk_size, max_chunk_size, chunk_overlap)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"DOI job submitted with ID: {job_id}")
        return {"message": f"DOI job submitted with ID: {job_id}", "job_id": job_id}
    
    except Exception as e:
        logger.exception(f"Error submitting DOI job: {str(e)}")
        return {"error": f"Error submitting DOI job: {str(e)}"}

def submit_pdf_job(
    pdf_file: str,
    fulltext: bool = True,
    parse_figures: bool = True,
    as_list: bool = False,
    grobid_url: str = "https://kermitt2-grobid.hf.space",
    enable_embedding: bool = False,
    enable_haiku: bool = False,
    surrealdb_enabled: bool = False,
    target_chunk_size: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> Dict[str, Any]:
    """Submit a job to process an uploaded PDF file"""
    # Generate a job ID
    job_id = f"job_{int(time.time())}"
    
    logger.info(f"Submitting PDF job with ID: {job_id}, file: {pdf_file}")
    
    # Create a job entry
    jobs[job_id] = {
        "id": job_id,
        "type": "pdf",
        "pdf_file": pdf_file,
        "fulltext": fulltext,
        "parse_figures": parse_figures,
        "as_list": as_list,
        "grobid_url": grobid_url,
        "enable_embedding": enable_embedding,
        "enable_haiku": enable_haiku,
        "surrealdb_enabled": surrealdb_enabled,
        "target_chunk_size": target_chunk_size,
        "min_chunk_size": min_chunk_size,
        "max_chunk_size": max_chunk_size,
        "chunk_overlap": chunk_overlap,
        "status": "pending",
        "result": None,
        "error": None,
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        # Copy the uploaded file to the workspace directory
        pdf_dir = WORKSPACE_DIR / "pdfs"
        pdf_dir.mkdir(exist_ok=True)
        
        # Ensure the file exists
        if not os.path.exists(pdf_file):
            logger.error(f"PDF file does not exist: {pdf_file}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"PDF file does not exist: {pdf_file}"
            return {"error": "PDF job failed: File does not exist", "job_id": job_id}
        
        # Copy the file
        pdf_path = pdf_dir / f"{job_id}_{os.path.basename(pdf_file)}"
        logger.info(f"Copying PDF from {pdf_file} to {pdf_path}")
        shutil.copy2(pdf_file, pdf_path)
        
        # Start a thread to run the job
        thread = threading.Thread(
            target=run_pdf_job,
            args=(job_id, pdf_path, fulltext, parse_figures, as_list, grobid_url,
                  enable_embedding, enable_haiku, surrealdb_enabled,
                  target_chunk_size, min_chunk_size, max_chunk_size, chunk_overlap)
        )
        thread.daemon = True
        thread.start()
        
        return {"message": f"PDF job submitted with ID: {job_id}", "job_id": job_id, "status": "pending"}
    
    except Exception as e:
        logger.exception(f"Error submitting PDF job: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        return {"error": f"PDF job failed: {str(e)}", "job_id": job_id}

def submit_query_job(
    query: str,
    sources: List[str],
    max_results: int = 5,
    fulltext: bool = True,
    parse_figures: bool = True,
    as_list: bool = False,
    grobid_url: str = "https://kermitt2-grobid.hf.space",
    enable_embedding: bool = False,
    enable_haiku: bool = False,
    surrealdb_enabled: bool = False,
    target_chunk_size: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> Dict[str, Any]:
    """Submit a job to process PDFs from a search query"""
    logger.info(f"Submit query job called with query: {query}, sources: {sources}")
    
    if not query:
        logger.error("No query provided")
        return {"error": "No query provided"}
    
    if not sources:
        logger.error("No sources selected")
        return {"error": "No sources selected"}
    
    try:
        # Generate a job ID
        job_id = f"job_{int(time.time())}"
        
        # Create a job entry
        jobs[job_id] = {
            "id": job_id,
            "type": "query",
            "query": query,
            "sources": sources,
            "max_results": max_results,
            "fulltext": fulltext,
            "parse_figures": parse_figures,
            "as_list": as_list,
            "grobid_url": grobid_url,
            "enable_embedding": enable_embedding,
            "enable_haiku": enable_haiku,
            "surrealdb_enabled": surrealdb_enabled,
            "target_chunk_size": target_chunk_size,
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size,
            "chunk_overlap": chunk_overlap,
            "status": "pending",
            "result": None,
            "error": None,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Start a thread to run the job
        thread = threading.Thread(
            target=run_query_job,
            args=(job_id, query, sources, max_results, fulltext, parse_figures, as_list, grobid_url,
                  enable_embedding, enable_haiku, surrealdb_enabled,
                  target_chunk_size, min_chunk_size, max_chunk_size, chunk_overlap)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Query job submitted with ID: {job_id}")
        return {"message": f"Query job submitted with ID: {job_id}", "job_id": job_id, "status": "pending"}
    
    except Exception as e:
        logger.exception(f"Error submitting query job: {str(e)}")
        return {"error": f"Error submitting query job: {str(e)}"}

def get_job_status() -> List[Dict[str, Any]]:
    """Get the status of all jobs"""
    return list(jobs.values())

def create_web_interface() -> gr.Blocks:
    """Create a Gradio web interface"""
    with gr.Blocks(title="LitPipe Web Interface") as interface:
        gr.Markdown("# LitPipe Web Interface")
        gr.Markdown("Search for papers, process PDFs, and extract content using direct Python API calls.")
        
        with gr.Tab("Search Papers"):
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter a search query",
                        lines=2
                    )
                    
                    with gr.Row():
                        query_sources = gr.CheckboxGroup(
                            label="Sources",
                            choices=["arxiv", "pubmed"],
                            value=["arxiv", "pubmed"]
                        )
                        
                        query_max_results = gr.Slider(
                            label="Max Results",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1
                        )
                    
                    with gr.Row():
                        query_fulltext = gr.Checkbox(
                            label="Extract Full Text",
                            value=True
                        )
                        
                        query_parse_figures = gr.Checkbox(
                            label="Parse Figures",
                            value=True
                        )
                        
                        query_as_list = gr.Checkbox(
                            label="Return Sections as Lists",
                            value=False
                        )
                    
                    query_grobid_url = gr.Textbox(
                        label="GROBID URL",
                        value="https://kermitt2-grobid.hf.space"
                    )
                    
                    with gr.Row():
                        query_enable_embedding = gr.Checkbox(
                            label="Generate Embeddings",
                            value=config.ENABLE_EMBEDDING
                        )
                        
                        query_enable_haiku = gr.Checkbox(
                            label="Generate Context",
                            value=config.ENABLE_HAIKU
                        )
                        
                        query_surrealdb_enabled = gr.Checkbox(
                            label="Store in SurrealDB",
                            value=config.SURREAL["enabled"]
                        )
                    
                    with gr.Accordion("Chunking Options", open=False):
                        query_target_chunk_size = gr.Slider(
                            label="Target Chunk Size (tokens)",
                            minimum=100,
                            maximum=2000,
                            value=config.CHUNKING["target_tokens"],
                            step=50
                        )
                        
                        query_min_chunk_size = gr.Slider(
                            label="Min Chunk Size (tokens)",
                            minimum=50,
                            maximum=1000,
                            value=config.CHUNKING["min_tokens"],
                            step=50
                        )
                        
                        query_max_chunk_size = gr.Slider(
                            label="Max Chunk Size (tokens)",
                            minimum=200,
                            maximum=4000,
                            value=config.CHUNKING["max_tokens"],
                            step=50
                        )
                        
                        query_chunk_overlap = gr.Slider(
                            label="Chunk Overlap (tokens)",
                            minimum=0,
                            maximum=200,
                            value=config.CHUNKING["overlap_tokens"],
                            step=10
                        )
                    
                    query_submit = gr.Button("Search and Process")
                
                with gr.Column():
                    query_output = gr.JSON(label="Results")
            
            query_submit.click(
                fn=submit_query_job,
                inputs=[
                    query_input, query_sources, query_max_results,
                    query_fulltext, query_parse_figures, query_as_list,
                    query_grobid_url, query_enable_embedding, query_enable_haiku,
                    query_surrealdb_enabled, query_target_chunk_size,
                    query_min_chunk_size, query_max_chunk_size, query_chunk_overlap
                ],
                outputs=query_output
            )
        
        with gr.Tab("Process DOI"):
            with gr.Row():
                with gr.Column():
                    doi_input = gr.Textbox(
                        label="DOI",
                        placeholder="Enter a DOI",
                        lines=1
                    )
                    
                    with gr.Row():
                        doi_fulltext = gr.Checkbox(
                            label="Extract Full Text",
                            value=True
                        )
                        
                        doi_parse_figures = gr.Checkbox(
                            label="Parse Figures",
                            value=True
                        )
                        
                        doi_as_list = gr.Checkbox(
                            label="Return Sections as Lists",
                            value=False
                        )
                    
                    doi_grobid_url = gr.Textbox(
                        label="GROBID URL",
                        value="https://kermitt2-grobid.hf.space"
                    )
                    
                    with gr.Row():
                        doi_enable_embedding = gr.Checkbox(
                            label="Generate Embeddings",
                            value=config.ENABLE_EMBEDDING
                        )
                        
                        doi_enable_haiku = gr.Checkbox(
                            label="Generate Context",
                            value=config.ENABLE_HAIKU
                        )
                        
                        doi_surrealdb_enabled = gr.Checkbox(
                            label="Store in SurrealDB",
                            value=config.SURREAL["enabled"]
                        )
                    
                    with gr.Accordion("Chunking Options", open=False):
                        doi_target_chunk_size = gr.Slider(
                            label="Target Chunk Size (tokens)",
                            minimum=100,
                            maximum=2000,
                            value=config.CHUNKING["target_tokens"],
                            step=50
                        )
                        
                        doi_min_chunk_size = gr.Slider(
                            label="Min Chunk Size (tokens)",
                            minimum=50,
                            maximum=1000,
                            value=config.CHUNKING["min_tokens"],
                            step=50
                        )
                        
                        doi_max_chunk_size = gr.Slider(
                            label="Max Chunk Size (tokens)",
                            minimum=200,
                            maximum=4000,
                            value=config.CHUNKING["max_tokens"],
                            step=50
                        )
                        
                        doi_chunk_overlap = gr.Slider(
                            label="Chunk Overlap (tokens)",
                            minimum=0,
                            maximum=200,
                            value=config.CHUNKING["overlap_tokens"],
                            step=10
                        )
                    
                    doi_submit = gr.Button("Process DOI")
                
                with gr.Column():
                    doi_output = gr.JSON(label="Results")
            
            doi_submit.click(
                fn=submit_doi_job,
                inputs=[
                    doi_input, doi_fulltext, doi_parse_figures, doi_as_list,
                    doi_grobid_url, doi_enable_embedding, doi_enable_haiku,
                    doi_surrealdb_enabled, doi_target_chunk_size,
                    doi_min_chunk_size, doi_max_chunk_size, doi_chunk_overlap
                ],
                outputs=doi_output
            )
        
        with gr.Tab("Process PDF"):
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(
                        label="PDF File",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    
                    with gr.Row():
                        pdf_fulltext = gr.Checkbox(
                            label="Extract Full Text",
                            value=True
                        )
                        
                        pdf_parse_figures = gr.Checkbox(
                            label="Parse Figures",
                            value=True
                        )
                        
                        pdf_as_list = gr.Checkbox(
                            label="Return Sections as Lists",
                            value=False
                        )
                    
                    pdf_grobid_url = gr.Textbox(
                        label="GROBID URL",
                        value="https://kermitt2-grobid.hf.space"
                    )
                    
                    with gr.Row():
                        pdf_enable_embedding = gr.Checkbox(
                            label="Generate Embeddings",
                            value=config.ENABLE_EMBEDDING
                        )
                        
                        pdf_enable_haiku = gr.Checkbox(
                            label="Generate Context",
                            value=config.ENABLE_HAIKU
                        )
                        
                        pdf_surrealdb_enabled = gr.Checkbox(
                            label="Store in SurrealDB",
                            value=config.SURREAL["enabled"]
                        )
                    
                    with gr.Accordion("Chunking Options", open=False):
                        pdf_target_chunk_size = gr.Slider(
                            label="Target Chunk Size (tokens)",
                            minimum=100,
                            maximum=2000,
                            value=config.CHUNKING["target_tokens"],
                            step=50
                        )
                        
                        pdf_min_chunk_size = gr.Slider(
                            label="Min Chunk Size (tokens)",
                            minimum=50,
                            maximum=1000,
                            value=config.CHUNKING["min_tokens"],
                            step=50
                        )
                        
                        pdf_max_chunk_size = gr.Slider(
                            label="Max Chunk Size (tokens)",
                            minimum=200,
                            maximum=4000,
                            value=config.CHUNKING["max_tokens"],
                            step=50
                        )
                        
                        pdf_chunk_overlap = gr.Slider(
                            label="Chunk Overlap (tokens)",
                            minimum=0,
                            maximum=200,
                            value=config.CHUNKING["overlap_tokens"],
                            step=10
                        )
                    
                    pdf_submit = gr.Button("Process PDF")
                
                with gr.Column():
                    pdf_output = gr.JSON(label="Results")
            
            def process_pdf(pdf_file, fulltext, parse_figures, as_list, grobid_url,
                           enable_embedding, enable_haiku, surrealdb_enabled,
                           target_chunk_size, min_chunk_size, max_chunk_size, chunk_overlap):
                logger.info(f"Process PDF called with file: {pdf_file}, type: {type(pdf_file)}")
                
                if pdf_file is None:
                    logger.error("No PDF file uploaded")
                    return {"error": "No PDF file uploaded"}
                
                try:
                    # Ensure the file exists
                    if not os.path.exists(pdf_file):
                        logger.error(f"PDF file does not exist: {pdf_file}")
                        return {"error": f"PDF file does not exist: {pdf_file}"}
                    
                    logger.info(f"PDF file exists at: {pdf_file}, size: {os.path.getsize(pdf_file)} bytes")
                    
                    # Submit the job
                    result = submit_pdf_job(
                        pdf_file, fulltext, parse_figures, as_list, grobid_url,
                        enable_embedding, enable_haiku, surrealdb_enabled,
                        target_chunk_size, min_chunk_size, max_chunk_size, chunk_overlap
                    )
                    
                    logger.info(f"PDF job submission result: {result}")
                    return result
                
                except Exception as e:
                    logger.exception(f"Error processing PDF: {str(e)}")
                    return {"error": f"Error processing PDF: {str(e)}"}
            
            pdf_submit.click(
                fn=process_pdf,
                inputs=[
                    pdf_input, pdf_fulltext, pdf_parse_figures, pdf_as_list,
                    pdf_grobid_url, pdf_enable_embedding, pdf_enable_haiku,
                    pdf_surrealdb_enabled, pdf_target_chunk_size,
                    pdf_min_chunk_size, pdf_max_chunk_size, pdf_chunk_overlap
                ],
                outputs=pdf_output
            )
        
        with gr.Tab("Job Status"):
            refresh_button = gr.Button("Refresh Job Status")
            job_status_output = gr.JSON(label="Job Status")
            
            refresh_button.click(
                fn=get_job_status,
                inputs=[],
                outputs=job_status_output
            )
    
    return interface

def main():
    """Main function"""
    interface = create_web_interface()
    interface.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    main()