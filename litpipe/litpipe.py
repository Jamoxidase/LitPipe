#!/usr/bin/env python3
"""
LitPipe: PDF Paper Processing Pipeline

This script processes academic PDFs into chunked, contextualized, and embedded format.
It can process PDFs from:
1. Local directory
2. DOIs
3. Search queries

Input:
- PDF files in INPUT_PDF_DIR (configurable in config.py)
- DOIs provided via command line
- Search queries provided via command line

Output:
- Intermediate parsed JSONs in PARSED_OUTPUT_DIR
- Final processed chunks in CHUNKS_OUTPUT_PATH

Usage:
    python litpipe.py --pdf-dir /path/to/pdfs
    python litpipe.py --doi 10.1093/bioinformatics/bti1134
    python litpipe.py --query "quantum computing"
"""

import os
import json
import logging
import tempfile
import argparse
import uuid
import shutil
import requests
import time
import re
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import nltk

# Import configuration
from litpipe.config import (
    SCIPDF, PROJECT_ROOT, INPUT_PDF_DIR, PARSED_OUTPUT_DIR, CHUNKS_OUTPUT_PATH, TEMP_DIR,
    ANTHROPIC_API_KEY, VOYAGE_API_KEY, TARGET_CHUNK_TOKENS, TARGET_CONTEXT_TOKENS,
    MAX_RETRIES, RETRY_DELAY, ENABLE_HAIKU, ENABLE_EMBEDDING, HAIKU_MODEL,
    EMBEDDING_MODEL, SURREAL_ENABLED, SURREAL_URL, SURREAL_CREDENTIALS,
    SURREAL_NAMESPACE, SURREAL_DATABASE, SURREAL_TABLE, DEFAULT_GROUP
)

# Import additional configuration variables
from litpipe.config import CHUNKING, CONTEXT, EMBEDDING, OUTPUT, SURREAL, API

# Ensure directories exist
INPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
PARSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)

# ============= Logging Setup =============

def setup_logging():
    """Set up logging with file and console output"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"litpipe_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ============= Data Structures =============

class DocumentChunk:
    """Represents a processed chunk of a document"""
    def __init__(
        self,
        paper_title: str,
        chunk_text: str = "",
        context: Optional[str] = None,
        abstract: Optional[str] = None,
        embedding: Optional[Union[np.ndarray, List[float]]] = None,
        uuid: Optional[str] = None,
        group: str = DEFAULT_GROUP,
        doi: Optional[str] = None,
        url: Optional[str] = None,
        temp_path: Optional[str] = None,
        section_heading: Optional[str] = None,
        section_index: Optional[int] = None,  
        timestamp: Optional[str] = None
    ):
        self.paper_title = paper_title
        self.chunk_text = chunk_text
        self.context = context
        self.uuid = uuid
        self.group = group
        self.doi = doi
        self.url = url
        self.temp_path = temp_path
        self.section_heading = section_heading
        self.section_index = section_index, 
        self.timestamp = timestamp
        
        # Handle embedding specially to ensure it's a list
        if embedding is not None:
            if isinstance(embedding, np.ndarray):
                self.embedding = embedding.tolist()
            else:
                self.embedding = embedding
        else:
            self.embedding = None

# ============= Processing Classes =============

class ContextualChunker:
    """Handles the chunking, context generation, and embedding of document text"""
    
    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        voyage_api_key: Optional[str] = None,
        target_chunk_tokens: int = TARGET_CHUNK_TOKENS,
        target_context_tokens: int = TARGET_CONTEXT_TOKENS,
        save_path: Path = CHUNKS_OUTPUT_PATH,
        enable_haiku: bool = ENABLE_HAIKU,
        enable_embedding: bool = ENABLE_EMBEDDING,
        haiku_model: str = HAIKU_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        chunking_config: Optional[Dict[str, Any]] = None,
        context_config: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        group: str = DEFAULT_GROUP
    ):
        """
        Initialize the chunker with API clients and parameters
        
        Args:
            anthropic_api_key: API key for Anthropic (Claude)
            voyage_api_key: API key for Voyage AI
            target_chunk_tokens: Target size for each chunk
            target_context_tokens: Target size for generated context
            save_path: Path to save processed chunks
            enable_haiku: Whether to use Haiku for context generation
            enable_embedding: Whether to generate embeddings
            haiku_model: Model to use for Haiku
            embedding_model: Model to use for embeddings
            chunking_config: Configuration for chunking
            context_config: Configuration for context generation
            embedding_config: Configuration for embeddings
            output_config: Configuration for output
            group: Group name for metadata
        """
        self.logger = logging.getLogger(__name__)
        
        # Store configuration
        self.chunking_config = chunking_config or CHUNKING.copy()
        self.context_config = context_config or CONTEXT.copy()
        self.embedding_config = embedding_config or EMBEDDING.copy()
        self.output_config = output_config or OUTPUT.copy()
        self.group = group
        
        # Initialize API clients if enabled
        self.client = None
        self.voyage_client = None
        
        if enable_haiku:
            try:
                import anthropic
                self.client = anthropic.Anthropic(
                    api_key=anthropic_api_key or ANTHROPIC_API_KEY
                )
            except ImportError:
                self.logger.warning("Anthropic package not installed. Context generation disabled.")
                enable_haiku = False
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic client: {str(e)}")
                enable_haiku = False
        
        # Set embedding model from config or parameter
        self.embedding_model = embedding_model or self.embedding_config["model"]
        
        if enable_embedding:
            try:
                import voyageai
                self.logger.info(f"Initializing Voyage client with API key: {'*****' + (voyage_api_key or VOYAGE_API_KEY)[-5:] if (voyage_api_key or VOYAGE_API_KEY) else 'None'}")
                self.logger.info(f"Using embedding model: {self.embedding_model}")
                self.voyage_client = voyageai.Client(
                    api_key=voyage_api_key or VOYAGE_API_KEY
                )
                # Test the client with a simple embedding
                test_embedding = self.voyage_client.embed(["Test"], model=self.embedding_model).embeddings[0]
                self.logger.info(f"Test embedding successful, dimensions: {len(test_embedding)}")
            except ImportError:
                self.logger.warning("VoyageAI package not installed. Embedding disabled.")
                enable_embedding = False
            except Exception as e:
                self.logger.warning(f"Failed to initialize Voyage client: {str(e)}")
                self.logger.warning(f"VOYAGE_API_KEY environment variable: {'set' if os.environ.get('VOYAGE_API_KEY') else 'not set'}")
                enable_embedding = False
        
        # Configuration
        self.target_chunk_chars = self.chunking_config["target_tokens"] * self.chunking_config["chars_per_token"]
        self.min_chunk_chars = self.chunking_config["min_tokens"] * self.chunking_config["chars_per_token"]
        self.max_chunk_chars = self.chunking_config["max_tokens"] * self.chunking_config["chars_per_token"]
        self.overlap_chars = self.chunking_config["overlap_tokens"] * self.chunking_config["chars_per_token"]
        self.preserve_sentences = self.chunking_config["preserve_sentences"]
        self.preserve_paragraphs = self.chunking_config["preserve_paragraphs"]
        self.section_aware = self.chunking_config["section_aware"]
        
        self.target_context_tokens = self.context_config["target_tokens"]
        self.max_context_tokens = self.context_config["max_tokens"]
        self.context_temperature = self.context_config["temperature"]
        self.include_title = self.context_config["include_title"]
        self.include_abstract = self.context_config["include_abstract"]
        self.include_section_heading = self.context_config["include_section_heading"]
        
        self.save_path = Path(save_path)
        self.chunks = []
        self.enable_haiku = enable_haiku
        self.enable_embedding = enable_embedding
        self.haiku_model = haiku_model or self.context_config["model"]
        # embedding_model is now set earlier in the initialization
        self.include_context_in_embedding = self.embedding_config["include_context"]
        self.normalize_embeddings = self.embedding_config["normalize"]

        # Prompts for context generation
        self.doc_prompt = """
        <document>
        {doc_content}
        </document>
        """
        
        self.chunk_prompt = """
        Here is the chunk we want to situate within the whole document/section
        <chunk>
        {chunk_content}
        </chunk>

        Please provide succinct context to situate this chunk within the overall document/section for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

    def chunk_text(self, text: str, section_heading: Optional[str] = None, type: Optional[str] = None) -> List[str]:
        """
        Split text into chunks of approximately target size
        
        Args:
            text: Text to split into chunks
            section_heading: Optional section heading for section-aware chunking
            
        Returns:
            List of text chunks
        """
        self.logger.debug(f"Chuunking section heading: {section_heading}, type: {type}, preview= {text[:50]}...")
        # If we're preserving paragraphs, split by paragraphs first
        if self.preserve_paragraphs:
            paragraphs = re.split(r'\n\s*\n', text)
            chunks = []
            
            for paragraph in paragraphs:
                if type != "figure":
                    # Skip empty paragraphs
                    if not paragraph.strip():
                        continue
                
                # If paragraph is smaller than max chunk size, add it as a chunk
                if len(paragraph) <= self.max_chunk_chars:
                    chunks.append(paragraph)
                else:
                    # Otherwise, split the paragraph into smaller chunks
                    paragraph_chunks = self._chunk_by_sentences(paragraph)
                    chunks.extend(paragraph_chunks)
            
            return chunks
        
        # If we're not preserving paragraphs, just split by sentences
        return self._chunk_by_sentences(text)
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks by sentences
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        # If we're preserving sentences, use NLTK to split by sentences
        if self.preserve_sentences:
            try:
                sentences = nltk.sent_tokenize(text)
            except LookupError as e:
                self.logger.warning(f"NLTK punkt tokenizer not found. Downloading now...")
                nltk.download('punkt_tab', quiet=True)
                try:
                    sentences = nltk.sent_tokenize(text)
                except Exception as e2:
                    self.logger.warning(f"Error tokenizing sentences with NLTK even after download: {str(e2)}")
                    # Fall back to simple regex-based sentence splitting
                    sentences = re.split(r'(?<=[.!?])\s+', text)
            except Exception as e:
                self.logger.warning(f"Error tokenizing sentences with NLTK: {str(e)}")
                # Fall back to simple regex-based sentence splitting
                sentences = re.split(r'(?<=[.!?])\s+', text)
        else:
            # Otherwise, just split by a fixed number of characters
            sentences = []
            for i in range(0, len(text), self.target_chunk_chars // 2):
                sentences.append(text[i:i + self.target_chunk_chars // 2])
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed the max chunk size and we already have content
            if current_length + sentence_length > self.max_chunk_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # If we're using overlap, include some of the previous sentences in the next chunk
                if self.overlap_chars > 0 and len(current_chunk) > 1:
                    # Calculate how many sentences to include in the overlap
                    overlap_length = 0
                    overlap_sentences = []
                    
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.overlap_chars:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    
                    current_chunk = overlap_sentences + [sentence]
                    current_length = overlap_length + sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            
            # If this is the first sentence and it's longer than the max chunk size
            elif sentence_length > self.max_chunk_chars and not current_chunk:
                # Split the sentence into smaller chunks
                for i in range(0, sentence_length, self.max_chunk_chars):
                    chunk = sentence[i:i + self.max_chunk_chars]
                    if chunk:
                        chunks.append(chunk)
            
            # If adding this sentence would make the chunk smaller than the target size
            elif current_length + sentence_length <= self.target_chunk_chars or not current_chunk:
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # If adding this sentence would make the chunk larger than the target size
            # but still smaller than the max size
            elif current_length + sentence_length <= self.max_chunk_chars:
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # If adding this sentence would exceed the max chunk size
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Add the last chunk if it has content and meets the minimum size
        if current_chunk and current_length >= 1:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def generate_context(self, doc_content: str, chunk_content: str, 
                       title: Optional[str] = None, abstract: Optional[str] = None, 
                       section_heading: Optional[str] = None) -> Optional[str]:
        """
        Generate context for a chunk using Claude with retries
        
        Args:
            doc_content: Full document content
            chunk_content: Content of the chunk
            title: Document title
            abstract: Document abstract
            section_heading: Section heading
            
        Returns:
            Generated context, or None if generation failed
        """
        if not self.enable_haiku or not self.client:
            self.logger.debug("Context generation disabled")
            return None
        
        # Prepare the document content with optional metadata
        enhanced_doc_content = doc_content
        
        # Add title if available and configured
        if title and self.include_title:
            enhanced_doc_content = f"Title: {title}\n\n{enhanced_doc_content}"
        
        # Add abstract if available and configured
        if abstract and self.include_abstract:
            enhanced_doc_content = f"{enhanced_doc_content}\n\nAbstract: {abstract}"
        
        # Add section heading if available and configured
        if section_heading and self.include_section_heading:
            enhanced_chunk_content = f"Section: {section_heading}\n\n{chunk_content}"
        else:
            enhanced_chunk_content = chunk_content
            
        
        for attempt in range(API["max_retries"]):
            try:
                
                response = self.client.messages.create(
                    model=self.haiku_model,
                    max_tokens=self.max_context_tokens,
                    temperature=self.context_temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.doc_prompt.format(doc_content=enhanced_doc_content),
                                    "cache_control": {"type": "ephemeral"}
                                },
                                {
                                    "type": "text",
                                    "text": self.chunk_prompt.format(chunk_content=enhanced_chunk_content)
                                }
                            ]
                        }
                    ]
                )
                return response.content[0].text

            except Exception as e:
                self.logger.error(f"Error generating context (attempt {attempt + 1}/{API['max_retries']}): {str(e)}")
                if attempt < API["max_retries"] - 1:
                    time.sleep(API["retry_delay"])
                else:
                    return "Failed to generate context"

    def create_embedding(self, context: str, chunk: str) -> Optional[np.ndarray]:
        """
        Create embedding for context and chunk with retries
        
        Args:
            context: Context for the chunk
            chunk: Content of the chunk
            
        Returns:
            Embedding vector, or None if embedding is disabled or failed
        """
        if not self.enable_embedding:
            self.logger.debug("Embedding is disabled, returning None")
            return None
            
        if not self.voyage_client:
            self.logger.warning("Voyage client is not initialized, returning None")
            return None
        
        # Prepare the text for embedding
        if self.include_context_in_embedding:
            text_to_embed = f"{context}\n\n{chunk}"
        else:
            text_to_embed = chunk
        
        self.logger.info(f"Creating embedding with model: {self.embedding_model}")
        self.logger.info(f"Text length: {len(text_to_embed)} chars")
            
        for attempt in range(API["max_retries"]):
            try:
                self.logger.info(f"Embedding attempt {attempt + 1}/{API['max_retries']}")
                embedding = self.voyage_client.embed([text_to_embed], model=self.embedding_model).embeddings[0]
                
                # Normalize the embedding if configured
                if self.normalize_embeddings:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                
                self.logger.info(f"Embedding successful, dimensions: {len(embedding)}")
                self.logger.info(f"First 5 values: {embedding[:5]}")
                return embedding
            except Exception as e:
                self.logger.error(f"Error creating embedding (attempt {attempt + 1}/{API['max_retries']}): {str(e)}")
                if attempt < API["max_retries"] - 1:
                    time.sleep(API["retry_delay"])
                else:
                    self.logger.error("All embedding attempts failed, returning zeros")
                    return np.zeros(self.embedding_config["dimensions"])  # Return dummy embedding on failure

    def save_chunk(self, chunk: DocumentChunk):
        """
        Save a chunk to the output file
        
        Args:
            chunk: DocumentChunk to save
        """
        # Debug logging for embedding
        if chunk.embedding is not None:
            self.logger.info(f"Saving chunk with embedding, type: {type(chunk.embedding)}")
            self.logger.info(f"Embedding dimensions: {len(chunk.embedding) if hasattr(chunk.embedding, '__len__') else 'unknown'}")
            self.logger.info(f"First 5 values: {chunk.embedding[:5] if hasattr(chunk.embedding, '__getitem__') else 'unknown'}")
            self.logger.info(f"Are all zeros? {np.all(chunk.embedding == 0) if isinstance(chunk.embedding, np.ndarray) else 'unknown'}")
        else:
            self.logger.warning("Chunk has no embedding")
            
        # Create the chunk dictionary based on output configuration
        chunk_dict = {}
        
        # Set timestamp if not already set
        if not chunk.timestamp:
            chunk.timestamp = datetime.now().isoformat()
        
        # Map of field names to their values and inclusion flags
        
        # The DocumentChunk class now handles embedding conversion in __post_init__
        # So we can directly use chunk.embedding
        
        field_map = {
            "uuid": (chunk.uuid, self.output_config["include_uuid"]),
            "paper_title": (chunk.paper_title, self.output_config["include_title"]),
            "temp_path": (chunk.temp_path, self.output_config["include_temp_path"]),
            "doi": (chunk.doi, self.output_config["include_doi"]),
            "url": (chunk.url, self.output_config["include_url"]),
            "group": (chunk.group, self.output_config["include_group"]),
            "timestamp": (chunk.timestamp, self.output_config["include_timestamp"]),
            "chunk_text": (chunk.chunk_text, self.output_config["include_chunk_text"]),
            "context": (chunk.context, self.output_config["include_context"]),
            "section_heading": (chunk.section_heading, self.output_config["include_section_info"]),
            "section_index": (chunk.section_index, self.output_config.get("include_section_index", True)), 
            "embedding": (chunk.embedding, self.output_config["include_embedding"])
        }
        
        # If specific fields are specified, use only those
        if self.output_config["fields"]:
            for field in self.output_config["fields"]:
                if field in field_map and field_map[field][0] is not None and field_map[field][1]:
                    chunk_dict[field] = field_map[field][0]
        else:
            # Otherwise, include all fields based on inclusion flags
            for field, (value, include) in field_map.items():
                if value is not None and include:
                    chunk_dict[field] = value
        
        # Save the chunk based on the output format
        if self.output_config["format"] == "jsonl":
            # Debug the chunk_dict before saving
            if "embedding" in chunk_dict:
                self.logger.info(f"Chunk dict embedding type: {type(chunk_dict['embedding'])}")
                self.logger.info(f"Chunk dict embedding length: {len(chunk_dict['embedding']) if hasattr(chunk_dict['embedding'], '__len__') else 'unknown'}")
                self.logger.info(f"Chunk dict embedding first 5 values: {chunk_dict['embedding'][:5] if hasattr(chunk_dict['embedding'], '__getitem__') else 'unknown'}")
                self.logger.info(f"Chunk dict embedding all zeros? {all(v == 0 for v in chunk_dict['embedding']) if hasattr(chunk_dict['embedding'], '__iter__') else 'unknown'}")
            
            # Create a direct dictionary for JSON serialization
            direct_dict = {}
            for field, (value, include) in field_map.items():
                if value is not None and include:
                    direct_dict[field] = value
            
            # Debug the direct dictionary
            if "embedding" in direct_dict:
                self.logger.info(f"Direct dict embedding type: {type(direct_dict['embedding'])}")
                self.logger.info(f"Direct dict embedding length: {len(direct_dict['embedding'])}")
                self.logger.info(f"Direct dict embedding first 5 values: {direct_dict['embedding'][:5]}")
                self.logger.info(f"Direct dict embedding all zeros? {all(v == 0 for v in direct_dict['embedding'])}")
            
            # Write the direct dictionary to the file
            with open(self.save_path, 'a') as f:
                f.write(json.dumps(direct_dict) + '\n')
        
        elif self.output_config["format"] == "json":
            # For JSON format, we need to read the existing file, add the chunk, and write it back
            chunks = []
            
            if self.save_path.exists():
                try:
                    with open(self.save_path, 'r') as f:
                        chunks = json.load(f)
                except json.JSONDecodeError:
                    # If the file exists but is not valid JSON, start with an empty list
                    chunks = []
            
            chunks.append(chunk_dict)
            
            with open(self.save_path, 'w') as f:
                json.dump(chunks, f, indent=2)
        
        elif self.output_config["format"] == "csv":
            # For CSV format, we need to flatten the embedding
            import csv
            
            # Check if the file exists
            file_exists = self.save_path.exists()
            
            with open(self.save_path, 'a', newline='') as f:
                # Flatten the embedding if it exists
                if "embedding" in chunk_dict:
                    embedding = chunk_dict.pop("embedding")
                    for i, val in enumerate(embedding):
                        chunk_dict[f"embedding_{i}"] = val
                
                writer = csv.DictWriter(f, fieldnames=chunk_dict.keys())
                
                # Write header if the file is new
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(chunk_dict)

    def process_document(self, doc_path: str, doc_data: Dict[str, Any], url: Optional[str] = None, 
                       scipdf_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Process a single document into chunks
        
        Args:
            doc_path: Path to the document (temporary local path)
            doc_data: Parsed document data
            url: URL where the document was downloaded from
            scipdf_config: Configuration for ScipdfParser
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            self.logger.info(f"Processing document: {doc_path}")
            
            # Use default ScipdfParser config if none provided
            if scipdf_config is None:
                scipdf_config = SCIPDF.copy()
                
            # Extract document metadata
            paper_title = doc_data.get("title", "")
            doi = doc_data.get("doi", "")
            abstract = doc_data.get("abstract", "")
            
            self.logger.info(f"Document metadata - Title: {paper_title}, DOI: {doi}")
            
            # Filter sections based on configuration
            filtered_sections = []
            section_filters = scipdf_config.get("section_filters", [])
            exclude_sections = scipdf_config.get("exclude_sections", [])


            
            for section in doc_data["sections"]:
                section_heading = section.get("heading", "")
                


                self.logger.info(f"Processing: {section_heading} Preview: {section.get('text')[:65]}...")
            
                # Skip if section is in exclude list
                if section_heading in exclude_sections:
                    continue
                
                # Include if section filters is empty or section is in filters
                if not section_filters or section_heading in section_filters:
                    filtered_sections.append(section)
            for figure in doc_data["figures"]:
                figure_label = figure.get("figure_label", "")
                figure_caption = figure.get("figure_caption", "")


                self.logger.info(f"Processing figure: {figure_label} Preview: {figure_caption[:65]}...")
                filtered_sections.append({
                    "type": "figure",
                    "heading": figure_label,
                    "text": figure_caption
                    })

            # Combine all section text for full document context
            con_text = ""
            
            if  self.context_config["enabled"] and self.context_config["full_text"]:

                if self.section_aware:
                    # Include section headings in the full document text
                    con_text = "\n\n".join(
                        f"{section.get('heading', 'Section')}\n{section['text'] if isinstance(section['text'], str) else ' '.join(section['text'])}"
                        for section in filtered_sections
                    )
                else:
                    # Just concatenate all section text
                    con_text = "\n\n".join(
                        section["text"] if isinstance(section["text"], str) else " ".join(section["text"])
                        for section in filtered_sections
                    )
                
            # Process each section
            self.logger.info(f"Processing {len(filtered_sections)} sections")
            
            for i, section in enumerate(filtered_sections):
                section_heading = section.get("heading", "")
                section_text = section["text"] if isinstance(section["text"], str) else " ".join(section["text"])

                if con_text == "":
                    if self.context_config["enabled"]:
                        con_text = section_text

                

                
                # Skip empty sections
                if not section_text.strip():
                    self.logger.warning(f"Skipping empty section: {section_heading}")
                    continue
                
                self.logger.info(f"Processing section {i+1}/{len(filtered_sections)}: {section_heading} ({len(section_text)} chars)")
                
                # Check if we should use whole sections
                if scipdf_config.get("whole_sections", False):
                    # Use the entire section as a single chunk
                    chunks = [section_text]
                    self.logger.info(f"Using whole section as a single chunk: {len(section_text)} chars")
                else:
                    # Chunk the section text
                    if self.section_aware:
                        chunks = self.chunk_text(section_text, section_heading, type=section.get("type"))
                    else:
                        chunks = self.chunk_text(section_text, type=section.get("type"))
                    self.logger.info(f"Created {len(chunks)} chunks from section")
                
                for chunk_index, chunk_text in enumerate(chunks):
                    # Generate context with retries
                    context = self.generate_context(
                        con_text, 
                        chunk_text, 
                        title=paper_title if self.include_title else None,
                        abstract=abstract if self.include_abstract else None,
                        section_heading=section_heading if self.include_section_heading else None
                    )
                    
                    # Create embedding with retries
                    embedding = self.create_embedding(context, chunk_text)
                    
                    # Debug logging for embedding before chunk creation
                    if embedding is not None:
                        self.logger.info(f"Embedding before chunk creation, type: {type(embedding)}")
                        self.logger.info(f"Embedding dimensions: {len(embedding) if hasattr(embedding, '__len__') else 'unknown'}")
                        self.logger.info(f"First 5 values: {embedding[:5] if hasattr(embedding, '__getitem__') else 'unknown'}")
                        self.logger.info(f"Are all zeros? {np.all(embedding == 0) if isinstance(embedding, np.ndarray) else 'unknown'}")
                    else:
                        self.logger.warning("Embedding is None before chunk creation")
                    
                    # Create timestamp
                    timestamp = datetime.now().isoformat()
                    
                    # Create and save chunk
                    chunk = DocumentChunk(
                        paper_title=paper_title,
                        chunk_text=chunk_text,
                        context=context,
                        abstract=abstract,
                        embedding=embedding,
                        uuid=str(uuid.uuid4()) if self.output_config["include_uuid"] else None,
                        group=self.group,
                        doi=doi,
                        url=url,  # Use the URL as the primary source identifier
                        temp_path=doc_path,  # Store the temporary path separately
                        section_heading=section_heading,
                        section_index=chunk_index, 
                        timestamp=timestamp
                    )
                    
                    self.chunks.append(chunk)
                    self.save_chunk(chunk)
                
                if  self.context_config["enabled"] and (self.context_config["full_text"] == False):
                    con_text = "" 
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing document {doc_path}: {str(e)}")
            return False

# ============= PDF Processing Functions =============

def download_pdf_from_url(url: str, output_dir: Path = TEMP_DIR) -> Optional[Path]:
    """
    Download a PDF from a URL to a temporary file
    
    Args:
        url: URL to download from
        output_dir: Directory to save the PDF to
        
    Returns:
        Path to the downloaded PDF, or None if download failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create a temporary file with a .pdf extension
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=output_dir) as temp_file:
            temp_path = Path(temp_file.name)
        
        # Download the PDF
        logger.info(f"Downloading PDF from {url}")
        
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,*/*',
        }
        
        # Use a session to handle redirects and cookies
        with requests.Session() as session:
            # First make a HEAD request to check content type
            head_response = session.head(url, headers=headers, timeout=API["timeout"])
            
            # Check if the content is a PDF
            content_type = head_response.headers.get('Content-Type', '')
            is_pdf = 'application/pdf' in content_type or url.lower().endswith('.pdf')
            
            if not is_pdf:
                logger.warning(f"URL may not be a PDF: {url} (Content-Type: {content_type})")
                # Continue anyway, as sometimes content type is not correctly set
            
            # Download the file
            response = session.get(url, headers=headers, stream=True, timeout=API["timeout"])
            response.raise_for_status()
            
            # Save the PDF
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify the file is a PDF by checking the magic bytes
            with open(temp_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    logger.warning(f"Downloaded file is not a PDF (magic bytes: {header})")
                    # Continue anyway, as some PDFs might not have the correct header
        
        logger.info(f"PDF downloaded to {temp_path}")
        return temp_path
    
    except Exception as e:
        logger.error(f"Error downloading PDF from {url}: {str(e)}")
        # Clean up the temporary file if it exists
        try:
            if 'temp_path' in locals():
                temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

def process_pdf_file(pdf_path: Path, chunker: ContextualChunker, parsed_output_dir: Path, 
                 scipdf_config: Dict[str, Any] = None, url: Optional[str] = None) -> bool:
    """
    Process a single PDF file
    
    Args:
        pdf_path: Path to the PDF file
        chunker: ContextualChunker instance
        parsed_output_dir: Directory to save parsed output
        scipdf_config: Configuration for ScipdfParser
        url: Optional URL where the PDF was downloaded from
        
    Returns:
        True if processing was successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Use default ScipdfParser config if none provided
        if scipdf_config is None:
            scipdf_config = SCIPDF.copy()
        
        # 1. Parse PDF
        pdf_path_obj = Path(pdf_path)
        parsed_output_path = parsed_output_dir / f"{pdf_path_obj.stem}.json"
        
        # Make sure the parsed_output_dir exists
        parsed_output_dir.mkdir(parents=True, exist_ok=True)
        
        if not parsed_output_path.exists():
            
            try:
                # Try importing from the litpipe.scipdf_parser
                from litpipe.scipdf_parser.scipdf.pdf.parse_pdf import parse_pdf_to_dict
                logger.info(f"Using litpipe.scipdf_parser for {pdf_path}")
            except ImportError:
                logger.error("Failed to import scipdf_parser. Make sure it's installed.")
                return False
            
            # Parse PDF with ScipdfParser configuration
            logger.info(f"Parsing PDF: {pdf_path}")
            try:
                article_dict = parse_pdf_to_dict(
                    str(pdf_path),
                    fulltext=True,
                    soup=True,
                    as_list=False,
                    return_coordinates=True,
                    parse_figures=scipdf_config["extract_figures"]
                )
                logger.info(f"Successfully parsed PDF: {pdf_path}")
            except Exception as e:
                logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
                return False
            
            if scipdf_config["extract_images"]:
                pass
            
            # Apply additional ScipdfParser configuration
            if not scipdf_config["extract_metadata"]:
                # Remove metadata fields
                for field in ["title", "authors", "doi", "journal", "year"]:
                    if field in article_dict:
                        del article_dict[field]
            
            if not scipdf_config["extract_sections"]:
                # Remove sections
                article_dict["sections"] = []
            
            if not scipdf_config["extract_abstract"]:
                # Remove abstract
                if "abstract" in article_dict:
                    del article_dict["abstract"]
            
            if not scipdf_config["extract_authors"]:
                # Remove authors
                if "authors" in article_dict:
                    del article_dict["authors"]
            
            if not scipdf_config["extract_doi"]:
                # Remove DOI
                if "doi" in article_dict:
                    del article_dict["doi"]
            
            # Save parsed result
            with open(parsed_output_path, 'w') as f:
                json.dump({str(pdf_path): article_dict}, f, indent=2)
                
        else:
            # Load existing parsed result
            with open(parsed_output_path) as f:
                data = json.load(f)
                article_dict = data[str(pdf_path)]
        
        # 2. Process into chunks
        success = chunker.process_document(str(pdf_path), article_dict, url, scipdf_config)
        if not success:
            logger.error(f"Failed to process document: {pdf_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return False

def process_pdfs_from_directory(pdf_dir: Path, chunker: ContextualChunker, parsed_output_dir: Path, 
                           scipdf_config: Dict[str, Any] = None) -> int:
    """
    Process all PDFs in a directory
    
    Args:
        pdf_dir: Directory containing PDFs
        chunker: ContextualChunker instance
        parsed_output_dir: Directory to save parsed output
        scipdf_config: Configuration for ScipdfParser
        
    Returns:
        Number of successfully processed PDFs
    """
    logger = logging.getLogger(__name__)
    
    # Get list of PDFs
    pdf_paths = list(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.error(f"No PDF files found in {pdf_dir}")
        return 0
    
    logger.info(f"Found {len(pdf_paths)} PDF files")
    
    # Sort by size to process smaller files first
    pdf_paths.sort(key=lambda x: x.stat().st_size)
    
    # Process each PDF
    success_count = 0
    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        # For local PDFs, we use a file:// URL as the source identifier
        file_url = f"file://{pdf_path.absolute()}"
        
        if process_pdf_file(pdf_path, chunker, parsed_output_dir, scipdf_config, url=file_url):
            success_count += 1
    
    return success_count

def process_pdf_from_doi(doi: str, chunker: ContextualChunker, parsed_output_dir: Path, 
                        scipdf_config: Dict[str, Any] = None) -> bool:
    """
    Process a PDF from a DOI
    
    Args:
        doi: DOI to process
        chunker: ContextualChunker instance
        parsed_output_dir: Directory to save parsed output
        scipdf_config: Configuration for ScipdfParser
        
    Returns:
        True if processing was successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Import DOI resolver
        from doi_resolver import resolve_doi_to_pdf_url
        
        # Resolve DOI to PDF URL
        pdf_url = resolve_doi_to_pdf_url(doi)
        if not pdf_url:
            logger.error(f"Failed to resolve DOI {doi} to PDF URL")
            return False
        
        # Process the PDF from the URL
        return process_pdf_from_url(pdf_url, chunker, parsed_output_dir, scipdf_config, doi=doi)
    
    except ImportError:
        logger.error("DOI resolver not available. Please implement doi_resolver.py")
        return False
    except Exception as e:
        logger.error(f"Error processing DOI {doi}: {str(e)}")
        return False

def process_pdf_from_url(url: str, chunker: ContextualChunker, parsed_output_dir: Path, 
                        scipdf_config: Dict[str, Any] = None, doi: Optional[str] = None) -> bool:
    """
    Process a PDF from a URL
    
    Args:
        url: URL to process
        chunker: ContextualChunker instance
        parsed_output_dir: Directory to save parsed output
        scipdf_config: Configuration for ScipdfParser
        doi: Optional DOI of the document
        
    Returns:
        True if processing was successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing PDF from URL: {url}")
    
    try:
        # Check if the URL is a PDF
        if url.lower().endswith('.pdf'):
            # Download the PDF directly
            pdf_path = download_pdf_from_url(url)
            if not pdf_path:
                logger.error(f"Failed to download PDF from {url}")
                return False
            
            # Process the PDF
            success = process_pdf_file(pdf_path, chunker, parsed_output_dir, scipdf_config, url)
            
            # Clean up the temporary file
            try:
                pdf_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {pdf_path}: {str(e)}")
            
            return success
        else:
            # For non-PDF URLs, we need to find the PDF link on the page
            logger.info(f"URL is not a direct PDF link: {url}")
            
            # Special handling for arXiv URLs
            if "arxiv.org" in url:
                # Convert arXiv URLs to direct PDF links
                if "/abs/" in url:
                    # Convert abstract URL to PDF URL
                    pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
                    logger.info(f"Converting arXiv abstract URL to PDF URL: {pdf_url}")
                elif "/pdf/" in url and not url.endswith(".pdf"):
                    # Add .pdf extension if missing
                    pdf_url = url + ".pdf"
                    logger.info(f"Adding .pdf extension to arXiv PDF URL: {pdf_url}")
                else:
                    pdf_url = url
                
                # Download the PDF directly
                pdf_path = download_pdf_from_url(pdf_url)
                if not pdf_path:
                    logger.error(f"Failed to download PDF from {pdf_url}")
                    return False
                
                # Process the PDF
                success = process_pdf_file(pdf_path, chunker, parsed_output_dir, scipdf_config, url)
                
                # Clean up the temporary file
                try:
                    pdf_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {pdf_path}: {str(e)}")
                
                return success
            
            # We already imported requests and BeautifulSoup at the top
            
            # Get the page content
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for PDF links
            pdf_links = []
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href.lower().endswith('.pdf'):
                    pdf_links.append(href)
            
            if not pdf_links:
                logger.error(f"No PDF links found on page: {url}")
                return False
            
            # Use the first PDF link
            pdf_url = pdf_links[0]
            
            # If the link is relative, make it absolute
            if not pdf_url.startswith('http'):
                from urllib.parse import urljoin
                pdf_url = urljoin(url, pdf_url)
            
            logger.info(f"Found PDF link: {pdf_url}")
            
            # Download and process the PDF
            pdf_path = download_pdf_from_url(pdf_url)
            if not pdf_path:
                logger.error(f"Failed to download PDF from {pdf_url}")
                return False
            
            # Process the PDF
            success = process_pdf_file(pdf_path, chunker, parsed_output_dir, scipdf_config, pdf_url)
            
            # Clean up the temporary file
            try:
                pdf_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {pdf_path}: {str(e)}")
            
            return success
    
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return False

def process_pdfs_from_query(query: str, chunker: ContextualChunker, parsed_output_dir: Path, 
                           scipdf_config: Dict[str, Any] = None, max_results: int = 5,
                           sources: List[str] = None, filter_by_pdf: bool = True,
                           enrich_with_unpaywall: bool = True, sort_by: str = "relevance") -> int:
    """
    Process PDFs from a search query
    
    Args:
        query: Search query
        chunker: ContextualChunker instance
        parsed_output_dir: Directory to save parsed output
        scipdf_config: Configuration for ScipdfParser
        max_results: Maximum number of results to process
        sources: List of sources to search (default: ["arxiv", "pubmed", "semantic_scholar", "crossref"])
        filter_by_pdf: Whether to filter out papers without PDF URLs
        enrich_with_unpaywall: Whether to enrich papers with PDF URLs from Unpaywall
        sort_by: How to sort the results (relevance, date)
        
    Returns:
        Number of successfully processed PDFs
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Import paper search
        from paper_search import search_papers
        
        # Search for papers
        papers = search_papers(
            query=query, 
            max_results=max_results,
            sources=sources,
            filter_by_pdf=filter_by_pdf,
            enrich_with_unpaywall=enrich_with_unpaywall,
            sort_by=sort_by
        )
        
        if not papers:
            logger.error(f"No papers found for query: {query}")
            return 0
        
        logger.info(f"Found {len(papers)} papers for query: {query}")
        
        # Process each paper with a PDF
        success_count = 0
        for i, paper in enumerate(papers):
            if paper.get('pdf_available') and paper.get('pdf_url'):
                logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.get('title')}")
                
                # Get metadata
                doi = paper.get('doi')
                source_url = paper.get('source_url')
                
                # Use the source URL if available, otherwise use the PDF URL
                url = source_url if source_url else paper.get('pdf_url')
                
                if process_pdf_from_url(paper['pdf_url'], chunker, parsed_output_dir, scipdf_config, doi=doi, url=url):
                    success_count += 1
                    logger.info(f"Successfully processed paper: {paper.get('title')}")
                else:
                    logger.error(f"Failed to process paper: {paper.get('title')}")
        
        return success_count
    
    except ImportError:
        logger.error("Paper search not available. Please implement paper_search.py")
        return 0
    except Exception as e:
        logger.error(f"Error processing query {query}: {str(e)}")
        return 0

# ============= SurrealDB Functions =============
'''
def store_in_surrealdb_rest(chunks_path: Path, surreal_config: Dict[str, Any] = None):
    """
    Store processed chunks in SurrealDB using REST API
    
    Args:
        chunks_path: Path to the processed chunks JSONL file
        surreal_config: Configuration for SurrealDB
        
    Returns:
        True if successful, False otherwise
    """
    # Use default SurrealDB config if none provided
    if surreal_config is None:
        surreal_config = SURREAL.copy()
    
    # Skip if SurrealDB is disabled
    if not surreal_config["enabled"]:
        return False
    
    logger = logging.getLogger(__name__)
    logger.info("Storing chunks in SurrealDB using REST API")
    
    try:
        import requests
        import json
        
        # Check if the file exists
        if not os.path.exists(chunks_path):
            logger.error(f"Chunks file not found: {chunks_path}")
            return False
            
        # Check if the file is empty
        if os.path.getsize(chunks_path) == 0:
            logger.warning(f"Chunks file is empty: {chunks_path}")
            return False
        
        # Set up authentication
        auth = (surreal_config["credentials"]["username"], surreal_config["credentials"]["password"])
        
        # Base URL - convert WebSocket URL to HTTP URL if needed
        base_url = surreal_config["url"]
        if base_url.startswith("ws://"):
            base_url = "http://" + base_url[5:]
        elif base_url.startswith("wss://"):
            base_url = "https://" + base_url[6:]
        
        if base_url.endswith("/rpc"):
            base_url = base_url[:-4]
            
        logger.info(f"Using REST API URL: {base_url}")
        
        # Test connection
        logger.info(f"Testing connection to SurrealDB at {base_url}")
        test_url = f"{base_url}/health"
        try:
            test_response = requests.get(test_url)
            test_response.raise_for_status()
            logger.info(f"Connection successful: {test_response.text}")
        except Exception as e:
            logger.warning(f"Health check failed, but continuing: {str(e)}")
        
        # Select namespace and database
        logger.info(f"Using namespace '{surreal_config['namespace']}' and database '{surreal_config['database']}'")
        
        # Set NS and DB headers
        headers = {
            "NS": surreal_config['namespace'],
            "DB": surreal_config['database']
        }
        
        # Create table
        logger.info(f"Creating table '{surreal_config['table']}'")
        sql_url = f"{base_url}/sql"
        
        try:
            table_response = requests.post(
                sql_url,
                auth=auth,
                headers=headers,
                data=f"DEFINE TABLE {surreal_config['table']};"
            )
            table_response.raise_for_status()
            logger.info(f"Table created or already exists: {table_response.text}")
        except Exception as e:
            logger.warning(f"Error creating table, but continuing: {str(e)}")
        
        # Import data
        logger.info(f"Importing data from {chunks_path}")
        
        success_count = 0
        error_count = 0
        
        with open(chunks_path, 'r') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                    
                try:
                    # Process each line
                    record = json.loads(line)
                    
                    # Create the record
                    record_id = record.get("uuid", str(uuid.uuid4()))
                    record_url = f"{base_url}/key/{surreal_config['namespace']}/{surreal_config['database']}/{surreal_config['table']}/{record_id}"
                    
                    record_response = requests.put(
                        record_url,
                        auth=auth,
                        headers=headers,
                        json=record
                    )
                    record_response.raise_for_status()
                    
                    success_count += 1
                    
                    # Print progress every 10 records
                    if success_count % 10 == 0:
                        logger.info(f"Imported {success_count} records so far...")
                    
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # Only show the first few errors
                        logger.error(f"Error importing record {i}: {e}")
                    elif error_count == 6:
                        logger.info("Suppressing further error messages...")
        
        logger.info(f"Import complete: {success_count} records imported successfully, {error_count} errors")
        
        # Add mtree index if configured
        if surreal_config["create_mtree_index"]:
            logger.info("Adding mtree index on embedding field")
            try:
                index_response = requests.post(
                    sql_url,
                    auth=auth,
                    headers=headers,
                    data=f"""
                    DEFINE INDEX IF NOT EXISTS embedding_mtree 
                    ON TABLE {surreal_config['table']} 
                    FIELDS embedding 
                    MTREE DIMENSION {surreal_config['mtree_dimension']} 
                    DIST {surreal_config['mtree_distance']} 
                    TYPE {surreal_config['mtree_type']};
                    """
                )
                index_response.raise_for_status()
                logger.info("mtree index added successfully")
            except Exception as e:
                logger.warning(f"Error adding mtree index, but continuing: {str(e)}")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error storing chunks in SurrealDB: {str(e)}")
        return False'''

async def store_in_surrealdb(chunks_path: Path, surreal_config: Dict[str, Any] = None):
    """
    Store processed chunks in SurrealDB using WebSocket API
    
    Args:
        chunks_path: Path to the processed chunks JSONL file
        surreal_config: Configuration for SurrealDB
        
    Returns:
        True if successful, False otherwise
    """
    # Use default SurrealDB config if none provided
    if surreal_config is None:
        surreal_config = SURREAL.copy()
    
    # Skip if SurrealDB is disabled
    if not surreal_config["enabled"]:
        return False
    
    logger = logging.getLogger(__name__)
    logger.info("Storing chunks in SurrealDB using WebSocket API")
    
    # Check if the file exists
    if not os.path.exists(chunks_path):
        logger.error(f"Chunks file not found: {chunks_path}")
        return False
        
    # Check if the file is empty
    if os.path.getsize(chunks_path) == 0:
        logger.warning(f"Chunks file is empty: {chunks_path}")
        return False
    
    # Try using the surrealdb package
    try:
        try:
            from surrealdb import AsyncSurreal
        except ImportError:
            logger.error("surrealdb package not installed. Please install it with 'pip install surrealdb'")
            return False
        
        logger.info(f"Connecting to SurrealDB at {surreal_config['url']}")
        db = AsyncSurreal(url=surreal_config["url"])
        
        try:
            # Authenticate
            logger.info("Authenticating with SurrealDB")
            await db.signin(surreal_config["credentials"])
            
            # Select namespace and database
            logger.info(f"Using namespace '{surreal_config['namespace']}' and database '{surreal_config['database']}'")
            await db.use(surreal_config["namespace"], surreal_config["database"])
            
            # Create table
            logger.info(f"Creating table '{surreal_config['table']}'")
            await db.query(f"DEFINE TABLE {surreal_config['table']};")
            
            # Import data
            logger.info(f"Importing data from {chunks_path}")
            
            success_count = 0
            error_count = 0
            
            with open(chunks_path, 'r') as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                        
                    try:
                        # Process each line
                        record = json.loads(line)
                        
                        # Create the record
                        query = f"""
                        CREATE type::thing("{surreal_config['table']}", $id) CONTENT {{
                            paper_title: $paper_title,
                            pdf_path: $pdf_path,
                            doi: $doi,
                            url: $url,
                            chunk_text: $chunk_text,
                            context: $context,
                            embedding: $embedding,
                            uuid: $uuid,
                            group: $group,
                            section_heading: $section_heading,
                            section_index: $section_index,
                            timestamp: $timestamp
                        }};
                        """
                        
                        # Generate a UUID if not present
                        record_uuid = record.get("uuid", str(uuid.uuid4()))
                        
                        params = {
                            "id": record_uuid,
                            "paper_title": record.get("paper_title", ""),
                            "pdf_path": record.get("pdf_path", ""),
                            "doi": record.get("doi", ""),
                            "url": record.get("url", ""),
                            "chunk_text": record.get("chunk_text", ""),
                            "context": record.get("context", ""),
                            "embedding": record.get("embedding", []),
                            "uuid": record_uuid,
                            "group": record.get("group", DEFAULT_GROUP),
                            "section_heading": record.get("section_heading", ""),
                            "section_index": record.get("section_index", 0),
                            "timestamp": record.get("timestamp", datetime.now().isoformat())
                        }
                        
                        await db.query(query, params)
                        
                        success_count += 1
                        
                        # Print progress every 10 records
                        if success_count % 10 == 0:
                            logger.info(f"Imported {success_count} records so far...")
                        
                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:  # Only show the first few errors
                            logger.error(f"Error importing record {i}: {e}")
                        elif error_count == 6:
                            logger.info("Suppressing further error messages...")
            
            logger.info(f"Import complete: {success_count} records imported successfully, {error_count} errors")
            
            # Add mtree index if configured
            if surreal_config["create_mtree_index"]:
                logger.info("Adding mtree index on embedding field")
                try:
                    await db.query(f"""
                    DEFINE INDEX IF NOT EXISTS embedding_mtree 
                    ON TABLE {surreal_config['table']} 
                    FIELDS embedding 
                    MTREE DIMENSION {surreal_config['mtree_dimension']} 
                    DIST {surreal_config['mtree_distance']} 
                    TYPE {surreal_config['mtree_type']};
                    """)
                    logger.info("mtree index added successfully")
                except Exception as e:
                    logger.warning(f"Error adding mtree index, but continuing: {str(e)}")
            
            # Close the connection
            logger.info("Closing connection to SurrealDB")
            await db.close()
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error in SurrealDB operations: {str(e)}")
            # Try to close the connection if it's open
            try:
                await db.close()
            except:
                pass
            return False
            
    except ImportError:
        logger.error("surrealdb package not installed. SurrealDB storage disabled.")
        return False
    except Exception as e:
        logger.error(f"Error storing chunks in SurrealDB: {str(e)}")
        return False

# ============= Main Function =============

def main():
    """Main function to process PDFs"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process PDFs into chunks")
    
    # Input source arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdf-dir", type=str, help="Directory containing PDFs to process")
    input_group.add_argument("--doi", type=str, help="DOI to process")
    input_group.add_argument("--url", type=str, help="URL to process")
    input_group.add_argument("--query", type=str, help="Search query to process")
    
    # Search parameters
    search_group = parser.add_argument_group("Search parameters")
    search_group.add_argument("--max-results", type=int, default=5, 
                             help="Maximum number of results to process")
    search_group.add_argument("--sources", type=str, 
                             help="Comma-separated list of sources to search (arxiv, pubmed, semantic_scholar, crossref)")
    search_group.add_argument("--no-filter-by-pdf", action="store_true", 
                             help="Don't filter out papers without PDF URLs")
    search_group.add_argument("--no-enrich-with-unpaywall", action="store_true", 
                             help="Don't enrich papers with PDF URLs from Unpaywall")
    search_group.add_argument("--sort-by", type=str, choices=["relevance", "date"], default="relevance", 
                             help="How to sort the results (relevance, date)")
    
    # Output arguments
    parser.add_argument("--output", type=str, help="Output path for processed chunks")
    parser.add_argument("--output-format", type=str, choices=["jsonl", "json", "csv"], 
                        help="Output format (default: jsonl)")
    
    # Feature flags
    parser.add_argument("--no-haiku", action="store_true", help="Disable Haiku context generation")
    parser.add_argument("--no-embedding", action="store_true", help="Disable embedding generation")
    parser.add_argument("--surreal", action="store_true", help="Store chunks in SurrealDB")
    
    # Chunking parameters
    chunking_group = parser.add_argument_group("Chunking parameters")
    chunking_group.add_argument("--chunk-target", type=int, help=f"Target chunk size in tokens (default: {CHUNKING['target_tokens']})")
    chunking_group.add_argument("--chunk-min", type=int, help=f"Minimum chunk size in tokens (default: {CHUNKING['min_tokens']})")
    chunking_group.add_argument("--chunk-max", type=int, help=f"Maximum chunk size in tokens (default: {CHUNKING['max_tokens']})")
    chunking_group.add_argument("--chunk-overlap", type=int, help=f"Chunk overlap in tokens (default: {CHUNKING['overlap_tokens']})")
    chunking_group.add_argument("--no-preserve-sentences", action="store_true", 
                               help="Don't preserve sentence boundaries when chunking")
    chunking_group.add_argument("--preserve-paragraphs", action="store_true", 
                               help="Preserve paragraph boundaries when chunking")
    chunking_group.add_argument("--no-section-aware", action="store_true", 
                               help="Don't be aware of document sections when chunking")
    
    # Context generation parameters
    context_group = parser.add_argument_group("Context generation parameters")
    context_group.add_argument("--context-target", type=int, 
                              help=f"Target context size in tokens (default: {CONTEXT['target_tokens']})")
    context_group.add_argument("--context-max", type=int, 
                              help=f"Maximum context size in tokens (default: {CONTEXT['max_tokens']})")
    context_group.add_argument("--context-temperature", type=float, 
                              help=f"Temperature for context generation (default: {CONTEXT['temperature']})")
    context_group.add_argument("--context-model", type=str, 
                              help=f"Model for context generation (default: {CONTEXT['model']})")
    context_group.add_argument("--no-include-title", action="store_true", 
                              help="Don't include title in context")
    context_group.add_argument("--no-include-abstract", action="store_true", 
                              help="Don't include abstract in context")
    context_group.add_argument("--no-include-section-heading", action="store_true", 
                              help="Don't include section heading in context")
    
    # Embedding parameters
    embedding_group = parser.add_argument_group("Embedding parameters")
    embedding_group.add_argument("--embedding-model", type=str, 
                                help=f"Model for embeddings (default: {EMBEDDING['model']})")
    embedding_group.add_argument("--no-include-context-in-embedding", action="store_true", 
                                help="Don't include context in embedding input")
    embedding_group.add_argument("--normalize-embeddings", action="store_true", 
                                help="Normalize embeddings")
    
    # ScipdfParser parameters
    scipdf_group = parser.add_argument_group("ScipdfParser parameters")
    scipdf_group.add_argument("--no-extract-references", action="store_true", 
                             help="Don't extract references")
    scipdf_group.add_argument("--extract-figures", action="store_true", 
                             help="Extract figures")
    scipdf_group.add_argument("--extract-tables", action="store_true", 
                             help="Extract tables")
    scipdf_group.add_argument("--extract-equations", action="store_true", 
                             help="Extract equations")
    scipdf_group.add_argument("--no-extract-metadata", action="store_true", 
                             help="Don't extract metadata")
    scipdf_group.add_argument("--no-extract-sections", action="store_true", 
                             help="Don't extract sections")
    scipdf_group.add_argument("--no-extract-abstract", action="store_true", 
                             help="Don't extract abstract")
    scipdf_group.add_argument("--no-extract-authors", action="store_true", 
                             help="Don't extract authors")
    scipdf_group.add_argument("--no-extract-doi", action="store_true", 
                             help="Don't extract DOI")
    
    # Section handling parameters
    section_group = parser.add_argument_group("Section handling parameters")
    section_group.add_argument("--whole-sections", action="store_true",
                             help="Return whole sections instead of chunking")
    section_group.add_argument("--section-filters", type=str,
                             help="Comma-separated list of section names to include (empty = all)")
    section_group.add_argument("--exclude-sections", type=str,
                             help="Comma-separated list of section names to exclude")
    
    # Output parameters
    output_group = parser.add_argument_group("Output parameters")
    output_group.add_argument("--no-include-embedding-in-output", action="store_true", 
                             help="Don't include embeddings in output")
    output_group.add_argument("--no-include-metadata", action="store_true", 
                             help="Don't include metadata in output")
    output_group.add_argument("--no-include-context-in-output", action="store_true", 
                             help="Don't include context in output")
    output_group.add_argument("--no-include-chunk-text", action="store_true", 
                             help="Don't include chunk text in output")
    output_group.add_argument("--no-include-section-info", action="store_true", 
                             help="Don't include section info in output")
    output_group.add_argument("--no-include-uuid", action="store_true", 
                             help="Don't include UUID in output")
    output_group.add_argument("--no-include-group", action="store_true", 
                             help="Don't include group in output")
    output_group.add_argument("--no-include-url", action="store_true", 
                             help="Don't include URL in output")
    output_group.add_argument("--no-include-timestamp", action="store_true", 
                             help="Don't include timestamp in output")
    output_group.add_argument("--no-include-doi", action="store_true", 
                             help="Don't include DOI in output")
    output_group.add_argument("--no-include-title-in-output", action="store_true", 
                             help="Don't include title in output")
    output_group.add_argument("--include-temp-path", action="store_true", 
                             help="Include temporary file path in output")
    output_group.add_argument("--fields", type=str, 
                             help="Comma-separated list of fields to include in output (overrides other include flags)")
    output_group.add_argument("--group", type=str, 
                             help=f"Group name (default: {DEFAULT_GROUP})")
    
    # SurrealDB parameters
    surreal_group = parser.add_argument_group("SurrealDB parameters")
    surreal_group.add_argument("--surreal-url", type=str, 
                              help=f"SurrealDB URL (default: {SURREAL['url']})")
    surreal_group.add_argument("--surreal-namespace", type=str, 
                              help=f"SurrealDB namespace (default: {SURREAL['namespace']})")
    surreal_group.add_argument("--surreal-database", type=str, 
                              help=f"SurrealDB database (default: {SURREAL['database']})")
    surreal_group.add_argument("--surreal-table", type=str, 
                              help=f"SurrealDB table (default: {SURREAL['table']})")
    surreal_group.add_argument("--surreal-username", type=str, 
                              help="SurrealDB username")
    surreal_group.add_argument("--surreal-password", type=str, 
                              help="SurrealDB password")
    surreal_group.add_argument("--no-create-mtree-index", action="store_true", 
                              help="Don't create mtree index in SurrealDB")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting PDF processing pipeline")
    
    # Override configuration with command line arguments
    
    # Chunking parameters
    chunking_config = CHUNKING.copy()
    if args.chunk_target is not None:
        chunking_config["target_tokens"] = args.chunk_target
    if args.chunk_min is not None:
        chunking_config["min_tokens"] = args.chunk_min
    if args.chunk_max is not None:
        chunking_config["max_tokens"] = args.chunk_max
    if args.chunk_overlap is not None:
        chunking_config["overlap_tokens"] = args.chunk_overlap
    if args.no_preserve_sentences:
        chunking_config["preserve_sentences"] = False
    if args.preserve_paragraphs:
        chunking_config["preserve_paragraphs"] = True
    if args.no_section_aware:
        chunking_config["section_aware"] = False
    
    # Context parameters
    context_config = CONTEXT.copy()
    context_config["enabled"] = not args.no_haiku and CONTEXT["enabled"]
    if args.context_target is not None:
        context_config["target_tokens"] = args.context_target
    if args.context_max is not None:
        context_config["max_tokens"] = args.context_max
    if args.context_temperature is not None:
        context_config["temperature"] = args.context_temperature
    if args.context_model is not None:
        context_config["model"] = args.context_model
    if args.no_include_title:
        context_config["include_title"] = False
    if args.no_include_abstract:
        context_config["include_abstract"] = False
    if args.no_include_section_heading:
        context_config["include_section_heading"] = False
    
    # Embedding parameters
    embedding_config = EMBEDDING.copy()
    embedding_config["enabled"] = not args.no_embedding and EMBEDDING["enabled"]
    if args.embedding_model is not None:
        embedding_config["model"] = args.embedding_model
    if args.no_include_context_in_embedding:
        embedding_config["include_context"] = False
    if args.normalize_embeddings:
        embedding_config["normalize"] = True
    
    # ScipdfParser parameters
    scipdf_config = SCIPDF.copy()
    if args.no_extract_references:
        scipdf_config["extract_references"] = False
    if args.extract_figures:
        scipdf_config["extract_figures"] = True
    if args.extract_tables:
        scipdf_config["extract_tables"] = True
    if args.extract_equations:
        scipdf_config["extract_equations"] = True
    if args.no_extract_metadata:
        scipdf_config["extract_metadata"] = False
    if args.no_extract_sections:
        scipdf_config["extract_sections"] = False
    if args.no_extract_abstract:
        scipdf_config["extract_abstract"] = False
    if args.no_extract_authors:
        scipdf_config["extract_authors"] = False
    if args.no_extract_doi:
        scipdf_config["extract_doi"] = False
    
    # Section handling parameters
    if args.whole_sections:
        scipdf_config["whole_sections"] = True
        logger.info("Using whole sections instead of chunking")
    
    if args.section_filters:
        section_filters = [s.strip() for s in args.section_filters.split(',')]
        scipdf_config["section_filters"] = section_filters
        logger.info(f"Filtering sections to include only: {', '.join(section_filters)}")
    
    if args.exclude_sections:
        exclude_sections = [s.strip() for s in args.exclude_sections.split(',')]
        scipdf_config["exclude_sections"] = exclude_sections
        logger.info(f"Excluding sections: {', '.join(exclude_sections)}")
    
    # Output parameters
    output_config = OUTPUT.copy()
    if args.output_format is not None:
        output_config["format"] = args.output_format
    if args.no_include_embedding_in_output:
        output_config["include_embedding"] = False
    if args.no_include_metadata:
        output_config["include_metadata"] = False
    if args.no_include_context_in_output:
        output_config["include_context"] = False
    if args.no_include_chunk_text:
        output_config["include_chunk_text"] = False
    if args.no_include_section_info:
        output_config["include_section_info"] = False
    if args.no_include_uuid:
        output_config["include_uuid"] = False
    if args.no_include_group:
        output_config["include_group"] = False
    if args.no_include_url:
        output_config["include_url"] = False
    if args.no_include_timestamp:
        output_config["include_timestamp"] = False
    if args.no_include_doi:
        output_config["include_doi"] = False
    if args.no_include_title_in_output:
        output_config["include_title"] = False
    if args.include_temp_path:
        output_config["include_temp_path"] = True
    
    # Process fields argument if provided
    if args.fields:
        # Split the comma-separated list of fields
        fields = [field.strip() for field in args.fields.split(',')]
        output_config["fields"] = fields
    
    # SurrealDB parameters
    surreal_config = SURREAL.copy()
    surreal_config["enabled"] = args.surreal or SURREAL["enabled"]
    if args.surreal_url is not None:
        surreal_config["url"] = args.surreal_url
    if args.surreal_namespace is not None:
        surreal_config["namespace"] = args.surreal_namespace
    if args.surreal_database is not None:
        surreal_config["database"] = args.surreal_database
    if args.surreal_table is not None:
        surreal_config["table"] = args.surreal_table
    if args.surreal_username is not None and args.surreal_password is not None:
        surreal_config["credentials"] = {
            "username": args.surreal_username,
            "password": args.surreal_password
        }
    if args.no_create_mtree_index:
        surreal_config["create_mtree_index"] = False
    
    # Group
    group = args.group if args.group is not None else DEFAULT_GROUP
    
    # Output path
    output_path = Path(args.output) if args.output else CHUNKS_OUTPUT_PATH
    
    # Initialize chunker with updated configuration
    chunker = ContextualChunker(
        enable_haiku=context_config["enabled"],
        enable_embedding=embedding_config["enabled"],
        target_chunk_tokens=chunking_config["target_tokens"],
        target_context_tokens=context_config["target_tokens"],
        save_path=output_path,
        haiku_model=context_config["model"],
        embedding_model=embedding_config["model"],
        chunking_config=chunking_config,
        context_config=context_config,
        embedding_config=embedding_config,
        output_config=output_config,
        group=group
    )
    
    # Process PDFs based on input type
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        logger.info(f"Processing PDFs from directory: {pdf_dir}")
        success_count = process_pdfs_from_directory(pdf_dir, chunker, PARSED_OUTPUT_DIR, scipdf_config)
        logger.info(f"Successfully processed {success_count} PDFs")
    
    elif args.doi:
        logger.info(f"Processing PDF from DOI: {args.doi}")
        success = process_pdf_from_doi(args.doi, chunker, PARSED_OUTPUT_DIR, scipdf_config)
        logger.info(f"DOI processing {'successful' if success else 'failed'}")
    
    elif args.url:
        logger.info(f"Processing PDF from URL: {args.url}")
        success = process_pdf_from_url(args.url, chunker, PARSED_OUTPUT_DIR, scipdf_config)
        logger.info(f"URL processing {'successful' if success else 'failed'}")
    
    elif args.query:
        logger.info(f"Processing PDFs from query: {args.query}")
        
        # Process search parameters
        max_results = args.max_results
        sources = [s.strip() for s in args.sources.split(',')] if args.sources else None
        filter_by_pdf = not args.no_filter_by_pdf
        enrich_with_unpaywall = not args.no_enrich_with_unpaywall
        sort_by = args.sort_by
        
        success_count = process_pdfs_from_query(
            query=args.query,
            chunker=chunker,
            parsed_output_dir=PARSED_OUTPUT_DIR,
            scipdf_config=scipdf_config,
            max_results=max_results,
            sources=sources,
            filter_by_pdf=filter_by_pdf,
            enrich_with_unpaywall=enrich_with_unpaywall,
            sort_by=sort_by
        )
        
        logger.info(f"Successfully processed {success_count} PDFs from query")
    
    # Store in SurrealDB if requested
    if surreal_config["enabled"]:
        import asyncio
        asyncio.run(store_in_surrealdb(output_path, surreal_config))
    
    logger.info("Processing complete")
    logger.info(f"Total chunks generated: {len(chunker.chunks)}")
    logger.info(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()