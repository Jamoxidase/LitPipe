"""
DOI Resolver

This module provides functions to resolve DOIs to PDF URLs.
"""

import requests
import logging
from typing import Optional, Dict, Any
import re
import time

def resolve_doi_to_metadata(doi: str) -> Optional[Dict[str, Any]]:
    """
    Resolve a DOI to metadata using the Crossref API
    
    Args:
        doi: DOI to resolve
        
    Returns:
        Dictionary of metadata, or None if resolution failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Clean the DOI
        doi = doi.strip()
        
        # Construct the URL
        url = f"https://api.crossref.org/works/{doi}"
        
        # Make the request
        logger.info(f"Resolving DOI: {doi}")
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        
        # Extract the metadata
        if "message" in data:
            return data["message"]
        
        return None
    
    except Exception as e:
        logger.error(f"Error resolving DOI {doi}: {str(e)}")
        return None

def resolve_doi_to_pdf_url(doi: str) -> Optional[str]:
    """
    Resolve a DOI to a PDF URL
    
    Args:
        doi: DOI to resolve
        
    Returns:
        PDF URL, or None if resolution failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # First try Unpaywall
        pdf_url = _resolve_doi_with_unpaywall(doi)
        if pdf_url:
            return pdf_url
        
        # Then try SciHub (if available)
        pdf_url = _resolve_doi_with_scihub(doi)
        if pdf_url:
            return pdf_url
        
        # Finally, try direct DOI resolution
        pdf_url = _resolve_doi_directly(doi)
        if pdf_url:
            return pdf_url
        
        return None
    
    except Exception as e:
        logger.error(f"Error resolving DOI {doi} to PDF URL: {str(e)}")
        return None

def _resolve_doi_with_unpaywall(doi: str) -> Optional[str]:
    """
    Resolve a DOI to a PDF URL using Unpaywall
    
    Args:
        doi: DOI to resolve
        
    Returns:
        PDF URL, or None if resolution failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Construct the URL
        url = f"https://api.unpaywall.org/v2/{doi}?email=litpipe@example.com"
        
        # Make the request
        logger.info(f"Resolving DOI with Unpaywall: {doi}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        
        # Check if there's a best OA location
        if data.get("best_oa_location") and data["best_oa_location"].get("url_for_pdf"):
            return data["best_oa_location"]["url_for_pdf"]
        
        # Check all OA locations
        for location in data.get("oa_locations", []):
            if location.get("url_for_pdf"):
                return location["url_for_pdf"]
        
        return None
    
    except Exception as e:
        logger.error(f"Error resolving DOI {doi} with Unpaywall: {str(e)}")
        return None

def _resolve_doi_with_scihub(doi: str) -> Optional[str]:
    """
    Resolve a DOI to a PDF URL using Sci-Hub
    
    Args:
        doi: DOI to resolve
        
    Returns:
        PDF URL, or None if resolution failed
    """
    # Note: This is a placeholder. Actual implementation would depend on
    # the current Sci-Hub domain and may raise ethical/legal questions.
    return None

def _resolve_doi_directly(doi: str) -> Optional[str]:
    """
    Resolve a DOI directly to a PDF URL
    
    Args:
        doi: DOI to resolve
        
    Returns:
        PDF URL, or None if resolution failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Construct the URL
        url = f"https://doi.org/{doi}"
        
        # Make the request
        logger.info(f"Resolving DOI directly: {doi}")
        response = requests.get(url, allow_redirects=True, timeout=30)
        
        # Check if the final URL is a PDF
        if response.url.lower().endswith(".pdf"):
            return response.url
        
        # Check if there's a PDF link in the page
        pdf_links = re.findall(r'href="([^"]+\.pdf)"', response.text, re.IGNORECASE)
        if pdf_links:
            # Convert relative URLs to absolute
            from urllib.parse import urljoin
            return urljoin(response.url, pdf_links[0])
        
        return None
    
    except Exception as e:
        logger.error(f"Error resolving DOI {doi} directly: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python doi_resolver.py <doi>")
        sys.exit(1)
    
    doi = sys.argv[1]
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Resolve the DOI
    pdf_url = resolve_doi_to_pdf_url(doi)
    
    if pdf_url:
        print(f"PDF URL: {pdf_url}")
    else:
        print(f"Failed to resolve DOI {doi} to PDF URL")
        
        # Try to get metadata
        metadata = resolve_doi_to_metadata(doi)
        if metadata:
            print("Metadata:")
            print(f"  Title: {metadata.get('title', [''])[0]}")
            print(f"  DOI: {metadata.get('DOI', '')}")
            print(f"  URL: {metadata.get('URL', '')}")
            print(f"  Publisher: {metadata.get('publisher', '')}")
            
            # Print authors
            authors = metadata.get("author", [])
            if authors:
                print("  Authors:")
                for author in authors[:5]:  # Limit to first 5 authors
                    given = author.get("given", "")
                    family = author.get("family", "")
                    print(f"    {given} {family}")
                if len(authors) > 5:
                    print(f"    ... and {len(authors) - 5} more")