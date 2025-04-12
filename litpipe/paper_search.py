"""
Paper Search

This module provides functions to search for papers using various APIs.
"""

import os
import requests
import logging
import json
from typing import List, Dict, Any, Optional
import time

def search_arxiv(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of paper metadata dictionaries
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Construct the URL
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        # Make the request
        logger.info(f"Searching arXiv for: {query}")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse the response
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        # Extract the papers
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            # Extract the metadata
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            published = entry.find("{http://www.w3.org/2005/Atom}published").text.strip()
            
            # Extract authors
            authors = []
            for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
                name = author.find("{http://www.w3.org/2005/Atom}name").text.strip()
                authors.append(name)
            
            # Extract links
            links = {}
            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                rel = link.get("rel")
                href = link.get("href")
                title = link.get("title")
                
                if rel == "alternate":
                    links["html"] = href
                elif rel == "related" and title == "pdf":
                    # Fix arXiv PDF links
                    if "arxiv.org/pdf/" in href:
                        links["pdf"] = href.replace("http://", "https://")
                    else:
                        links["pdf"] = href
            
            # Extract DOI if available
            doi = None
            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                href = link.get("href", "")
                if "doi.org" in href:
                    doi = href.split("doi.org/")[1]
            
            # Extract arXiv ID
            id_url = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
            arxiv_id = id_url.split("/abs/")[1]
            
            # Create the paper dictionary
            paper = {
                "title": title,
                "authors": authors,
                "abstract": summary,
                "publication_date": published,
                "doi": doi,
                "arxiv_id": arxiv_id,
                "pdf_available": "pdf" in links,
                "pdf_url": links.get("pdf"),
                "source": "arxiv",
                "source_url": links.get("html")
            }
            
            papers.append(paper)
        
        return papers
    
    except Exception as e:
        logger.error(f"Error searching arXiv for {query}: {str(e)}")
        return []

def search_pubmed(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search PubMed for papers
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of paper metadata dictionaries
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Construct the URL for search
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance"
        }
        
        # Make the search request
        logger.info(f"Searching PubMed for: {query}")
        search_response = requests.get(base_url, params=search_params, timeout=30)
        search_response.raise_for_status()
        
        # Parse the search response
        search_data = search_response.json()
        
        # Extract the PMIDs
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        
        # Construct the URL for fetching details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json"
        }
        
        # Make the fetch request
        logger.info(f"Fetching details for {len(pmids)} PubMed articles")
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
        fetch_response.raise_for_status()
        
        # Parse the fetch response
        fetch_data = fetch_response.json()
        
        # Extract the papers
        papers = []
        for pmid in pmids:
            if pmid not in fetch_data.get("result", {}):
                continue
                
            article = fetch_data["result"][pmid]
            
            # Extract the metadata
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            
            # Extract authors
            authors = []
            for author in article.get("authors", []):
                name = author.get("name", "")
                if name:
                    authors.append(name)
            
            # Extract publication date
            pub_date = article.get("pubdate", "")
            
            # Extract DOI
            doi = None
            for id_obj in article.get("articleids", []):
                if id_obj.get("idtype") == "doi":
                    doi = id_obj.get("value")
            
            # Create the paper dictionary
            paper = {
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "publication_date": pub_date,
                "doi": doi,
                "pmid": pmid,
                "pdf_available": False,  # PubMed doesn't provide PDF links directly
                "pdf_url": None,
                "source": "pubmed",
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            
            papers.append(paper)
        
        return papers
    
    except Exception as e:
        logger.error(f"Error searching PubMed for {query}: {str(e)}")
        return []

def search_semantic_scholar(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar for papers
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of paper metadata dictionaries
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Semantic Scholar API endpoint
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        # Parameters
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,url,year,venue,publicationDate,openAccessPdf,doi"
        }
        
        # Headers
        headers = {
            "Accept": "application/json"
        }
        
        # API key (optional)
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key
        
        # Execute search
        logger.info(f"Searching Semantic Scholar for: {query}")
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Convert results to dictionaries
        papers = []
        for result in data.get("data", []):
            paper = {
                "title": result.get("title", ""),
                "authors": [author.get("name", "") for author in result.get("authors", [])],
                "abstract": result.get("abstract", ""),
                "publication_date": result.get("publicationDate", ""),
                "doi": result.get("doi"),
                "pdf_available": bool(result.get("openAccessPdf", {}).get("url")),
                "pdf_url": result.get("openAccessPdf", {}).get("url"),
                "source": "semantic_scholar",
                "source_url": result.get("url")
            }
            papers.append(paper)
        
        return papers
    
    except Exception as e:
        logger.error(f"Error searching Semantic Scholar for {query}: {str(e)}")
        return []

def search_crossref(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search Crossref for papers
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of paper metadata dictionaries
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Crossref API endpoint
        url = "https://api.crossref.org/works"
        
        # Parameters
        params = {
            "query": query,
            "rows": max_results,
            "sort": "relevance",
            "select": "DOI,title,author,abstract,URL,published-online,link"
        }
        
        # Headers
        headers = {
            "Accept": "application/json"
        }
        
        # Execute search
        logger.info(f"Searching Crossref for: {query}")
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Convert results to dictionaries
        papers = []
        for result in data.get("message", {}).get("items", []):
            # Find PDF URL if available
            pdf_url = None
            for link in result.get("link", []):
                if link.get("content-type") == "application/pdf":
                    pdf_url = link.get("URL")
                    break
            
            paper = {
                "title": " ".join(result.get("title", [])),
                "authors": [f"{author.get('given', '')} {author.get('family', '')}" for author in result.get("author", [])],
                "abstract": result.get("abstract", ""),
                "publication_date": result.get("published-online", {}).get("date-time"),
                "doi": result.get("DOI"),
                "pdf_available": bool(pdf_url),
                "pdf_url": pdf_url,
                "source": "crossref",
                "source_url": result.get("URL")
            }
            papers.append(paper)
        
        return papers
    
    except Exception as e:
        logger.error(f"Error searching Crossref for {query}: {str(e)}")
        return []

def search_unpaywall(doi: str) -> Optional[Dict[str, Any]]:
    """
    Search for a paper on Unpaywall by DOI
    
    Args:
        doi: DOI of the paper
        
    Returns:
        Paper metadata dictionary, or None if not found
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Clean the DOI - remove any URL prefixes
        if doi.startswith("http"):
            doi = doi.split("doi.org/")[-1]
        
        # Unpaywall API endpoint
        url = f"https://api.unpaywall.org/v2/{doi}"
        
        # Parameters
        params = {
            "email": "superswagviking@gmail.com"  # Replace with your email
        }
        
        # Execute search
        logger.info(f"Searching Unpaywall for DOI: {doi}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Find best open access URL
        pdf_url = None
        if data.get("is_oa"):
            # First try to get the URL from best_oa_location
            if data.get("best_oa_location", {}).get("url_for_pdf"):
                pdf_url = data.get("best_oa_location", {}).get("url_for_pdf")
                logger.info(f"Found PDF URL from best_oa_location: {pdf_url}")
            
            # If not found, try all locations
            if not pdf_url:
                for location in data.get("oa_locations", []):
                    if location.get("url_for_pdf"):
                        pdf_url = location.get("url_for_pdf")
                        logger.info(f"Found PDF URL from oa_locations: {pdf_url}")
                        break
        
        # If still no PDF URL but we have a URL, try that
        if not pdf_url and data.get("best_oa_location", {}).get("url"):
            pdf_url = data.get("best_oa_location", {}).get("url")
            logger.info(f"Using non-PDF URL as fallback: {pdf_url}")
        
        logger.info(f"Final PDF URL for DOI {doi}: {pdf_url}")
        
        return {
            "pdf_url": pdf_url,
            "pdf_available": bool(pdf_url),
            "title": data.get("title"),
            "doi": doi,
            "source": "unpaywall"
        }
    
    except Exception as e:
        logger.error(f"Error searching Unpaywall for DOI {doi}: {str(e)}")
        logger.exception(e)
        return None

def enrich_papers_with_pdf_urls(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich papers with PDF URLs from Unpaywall
    
    Args:
        papers: List of paper metadata dictionaries
        
    Returns:
        Enriched list of paper metadata dictionaries
    """
    logger = logging.getLogger(__name__)
    
    enriched_papers = []
    for paper in papers:
        # Skip if already has PDF URL
        if paper.get("pdf_url"):
            enriched_papers.append(paper)
            continue
        
        # Skip if no DOI
        if not paper.get("doi"):
            enriched_papers.append(paper)
            continue
        
        # Search Unpaywall
        logger.info(f"Enriching paper with DOI {paper['doi']} with PDF URL")
        unpaywall_data = search_unpaywall(paper["doi"])
        if unpaywall_data and unpaywall_data.get("pdf_url"):
            paper["pdf_url"] = unpaywall_data["pdf_url"]
            paper["pdf_available"] = True
        
        enriched_papers.append(paper)
        
        # Rate limiting
        time.sleep(1)
    
    return enriched_papers

def search_papers(query: str, max_results: int = 10, sources: List[str] = None, 
                 filter_by_pdf: bool = True, enrich_with_unpaywall: bool = True,
                 sort_by: str = "relevance") -> List[Dict[str, Any]]:
    """
    Search for papers using multiple sources
    
    Args:
        query: Search query
        max_results: Maximum number of results to return per source
        sources: List of sources to search (default: ["arxiv", "pubmed", "semantic_scholar", "crossref"])
        filter_by_pdf: Whether to filter out papers without PDF URLs
        enrich_with_unpaywall: Whether to enrich papers with PDF URLs from Unpaywall
        sort_by: How to sort the results (relevance, date)
        
    Returns:
        List of paper metadata dictionaries
    """
    logger = logging.getLogger(__name__)
    
    # Default sources
    if sources is None:
        sources = ["arxiv", "pubmed", "semantic_scholar", "crossref"]
    
    # Initialize results
    all_papers = []
    
    # Search each source
    for source in sources:
        try:
            if source == "arxiv":
                papers = search_arxiv(query, max_results)
            elif source == "pubmed":
                papers = search_pubmed(query, max_results)
            elif source == "semantic_scholar":
                papers = search_semantic_scholar(query, max_results)
            elif source == "crossref":
                papers = search_crossref(query, max_results)
            else:
                logger.warning(f"Unknown source: {source}")
                continue
            
            all_papers.extend(papers)
            
            # Add a small delay between API calls
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error searching {source} for {query}: {str(e)}")
    
    # Enrich papers with PDF URLs from Unpaywall
    if enrich_with_unpaywall:
        all_papers = enrich_papers_with_pdf_urls(all_papers)
    
    # Filter out papers without PDF URLs
    if filter_by_pdf:
        all_papers = [paper for paper in all_papers if paper.get("pdf_available")]
    
    # Sort papers
    if sort_by == "date":
        # Sort by publication date (newest first)
        all_papers.sort(key=lambda p: p.get("publication_date", ""), reverse=True)
    # Default is relevance, which is the order from the APIs
    
    # Limit to max_results
    return all_papers[:max_results]

if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python paper_search.py <query> [max_results]")
        sys.exit(1)
    
    query = sys.argv[1]
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Search for papers
    papers = search_papers(query, max_results)
    
    # Print the results
    print(f"Found {len(papers)} papers for query: {query}")
    for i, paper in enumerate(papers):
        print(f"\n{i+1}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Source: {paper['source']}")
        print(f"   URL: {paper['source_url']}")
        if paper['pdf_available']:
            print(f"   PDF: {paper['pdf_url']}")
        if paper['doi']:
            print(f"   DOI: {paper['doi']}")
        print(f"   Abstract: {paper['abstract'][:200]}...")