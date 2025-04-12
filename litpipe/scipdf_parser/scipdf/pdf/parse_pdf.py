import re
import os
import os.path as op
from glob import glob
import urllib
from typing import Dict 
import subprocess
import requests
import fitz  
from PIL import Image
import io
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm, tqdm_notebook


GROBID_URL = "https://kermitt2-grobid.hf.space"
DIR_PATH = op.dirname(op.abspath(__file__))
PDF_FIGURES_JAR_PATH = op.join(
    DIR_PATH, "pdffigures2", "pdffigures2-assembly-0.0.12-SNAPSHOT.jar"
)


def list_pdf_paths(pdf_folder: str):
    """
    list of pdf paths in pdf folder
    """
    return glob(op.join(pdf_folder, "*", "*", "*.pdf"))


def validate_url(path: str):
    """
    Validate a given ``path`` if it is URL or not
    """
    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(regex, path) is not None


def parse_pdf(
    pdf_path,  # Can be str, bytes, or pathlib.Path
    fulltext: bool = True,
    soup: bool = False,
    return_coordinates: bool = False,
    grobid_url: str = GROBID_URL,
):
    """
    Function to parse PDF to XML or BeautifulSoup using GROBID tool

    You can see http://grobid.readthedocs.io/en/latest/Install-Grobid/ on how to run GROBID locally
    After loading GROBID zip file, you can run GROBID by using the following
    >> ./gradlew run

    Parameters
    ==========
    pdf_path: str, bytes, or pathlib.Path, path or URL to publication or article or bytes string of PDF
    fulltext: bool, option for parsing, if True, parse full text of the article
        if False, parse only header
    grobid_url: str, url to GROBID parser, default at 'http://localhost:8070'
        This could be changed to "https://cloud.science-miner.com/grobid/" for the cloud service
    soup: bool, if True, return BeautifulSoup of the article

    Output
    ======
    parsed_article: if soup is False, return parsed XML in text format,
        else return BeautifulSoup of the XML
    Example
    =======
    >> parsed_article = parse_pdf(pdf_path, fulltext=True, soup=True)
    """
    import logging
    import pathlib
    logger = logging.getLogger(__name__)
    

    logger.info("Starting PDF parsing...")
    try:
        # GROBID URL
        if fulltext:
            url = "%s/api/processFulltextDocument" % grobid_url
        else:
            url = "%s/api/processHeaderDocument" % grobid_url
        
        logger.info(f"Using GROBID endpoint: {url}")
        
        files = []
        if return_coordinates:
            files += [
                ("teiCoordinates", (None, "persName")),
                ("teiCoordinates", (None, "figure")),
                ("teiCoordinates", (None, "ref")),
                ("teiCoordinates", (None, "formula")),
                ("teiCoordinates", (None, "biblStruct")),
            ]

        # Convert pathlib.Path to string if needed
        if isinstance(pdf_path, pathlib.Path):
            logger.info(f"Converting pathlib.Path to string: {pdf_path}")
            pdf_path = str(pdf_path)

        if isinstance(pdf_path, str):
            if validate_url(pdf_path) and op.splitext(pdf_path)[-1].lower() != ".pdf":
                logger.error(f"The input URL has to end with .pdf: {pdf_path}")
                return None
            elif validate_url(pdf_path) and op.splitext(pdf_path)[-1] == ".pdf":
                logger.info(f"Downloading PDF from URL: {pdf_path}")
                try:
                    page = urllib.request.urlopen(pdf_path).read()
                    files += [("input", page)]
                    response = requests.post(url, files=files, timeout=60)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    parsed_article = response.text
                    logger.info(f"Successfully parsed PDF from URL: {pdf_path}")
                except Exception as e:
                    logger.error(f"Error downloading or parsing PDF from URL {pdf_path}: {str(e)}")
                    return None
            elif op.exists(pdf_path):
                logger.info(f"Reading PDF from file: {pdf_path}")
                try:
                    with open(pdf_path, "rb") as f:
                        files += [("input", f)]
                        response = requests.post(url, files=files, timeout=60)
                        response.raise_for_status()  # Raise an exception for HTTP errors
                        parsed_article = response.text
                        logger.info(f"Successfully parsed PDF file: {pdf_path}")
                except Exception as e:
                    logger.error(f"Error parsing PDF file {pdf_path}: {str(e)}")
                    return None
            else:
                logger.error(f"PDF file does not exist: {pdf_path}")
                return None
        elif isinstance(pdf_path, bytes):
            # assume that incoming is byte string
            logger.info("Processing PDF from bytes")
            try:
                files += [("input", pdf_path)]
                response = requests.post(url, files=files, timeout=60)
                response.raise_for_status()  # Raise an exception for HTTP errors
                parsed_article = response.text
                logger.info("Successfully parsed PDF from bytes")
            except Exception as e:
                logger.error(f"Error parsing PDF from bytes: {str(e)}")
                return None
        else:
            logger.error(f"Unsupported PDF path type: {type(pdf_path)}")
            return None

        if soup and parsed_article is not None:
            try:
                parsed_article = BeautifulSoup(parsed_article, "lxml")
                logger.info("Successfully converted XML to BeautifulSoup")
            except Exception as e:
                logger.error(f"Error converting XML to BeautifulSoup: {str(e)}")
                return None

        return parsed_article
        
    except Exception as e:
        logger.error(f"Unexpected error in parse_pdf: {str(e)}")
        logger.exception(e)  # Log the full stack trace
        return None


def parse_authors(article):
    """
    Parse authors from a given BeautifulSoup of an article
    """
    author_names = article.find("sourcedesc").findAll("persname")
    authors = []
    for author in author_names:
        firstname = author.find("forename", {"type": "first"})
        firstname = firstname.text.strip() if firstname is not None else ""
        middlename = author.find("forename", {"type": "middle"})
        middlename = middlename.text.strip() if middlename is not None else ""
        lastname = author.find("surname")
        lastname = lastname.text.strip() if lastname is not None else ""
        if middlename != "":
            authors.append(firstname + " " + middlename + " " + lastname)
        else:
            authors.append(firstname + " " + lastname)
    authors = "; ".join(authors)
    return authors


def parse_date(article):
    """
    Parse date from a given BeautifulSoup of an article
    """
    pub_date = article.find("publicationstmt")
    year = pub_date.find("date")
    year = year.attrs.get("when") if year is not None else ""
    return year


def parse_abstract(article):
    """
    Parse abstract from a given BeautifulSoup of an article
    """
    div = article.find("abstract")
    abstract = ""
    for p in list(div.children):
        if not isinstance(p, NavigableString) and len(list(p)) > 0:
            abstract += " ".join(
                [elem.text for elem in p if not isinstance(elem, NavigableString)]
            )
    return abstract


def find_references(div):
    """
    For a given section, find references made in the section for publications, figures, tables
    """
    publication_ref = [ref.attrs.get("target").strip("#") for ref in div.find_all("ref") if ref.attrs.get("type") == "bibr" and "target" in ref.attrs]
    figure_ref = [ref.attrs.get("target").strip("#") for ref in div.find_all("ref") if ref.attrs.get("type") == "figure" and "target" in ref.attrs]
    table_ref = [ref.attrs.get("target").strip("#") for ref in div.find_all("ref") if ref.attrs.get("type") == "table" and "target" in ref.attrs]
    return {"publication_ref": publication_ref, "figure_ref": figure_ref, "table_ref": table_ref}


def parse_sections(article, as_list: bool = False):
    """
    Parse list of sections from a given BeautifulSoup of an article

    Parameters
    ==========
    as_list: bool, if True, output text as a list of paragraph instead
        of joining it together as one single text
    """
    article_text = article.find("text")
    divs = article_text.find_all("div", attrs={"xmlns": "http://www.tei-c.org/ns/1.0"})
    sections = []
    for div in divs:
        div_list = list(div.children)
        if len(div_list) == 0:
            heading = ""
            text = ""
        elif len(div_list) == 1:
            if isinstance(div_list[0], NavigableString):
                heading = str(div_list[0])
                text = ""
            else:
                heading = ""
                text = div_list[0].text
        else:
            text = []
            heading = div_list[0]
            if isinstance(heading, NavigableString):
                heading = str(heading)
                p_all = list(div.children)[1:]
            else:
                heading = ""
                p_all = list(div.children)
            for p in p_all:
                if p is not None:
                    try:
                        text.append(p.text)
                    except:
                        pass
            if not as_list:
                text = "\n".join(text)

        if heading != "" or text != "":
            ref_dict = find_references(div)
            sections.append(
                {
                    "heading": heading,
                    "text": text,
                    "publication_ref": ref_dict["publication_ref"],
                    "figure_ref": ref_dict["figure_ref"],
                    "table_ref": ref_dict["table_ref"],
                }
            )
    return sections


def parse_references(article):
    """
    Parse list of references from a given BeautifulSoup of an article
    """
    reference_list = []
    references = article.find("text").find("div", attrs={"type": "references"})
    references = references.find_all("biblstruct") if references is not None else []
    reference_list = []
    for reference in references:
        ref_id = reference.get('xml:id', "")
        title = reference.find("title", attrs={"level": "a"})
        if title is None:
            title = reference.find("title", attrs={"level": "m"})
        title = title.text if title is not None else ""
        journal = reference.find("title", attrs={"level": "j"})
        journal = journal.text if journal is not None else ""
        if journal == "":
            journal = reference.find("publisher")
            journal = journal.text if journal is not None else ""
        year = reference.find("date")
        year = year.attrs.get("when") if year is not None else ""
        authors = []
        for author in reference.find_all("author"):
            firstname = author.find("forename", {"type": "first"})
            firstname = firstname.text.strip() if firstname is not None else ""
            middlename = author.find("forename", {"type": "middle"})
            middlename = middlename.text.strip() if middlename is not None else ""
            lastname = author.find("surname")
            lastname = lastname.text.strip() if lastname is not None else ""
            if middlename != "":
                authors.append(firstname + " " + middlename + " " + lastname)
            else:
                authors.append(firstname + " " + lastname)
        authors = "; ".join(authors)
        reference_list.append(
            {"ref_id": ref_id, "title": title, "journal": journal, "year": year, "authors": authors}
        )
    return reference_list


def parse_figure_caption(article):
    """
    Parse list of figures/tables from a given BeautifulSoup of an article
    """
    figures_list = []
    figures = article.find_all("figure")
    for figure in figures:
        figure_type = figure.attrs.get("type") or "figure"
        figure_id = figure.attrs.get("xml:id") or ""
        label = figure.find("label").text
        if figure_type == "table":
            caption = figure.find("figdesc").text
            data = figure.table.text
        else:
            caption = figure.text
            data = ""
        figures_list.append(
            {
                "figure_label": label,
                "figure_type": figure_type,
                "figure_id": figure_id,
                "figure_caption": caption,
                "figure_data": data,
            }
        )
    return figures_list


def parse_formulas(article):
    """
    Parse list of formulas from a given BeautifulSoup of an article
    """
    formulas_list = []
    formulas = article.find_all("formula")
    for formula in formulas:
        formula_id = formula.attrs["xml:id"] or ""
        formula_text = formula.text
        formula_coordinates = formula.attrs.get("coords") or ""
        if formula_coordinates != "":
            formula_coordinates = [float(x) for x in formula_coordinates.split(",")]
            formulas_list.append(
                {
                    "formula_id": formula_id,
                    "formula_text": formula_text,
                    "formula_coordinates": formula_coordinates,
                }
            )
    return formulas_list


def convert_article_soup_to_dict(article, as_list: bool = False):
    """
    Function to convert BeautifulSoup to JSON format
    similar to the output from https://github.com/allenai/science-parse/

    Parameters
    ==========
    article: BeautifulSoup

    Output
    ======
    article_json: dict, parsed dictionary of a given article in the following format
        {
            'title': ...,
            'abstract': ...,
            'sections': [
                {'heading': ..., 'text': ...},
                {'heading': ..., 'text': ...},
                ...
            ],
            'references': [
                {'title': ..., 'journal': ..., 'year': ..., 'authors': ...},
                {'title': ..., 'journal': ..., 'year': ..., 'authors': ...},
                ...
            ],
            'figures': [
                {'figure_label': ..., 'figure_type': ..., 'figure_id': ..., 'figure_caption': ..., 'figure_data': ...},
                ...
            ]
        }
    """
    article_dict = {}
    if article is not None:
        title = article.find("title", attrs={"type": "main"})
        title = title.text.strip() if title is not None else ""

        article_dict["title"] = title
        article_dict["authors"] = parse_authors(article)
        article_dict["pub_date"] = parse_date(article)
        article_dict["abstract"] = parse_abstract(article)
        article_dict["sections"] = parse_sections(article, as_list=as_list)
        article_dict["references"] = parse_references(article)
        article_dict["figures"] = parse_figure_caption(article)
        article_dict["formulas"] = parse_formulas(article)

        doi = article.find("idno", attrs={"type": "DOI"})
        doi = doi.text if doi is not None else ""
        article_dict["doi"] = doi

        return article_dict
    else:
        return None


def parse_pdf_to_dict(
    pdf_path: str,
    fulltext: bool = True,
    soup: bool = True,
    as_list: bool = False,
    return_coordinates: bool = True,
    grobid_url: str = GROBID_URL,
    parse_figures: bool = True,
):
    """
    Parse the given PDF and return dictionary of the parsed article

    Parameters
    ==========
    pdf_path: str, path to publication or article
    fulltext: bool, whether to extract fulltext or not
    soup: bool, whether to return BeautifulSoup or not
    as_list: bool, whether to return list of sections or not
    grobid_url: str, url to grobid server, default is `GROBID_URL`
        This could be changed to "https://kermitt2-grobid.hf.space" for the cloud service

    Ouput
    =====
    article_dict: dict, dictionary of an article
    """
    import logging
    import json
    import pprint
    logger = logging.getLogger(__name__)
    
    try:
        # Check if the PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file does not exist: {pdf_path}")
            return None
            
        # Log the GROBID URL being used
        logger.info(f"Using GROBID URL: {grobid_url}")
        
        # Parse the PDF
        parsed_article = parse_pdf(
            pdf_path,
            fulltext=fulltext,
            soup=soup,
            return_coordinates=return_coordinates,
            grobid_url=grobid_url,
        )
        
        # Check if parsing was successful
        if parsed_article is None:
            logger.error(f"Failed to parse PDF: {pdf_path}")
            return None
            
        # Convert to dictionary
        article_dict = convert_article_soup_to_dict(parsed_article, as_list=as_list)
        
        # Check if conversion was successful
        if article_dict is None:
            logger.error(f"Failed to convert parsed article to dictionary: {pdf_path}")
            return None
        
        # Log the structure and fields of the parsed data
        logger.info("=== PARSED DATA STRUCTURE ===")
        logger.info(f"Article has {len(article_dict.keys())} top-level fields")
        
        # Log the top-level fields
        logger.info("Top-level fields:")
        for key in article_dict.keys():
            field_type = type(article_dict[key]).__name__
            if isinstance(article_dict[key], list):
                field_len = len(article_dict[key])
                logger.info(f"  - {key}: {field_type} with {field_len} items")
            elif isinstance(article_dict[key], dict):
                field_len = len(article_dict[key].keys())
                logger.info(f"  - {key}: {field_type} with {field_len} keys")
            elif isinstance(article_dict[key], str) and len(article_dict[key]) > 100:
                logger.info(f"  - {key}: {field_type} with {len(article_dict[key])} characters")
            else:
                logger.info(f"  - {key}: {field_type} = {article_dict[key]}")
        
        # Log more details about sections if they exist
        if "sections" in article_dict and article_dict["sections"]:
            sections = article_dict["sections"]
            logger.info(f"Found {len(sections)} sections in the document")
            for i, section in enumerate(sections[:3]):  # Log only first 3 sections to avoid too much output
                if isinstance(section, dict):
                    logger.info(f"  Section {i+1}: {section.get('heading', 'No heading')} - {len(section.get('text', ''))} characters")
                else:
                    logger.info(f"  Section {i+1}: {type(section).__name__}")
            if len(sections) > 3:
                logger.info(f"  ... and {len(sections) - 3} more sections")
        
        # Log references if they exist
        if "references" in article_dict and article_dict["references"]:
            refs = article_dict["references"]
            logger.info(f"Found {len(refs)} references in the document")
            for i, ref in enumerate(refs[:3]):  # Log only first 3 references
                if isinstance(ref, dict):
                    logger.info(f"  Reference {i+1}: {ref.get('title', 'No title')}")
                else:
                    logger.info(f"  Reference {i+1}: {type(ref).__name__}")
            if len(refs) > 3:
                logger.info(f"  ... and {len(refs) - 3} more references")
                
        # Return the article dictionary
        return article_dict
        
    except Exception as e:
        logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
        logger.exception(e)  # Log the full stack trace
        return None


def parse_figures(
    pdf_folder: str,
    jar_path: str = PDF_FIGURES_JAR_PATH,
    resolution: int = 300,
    output_folder: str = "figures",
):
    """
    Parse figures from the given scientific PDF using pdffigures2

    Parameters
    ==========
    pdf_folder: str, path to a folder that contains PDF files. A folder must contains only PDF files
    jar_path: str, default path to pdffigures2-assembly-0.0.12-SNAPSHOT.jar file
    resolution: int, resolution of the output figures
    output_folder: str, path to folder that we want to save parsed data (related to figures) and figures

    Output
    ======
    folder: making a folder of output_folder/data and output_folder/figures of parsed data and figures relatively
    """
    if not op.isdir(output_folder):
        os.makedirs(output_folder)

    # create ``data`` and ``figures`` subfolder within ``output_folder``
    data_path = op.join(output_folder, "data")
    figure_path = op.join(output_folder, "figures")
    if not op.exists(data_path):
        os.makedirs(data_path)
    if not op.exists(figure_path):
        os.makedirs(figure_path)

    if op.isdir(data_path) and op.isdir(figure_path):
        args = [
            "java",
            "-jar",
            jar_path,
            pdf_folder,
            "-i",
            str(resolution),
            "-d",
            op.join(op.abspath(data_path), ""),
            "-m",
            op.join(op.abspath(figure_path), ""),  # end path with "/"
        ]
        _ = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20
        )
        print("Done parsing figures from PDFs!")
        print("Data Path: ", data_path)
        print("Figure Path: ", figure_path)
        print("PDF Folder: ", pdf_folder)   
    else:
        print(
            "You may have to check of ``data`` and ``figures`` in the the output folder path."
        )
