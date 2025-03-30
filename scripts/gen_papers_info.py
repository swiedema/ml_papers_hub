import os
import pathlib
import tempfile
import logging
from datetime import datetime
import concurrent.futures
from functools import partial
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm import tqdm
from src.arxiv import Arxiv
from src.firebase import (
    fetch_specific_attributes_from_collection,
    initialize_firebase_client,
    update_paper_in_firestore,
    update_all_papers_status,
)
from src.pdf_parser import create_pdf_thumbnail, compress_pdf


os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "0"
PROJECT_DIR = os.getenv("PROJECT_DIR")
if not PROJECT_DIR:
    raise ValueError("PROJECT_DIR environment variable not set")


class PaperAnalysis(BaseModel):
    summary: str
    short_description: str
    problem_description: str
    proposed_method: str
    conclusion_and_limitations: str


def setup_logging(log_level=logging.INFO):
    """Configure logging with timestamp and level."""
    log_dir = os.path.join(PROJECT_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"paper_processing_{timestamp}.log")

    # Suppress logs from libraries that make API requests (including warnings)
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("arxiv").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    # Suppress google_genai logs
    logging.getLogger("google_genai").setLevel(logging.ERROR)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger("paper_processing")


def initialize_services(logger=None):
    """Initialize Firebase and Gemini services."""
    # Initialize Firebase
    initialize_firebase_client(logger=logger)

    # Initialize Gemini
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    gemini_client = genai.Client(api_key=gemini_api_key)

    return gemini_client


def get_paper_analysis(gemini_client, pdf_path, logger):
    """Generate paper analysis using Gemini API."""
    filepath = pathlib.Path(pdf_path)

    prompt = """
    Analyze this academic machine learning research paper and provide a comprehensive analysis with the following structure:

    1. SUMMARY: Provide a thorough overview of the paper's key contributions, methodology, and findings.

    2. SHORT DESCRIPTION (TLDR - 50-100 words): Provide a concise overview that covers:
       - The problem/challenge being addressed
       - The proposed solution/methodology
       - Key results/findings
       - Main limitations or future work directions
       Make it accessible while maintaining technical accuracy.

    3. PROBLEM STATEMENT: Clearly articulate the specific research gap or challenge the paper addresses, including why this problem is significant.

    4. METHODOLOGY:
       - Detail the novel technical approach proposed
       - Highlight key algorithmic innovations
       - Explain the theoretical foundations
       - Describe implementation details critical to understanding the method

    5. RESULTS:
       - Summarize quantitative performance metrics
       - Compare against relevant baselines
       - Note any particularly impressive or unexpected findings

    6. LIMITATIONS AND FUTURE WORK:
       - Identify constraints or weaknesses acknowledged by the authors
       - Note any unaddressed edge cases or scenarios
       - Mention proposed directions for future research

    Use precise technical terminology appropriate for an expert in machine learning. Focus on conveying the technical depth and nuance of the research without simplification.
    """

    try:
        logger.info(f"Generating analysis for {pdf_path}")
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            config={
                "response_mime_type": "application/json",
                "response_schema": PaperAnalysis,
            },
            contents=[
                types.Part.from_bytes(
                    data=filepath.read_bytes(),
                    mime_type="application/pdf",
                ),
                prompt,
            ],
        )
        logger.info("Successfully generated analysis")

        # Convert response to PaperAnalysis model and then to dict
        try:
            analysis_dict = PaperAnalysis.model_validate_json(
                response.text
            ).model_dump()
            logger.debug(f"Analysis dict: {analysis_dict}")
            return analysis_dict
        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
            logger.debug(f"Raw response: {response.text}")
            return None

    except Exception as e:
        logger.error(f"Error generating analysis for {pdf_path}: {e}", exc_info=True)
        return None


def process_single_paper(paper_info, temp_dir, paper_paths, gemini_client, logger):
    """Process a single paper with all the steps (thumbnail, compression, analysis)"""
    paper_id = paper_info["document_id"]
    paper_arxiv_id = paper_info["arxiv_data.entry_id"].split("/")[-1].split("v")[0]
    pdf_path = [p for p in paper_paths if paper_arxiv_id in p]

    if len(pdf_path) == 0:
        logger.warning(
            f"PDF not found for paper {paper_arxiv_id}. Probably didnt download. Skipping..."
        )
        return
    pdf_path = pdf_path[0]
    if not os.path.exists(pdf_path):
        logger.warning(
            f"PDF not found for paper {paper_arxiv_id} with path {pdf_path}. Skipping..."
        )
        return

    try:
        logger.info(f"Processing paper {paper_id}")

        # Generate thumbnail in temporary directory
        thumb_path = os.path.join(temp_dir, f"{paper_id}_thumb.png")
        thumbnail_bytes = create_pdf_thumbnail(
            pdf_path, thumb_path, max_size_bytes=300 * 1024  # 300 kb
        )
        logger.info(f"Generated thumbnail for paper {paper_id}")

        # Compress PDF
        pdf_path, compression_ratio = compress_pdf(
            pdf_path, pdf_path, remove_images=True
        )
        logger.info(f"Compressed PDF for paper {paper_id}")
        logger.info(
            f"Size of compressed pdf: {os.path.getsize(pdf_path) / (1024 * 1024):.2f} MB"
        )
        logger.info(f"Compression ratio: {compression_ratio:.2f}%")

        # Generate paper analysis
        analysis = get_paper_analysis(gemini_client, pdf_path, logger)

        if analysis:
            # Update paper with new data
            update_successful = update_paper_in_firestore(
                paper_id=paper_id,
                update_data={
                    "status": "processed",
                    "thumbnail": thumbnail_bytes,
                    "analysis": analysis,
                },
                logger=logger,
            )

            if not update_successful:
                logger.error(f"Failed to update paper {paper_id} in Firestore")
                return
        else:
            logger.error(f"Failed to generate analysis for paper {paper_id}")
            logger.info(f"size of pdf: {os.path.getsize(pdf_path)}")
            update_paper_in_firestore(
                paper_id=paper_id,
                update_data={"status": "ERROR-gen_analysis"},
                logger=logger,
            )
            return

    except Exception as e:
        logger.error(f"Error processing paper {paper_id}: {e}", exc_info=True)
        return


def process_papers():
    """Main function to process papers."""
    logger = setup_logging()
    logger.info("Starting paper processing pipeline")

    # Initialize services
    gemini_client = initialize_services(logger=logger)
    arxiv = Arxiv()

    # update_all_papers_status(target_status="new", logger=logger)

    # Fetch new papers from Firestore
    fb_papers_info = fetch_specific_attributes_from_collection(
        attributes=["arxiv_data.title", "arxiv_data.entry_id", "status"],
        filters=[("status", "==", "new")],
    )
    logger.info(f"Found {len(fb_papers_info)} new papers to process")
    if len(fb_papers_info) == 0:
        logger.info("No new papers to process. Exiting...")
        return

    # Get papers from arXiv
    queries = [
        paper["arxiv_data.entry_id"].split("/")[-1].split("v")[0]
        for paper in fb_papers_info
    ]
    logger.debug(f"Queries: {queries}")
    papers = arxiv.get_papers(queries, verbose=True)
    logger.info(f"Fetched {len(papers)} papers from arXiv")

    # Download papers to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Downloading papers to temp dir {temp_dir}...")
        paper_paths, not_downloaded = arxiv.download_papers(
            dirpath=temp_dir, verbose=True, logger=logger
        )
        logger.info(f"Downloaded {len(paper_paths)} papers to {temp_dir}")
        logger.info(f"Download not successful for {len(not_downloaded)} papers")

        # Update status to download_error for papers that failed to download
        not_downloaded_ids = []
        for paper in not_downloaded:
            paper_id = [
                p["document_id"]
                for p in fb_papers_info
                if p["arxiv_data.entry_id"] == paper.entry_id
            ][0]
            update_paper_in_firestore(
                paper_id=paper_id,
                update_data={"status": "ERROR-download"},
                logger=logger,
            )
            logger.info(f"Updating status to ERROR-download for paper {paper_id}")
            not_downloaded_ids.append(paper_id)

        # remove papers that failed to download
        fb_papers_info = [
            p for p in fb_papers_info if p["document_id"] not in not_downloaded_ids
        ]

        # Process papers
        parallel = True  # Set to False for sequential processing

        if parallel:
            max_workers = 4
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Create a partial function with the common arguments
                process_paper_partial = partial(
                    process_single_paper,
                    temp_dir=temp_dir,
                    paper_paths=paper_paths,
                    gemini_client=gemini_client,
                    logger=logger,
                )

                # Process papers in parallel with progress bar
                list(
                    tqdm(
                        executor.map(process_paper_partial, fb_papers_info),
                        total=len(fb_papers_info),
                        desc="Processing papers",
                    )
                )
        else:
            # Sequential processing with progress bar
            for paper_info in tqdm(fb_papers_info, desc="Processing papers"):
                process_single_paper(
                    paper_info,
                    temp_dir=temp_dir,
                    paper_paths=paper_paths,
                    gemini_client=gemini_client,
                    logger=logger,
                )

    logger.info("Completed paper processing pipeline")


if __name__ == "__main__":
    process_papers()
