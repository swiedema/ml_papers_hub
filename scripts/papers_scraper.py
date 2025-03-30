#!/usr/bin/env python3

from src.scrapers.ak_scraper import scrape_ak_huggingface_daily_paper_titles
from src.scrapers.gmail_scraper import scrape_paper_urls_from_gmail
from src.arxiv import Arxiv, is_arxiv_id
from src.cache import PaperCache
import logging
import os
import sys
from datetime import datetime, timedelta
from src.firebase import (
    initialize_firebase_client,
    fetch_specific_attributes_from_collection,
    add_papers_to_firestore,
)

# Get PROJECT_DIR from environment variable
PROJECT_DIR = os.getenv("PROJECT_DIR")
if not PROJECT_DIR:
    raise ValueError("PROJECT_DIR environment variable not set")


# Configure logging
def setup_logging(log_level=logging.INFO):
    log_dir = os.path.join(PROJECT_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"papers_scraper_{timestamp}.log")

    # Suppress logs from libraries that make API requests (including warnings)
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("arxiv").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger("papers_scraper")


def scrape_papers():
    logger = setup_logging()
    logger.info("Starting papers scraper")

    try:
        # initialize firebase client
        initialize_firebase_client(logger=logger)

        # initialize arxiv client
        arxiv = Arxiv()
        logger.info("Initialized Arxiv client")

        # initialize cache
        cache = PaperCache()
        logger.info("Initialized paper cache")

        # scrape ak huggingface daily paper titles
        logger.info("Scraping Hugging Face daily paper titles")
        paper_titles = scrape_ak_huggingface_daily_paper_titles()
        logger.info(f"Found {len(paper_titles)} paper titles from Hugging Face")

        # scrape google scholar papers from gmail
        logger.info("Finding new papers from Gmail")
        arxiv_urls = scrape_paper_urls_from_gmail(only_arxiv_urls=True)
        logger.info(f"Found {len(arxiv_urls)} URLs from Gmail")
        arxiv_queries = [url.split("/")[-1] for url in arxiv_urls]
        arxiv_queries = [
            arxiv_id for arxiv_id in arxiv_queries if is_arxiv_id(arxiv_id)
        ]
        arxiv_queries = list(set(arxiv_queries))
        logger.info(f"{len(arxiv_queries)} are valid arXiv IDs")

        # get papers from arxiv
        queries = paper_titles + arxiv_queries
        logger.info(f"Fetching {len(queries)} papers from arXiv")
        arxiv.get_papers(queries, verbose=True)
        logger.info(f"Successfully retrieved {len(arxiv.papers)} papers")

        # convert papers to dict
        papers_dict = arxiv.papers_to_dict()
        papers_dict = [
            {"arxiv_data": paper_dict, "status": "new"} for paper_dict in papers_dict
        ]
        logger.info(f"Converted {len(papers_dict)} papers to dict")

        # filter out papers that are in cache
        papers_dict = cache.filter_papers(papers_dict)
        logger.info(f"{len(papers_dict)} papers remaining after cache filtering")

        # save papers to firestore
        if papers_dict:
            try:
                # fetch papers from firestore
                x_days_ago = datetime.now() - timedelta(days=7)
                fb_papers_dict = fetch_specific_attributes_from_collection(
                    attributes=["arxiv_data.title", "status", "created_at"],
                    filters=[
                        ("created_at", ">=", x_days_ago),
                    ],
                )
                fb_papers_dict = [
                    paper for paper in fb_papers_dict if paper["status"] != "test"
                ]
                logger.info(f"Fetched {len(fb_papers_dict)} papers from firestore")

                # remove papers from papers_dict that are already in fb_papers_dict
                fb_papers_titles = [
                    paper["arxiv_data.title"] for paper in fb_papers_dict
                ]
                papers_dict = [
                    paper
                    for paper in papers_dict
                    if paper["arxiv_data"]["title"] not in fb_papers_titles
                ]
                logger.info(f"{len(papers_dict)} new papers left to save to firestore")

                # save papers to firestore
                add_papers_to_firestore(papers_dict, collection_name="papers_app")
                logger.info(f"Saved {len(papers_dict)} papers to firestore")

                # Add successfully processed papers to cache
                for paper in papers_dict:
                    cache.add_processed_paper(paper["arxiv_data"]["title"])
            except Exception as e:
                logger.error(f"Error saving papers to firestore: {e}")

                # Add failed papers to error cache
                for paper in papers_dict:
                    cache.add_error_paper(paper["arxiv_data"]["title"])
                raise

        logger.info("Papers scraper completed successfully")

    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    scrape_papers()
