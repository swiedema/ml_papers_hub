#!/usr/bin/env python3

import os
import sys
import time
import logging
import pytz
from datetime import datetime, timedelta
from papers_scraper import scrape_papers
from gen_papers_info import process_papers
import argparse
from src.scrapers.ak_scraper import scrape_ak_huggingface_daily_paper_titles
from src.scrapers.gmail_scraper import scrape_paper_urls_from_gmail
from src.arxiv import Arxiv, is_arxiv_id
from src.cache import PaperCache
from src.firebase import (
    initialize_firebase_client,
    fetch_specific_attributes_from_collection,
    add_papers_to_firestore,
)

PROJECT_DIR = os.getenv("PROJECT_DIR")
if not PROJECT_DIR:
    raise ValueError("PROJECT_DIR environment variable not set")


def setup_logging(log_level=logging.INFO):
    """Configure logging with timestamp and level."""
    log_dir = os.path.join(PROJECT_DIR, "logs")
    print(f"Log dir: {log_dir}")
    print(os.path.dirname(__file__))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"main_scraper_{timestamp}.log")

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

    return logging.getLogger("main_scraper")


def run_scrapers():
    """Run both scrapers in sequence."""
    logger = setup_logging()
    pt_time = datetime.now(pytz.timezone("America/Los_Angeles"))
    time_str = pt_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(f"Starting scraping cycle at {time_str}")

    try:
        # Run papers scraper first to collect new papers
        logger.info("Running papers scraper")
        scrape_papers()

        # Run papers info generator to process new papers
        logger.info("Running papers info generator")
        process_papers()

        logger.info("Completed scraping cycle successfully")

    except Exception as e:
        logger.exception(f"An error occurred during scraping cycle: {str(e)}")


def parse_time(time_str):
    """Validate and parse time string in HH:MM format."""
    try:
        return datetime.strptime(time_str, "%H:%M").time()
    except ValueError:
        raise argparse.ArgumentTypeError("Time must be in HH:MM format (24-hour)")


def main():
    logger = setup_logging()
    pt_timezone = pytz.timezone("America/Los_Angeles")
    current_time = datetime.now(pt_timezone)
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(f"Starting main scraper service at {time_str}")

    # Calculate next hour
    next_run = current_time.replace(minute=0, second=0, microsecond=0)
    if next_run <= current_time:
        next_run += timedelta(hours=1)

    logger.info(f"Next run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    while True:
        try:
            current_time = datetime.now(pt_timezone)

            if current_time >= next_run:
                run_scrapers()
                next_run = current_time.replace(
                    minute=0, second=0, microsecond=0
                ) + timedelta(hours=1)
                logger.info(
                    f"Next run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                )

            # Sleep for 60 seconds before checking again
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"An error occurred in the main loop: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying on error
            continue


if __name__ == "__main__":
    main()
