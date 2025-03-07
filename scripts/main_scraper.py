#!/usr/bin/env python3

import os
import sys
import time
import logging
import schedule
import pytz
from datetime import datetime
from scripts.papers_scraper import scrape_papers
from scripts.gen_papers_info import process_papers


PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def setup_logging(log_level=logging.INFO):
    """Configure logging with timestamp and level."""
    log_dir = os.path.join(PROJECT_DIR, "logs")
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
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("main_scraper")


def run_scrapers():
    """Run both scrapers in sequence."""
    logger = setup_logging()
    pt_time = datetime.now(pytz.timezone('America/Los_Angeles'))
    time_str = pt_time.strftime('%Y-%m-%d %H:%M:%S %Z')
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


def main():
    logger = setup_logging()
    pt_timezone = pytz.timezone('America/Los_Angeles')
    current_time = datetime.now(pt_timezone)
    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S %Z')
    logger.info(f"Starting main scraper service at {time_str}")

    # Schedule job to run at midnight Pacific Time
    schedule.every().day.at("00:00").do(run_scrapers).tag('pacific-time')

    # Run immediately if requested (commented out by default)
    # run_scrapers()

    # Keep the script running
    while True:
        try:
            next_run = schedule.next_run()
            if next_run:
                next_run_pt = next_run.astimezone(pt_timezone)
                time_str = next_run_pt.strftime('%Y-%m-%d %H:%M:%S %Z')
                logger.debug(f"Next run scheduled for: {time_str}")
            
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"An error occurred in the main loop: {str(e)}")
            # Continue running despite errors
            continue


if __name__ == "__main__":
    main()
