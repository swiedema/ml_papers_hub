import json
import os
from typing import List, Dict, Set
import logging
from datetime import datetime


class PaperCache:
    def __init__(self, cache_file: str = None, max_size: int = 100):
        if cache_file is None:
            project_dir = os.getenv("PROJECT_DIR")
            if not project_dir:
                raise ValueError("PROJECT_DIR environment variable not set")

            cache_dir = os.path.join(project_dir, "cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "paper_cache.json")

        self.cache_file = cache_file
        self.max_size = max_size
        self.processed_papers: Set[str] = set()
        self.error_papers: Set[str] = set()
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    self.processed_papers = set(data.get("processed_papers", []))
                    self.error_papers = set(data.get("error_papers", []))
            except Exception as e:
                logging.error(f"Error loading cache: {e}")
                self.processed_papers = set()
                self.error_papers = set()

    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(
                    {
                        "processed_papers": list(self.processed_papers),
                        "error_papers": list(self.error_papers),
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                )
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

    def add_processed_paper(self, paper_title: str) -> None:
        """Add a paper to the processed papers cache."""
        self.processed_papers.add(paper_title)
        if len(self.processed_papers) > self.max_size:
            # Remove oldest entries if we exceed max size
            self.processed_papers = set(list(self.processed_papers)[-self.max_size :])
        self._save_cache()

    def add_error_paper(self, paper_title: str) -> None:
        """Add a paper to the error papers cache."""
        self.error_papers.add(paper_title)
        if len(self.error_papers) > self.max_size:
            # Remove oldest entries if we exceed max size
            self.error_papers = set(list(self.error_papers)[-self.max_size :])
        self._save_cache()

    def is_paper_processed(self, paper_title: str) -> bool:
        """Check if a paper has been processed."""
        return paper_title in self.processed_papers

    def is_paper_error(self, paper_title: str) -> bool:
        """Check if a paper has errored."""
        return paper_title in self.error_papers

    def should_process_paper(self, paper_title: str) -> bool:
        """Determine if a paper should be processed based on cache status."""
        return not (
            self.is_paper_processed(paper_title) or self.is_paper_error(paper_title)
        )

    def filter_papers(self, papers: List[Dict]) -> List[Dict]:
        """Filter out papers that are in the cache."""
        return [
            paper
            for paper in papers
            if self.should_process_paper(paper["arxiv_data"]["title"])
        ]
