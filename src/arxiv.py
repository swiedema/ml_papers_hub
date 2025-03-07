import arxiv
from tqdm import tqdm
import concurrent.futures
import os


class Arxiv:
    @staticmethod
    def to_dict(paper):
        paper_dict = {}

        # Basic metadata
        if hasattr(paper, "entry_id"):
            paper_dict["entry_id"] = paper.entry_id
        if hasattr(paper, "updated"):
            paper_dict["updated"] = paper.updated
        if hasattr(paper, "published"):
            paper_dict["published"] = paper.published
        if hasattr(paper, "title"):
            paper_dict["title"] = paper.title

        # Authors
        if hasattr(paper, "authors"):
            paper_dict["authors"] = [author.name for author in paper.authors]

        # Content
        if hasattr(paper, "summary"):
            paper_dict["abstract"] = paper.summary
        if hasattr(paper, "comment"):
            paper_dict["comment"] = paper.comment
        if hasattr(paper, "journal_ref"):
            paper_dict["journal_ref"] = paper.journal_ref
        if hasattr(paper, "doi"):
            paper_dict["doi"] = paper.doi

        # Categories
        if hasattr(paper, "primary_category"):
            paper_dict["primary_category"] = paper.primary_category
        if hasattr(paper, "categories"):
            paper_dict["categories"] = paper.categories

        # Links
        if hasattr(paper, "links"):
            paper_dict["links"] = []
            for link in paper.links:
                link_dict = {}
                if hasattr(link, "href"):
                    link_dict["url"] = link.href
                if hasattr(link, "title"):
                    link_dict["title"] = link.title
                if hasattr(link, "rel"):
                    link_dict["rel"] = link.rel
                if hasattr(link, "content_type"):
                    link_dict["content_type"] = link.content_type
                if link_dict:  # Only add if link has any attributes
                    paper_dict["links"].append(link_dict)

        return paper_dict

    def __init__(self):
        self.client = arxiv.Client()
        self.papers = []

    def get_papers(
        self, queries: list[str], verbose: bool = False
    ) -> list[arxiv.Result]:
        # Define a helper function to fetch a single paper
        def fetch_paper(query):
            search = arxiv.Search(
                query=query,
                max_results=1,
            )
            results = self.client.results(search)
            all_results = list(results)
            return all_results[0] if all_results else None

        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all queries to the executor
            future_to_query = {
                executor.submit(fetch_paper, query): query for query in queries
            }

            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(future_to_query),
                total=len(queries),
                desc="Getting papers",
                disable=not verbose,
            ):
                paper = future.result()
                if paper:
                    self.papers.append(paper)

        self.clean_duplicate_papers()
        return self.papers

    def download_papers(self, dirpath: str, verbose: bool = False, logger=None):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        not_downloaded = []

        def download_single_paper(paper):
            try:
                paper.download_pdf(dirpath=dirpath)
                return None
            except Exception as e:
                if logger is not None:
                    logger.error(
                        f"Error downloading paper arxiv_id={paper.entry_id} title={paper.title}: {e}"
                    )
                else:
                    print(
                        f"Error downloading paper arxiv_id={paper.entry_id} title={paper.title}: {e}"
                    )
                return paper

        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all papers to the executor
            future_to_paper = {
                executor.submit(download_single_paper, paper): paper
                for paper in self.papers
            }

            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(future_to_paper),
                total=len(self.papers),
                desc="Downloading papers",
                disable=not verbose,
            ):
                result = future.result()
                if result is not None:  # If paper failed to download
                    not_downloaded.append(result)

        paper_paths = [
            os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith(".pdf")
        ]
        return paper_paths, not_downloaded

    def clean_duplicate_papers(self):
        # Create a dictionary with titles as keys to remove duplicates
        unique_papers = {}
        for paper in self.papers:
            unique_papers[paper.title] = paper
        self.papers = list(unique_papers.values())

    def print_papers_info(self, with_abstract: bool = False):
        for paper in self.papers:
            print(f"TITLE: {paper.title}")
            print(f"AUTHORS: {', '.join([author.name for author in paper.authors])}")
            if with_abstract:
                print(f"ABSTRACT: {paper.summary}")
            print()

    def papers_to_dict(self) -> list[dict]:
        return [self.to_dict(paper) for paper in self.papers]


def is_arxiv_id(id: str) -> bool:
    # Check if string matches format 'XXXX.XXXX' where X are digits
    if not isinstance(id, str):
        return False
    parts = id.split(".")
    if len(parts) != 2:
        return False
    parts_are_digits = parts[0].isdigit() and parts[1].isdigit()
    return parts_are_digits
