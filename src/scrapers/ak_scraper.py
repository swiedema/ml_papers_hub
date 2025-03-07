import requests
from bs4 import BeautifulSoup
import datetime


def scrape_ak_huggingface_daily_paper_titles(start_date: str = None):
    """
    Scrapes paper titles from HuggingFace papers page from start_date until today.

    Args:
        start_date (str, optional): The starting date in 'YYYY-MM-DD' format (e.g., '2024-03-01').
            If None, defaults to today's date.

    Returns:
        list: A list of paper titles published between start_date and today.
    """

    # If no start date provided, use today
    if start_date is None:
        start_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Convert start_date string to datetime object
    current_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    today = datetime.datetime.now()
    all_titles = []

    # Iterate through dates until today
    while current_date <= today:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f"https://huggingface.co/papers?date={date_str}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all paper entries
        paper_elements = soup.text.split("Submitted by")

        # Extract titles for current date
        for element in paper_elements[1:]:
            all_titles.append(element.split("\n\n")[3])

        # Move to next day
        current_date += datetime.timedelta(days=1)

    return all_titles
