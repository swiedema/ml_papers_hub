import requests
import time
import os
from dotenv import load_dotenv


load_dotenv()


def scrape_paper_urls_from_gmail(only_arxiv_urls=True):

    # first run gumloops pipeline to get the run_id
    url = "https://api.gumloop.com/api/v1/start_pipeline"
    payload = {
        "user_id": os.getenv("GUMLOOP_USER_ID"),
        "saved_item_id": os.getenv("GUMLOOP_SAVED_ITEM_ID"),
        "pipeline_inputs": [],
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('GUMLOOP_API_TOKEN')}",
        "Content-Type": "application/json",
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    try:
        run_id = response.json()["run_id"]
    except Exception as e:
        print(response.json())
        raise Exception("Failed to get run_id") from e

    # wait till the run is finished
    print(f"Waiting for run {run_id} to finish...")
    url = "https://api.gumloop.com/api/v1/get_pl_run"
    headers = {"Authorization": f"Bearer {os.getenv('GUMLOOP_API_TOKEN')}"}
    params = {
        "run_id": run_id,
        "user_id": os.getenv("GUMLOOP_USER_ID"),
    }
    run_status = "RUNNING"
    while not run_status == "DONE":
        response = requests.request("GET", url, headers=headers, params=params)
        run_status = response.json()["state"]
        if run_status == "FAILED":
            return []
        time.sleep(1)

    # then use the run_id to get the outputs
    url = "https://api.gumloop.com/api/v1/get_pl_run"
    headers = {
        "Authorization": f"Bearer {os.getenv('GUMLOOP_API_TOKEN')}",
    }
    params = {
        "run_id": run_id,
        "user_id": os.getenv("GUMLOOP_USER_ID"),
    }
    response = requests.request("GET", url, headers=headers, params=params)
    results = response.json()["outputs"]
    if results.get("output", None) is None:
        urls = []
    else:
        urls = results["output"]
        urls = [url for sublist in urls for url in sublist]
        if only_arxiv_urls:
            urls = [url for url in urls if "arxiv.org" in url]

    return urls
