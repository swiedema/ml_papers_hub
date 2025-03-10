# Papers Hub

A minimalistic web application for managing and analyzing research papers in bulk, featuring automated scraping, AI-powered analysis, and intelligent organization tools.

## Features

- **Paper Collection**: Multi-source paper scraping from Hugging Face, Gmail, arXiv, and direct uploads

- **Automated Processing**: Bulk paper analysis using Gemini, including summaries, key insights, and PDF processing, smart classification and custom labeling.

- **Research Tools**: Interactive paper viewing with AI-assisted analysis and discussion

## Setup

### Pre-requisites
The current implementation is using 3rd party services for scraping and processing papers.

For scraping papers from Gmail, the app uses the Gumloop api services.

For Database, the app uses Firebase.

For AI processing, the app uses Gemini from Google Cloud.

API keys for these services should be placed in the `.env` file.

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml_papers_hub
```

2. Create and activate a virtual environment:
```bash
./build.sh
```
This script will:
- Install uv (Python package installer)
- Create a virtual environment
- Install all dependencies
- Set up the project package

3. Set up environment variables:
Create a `.env` file with the following the env.example file.

### Running the Application

1. Start the Streamlit web interface:
```bash
streamlit run app/main.py
```

2. (Optional) Start the paper scraping service:
```bash
python scripts/main_scraper.py
```


