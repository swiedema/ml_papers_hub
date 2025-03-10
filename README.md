# ML Papers Hub

A minimalistic web application for managing and analyzing machine learning research papers, featuring automated scraping, AI-powered analysis, and intelligent organization tools.

## Features

### Paper Collection
- **Multi-Source Paper Scraping**
  - Automatic scraping of papers from Hugging Face's daily papers
  - Gmail integration for Google Scholar alerts
  - arXiv paper fetching with ID or title
  - Direct PDF upload support
  - URL-based paper import

### Automated Processing
- **Smart Paper Analysis**
  - AI-powered paper summarization using Google's Gemini
  - Automatic extraction of key components:
    - Problem description
    - Proposed methodology
    - Conclusions and limitations
    - TLDR summary
  - PDF processing with thumbnail generation and compression

### Organization & Management
- **Intelligent Paper Organization**
  - Automatic paper classification
  - Custom labeling system
  - Status tracking (new, processed, etc.)
  - Bulk processing capabilities

### Research Tools
- **Deep Dive Analysis**
  - Interactive paper viewing
  - AI-assisted paper discussion
  - Technical analysis with ML expertise
  - Cross-reference with related research

### User Interface
- Modern Streamlit-based web interface
- Responsive design with wide-layout support
- Easy navigation with sidebar
- Real-time paper management

## Setup

### Prerequisites
- Python 3.8 or higher
- Firebase account with Firestore database
- Google Cloud credentials for Gemini API
- (Optional) Gmail API credentials for paper scraping
- (Optional) Gumloop account for enhanced paper scraping

### Service Setup

Before running the application, you'll need to create accounts and obtain API credentials from the following services:

1. **Firebase** ([console.firebase.google.com](https://console.firebase.google.com))
   - Create a project and enable Firestore
   - Generate service account credentials

2. **Google Cloud & Gemini API** ([console.cloud.google.com](https://console.cloud.google.com))
   - Enable Gemini API
   - Create API credentials

3. **Gmail API** (Optional) ([console.cloud.google.com](https://console.cloud.google.com))
   - Enable Gmail API and create OAuth credentials

4. **Gumloop** (Optional) ([gumloop.com](https://gumloop.com))
   - Create account and obtain API credentials

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


