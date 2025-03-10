import streamlit as st
import tempfile
import os
import requests
from datetime import datetime
from urllib.parse import urlparse
import validators
from src.arxiv import Arxiv, is_arxiv_id, get_arxiv_id
from src.firebase import (
    initialize_firebase_client,
    add_single_paper_to_firestore,
    fetch_specific_attributes_from_collection,
)

# Initialize Firebase
initialize_firebase_client()

st.set_page_config(page_title="Upload Papers", page_icon="📤", layout="wide")

st.title("Upload Papers")
st.markdown("Upload research papers using any of the methods below.")

# Create tabs for different upload methods
tab1, tab2, tab3, tab4 = st.tabs(
    ["Upload PDF File", "PDF URL", "arXiv ID", "Paper Title"]
)


def check_duplicate_title(title):
    """Check if a paper with the same title already exists in the database."""
    existing_papers = fetch_specific_attributes_from_collection(
        attributes=["arxiv_data.title", "title"], filters=[("status", "!=", "test")]
    )

    title = title.lower().strip()
    for paper in existing_papers:
        # Check both arxiv papers and locally uploaded papers
        arxiv_title = paper.get("arxiv_data.title", "").lower().strip()
        local_title = paper.get("title", "").lower().strip()

        if title == arxiv_title or title == local_title:
            return True
    return False


def process_local_pdf(uploaded_file):
    """Process a locally uploaded PDF file"""
    if uploaded_file is None:
        return

    # Check for duplicate title
    title = os.path.splitext(uploaded_file.name)[0]  # Remove .pdf extension
    if check_duplicate_title(title):
        st.error("A paper with this title already exists in the database!")
        return

    # TODO: Implement Firebase Storage upload
    st.info("Note: PDF storage in Firebase Storage will be implemented soon.")

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Store paper in Firestore
    paper_data = {
        "title": title,
        "source": "local_upload",
        "status": "new",
        "created_at": datetime.now(),
    }

    paper_id = add_single_paper_to_firestore(paper_data)
    if paper_id:
        st.success(f"Paper uploaded successfully! ID: {paper_id}")
    else:
        st.error("Failed to upload paper to database.")

    # Clean up temp file
    os.unlink(tmp_path)


def process_pdf_url(pdf_url):
    """Process a PDF from a URL"""
    if not pdf_url:
        return

    if not validators.url(pdf_url):
        st.error("Please enter a valid URL")
        return

    if not pdf_url.lower().endswith(".pdf"):
        st.error("URL must point to a PDF file")
        return

    # Check for duplicate title
    title = os.path.splitext(os.path.basename(urlparse(pdf_url).path))[0]
    if check_duplicate_title(title):
        st.error("A paper with this title already exists in the database!")
        return

    try:
        # Download PDF to temporary file
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name

        # TODO: Implement Firebase Storage upload
        st.info("Note: PDF storage in Firebase Storage will be implemented soon.")

        # Store paper in Firestore
        paper_data = {
            "title": title,
            "source": "url_upload",
            "source_url": pdf_url,
            "status": "new",
            "created_at": datetime.now(),
        }

        paper_id = add_single_paper_to_firestore(paper_data)
        if paper_id:
            st.success(f"Paper downloaded and uploaded successfully! ID: {paper_id}")
        else:
            st.error("Failed to upload paper to database.")

        # Clean up temp file
        os.unlink(tmp_path)

    except Exception as e:
        st.error(f"Error downloading PDF: {str(e)}")


def process_arxiv_id(arxiv_id):
    """Process a paper using its arXiv ID"""
    if not arxiv_id:
        return

    # Clean and validate arXiv ID
    arxiv_id = arxiv_id.strip()
    if not is_arxiv_id(arxiv_id):
        st.error("Please enter a valid arXiv ID")
        return

    # Initialize Arxiv client and get paper
    arxiv = Arxiv()
    papers = arxiv.get_papers([arxiv_id])

    if not papers:
        st.error("Could not find paper on arXiv")
        return

    # Check for duplicate title
    papers_dict = arxiv.papers_to_dict()
    title = papers_dict[0]["title"]
    if check_duplicate_title(title):
        st.error("A paper with this title already exists in the database!")
        return

    # Store in Firestore
    paper_data = {
        "arxiv_data": papers_dict[0],
        "status": "new",
        "created_at": datetime.now(),
    }

    paper_id = add_single_paper_to_firestore(paper_data)
    if paper_id:
        st.success(f"Paper added successfully! ID: {paper_id}")
    else:
        st.error("Failed to add paper to database.")


def process_paper_title(title):
    """Process a paper using its title"""
    if not title:
        return

    # Check for duplicate title first
    if check_duplicate_title(title):
        st.error("A paper with this title already exists in the database!")
        return

    # Initialize Arxiv client and search for paper
    arxiv = Arxiv()
    papers = arxiv.get_papers([title])

    if not papers:
        st.error("Could not find paper on arXiv")
        return

    # Convert to dict and store in Firestore
    papers_dict = arxiv.papers_to_dict()
    paper_data = {
        "arxiv_data": papers_dict[0],
        "status": "new",
        "created_at": datetime.now(),
    }

    paper_id = add_single_paper_to_firestore(paper_data)
    if paper_id:
        st.success(f"Paper added successfully! ID: {paper_id}")
    else:
        st.error("Failed to add paper to database.")


# Tab 1: Upload PDF File
with tab1:
    st.header("Upload PDF File")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if st.button("Upload", key="upload_file"):
        process_local_pdf(uploaded_file)

# Tab 2: PDF URL
with tab2:
    st.header("PDF URL")
    pdf_url = st.text_input(
        "Enter PDF URL", placeholder="https://example.com/paper.pdf"
    )
    if st.button("Upload", key="upload_url"):
        process_pdf_url(pdf_url)

# Tab 3: arXiv ID
with tab3:
    st.header("arXiv ID")
    arxiv_id = st.text_input("Enter arXiv ID", placeholder="2102.12345")
    if st.button("Upload", key="upload_arxiv"):
        process_arxiv_id(arxiv_id)

# Tab 4: Paper Title
with tab4:
    st.header("Paper Title")
    paper_title = st.text_input(
        "Enter paper title", placeholder="Attention Is All You Need"
    )
    if st.button("Upload", key="upload_title"):
        process_paper_title(paper_title)

# Add information about paper processing
st.markdown("---")
st.markdown(
    """
### What happens after upload?
1. Your paper will be added to the database with status "new"
2. The paper processing pipeline will:
   - Generate a thumbnail
   - Compress the PDF
   - Generate an AI analysis of the paper
3. Once processed, the paper will be available in the "View Papers" section
"""
)
