import pandas as pd
import streamlit as st
from src.firebase import (
    initialize_firebase_client,
    fetch_specific_attributes_from_collection,
)
from src.arxiv import get_arxiv_id

# Set page configuration
st.set_page_config(
    page_title="ML Papers Hub",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Firebase client
initialize_firebase_client()

# Main app header
st.title("ML Papers Hub")
st.markdown("A simple application to manage your research papers.")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    st.markdown("Use the sidebar to navigate through different sections of the app.")

    # Add a separator
    st.divider()

    # About section in the sidebar
    st.markdown("### About")
    st.markdown(
        "This app helps you organize and manage your research papers efficiently."
    )

# Main content
st.header("Welcome to ML Papers Hub")
st.markdown(
    """
This application allows you to:
- Perform bulk operations like auto tagging, smart search, etc.
- Deep dive into a particular paper with AI assistance
- Quickly manually label newly scraped papers as intersting/reject
"""
)

# Example data display
st.subheader("Sample Papers")
# Fetch 5 random papers from Firestore
with st.spinner("Fetching papers..."):
    papers = fetch_specific_attributes_from_collection(
        attributes=[
            "arxiv_data.title",
            "arxiv_data.published",
            "arxiv_data.entry_id",
            "label",
            "status",
            "analysis.short_description",
        ],
        filters=[("status", "==", "processed")],  # Only fetch processed papers
        limit=5,  # Will return at most 5 documents
    )

    # Convert to DataFrame format and select 5 random papers
    papers_data = {
        "Title": [p.get("arxiv_data.title", "") for p in papers],
        "Description": [p.get("analysis.short_description", "") for p in papers],
        "Year": [
            (
                p.get("arxiv_data.published", "").year
                if p.get("arxiv_data.published")
                else None
            )
            for p in papers
        ],
        "Status": [p.get("status", "") for p in papers],
        "Label": [p.get("label", "") for p in papers],
        "arxiv_id": [get_arxiv_id(p.get("arxiv_data.entry_id", "")) for p in papers],
    }
    df = pd.DataFrame(papers_data).sample(n=min(5, len(papers_data["Title"])))

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Title": st.column_config.Column(
                "Title",
                width="large",
            ),
            "arxiv_id": st.column_config.Column(
                "arXiv ID",
                width="medium",
            ),
        },
    )

# Footer
st.markdown("---")
st.markdown("Papers Manager v0.1 | Created with Streamlit")
