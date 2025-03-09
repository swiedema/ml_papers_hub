import pandas as pd
import streamlit as st
from src.firebase import fetch_specific_attributes_from_collection
import tempfile
import os
from src.arxiv import Arxiv

st.set_page_config(page_title="View Papers", page_icon="📖", layout="wide")

# Create a row for the title and refresh button
title_col, refresh_col = st.columns([6, 1])
with title_col:
    st.title("View Papers")
with refresh_col:
    button_params = {
        "help": "Refresh Papers",
        "key": "refresh_button",
        "use_container_width": True,
    }
    if st.button("🔄", **button_params):
        st.session_state.papers_df = None
        st.rerun()

st.markdown("Browse and search through your collection of papers.")


# Fetch papers from Firestore
def get_papers():
    papers = fetch_specific_attributes_from_collection(
        attributes=[
            "arxiv_data.title",
            "arxiv_data.published",
            "label",
            "status",
            "analysis.short_description",
        ]
    )

    # Convert to DataFrame format
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
    }
    return pd.DataFrame(papers_data)


# Get papers from Firestore
if st.session_state.get("papers_df") is None:
    papers_df = get_papers()
    st.session_state.papers_df = papers_df
else:
    papers_df = st.session_state.papers_df

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Year range filter
    years = papers_df["Year"].dropna().astype(int)
    if not years.empty:
        min_year = int(years.min())
        max_year = int(years.max())
        year_range = st.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )

    # Status filter
    statuses = ["All"] + sorted(papers_df["Status"].unique().tolist())
    selected_status = st.selectbox("Status", statuses)

    # Label filter
    labels = ["All"] + sorted([l for l in papers_df["Label"].unique() if pd.notna(l)])
    selected_label = st.selectbox("Label", labels)

# Apply filters
filtered_df = papers_df.copy()

# Filter by year range
if not years.empty:
    filtered_df = filtered_df[
        (filtered_df["Year"] >= year_range[0]) & (filtered_df["Year"] <= year_range[1])
    ]

# Filter by status
if selected_status != "All":
    filtered_df = filtered_df[filtered_df["Status"] == selected_status]

# Filter by label
if selected_label != "All":
    filtered_df = filtered_df[filtered_df["Label"] == selected_label]

# Search box
search_term = st.text_input("Search papers", placeholder="Enter keywords...")
if search_term:
    filtered_df = filtered_df[
        filtered_df["Title"].str.contains(search_term, case=False)
        | filtered_df["Description"].str.contains(search_term, case=False)
    ]

# Display results
st.subheader(f"Papers ({len(filtered_df)} results)")
if not filtered_df.empty:
    # Display the table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "Title": st.column_config.Column(
                "Title",
                width="large",
            )
        },
    )

# Add paper title input field
paper_title = st.text_input("Enter paper title to view PDF:", "")

# Only process if a title was entered
if paper_title:
    # Create a temporary directory for PDF
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Initialize Arxiv client and get paper
        arxiv_client = Arxiv()
        papers = arxiv_client.get_papers([paper_title], verbose=False)

        if papers:
            # Download PDF
            paper_paths, not_downloaded = arxiv_client.download_papers(
                tmp_dir, verbose=False
            )

            if paper_paths:
                pdf_path = paper_paths[0]

                # Read PDF file
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()

                # Display PDF using Streamlit
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=f"{paper_title}.pdf",
                    mime="application/pdf",
                )

                st.pdf(pdf_bytes)
            else:
                st.error("Could not download the PDF for this paper.")
        else:
            st.error("Could not find the paper on arXiv.")
else:
    st.info("Enter a paper title above to view its PDF.")
