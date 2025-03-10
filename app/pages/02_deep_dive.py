import pandas as pd
import streamlit as st
from src.firebase import fetch_specific_attributes_from_collection
import tempfile
from src.arxiv import Arxiv, get_arxiv_id
from streamlit_pdf_viewer import pdf_viewer
from components.chat import GeminiChat, OpenAIChat
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


st.set_page_config(
    page_title="View Papers and Deep Dive", page_icon="📖", layout="wide"
)

# Create a row for the title and refresh button
title_col, refresh_col = st.columns([6, 1])
with title_col:
    st.title("View Papers and Deep Dive")
with refresh_col:
    button_params = {
        "help": "Refresh Papers",
        "key": "refresh_button",
        "use_container_width": True,
    }
    if st.button("🔄", **button_params):
        st.rerun()

st.markdown("Browse and search through your collection of papers.")


# Fetch papers from Firestore
with st.spinner("Fetching papers..."):

    def get_papers():
        papers = fetch_specific_attributes_from_collection(
            attributes=[
                "arxiv_data.title",
                "arxiv_data.published",
                "arxiv_data.entry_id",
                "label",
                "status",
                "analysis.short_description",
            ],
            filters=[("status", "==", "processed")],
        )

        # Convert to DataFrame format
        papers_data = {
            "Title": [p.get("arxiv_data.title", "") for p in papers],
            "analysis.short_description": [
                p.get("analysis.short_description", "") for p in papers
            ],
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
            "arxiv_id": [
                get_arxiv_id(p.get("arxiv_data.entry_id", "")) for p in papers
            ],
        }
        return pd.DataFrame(papers_data)

    papers_df = get_papers()

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
        | filtered_df["analysis.short_description"].str.contains(
            search_term, case=False
        )
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
            ),
            "arxiv_id": st.column_config.Column(
                "arXiv ID",
                width="medium",
            ),
        },
    )

# Add paper title input field
paper_title = st.text_input("Enter paper title or arXiv ID to view PDF:", "")

# Initialize pdf cache in session state if not exists
if "pdf_cache" not in st.session_state:
    st.session_state.pdf_cache = {}

# Only process if a title was entered
if paper_title:
    # Create a temporary directory for PDF
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Check if we already have this paper cached
        if paper_title in st.session_state.pdf_cache:
            with st.spinner("Loading paper from cache..."):
                pdf_bytes = st.session_state.pdf_cache[paper_title]
                # Write cached bytes to a temporary file
                pdf_path = os.path.join(tmp_dir, f"{paper_title}.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)
        else:
            # Initialize Arxiv client and get paper
            with st.spinner("Downloading paper from arXiv..."):
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

                        # Cache the PDF bytes
                        st.session_state.pdf_cache[paper_title] = pdf_bytes
                    else:
                        st.error("Could not download the PDF for this paper.")
                        pdf_bytes = None
                        pdf_path = None
                else:
                    st.error("Could not find the paper on arXiv.")
                    pdf_bytes = None
                    pdf_path = None

        # If we have valid PDF data (either from cache or new download), display it
        if pdf_bytes and pdf_path:
            # Display PDF using Streamlit
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"{paper_title}.pdf",
                mime="application/pdf",
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                pdf_viewer(pdf_path, width="90%", height=1000, render_text=True)
            with col2:
                # Initialize and render chat component
                paper_chat = GeminiChat(
                    session_key_prefix=f"paper_chat_{paper_title}", pdf_path=pdf_path
                )
                paper_chat.render(chat_title="Chat about this paper")
else:
    st.info("Enter a paper title above to view its PDF.")
