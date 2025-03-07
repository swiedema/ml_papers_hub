import pandas as pd
import streamlit as st

st.set_page_config(page_title="Upload Papers", page_icon="📤", layout="wide")

st.title("Upload Papers")
st.markdown("Upload your research papers and add metadata.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a PDF file", type="pdf", help="Upload a research paper in PDF format"
)

# Form for paper metadata
with st.form("paper_metadata_form"):
    st.subheader("Paper Metadata")

    # Two columns for form fields
    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Title", placeholder="Paper title")
        authors = st.text_input("Authors", placeholder="Comma-separated authors")
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)

    with col2:
        category = st.selectbox(
            "Category",
            options=[
                "Machine Learning",
                "Deep Learning",
                "NLP",
                "Computer Vision",
                "Other",
            ],
        )
        tags = st.text_input("Tags", placeholder="Comma-separated tags")
        rating = st.slider("Rating", min_value=1, max_value=5, value=3)

    notes = st.text_area("Notes", placeholder="Add your notes about this paper")

    # Submit button
    submitted = st.form_submit_button("Save Paper")

    if submitted:
        if uploaded_file is not None:
            st.success(f"Paper '{title}' uploaded successfully!")

            # Display the metadata as a table
            metadata = {
                "Title": [title],
                "Authors": [authors],
                "Year": [year],
                "Category": [category],
                "Tags": [tags],
                "Rating": [rating],
            }
            st.dataframe(pd.DataFrame(metadata))
        else:
            st.error("Please upload a PDF file first.")
