import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Papers Manager",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main app header
st.title("Papers Manager")
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
st.header("Welcome to Papers Manager")
st.markdown(
    """
This application allows you to:
- Upload and organize research papers
- Add metadata and notes to your papers
- Search through your collection
- Generate summaries and insights
"""
)

# Example data display
st.subheader("Sample Papers")
sample_data = {
    "Title": [
        "Machine Learning Basics",
        "Neural Networks and Deep Learning",
        "Natural Language Processing",
    ],
    "Authors": ["Smith et al.", "Johnson et al.", "Williams et al."],
    "Year": [2020, 2021, 2022],
    "Category": ["ML", "DL", "NLP"],
}
df = pd.DataFrame(sample_data)
st.dataframe(df)

# Footer
st.markdown("---")
st.markdown("Papers Manager v0.1 | Created with Streamlit")
