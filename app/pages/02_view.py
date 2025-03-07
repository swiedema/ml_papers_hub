import pandas as pd
import streamlit as st

st.set_page_config(page_title="View Papers", page_icon="📖", layout="wide")

st.title("View Papers")
st.markdown("Browse and search through your collection of papers.")


# Create some sample data
def get_sample_papers():
    return pd.DataFrame(
        {
            "Title": [
                "Machine Learning Basics",
                "Neural Networks and Deep Learning",
                "Natural Language Processing",
                "Computer Vision Techniques",
                "Reinforcement Learning",
            ],
            "Authors": [
                "Smith et al.",
                "Johnson et al.",
                "Williams et al.",
                "Brown et al.",
                "Davis et al.",
            ],
            "Year": [2020, 2021, 2022, 2021, 2023],
            "Category": ["ML", "DL", "NLP", "CV", "RL"],
            "Rating": [4, 5, 3, 4, 5],
        }
    )


# Get sample data
papers_df = get_sample_papers()

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Category filter
    categories = ["All"] + sorted(papers_df["Category"].unique().tolist())
    selected_category = st.selectbox("Category", categories)

    # Year range filter
    min_year = int(papers_df["Year"].min())
    max_year = int(papers_df["Year"].max())
    year_range = st.slider(
        "Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year)
    )

    # Rating filter
    min_rating = st.slider("Minimum Rating", 1, 5, 1)

# Apply filters
filtered_df = papers_df.copy()

# Filter by category
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

# Filter by year range
filtered_df = filtered_df[
    (filtered_df["Year"] >= year_range[0]) & (filtered_df["Year"] <= year_range[1])
]

# Filter by rating
filtered_df = filtered_df[filtered_df["Rating"] >= min_rating]

# Search box
search_term = st.text_input("Search papers", placeholder="Enter keywords...")
if search_term:
    filtered_df = filtered_df[
        filtered_df["Title"].str.contains(search_term, case=False)
        | filtered_df["Authors"].str.contains(search_term, case=False)
    ]

# Display results
st.subheader(f"Papers ({len(filtered_df)} results)")
if not filtered_df.empty:
    st.dataframe(filtered_df, use_container_width=True)
else:
    st.info("No papers match your search criteria.")
