import streamlit as st
from src.firebase import (
    fetch_specific_attributes_from_collection,
    update_paper_in_firestore,
)
from src.pdf_parser import bytes_to_pil_image

st.set_page_config(page_title="Label Papers", page_icon="🏷️", layout="wide")


def update_label(label_value):
    print(
        f"Updating label for paper {st.session_state.current_paper['document_id']} to {label_value}"
    )
    success = update_paper_in_firestore(
        paper_id=st.session_state.current_paper["document_id"],
        update_data={"label": label_value},
    )
    if success:
        print(f"Success: {success}")
        st.session_state.current_index += 1
        st.success(f"Paper labeled as {label_value}")
        st.rerun()  # Refresh the page to show the next paper
    else:
        st.error("Failed to update paper label. Please try again.")


st.title("Label Papers")

# Fetch papers that need labeling
if st.session_state.get("papers") is None:
    with st.spinner("Fetching unlabeled papers. This may take a few seconds..."):
        papers = fetch_specific_attributes_from_collection(
            attributes=[
                "arxiv_data.title",
                "arxiv_data.abstract",
                "analysis",
                "thumbnail",
                # "label",
            ],
            filters=[("status", "==", "processed")],
        )
        st.session_state["papers"] = papers
else:
    papers = st.session_state["papers"]

# Add this after the papers fetch but before the unlabeled_papers filter
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Filter out papers that already have labels
unlabeled_papers = [
    paper for paper in papers if "label" not in paper or paper["label"] is None
]

if not unlabeled_papers:
    st.info("No papers left to label! All papers have been reviewed.")
else:
    # Add progress indicators and label buttons in a side-by-side layout
    buttons_col, _, progress_col = st.columns([2, 1, 1])

    with progress_col:
        total_papers = len(unlabeled_papers)
        papers_done = st.session_state.current_index
        st.progress(papers_done / total_papers)
        st.markdown(f"{papers_done}/{total_papers} papers labeled")

    with buttons_col:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👎 Not Relevant", use_container_width=True):
                update_label("negative")
        with col2:
            if st.button("😐 Neutral", use_container_width=True):
                update_label("neutral")
        with col3:
            if st.button("👍 Interesting", use_container_width=True):
                update_label("positive")

    # Get the current unlabeled paper
    st.session_state.current_paper = unlabeled_papers[st.session_state.current_index]

    st.markdown("---")

    # Create two columns for layout
    left_col, right_col = st.columns([1, 2])

    with left_col:
        if "thumbnail" in st.session_state.current_paper:
            img = bytes_to_pil_image(st.session_state.current_paper["thumbnail"])
            st.image(img, use_container_width=True)

    with right_col:
        st.markdown("### TLDR")
        st.write(st.session_state.current_paper["analysis"]["short_description"])

    # Create expandable sections for detailed information
    with st.expander("🔍 Problem Description"):
        st.write(st.session_state.current_paper["analysis"]["problem_description"])

    with st.expander("⚙️ Proposed Method"):
        st.write(st.session_state.current_paper["analysis"]["proposed_method"])

    with st.expander("💭 Conclusions and Limitations"):
        st.write(
            st.session_state.current_paper["analysis"]["conclusion_and_limitations"]
        )

    with st.expander("📊 Detailed Summary"):
        st.write(st.session_state.current_paper["analysis"]["summary"])

    with st.expander("📝 Abstract"):
        st.write(st.session_state.current_paper["arxiv_data.abstract"])
