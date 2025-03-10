import pandas as pd
import streamlit as st
from src.firebase import (
    fetch_specific_attributes_from_collection,
    update_paper_in_firestore,
)
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key)


def classify_paper(paper, labels_list, model="gemini-2.0-flash"):
    """Classify a single paper into one of the given labels."""
    # Prepare paper context
    paper_context = f"""
    Title: {paper['Title']}
    Abstract: {paper['Abstract']}
    
    Analysis Details:
    - Summary: {paper['Summary']}
    - Problem Description: {paper['Problem']}
    - Proposed Method: {paper['Method']}
    - Short Description: {paper['TLDR']}
    - Conclusions and Limitations: {paper['Conclusions']}
    """

    # Generate classification prompt
    prompt = f"""
    Based on the following paper information, classify it into one of these categories: {', '.join(labels_list)}
    
    Paper information:
    {paper_context}
    
    Consider all aspects of the paper, including its problem statement, methodology, and conclusions.
    Respond with ONLY the category name, nothing else.
    """

    # Get classification from Gemini
    response = gemini_client.models.generate_content(model=model, contents=prompt)

    predicted_label = response.text.strip()
    return predicted_label if predicted_label in labels_list else None


def generate_group_labels(papers_df, num_groups=None, model="gemini-2.0-flash"):
    """Generate group labels based on the papers content."""
    # Prepare papers context
    papers_context = "\n\n".join(
        [
            f"""Paper {i+1}:
        Title: {paper['Title']}
        Abstract: {paper['Abstract']}
        Summary: {paper['Summary']}
        Problem: {paper['Problem']}
        Method: {paper['Method']}
        TLDR: {paper['TLDR']}
        Conclusions: {paper['Conclusions']}"""
            for i, paper in papers_df.iterrows()
        ]
    )

    # Generate group labels prompt
    if num_groups is None:
        group_prompt = f"""
        List of Papers:
        {papers_context}

        Based on the above set of {len(papers_df)} papers, determine the optimal number of groups 
        (between 2 and 10) and their descriptive labels. The groups should represent distinct 
        research directions or themes.
        
        Requirements:
        1. First determine the optimal number of groups based on paper similarity.
        2. Then generate descriptive labels that are:
           - Short and concise (max 3-4 words)
           - Capture main themes/topics
           - Distinct from each other
           - Easily understandable

        IMPORTANT: Make sure the list of labels is between 2 and 10 labels. Double check your response is in the correct format.

        Respond with ONLY a Python list of strings containing the labels, nothing else.
        Example format:
        ["Label 1", "Label 2", "Label 3"]
        """
    else:
        group_prompt = f"""
        List of Papers:
        {papers_context}

        Based on the above set of {len(papers_df)} papers, suggest {num_groups} descriptive group labels.
        
        Requirements for labels:
        1. Must be short and concise (max 3-4 words)
        2. Should capture the main theme/topic
        3. Should be distinct from each other
        4. Should be easily understandable

        Respond with ONLY a Python list of strings containing exactly {num_groups} labels, nothing else.
        Example format:
        ["Label 1", "Label 2", "Label 3"]
        """

    response = gemini_client.models.generate_content(model=model, contents=group_prompt)

    # Parse response as a Python list
    try:
        # Clean up the response text and evaluate it as a Python expression
        response_text = response.text.strip()
        # Remove any Python code block markers if present
        response_text = (
            response_text.replace("```python", "").replace("```", "").strip()
        )
        labels = eval(response_text)

        # Validate that we got a list of strings
        if (
            isinstance(labels, list)
            and all(isinstance(label, str) for label in labels)
            and (num_groups is None or len(labels) == num_groups)
        ):
            return labels
        else:
            return []
    except:
        return []


def auto_classify_papers(papers_df, labels_list):
    """Auto-classify multiple papers and return their labels."""
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (_, paper) in enumerate(papers_df.iterrows()):
        status_text.text(
            f"Processing paper {idx + 1}/{len(papers_df)}: {paper['Title'][:50]}..."
        )
        predicted_label = classify_paper(paper, labels_list)
        if predicted_label:
            results[paper["document_id"]] = predicted_label
        progress_bar.progress((idx + 1) / len(papers_df))

    progress_bar.empty()
    status_text.empty()
    return results


def auto_group_papers(papers_df, num_groups=None):
    """Auto-group papers and return their labels."""
    results = {}

    # First generate group labels
    with st.spinner("Generating group labels..."):
        group_labels = generate_group_labels(papers_df, num_groups)
        if not group_labels:
            st.error("Failed to generate group labels")
            return {}
        st.success(f"Generated {len(group_labels)} group labels")

    # Then classify each paper
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (_, paper) in enumerate(papers_df.iterrows()):
        status_text.text(
            f"Processing paper {idx + 1}/{len(papers_df)}: {paper['Title'][:50]}..."
        )
        predicted_label = classify_paper(paper, group_labels)
        if predicted_label:
            results[paper["document_id"]] = predicted_label
        progress_bar.progress((idx + 1) / len(papers_df))

    progress_bar.empty()
    status_text.empty()
    return results


def save_labels_to_db(papers_df):
    """Save AI labels to Firestore."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    success_count = 0

    for idx, (_, paper) in enumerate(papers_df.iterrows()):
        if pd.notna(paper.get("AI_label")):
            status_text.text(
                f"Saving paper {idx + 1}/{len(papers_df)}: {paper['Title'][:50]}..."
            )
            success = update_paper_in_firestore(
                paper_id=paper["document_id"], update_data={"label": paper["AI_label"]}
            )
            if success:
                success_count += 1
        progress_bar.progress((idx + 1) / len(papers_df))

    progress_bar.empty()
    status_text.empty()
    return success_count


def format_paper_details(paper):
    """Format paper details consistently for prompts."""
    return f"""
    Title: {paper['Title']}
    Abstract: {paper['Abstract']}
    
    Analysis Details:
    - Summary: {paper['Summary']}
    - Problem Description: {paper['Problem']}
    - Proposed Method: {paper['Method']}
    - Short Description: {paper['TLDR']}
    - Conclusions and Limitations: {paper['Conclusions']}
    """


def flag_paper(
    paper,
    prompt,
    positive_examples=None,
    negative_examples=None,
    model="gemini-2.0-flash",
):
    """Flag a paper based on prompt and examples."""
    # Prepare paper context
    paper_context = format_paper_details(paper)

    # Prepare examples context if provided
    examples_context = ""
    if positive_examples:
        examples_context += (
            "\nPositive examples (papers that should be flagged as True):\n"
        )
        for i, example in enumerate(positive_examples, 1):
            # Find the example paper in the DataFrame
            example_details = format_paper_details(example)
            examples_context += f"Example {i}:\n{example_details}\n"

    if negative_examples:
        examples_context += (
            "\nNegative examples (papers that should be flagged as False):\n"
        )
        for i, example in enumerate(negative_examples, 1):
            # Find the example paper in the DataFrame
            example_details = format_paper_details(example)
            examples_context += f"Example {i}:\n{example_details}\n"

    # Generate flagging prompt
    flag_prompt = f"""
    Your task is to analyze a research paper and determine if it matches the given criteria.
    
    Criteria to check:
    {prompt}
    {examples_context}
    
    Paper to analyze:
    {paper_context}
    
    Respond with ONLY 'True' or 'False', nothing else.
    True means the paper matches the criteria, False means it doesn't.
    """

    # Get classification from Gemini
    response = gemini_client.models.generate_content(model=model, contents=flag_prompt)

    result = response.text.strip().lower()
    return result == "true"


def auto_flag_papers(papers_df, prompt, positive_examples=None, negative_examples=None):
    """Auto-flag multiple papers based on prompt and examples."""
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Convert example titles/IDs to paper dictionaries
    pos_papers = []
    neg_papers = []

    if positive_examples:
        for ex in positive_examples:
            # Try to find paper by title
            matching_paper = papers_df[
                papers_df["Title"].str.contains(ex, case=False, na=False)
            ]
            if not matching_paper.empty:
                pos_papers.append(matching_paper.iloc[0])

    if negative_examples:
        for ex in negative_examples:
            # Try to find paper by title
            matching_paper = papers_df[
                papers_df["Title"].str.contains(ex, case=False, na=False)
            ]
            if not matching_paper.empty:
                neg_papers.append(matching_paper.iloc[0])

    for idx, (_, paper) in enumerate(papers_df.iterrows()):
        status_text.text(
            f"Processing paper {idx + 1}/{len(papers_df)}: {paper['Title'][:50]}..."
        )
        is_flagged = flag_paper(paper, prompt, pos_papers, neg_papers)
        results[paper["document_id"]] = is_flagged
        progress_bar.progress((idx + 1) / len(papers_df))

    progress_bar.empty()
    status_text.empty()
    return results


# Page setup
st.set_page_config(page_title="Bulk Process Papers", page_icon="🔄", layout="wide")

# Create a row for the title and refresh button
title_col, refresh_col = st.columns([6, 1])
with title_col:
    st.title("Bulk Process Papers")
with refresh_col:
    button_params = {
        "help": "Refresh Papers",
        "key": "refresh_button",
        "use_container_width": True,
    }
    if st.button("🔄", **button_params):
        st.rerun()

st.markdown("Bulk process your papers with auto-classification and auto-grouping.")

# Initialize session state for AI labels if not exists
if "ai_labels" not in st.session_state:
    st.session_state.ai_labels = {}


# Fetch papers from Firestore
def get_papers():
    # Define required attributes
    required_attributes = [
        "arxiv_data.title",
        "arxiv_data.abstract",
        "arxiv_data.published",
        "arxiv_data.entry_id",
        "label",
        "status",
        "analysis",
    ]

    # Check if papers are already in session state and have all required attributes
    if "papers_cache" in st.session_state:
        papers = st.session_state.papers_cache
        # Check if all required attributes are present in the first paper
        if papers and all(
            any(p.get(attr) is not None for p in papers) for attr in required_attributes
        ):
            return convert_to_dataframe(papers)

    # If not in cache or missing attributes, fetch from database
    papers = fetch_specific_attributes_from_collection(
        attributes=required_attributes,
        filters=[("status", "==", "processed")],
    )
    # Cache the papers in session state
    st.session_state.papers_cache = papers
    return convert_to_dataframe(papers)


def convert_to_dataframe(papers):
    # Convert to DataFrame format
    papers_data = {
        "Title": [p.get("arxiv_data.title", "") for p in papers],
        "Abstract": [p.get("arxiv_data.abstract", "") for p in papers],
        "Analysis": [p.get("analysis", {}) for p in papers],
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
        "document_id": [p.get("document_id", "") for p in papers],
    }
    return pd.DataFrame(papers_data)


# Get papers from Firestore
with st.spinner("Fetching papers..."):
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

    # AI Label filter
    ai_labels = ["All"] + sorted(
        [
            label
            for label in papers_df["document_id"]
            .map(st.session_state.ai_labels)
            .unique()
            if pd.notna(label)
        ]
    )
    selected_ai_label = st.selectbox("AI Label", ai_labels)

    # AI Flag filter
    if "ai_flags" not in st.session_state:
        st.session_state.ai_flags = {}

    flag_options = ["All", "Flagged", "Not Flagged"]
    selected_flag = st.selectbox("AI Flag", flag_options)

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

# Filter by AI label
if selected_ai_label != "All":
    filtered_df["AI_label"] = filtered_df["document_id"].map(st.session_state.ai_labels)
    filtered_df = filtered_df[filtered_df["AI_label"] == selected_ai_label]

# Filter by AI flag
if selected_flag != "All":
    filtered_df["AI_flag"] = filtered_df["document_id"].map(st.session_state.ai_flags)
    if selected_flag == "Flagged":
        filtered_df = filtered_df[filtered_df["AI_flag"] == True]
    else:  # Not Flagged
        filtered_df = filtered_df[filtered_df["AI_flag"] == False]

# Add checkboxes to select papers
if not filtered_df.empty:
    # Create a row for the title and save button
    title_col, save_col = st.columns([6, 2])
    with title_col:
        st.subheader(f"Papers ({len(filtered_df)} results)")
    with save_col:
        if st.session_state.ai_labels or st.session_state.ai_flags:
            if st.button("Save AI Labels and Flags to Database"):
                # Update the dataframe with current AI labels and flags
                filtered_df["AI_label"] = filtered_df["document_id"].map(
                    st.session_state.ai_labels
                )
                filtered_df["AI_flag"] = filtered_df["document_id"].map(
                    st.session_state.ai_flags
                )

                # Save labels
                success_count_labels = save_labels_to_db(
                    filtered_df[filtered_df["AI_label"].notna()]
                )

                # Save flags
                success_count_flags = 0
                for _, paper in filtered_df[filtered_df["AI_flag"].notna()].iterrows():
                    success = update_paper_in_firestore(
                        paper_id=paper["document_id"],
                        update_data={"flag": paper["AI_flag"]},
                    )
                    if success:
                        success_count_flags += 1

                st.success(
                    f"Successfully saved {success_count_labels} labels and {success_count_flags} flags "
                    "to the database!"
                )

                # Clear AI labels and flags from session state
                st.session_state.ai_labels = {}
                st.session_state.ai_flags = {}
                st.rerun()

    # Search box
    search_term = st.text_input("Search papers", placeholder="Enter keywords...")
    if search_term:
        filtered_df = filtered_df[
            filtered_df["Title"].str.contains(search_term, case=False)
        ]

    # Add select all checkbox
    select_all = st.checkbox("Select All")

    # Create columns for checkboxes, AI labels, flags, and analysis components
    filtered_df["Selected"] = select_all
    filtered_df["AI_label"] = filtered_df["document_id"].map(st.session_state.ai_labels)
    filtered_df["AI_flag"] = filtered_df["document_id"].map(st.session_state.ai_flags)

    # Extract analysis components
    filtered_df["TLDR"] = filtered_df["Analysis"].apply(
        lambda x: x.get("short_description", "") if isinstance(x, dict) else ""
    )
    filtered_df["Summary"] = filtered_df["Analysis"].apply(
        lambda x: x.get("summary", "") if isinstance(x, dict) else ""
    )
    filtered_df["Problem"] = filtered_df["Analysis"].apply(
        lambda x: x.get("problem_description", "") if isinstance(x, dict) else ""
    )
    filtered_df["Method"] = filtered_df["Analysis"].apply(
        lambda x: x.get("proposed_method", "") if isinstance(x, dict) else ""
    )
    filtered_df["Conclusions"] = filtered_df["Analysis"].apply(
        lambda x: x.get("conclusion_and_limitations", "") if isinstance(x, dict) else ""
    )

    # Display the table with checkboxes
    edited_df = st.data_editor(
        filtered_df,
        hide_index=True,
        column_config={
            "Selected": st.column_config.CheckboxColumn(
                "Select", default=False, help="Select papers for bulk processing"
            ),
            "Title": st.column_config.Column(
                "Title",
                width="large",
            ),
            "TLDR": st.column_config.Column(
                "TLDR",
                width="large",
            ),
            "AI_label": st.column_config.Column(
                "AI Label",
                width="medium",
            ),
            "Label": st.column_config.Column(
                "Current Label",
                width="medium",
            ),
            "AI_flag": st.column_config.CheckboxColumn(
                "AI Flag",
                default=False,
                disabled=True,
                help="Papers flagged by AI based on criteria",
            ),
            "Year": st.column_config.Column(
                "Year",
                width="small",
            ),
            "Status": st.column_config.Column(
                "Status",
                width="small",
            ),
            "Summary": st.column_config.Column(
                "Summary",
                width="large",
            ),
            "Problem": st.column_config.Column(
                "Problem Description",
                width="large",
            ),
            "Method": st.column_config.Column(
                "Proposed Method",
                width="large",
            ),
            "Conclusions": st.column_config.Column(
                "Conclusions & Limitations",
                width="large",
            ),
        },
        column_order=[
            "Selected",
            "Title",
            "TLDR",
            "AI_flag",
            "AI_label",
            "Label",
            "Year",
        ],
        disabled=[
            "Title",
            "TLDR",
            "AI_label",
            "Label",
            "AI_flag",
            "Year",
            "Status",
            "document_id",
            "Abstract",
            "Analysis",
            "Summary",
            "Problem",
            "Method",
            "Conclusions",
        ],
    )

    # Get selected papers
    selected_papers = edited_df[edited_df["Selected"]]
    num_selected = len(selected_papers)

    # Display bulk processing options if papers are selected
    if num_selected > 0:
        st.markdown(f"**{num_selected} papers selected**")

        # Auto-flagging section
        with st.expander("Auto-flagging", expanded=False):
            flag_prompt = st.text_area(
                "Enter flagging criteria",
                help="Describe what properties or criteria a paper should have to be flagged as True",
            )

            col1, col2 = st.columns(2)
            with col1:
                positive_examples = st.text_area(
                    "Positive examples (one per line)",
                    help="Enter paper titles or arXiv IDs that should be flagged as True",
                )
            with col2:
                negative_examples = st.text_area(
                    "Negative examples (one per line)",
                    help="Enter paper titles or arXiv IDs that should be flagged as False",
                )

            if st.button("Auto-flag Selected Papers") and flag_prompt:
                # Process examples
                pos_examples = (
                    [ex.strip() for ex in positive_examples.split("\n") if ex.strip()]
                    if positive_examples
                    else None
                )
                neg_examples = (
                    [ex.strip() for ex in negative_examples.split("\n") if ex.strip()]
                    if negative_examples
                    else None
                )

                # Run auto-flagging
                new_flags = auto_flag_papers(
                    selected_papers, flag_prompt, pos_examples, neg_examples
                )

                # Update session state
                st.session_state.ai_flags.update(new_flags)
                st.success(
                    "Auto-flagging complete! The table has been updated with AI flags."
                )
                st.rerun()

        # Auto-grouping section
        with st.expander("Auto-grouping", expanded=False):
            auto_groups = st.radio(
                "Number of groups",
                ["Auto-detect", "Manual"],
                help="Choose whether to automatically detect the optimal number of groups or specify manually",
            )

            num_groups = None
            if auto_groups == "Manual":
                num_groups = st.number_input(
                    "Specify number of groups",
                    min_value=2,
                    max_value=10,
                    value=5,
                    help="Number of groups to create",
                )

            if st.button("Auto-group Selected Papers"):
                # Run auto-grouping
                new_labels = auto_group_papers(selected_papers, num_groups)

                # Update session state
                st.session_state.ai_labels.update(new_labels)
                st.success(
                    "Auto-grouping complete! The table has been updated with AI labels."
                )
                st.rerun()

        # Auto-classification section
        with st.expander("Auto-classification", expanded=False):
            class_labels = st.text_area(
                "Enter class labels (one per line)",
                help="Enter the possible class labels for papers, one per line",
            )

            if st.button("Auto-classify Selected Papers") and class_labels:
                labels_list = [
                    label.strip() for label in class_labels.split("\n") if label.strip()
                ]

                if labels_list:
                    # Run auto-classification
                    new_labels = auto_classify_papers(selected_papers, labels_list)

                    # Update session state
                    st.session_state.ai_labels.update(new_labels)
                    st.success(
                        "Auto-classification complete! The table has been updated with AI labels."
                    )
                    st.rerun()

else:
    st.info("No papers found matching the filters.")
