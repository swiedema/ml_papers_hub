from firebase_admin import credentials, firestore
import firebase_admin
import os
import traceback
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()


def initialize_firebase_client(project_dir=None, logger=None):
    """Initialize Firebase Admin SDK."""
    if project_dir is None:
        PROJECT_DIR = os.environ.get("PROJECT_DIR")
    else:
        PROJECT_DIR = project_dir
    credentials_file_path = os.path.join(
        PROJECT_DIR, os.getenv("FIREBASE_CREDENTIALS_RELATIVE_PATH")
    )

    try:
        if not os.path.exists(credentials_file_path):
            logger.error(f"❌ Credentials file not found: {credentials_file_path}")
            raise FileNotFoundError(
                f"Credentials directory not found: {credentials_file_path}"
            )

        cred = credentials.Certificate(credentials_file_path)

        firebase_admin.initialize_app(
            cred,
            {
                "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
                "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
            },
        )
    except ValueError as e:
        if "The default Firebase app already exists." in str(e):
            pass  # App is already initialized
        else:
            logger.error(f"❌ Error initializing Firebase client: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")


def get_all_data_from_firestore_collection(
    collection_name: str = "papers_app",
) -> list[dict]:
    """
    Fetches all data from Firestore.
    """
    firestore_db = firebase_admin.firestore.client()
    papers_docs = firestore_db.collection(collection_name).get()
    return [{"paper_id": doc.id, **doc.to_dict()} for doc in papers_docs]


def fetch_specific_attributes_from_collection(
    attributes: list[str],
    collection_name: str = "papers_app",
    filters: list[tuple] = None,
    logger=None,
) -> list[dict]:
    """
    Fetches specific attributes from documents in a Firestore collection with optional filtering.

    Args:
        attributes (list[str]): List of field paths to fetch. Can include nested paths using dots.
                e.g. ['title', 'metadata.author', 'content.abstract']
        collection_name (str): Name of the Firestore collection
        filters (list[tuple]): Optional list of filter conditions, each as a tuple of (field, operator, value)
                e.g. [('status', '==', 'pending'), ('created_at', '>', '2023-01-01')]
                Supported operators: '==', '!=', '<', '<=', '>', '>=', 'array_contains', 'in', 'array_contains_any'
        logger: Optional logger instance

    Returns:
        list[dict]: List of dictionaries containing requested fields for each document
    """
    try:
        # Get reference to Firestore database
        firestore_db = firebase_admin.firestore.client()

        # Start with the collection reference
        query = firestore_db.collection(collection_name)

        # Apply filters if provided
        if filters:
            for field, operator, value in filters:
                query = query.where(field, operator, value)

        # Execute the query
        docs = query.get()

        results = []

        for doc in docs:
            # For each document, create a result with document_id
            result = {"document_id": doc.id}
            doc_data = doc.to_dict()

            # Process each requested field
            for field in attributes:
                if "." in field:
                    # Handle nested fields (e.g., 'metadata.author')
                    parts = field.split(".")
                    value = doc_data
                    for part in parts:
                        if (
                            value is not None
                            and isinstance(value, dict)
                            and part in value
                        ):
                            value = value.get(part)
                        else:
                            value = None
                            break
                    result[field] = value
                else:
                    # Handle top-level fields
                    result[field] = doc_data.get(field)

            results.append(result)

        if logger is not None:
            logger.info(f"Retrieved {len(results)} documents from {collection_name}")

        return results

    except Exception as e:
        if logger is not None:
            logger.error(f"❌ Error fetching data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            print(f"❌ Error fetching data: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
        return []


def add_papers_to_firestore(papers: list[dict], collection_name: str = "papers_app"):
    """
    Adds a paper to Firestore.
    """
    firestore_db = firebase_admin.firestore.client()
    for paper in papers:
        # Add timestamp for when paper was added to Firestore
        paper["created_at"] = datetime.now()
        firestore_db.collection(collection_name).add(paper)


def add_single_paper_to_firestore(
    paper: dict, collection_name: str = "papers_app", custom_id: str = None
):
    """
    Adds a single paper to Firestore.

    Args:
        paper (dict): Paper data to add to Firestore
        collection_name (str): Name of the Firestore collection
        custom_id (str, optional): Custom document ID. If None, Firestore will auto-generate an ID

    Returns:
        str: The document ID of the added paper
    """
    firestore_db = firebase_admin.firestore.client()

    if custom_id:
        # Use the specified custom ID
        doc_ref = firestore_db.collection(collection_name).document(custom_id)
        doc_ref.set(paper)
        return custom_id
    else:
        # Let Firestore generate a random ID
        doc_ref = firestore_db.collection(collection_name).add(paper)
        return doc_ref[1].id


def remove_all_papers_from_firestore(
    collection_name: str = "papers_app", exclude_ids: list[str] = None
):
    """
    Removes all papers from Firestore.
    """
    firestore_db = firebase_admin.firestore.client()

    # Get all documents except those in exclude_ids
    docs = firestore_db.collection(collection_name).stream()
    for doc in docs:
        doc_id = doc.id
        if exclude_ids is None or doc_id not in exclude_ids:
            doc.reference.delete()


def delete_paper_from_firestore(
    paper_id: str, collection_name: str = "papers_app", logger=None
):
    """
    Deletes a paper from Firestore.
    """
    firestore_db = firebase_admin.firestore.client()
    try:
        firestore_db.collection(collection_name).document(paper_id).delete()
        if logger is not None:
            logger.info(f"Successfully deleted paper {paper_id} from {collection_name}")
        else:
            print(f"Successfully deleted paper {paper_id} from {collection_name}")
        return True
    except Exception as e:
        if logger is not None:
            logger.error(
                f"Error deleting paper {paper_id} from {collection_name}: {str(e)}"
            )
            logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            print(f"Error deleting paper {paper_id} from {collection_name}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
        return False


def update_all_papers_status(
    target_status: str, collection_name: str = "papers_app", logger=None
) -> bool:
    """
    Updates the status field of all papers in Firestore to the target status.

    Args:
        target_status (str): The status value to set for all papers
        collection_name (str): Name of the Firestore collection
        logger: Optional logger instance

    Returns:
        bool: True if all updates were successful, False if any update failed
    """
    firestore_db = firebase_admin.firestore.client()
    success = True

    try:
        # Get all documents in the collection
        docs = firestore_db.collection(collection_name).stream()

        for doc in docs:
            # Skip test papers
            if doc.id.startswith("test_paper"):
                continue
            try:
                doc.reference.update({"status": target_status})
                if logger is not None:
                    logger.info(
                        f"Updated status to '{target_status}' for paper {doc.id}"
                    )
            except Exception as e:
                success = False
                if logger is not None:
                    logger.error(f"Error updating paper {doc.id}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    print(f"Error updating paper {doc.id}: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")

        return success

    except Exception as e:
        if logger is not None:
            logger.error(f"Error accessing collection {collection_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            print(f"Error accessing collection {collection_name}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
        return False


def update_paper_in_firestore(
    paper_id: str, update_data: dict, collection_name: str = "papers_app", logger=None
) -> bool:
    """
    Updates a paper document in Firestore with new data.

    Args:
        paper_id (str): The ID of the paper document to update
        update_data (dict): Dictionary containing the fields to update
        collection_name (str): Name of the Firestore collection
        logger: Optional logger instance

    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        firestore_db = firestore.client()
        doc_ref = firestore_db.collection(collection_name).document(paper_id)
        doc_ref.update(update_data)

        if logger is not None:
            logger.info(f"Successfully updated paper {paper_id}")
        return True

    except Exception as e:
        if logger is not None:
            logger.error(f"Error updating paper {paper_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return False
