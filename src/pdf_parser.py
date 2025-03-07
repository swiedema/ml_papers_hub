import os
import sys
import pymupdf as fitz  # Correct import for PyMuPDF
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from openai import OpenAI
from dotenv import load_dotenv
import json
import base64
from tqdm import tqdm
from PIL import Image
from io import BytesIO


load_dotenv()

# OpenAI API Key (set your own API key)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)


def classify_image_with_gpt4v(image_path):
    with open(image_path, "rb") as img_file:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI model that classifies images from research papers as figures or tables and extracts captions if available.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What type of content is in this image? If it's a figure or table, extract any visible captions.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
                            },
                        },
                    ],
                },
            ],
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI model that classifies images from research papers as figures or tables and extracts captions if available.",
                },
                {
                    "role": "user",
                    "content": "What type of content is in this image? If it's a figure or table, extract any visible captions.",
                },
                {"role": "user", "content": {"image": img_file.read()}},
            ],
        )

    classification = response["choices"][0]["message"]["content"].lower()
    if "table" in classification:
        return "table", classification
    elif "figure" in classification:
        return "figure", classification
    return "other", classification


def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    os.makedirs(output_folder, exist_ok=True)
    figures_folder = os.path.join(output_folder, "figures")
    tables_folder = os.path.join(output_folder, "tables")
    metadata_file = os.path.join(output_folder, "metadata.csv")
    os.makedirs(figures_folder, exist_ok=True)
    os.makedirs(tables_folder, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_data = []

    for page_num in range(len(doc)):
        for img_index, img in enumerate(doc[page_num].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            image_filename = os.path.join(
                output_folder, f"page_{page_num+1}_img_{img_index+1}.{img_ext}"
            )

            with open(image_filename, "wb") as f:
                f.write(image_bytes)

            category, caption = classify_image_with_gpt4v(image_filename)
            if category == "figure":
                new_path = os.path.join(
                    figures_folder, os.path.basename(image_filename)
                )
                os.rename(image_filename, new_path)
            elif category == "table":
                new_path = os.path.join(tables_folder, os.path.basename(image_filename))
                os.rename(image_filename, new_path)
            else:
                new_path = image_filename  # Keep in the main extracted folder

            image_data.append(
                [page_num + 1, os.path.basename(new_path), category, caption]
            )

    df = pd.DataFrame(image_data, columns=["Page", "Filename", "Category", "Caption"])
    df.to_csv(metadata_file, index=False)
    print(
        f"Extracted and classified {len(image_data)} images. Metadata saved to {metadata_file}."
    )


def extract_tables_with_bounding_boxes(pdf_path, output_folder="extracted_tables"):
    os.makedirs(output_folder, exist_ok=True)
    annotated_output_folder = os.path.join(output_folder, "annotated")
    os.makedirs(annotated_output_folder, exist_ok=True)

    images = convert_from_path(pdf_path)

    for page_num, img in enumerate(images):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 100 and h > 50:  # Adjust size threshold for tables
                table_crop = img_np[y : y + h, x : x + w]
                table_filename = os.path.join(
                    output_folder, f"page_{page_num+1}_table_{i+1}.png"
                )
                cv2.imwrite(table_filename, table_crop)

                # Draw bounding box on original image
                cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Save annotated image with bounding boxes
        annotated_filename = os.path.join(
            annotated_output_folder, f"page_{page_num+1}_annotated.png"
        )
        cv2.imwrite(annotated_filename, img_np)

    print("Tables extracted and bounding boxes overlaid on pages.")


def extract_captions_from_pdf(pdf_path, keywords=["Figure", "Fig.", "fig."]):
    images = convert_from_path(pdf_path)
    captions = []

    for page_num, img in tqdm(
        enumerate(images), total=len(images), desc="Processing pages"
    ):
        text = pytesseract.image_to_string(img)
        for line in text.split("\n"):
            if any(keyword in line for keyword in keywords):
                captions.append((page_num + 1, line))

    print("Extracted captions:")
    for caption in captions:
        print(f"Page {caption[0]}: {caption[1]}")

    return captions


def extract_visual_elements_with_gpt4v(pdf_path, output_folder="visual_elements"):
    """
    Extract and annotate images, figures, and tables from a PDF using GPT-4V.
    Returns coordinates of detected elements along with their classifications.

    Args:
        pdf_path: Path to the PDF file
        output_folder: Folder to save extracted elements and annotations

    Returns:
        List of dictionaries containing page number, element type, coordinates, and confidence
    """
    os.makedirs(output_folder, exist_ok=True)
    annotated_folder = os.path.join(output_folder, "annotated")
    elements_folder = os.path.join(output_folder, "elements")
    os.makedirs(annotated_folder, exist_ok=True)
    os.makedirs(elements_folder, exist_ok=True)

    # Convert PDF to images
    print(f"Converting PDF to images: {pdf_path}")
    images = convert_from_path(pdf_path)
    visual_elements = []

    for page_num, img in tqdm(
        enumerate(images), total=len(images), desc="Processing pages"
    ):
        # Save the page image for GPT-4V analysis
        page_image_path = os.path.join(output_folder, f"page_{page_num+1}.png")
        img.save(page_image_path)

        # Convert to numpy array for OpenCV processing
        img_np = np.array(img)
        img_annotated = img_np.copy()

        # Process with GPT-4V to identify visual elements
        prompt = """
        Analyze this page from a research paper and identify all visual elements (figures, tables, charts, diagrams).
        For each element:
        1. Specify its type (figure, table, chart, diagram)
        2. Provide the approximate bounding box coordinates as [x1, y1, x2, y2] where:
           - x1, y1 is the top-left corner
           - x2, y2 is the bottom-right corner
           - All values should be percentages of the image dimensions (0-100)
        3. Extract any visible caption
        
        Format your response as a valid JSON array of objects, where each object has these keys:
        - "type": string (one of: "figure", "table", "chart", "diagram")
        - "coordinates": array of 4 numbers [x1, y1, x2, y2] representing percentages
        - "caption": string (the element's caption text)
        
        Example response format:
        [
          {
            "type": "figure",
            "coordinates": [10, 20, 90, 45],
            "caption": "Figure 1: Example visualization of the proposed method"
          },
          {
            "type": "table",
            "coordinates": [15, 50, 85, 70],
            "caption": "Table 2: Experimental results comparing different approaches"
          }
        ]
        """

        with open(page_image_path, "rb") as img_file:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant that analyzes research paper pages to identify and locate visual elements.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
                                },
                            },
                        ],
                    },
                ],
            )

        # Parse the response
        try:
            response_content = response.choices[0].message.content

            # Check if the response contains a code block with JSON
            if (
                "```json" in response_content
                and "```" in response_content.split("```json", 1)[1]
            ):
                # Extract JSON from code block
                json_content = (
                    response_content.split("```json", 1)[1].split("```", 1)[0].strip()
                )
                result = json.loads(json_content)
            else:
                # Try parsing the raw response
                result = json.loads(response_content)

            # Handle different response formats
            if isinstance(result, list):
                elements = result
            elif isinstance(result, dict):
                elements = list(result.values())[0] if result else []
            else:
                elements = []

            # Process each detected element
            for i, element in enumerate(elements):
                element_type = element.get("type", "unknown")
                coords_percent = element.get("coordinates", [0, 0, 100, 100])
                caption = element.get("caption", "")

                # Convert percentage coordinates to pixel coordinates
                height, width = img_np.shape[:2]
                x1 = int(width * coords_percent[0] / 100)
                y1 = int(height * coords_percent[1] / 100)
                x2 = int(width * coords_percent[2] / 100)
                y2 = int(height * coords_percent[3] / 100)

                # Extract the element
                element_img = img_np[y1:y2, x1:x2]
                element_path = os.path.join(
                    elements_folder, f"page_{page_num+1}_{element_type}_{i+1}.png"
                )
                cv2.imwrite(element_path, element_img)

                # Draw bounding box on annotated image
                color = (0, 255, 0)  # Green for all elements
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    img_annotated,
                    element_type,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

                # Store element data
                visual_elements.append(
                    {
                        "page": page_num + 1,
                        "type": element_type,
                        "coordinates": [x1, y1, x2, y2],
                        "caption": caption,
                        "path": element_path,
                    }
                )

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing page {page_num+1}: {e}")
            continue

        # Save annotated image
        annotated_path = os.path.join(
            annotated_folder, f"page_{page_num+1}_annotated.png"
        )
        cv2.imwrite(annotated_path, img_annotated)

    # Save metadata
    metadata_path = os.path.join(output_folder, "visual_elements_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(visual_elements, f, indent=2)

    print(
        f"Extracted {len(visual_elements)} visual elements. Metadata saved to {metadata_path}."
    )
    return visual_elements


def extract_top_portion_of_first_page(
    pdf_path,
    output_path=None,
    top_percentage=0,
    bottom_percentage=50,
    compression_quality=100,
    max_size_bytes=1024 * 1024,
):
    """
    Extract a portion of the first page of a PDF and save it as an image.

    Args:
        pdf_path: Path to the PDF file
        output_path: Path where the output image will be saved
        top_percentage: Percentage from the top where to start the cut (0-100)
        bottom_percentage: Percentage from the top where to end the cut (0-100)
        compression_quality: Initial compression quality (0-100)
        max_size_bytes: Maximum allowed file size in bytes (default: 1MB)

    Returns:
        Path to the saved image
    """
    if not 0 <= top_percentage <= 100:
        raise ValueError("top_percentage must be between 0 and 100")
    if not 0 <= bottom_percentage <= 100:
        raise ValueError("bottom_percentage must be between 0 and 100")
    if top_percentage >= bottom_percentage:
        raise ValueError("top_percentage must be less than bottom_percentage")

    # Convert first page of PDF to image
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    if not images:
        raise ValueError(f"Failed to extract first page from {pdf_path}")

    first_page = images[0]
    img_np = np.array(first_page)

    # Calculate the region to keep
    height = img_np.shape[0]
    start_y = int(height * (top_percentage / 100))
    end_y = int(height * (bottom_percentage / 100))

    # Cut the image
    cropped_img = img_np[start_y:end_y, :]

    # Initialize scaling factor and quality
    scale = 1.0
    quality = compression_quality

    while True:
        # Scale image if needed
        if scale != 1.0:
            new_size = (
                int(cropped_img.shape[1] * scale),
                int(cropped_img.shape[0] * scale),
            )
            resized_img = cv2.resize(
                cropped_img, new_size, interpolation=cv2.INTER_AREA
            )
        else:
            resized_img = cropped_img

        # Encode image
        success, img_bytes = cv2.imencode(
            ".png", resized_img, [cv2.IMWRITE_PNG_COMPRESSION, quality]
        )
        if not success:
            raise ValueError("Failed to encode image")

        # Check size
        if len(img_bytes.tobytes()) <= max_size_bytes:
            break

        # Adjust parameters if file is too large
        if quality > 10:
            quality -= 10
        else:
            scale *= 0.8

        if scale < 0.1:  # Prevent infinite loop
            raise ValueError(
                "Could not achieve target file size even with maximum compression"
            )

    # Save the final image if output path is provided
    if output_path is not None:
        cv2.imwrite(output_path, resized_img, [cv2.IMWRITE_PNG_COMPRESSION, quality])

    # Convert to base64 string
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return img_base64


def identify_top_blank_space(pdf_path, threshold=245):
    """
    Identify the percentage of blank space at the very top of the first page of a PDF.

    Args:
        pdf_path: Path to the PDF file
        threshold: Pixel value threshold to consider as blank (0-255, higher means stricter)
                  Default is 245, meaning pixels with values above 245 are considered blank

    Returns:
        Float: Percentage of the page height that is blank at the top (0-100)
    """
    # Convert first page of PDF to image
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    if not images:
        raise ValueError(f"Failed to extract first page from {pdf_path}")

    first_page = images[0]
    img_np = np.array(first_page)

    # Convert to grayscale if it's a color image
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np

    height, width = gray.shape

    # Scan from top to bottom to find first non-blank row
    first_content_row = 0
    for y in range(height):
        # Check if row has content (any pixel below threshold)
        row = gray[y, :]
        if np.any(row < threshold):
            first_content_row = y
            break

    # Calculate percentage of blank space
    blank_percentage = (first_content_row / height) * 100

    return blank_percentage


def create_pdf_thumbnail(
    pdf_path, output_path, prct_range=30, max_size_bytes=1024 * 1024
):
    """
    Generate a thumbnail of the first page of a PDF.

    Args:
        pdf_path: Path to the PDF file
        output_path: Path where the output image will be saved
        prct_range: Percentage range to consider for the thumbnail (0-100)
                    Default is 30, meaning the thumbnail will be 30% of the page height

    Returns:
        Path to the saved image
    """

    blank_percentage = identify_top_blank_space(pdf_path)
    img_bytes = extract_top_portion_of_first_page(
        pdf_path,
        output_path,
        blank_percentage - 1,
        blank_percentage + prct_range,
        max_size_bytes=max_size_bytes,
    )
    return img_bytes


def bytes_to_pil_image(img_bytes_str):
    """
    Convert image bytes to a PIL Image object.

    Args:
        img_bytes_str: Base64 encoded string of an image

    Returns:
        PIL.Image: The converted PIL Image object
    """

    img_bytes = base64.b64decode(img_bytes_str)
    return Image.open(BytesIO(img_bytes))


def compress_pdf(
    input_path, output_path=None, compression_level=2, remove_images=False
):
    """
    Compress a PDF file using PyMuPDF.

    Args:
        input_path: Path to the input PDF file
        output_path: Path where the compressed PDF will be saved. If None, will overwrite the input file.
        compression_level: Level of compression (0-4)
            0: no compression
            1: basic compression
            2: compress streams and images (default)
            3: compress streams + clean content streams
            4: maximum compression (may take longer)
        remove_images: If True, removes all images from the PDF instead of compressing them

    Returns:
        tuple: (output_path, compression_ratio)
    """
    if not 0 <= compression_level <= 4:
        raise ValueError("compression_level must be between 0 and 4")

    # If input and output paths are the same, create a temporary file
    if output_path == input_path:
        temp_output = f"{input_path}.temp"
    else:
        temp_output = output_path or input_path

    # Get original file size
    original_size = os.path.getsize(input_path)

    # Open the PDF
    doc = fitz.open(input_path)

    if remove_images:
        # Remove all images from the PDF
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()

            for img_idx, img in enumerate(image_list):
                xref = img[0]
                try:
                    page.delete_image(xref)
                except Exception as e:
                    print(
                        f"Warning: Could not remove image {img_idx} on page {page_num + 1}: {str(e)}"
                    )
                    continue
    elif compression_level >= 2:
        # Compress images
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()

            for img_idx, img in enumerate(image_list):
                xref = img[0]

                try:
                    # Get image info
                    base_image = doc.extract_image(xref)
                    if base_image is None:
                        print(
                            f"Warning: Could not extract image {img_idx} on page {page_num + 1}"
                        )
                        continue

                    # Create pixmap from image data
                    img_bytes = base_image["image"]
                    pix = fitz.Pixmap(img_bytes)

                    # If pixmap has alpha channel, remove it
                    if pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    # Skip if image is invalid
                    if pix.width < 1 or pix.height < 1:
                        print(
                            f"Warning: Invalid image dimensions for image {img_idx} on page {page_num + 1}"
                        )
                        continue

                    # Reduce image quality based on compression level
                    quality = max(30, 90 - (compression_level * 15))

                    try:
                        # Convert to PIL Image
                        img = Image.frombytes(
                            "RGB", [pix.width, pix.height], pix.samples
                        )

                        # Resize if image is too large
                        max_size = 1500
                        if max(img.size) > max_size:
                            ratio = max_size / max(img.size)
                            new_size = tuple(int(dim * ratio) for dim in img.size)
                            img = img.resize(new_size, Image.Resampling.LANCZOS)

                        # Compress image
                        output = BytesIO()
                        img.save(output, format="JPEG", quality=quality, optimize=True)
                        img_bytes = output.getvalue()

                        # Replace the image in the PDF
                        page.delete_image(xref)
                        page.insert_image(
                            page.rect, stream=img_bytes, keep_proportion=True
                        )

                    except Exception as e:
                        print(
                            f"Warning: Error processing image {img_idx} on page {page_num + 1}: {str(e)}"
                        )
                        continue

                except Exception as e:
                    print(
                        f"Warning: Error extracting image {img_idx} on page {page_num + 1}: {str(e)}"
                    )
                    continue

    if compression_level >= 3:
        # Clean content streams
        for page in doc:
            page.clean_contents()

    if compression_level >= 4:
        # Maximum compression - additional optimizations
        doc.set_metadata({})  # Remove metadata
        for page in doc:
            page.clean_contents()
            page.wrap_contents()

    # Save the compressed PDF with appropriate options
    doc.save(
        temp_output,
        garbage=compression_level > 0,
        clean=compression_level > 2,
        deflate=True,
        deflate_images=True,
        deflate_fonts=True,
        pretty=False,
    )
    doc.close()

    # If we used a temporary file, replace the original
    if temp_output != output_path:
        os.replace(temp_output, output_path)

    # Calculate compression ratio
    compressed_size = os.path.getsize(output_path)
    compression_ratio = (compressed_size / original_size) * 100

    return output_path, compression_ratio


if __name__ == "__main__":
    PROJECT_DIR = "/Users/simon/workspace/projects/ml_papers_hub"
    DATA_DIR = os.path.join(PROJECT_DIR, "data", "papers")
    paper_paths = [
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pdf")
    ]

    # Example Usage:
    paper_path = paper_paths[0]
    OUTPUT_DIR = os.path.join(
        PROJECT_DIR,
        "data",
        "papers",
        f"{os.path.basename(paper_path).rsplit('.', 1)[0]}_outputs",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # extract_images_from_pdf(paper_path, OUTPUT_DIR)
    # extract_tables_with_bounding_boxes(paper_path, OUTPUT_DIR)
    # extract_captions_from_pdf(paper_path)

    # Add this to test the new function
    # visual_elements = extract_visual_elements_with_gpt4v(paper_path, os.path.join(OUTPUT_DIR, "visual_elements"))
    # print(visual_elements)

    max_size_bytes = 100 * 1024  # 500 kb
    img_bytes = create_pdf_thumbnail(
        paper_path,
        os.path.join(OUTPUT_DIR, "thumbnail.png"),
        30,
        max_size_bytes=max_size_bytes,
    )
    img = bytes_to_pil_image(img_bytes)
    print(
        f"Image size: {img.size[0] * img.size[1] * 3 / 1024:.2f} KB"
    )  # Assuming RGB image (3 bytes per pixel)
    img.show()

    # compressed_pdf_path = os.path.join(OUTPUT_DIR, "compressed.pdf")
    # output_path, compression_ratio = compress_pdf(paper_path, compressed_pdf_path, remove_images=True)
    # print(f"Compressed PDF saved to {output_path}")
    # print(f"Compression ratio (comp. size / orig. size): {compression_ratio}%")
    # print(f"Original size: {os.path.getsize(paper_path)} bytes")
    # print(f"Compressed size: {os.path.getsize(output_path)} bytes")
