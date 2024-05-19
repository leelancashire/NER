import fitz  # PyMuPDF. See https://pymupdf.readthedocs.io/en/latest/tutorial.html
import os

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF file.
    """
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def save_file(file, upload_folder):
    """
    Save an uploaded file to the specified folder.

    Args:
        file: Uploaded file.
        upload_folder (str): Directory to save the file.

    Returns:
        str: Path to the saved file.
    """
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    return file_path
