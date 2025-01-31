"""
This module provides functionality to extract text from PDF files. 
It uses the `PyPDF2` library to read and parse PDF files, allowing text extraction from each page.
"""

import PyPDF2
import io

def get_text(file):
    """
    Extracts text from a PDF file.

    Args:
        file (str or UploadedFile): The path to a PDF file or an uploaded file.

    Returns:
        str: The extracted text from the PDF file.
    """
    text = ""
    # If the file is a string, it is opened in binary mode
    if isinstance(file, str):
        with open(file, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    
    # If the file is an UploadedFile, BytesIO is used to read the data.
    else:
        with io.BytesIO(file.read()) as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

    return text

