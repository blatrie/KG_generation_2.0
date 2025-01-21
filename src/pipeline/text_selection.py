import PyPDF2
import io
from params import PATH_TO_PDF_FILES
import os

def get_text(file):
    """
    Extracts text from a PDF file.

    Args:
        file (str or UploadedFile): The path to a PDF file or an uploaded file.

    Returns:
        str: The extracted text from the PDF file.
    """
    text = ""
    # print(file)
    # Si le fichier est un chemin (string), on l'ouvre en mode binaire
    if isinstance(file, str):
        with open(file, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    
    # Si le fichier est un UploadedFile, on utilise BytesIO pour lire les données
    else:
        with io.BytesIO(file.read()) as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

    return text

