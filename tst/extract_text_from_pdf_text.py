import fitz  # PyMuPDF

def extract_text_from_pdf_fitz(filepath, output_file):
    text = ""
    with fitz.open(filepath) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()  
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)  

extract_text_from_pdf_fitz("pdf_files/Rapport_de_stage_final.pdf", "output_fitz.txt")



import pdfplumber

def extract_text_from_pdf_pdfplumber(filepath, output_file):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text()  
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)  

extract_text_from_pdf_pdfplumber("pdf_files/Rapport_de_stage_final.pdf", "output_pdfplumber.txt")



from PyPDF2 import PdfReader

def extract_text_from_pdf_pypdf2(filepath, output_file):
    text = ""
    with open(filepath, "rb") as file:
        pdf = PdfReader(file)
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text += page.extract_text() 
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text) 

extract_text_from_pdf_pypdf2("pdf_files/Rapport_de_stage_final.pdf", "output_pypdf2.txt")
