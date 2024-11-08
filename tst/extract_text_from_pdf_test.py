import fitz  # PyMuPDF

def extract_text_from_pdf(filepath):
    text = ""
    with fitz.open(filepath) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()  # Utilise get_text() pour récupérer tout le texte brut de la page
    return text



# Exemple d'utilisation
texte = extract_text_from_pdf("tst/pdf_files/Rapport_de_stage_final.pdf")
print(texte)

import pdfplumber

def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text()  # Récupère le texte brut de chaque page
    return text

# Exemple d'utilisation
texte = extract_text_from_pdf("tst/pdf_files/Rapport_de_stage_final.pdf")
print(texte)



from PyPDF2 import PdfReader

def extract_text_from_pdf(filepath):
    text = ""
    # Ouvrir le fichier PDF
    with open(filepath, "rb") as file:
        pdf = PdfReader(file)
        # Parcourir chaque page du PDF
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text += page.extract_text()  # Extraire le texte de la page
    return text

# Exemple d'utilisation
texte = extract_text_from_pdf("tst/pdf_files/Rapport_de_stage_final.pdf")
print(texte)