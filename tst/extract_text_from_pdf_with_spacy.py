import spacy

from spacypdfreader import pdf_reader

nlp = spacy.load("en_core_web_sm")
doc = pdf_reader("tests/data/test_pdf_01.pdf", nlp)

print(doc)