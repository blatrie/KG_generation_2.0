import spacy

from spacypdfreader import pdf_reader

nlp = spacy.load("en_core_web_sm")
doc = pdf_reader("src/finetuning/data/articles/01-Petit-181-216.pdf", nlp)

# print(doc)
# print(doc._.first_page)
print(doc._.page(1))