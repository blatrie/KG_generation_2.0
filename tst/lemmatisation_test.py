import sys
sys.path.insert(0, '../src/pipeline')
from KB_generation import get_kb
from text_selection import get_text

from langdetect import detect
from deep_translator import GoogleTranslator
import nltk
from nltk.stem import WordNetLemmatizer

import os
import pandas as pd



# Téléchargement des ressources NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')

class KB():
    def __init__(self):
        self.relations = []
        self.pdf_name = ""
        self.lemmatizer = WordNetLemmatizer()
        #self.translator = Translator()

    def detect_and_translate(self, text):
        """
        Détecte la langue de 'text' et le traduit en anglais si nécessaire.
        """
        try:
            lang = detect(text)  # Détection de la langue
            if lang != 'en':  # Si la langue n'est pas l'anglais
                translated = self.translator.translate(text, dest='en')
                return translated.text
            return text
        except:
            print(f"Erreur lors de la détection/traduction de la langue pour le texte : {text}")
            return text  # Retourne le texte original en cas d'erreur

    def lemmatize_relation(self, relation):
        """
        Lemmatiser chaque partie du triplet de relation (sujet, prédicat, objet) après traduction.
        """
        # # Traduction si nécessaire
        # subject = self.detect_and_translate(relation['head'])
        # predicate = self.detect_and_translate(relation['type'])
        # obj = self.detect_and_translate(relation['tail'])

        # Lemmatisation
        subject = self.lemmatizer.lemmatize(relation['head'])
        predicate = self.lemmatizer.lemmatize(relation['type'])
        obj = self.lemmatizer.lemmatize(relation['tail'])

        # Retourner la relation transformée
        return {
            'head': subject,
            'head_type': relation['head_type'],
            'type': predicate,
            'tail': obj,
            'tail_type': relation['tail_type'],
            'fname': relation['fname']
        }

    def add_relation(self, r):
        # Traduire et lemmatiser la relation avant de l'ajouter
        lemmatized_relation = self.lemmatize_relation(r)
        self.relations.append(lemmatized_relation)

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")

if __name__ == "__main__" :
    # directory_path = "./pdf_files"
    # files = [f for f in os.listdir(directory_path)]

    # kb = KB()
    # for idx, file in enumerate(files):
    #     text = get_text(f"{directory_path}/{file}")
    #     batch_size = 15000
    #     for i in range(0, len(text), batch_size):
    #         if i+batch_size > len(text) :
    #             text_part = text[i:]
    #         else :
    #             text_part = text[i:i+batch_size]
    #         kb, partial_model_time = get_kb(text_part, verbose=False, kb=kb)

    # kb.print()



    # # # Convertir en DataFrame
    # df = pd.DataFrame(kb.relations)

    # # Supprimer les doublons
    # df.drop_duplicates(subset=['head', 'type', 'tail'], inplace=True)

    # # Filtrer selon des critères (ex. supprimer les auto-références)
    # df = df[df['head'] != df['tail']]

    # # Afficher les relations finales
    # final_relations = df.to_dict(orient='records')
    # print(final_relations)

    # text = "université"
    # lang = detect(text)
    # print(f"Langue détectée : {lang}")

    # # Si la langue n'est pas l'anglais, on effectue la traduction
    # if lang != 'en':  
    #     translated = GoogleTranslator(source='auto', target='en').translate(text)
    #     print(f"Texte traduit : {translated}")

    lemmatizer = WordNetLemmatizer()
    print("running :", lemmatizer.lemmatize("running", pos="v"))

    print("rocks :", lemmatizer.lemmatize("rocks"))
    print("corpora :", lemmatizer.lemmatize("corpora"))
 
    # a denotes adjective in "pos"
    print("better :", lemmatizer.lemmatize("better", pos="a"))

    print("part of :", lemmatizer.lemmatize("part of", pos="v"))

    text = "maison"
    translated = GoogleTranslator(source='fr', target='en').translate(text)
    print(f"Texte traduit : {translated}")