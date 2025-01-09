import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import sys
sys.path.insert(0, '../pipeline')
from semantic_segmentation import segment_text
from text_selection import get_text
from llama import extract_triplets
from translation import detect_and_translate
import spacy

# def process_all_pdfs(directory, output_json):
#     """
#     Parcourt tous les fichiers PDF dans un répertoire et ajoute les chunks dans un fichier JSON,
#     avec comme objectif de savoir si un chunk est exploitable ou non.
#     """
#     # Vérifier si le fichier JSON existe, sinon le créer avec une liste vide
#     if not os.path.exists(output_json):
#         with open(output_json, 'w', encoding='utf-8') as f:
#             json.dump([], f, indent=4, ensure_ascii=False)

#     # Charger le fichier JSON existant une seule fois
#     with open(output_json, 'r', encoding='utf-8') as f:
#         current_data = json.load(f)

#     for file_name in os.listdir(directory):
#         if file_name.endswith('.pdf'):
#             file_path = os.path.join(directory, file_name)
#             print(f"Traitement du fichier : {file_path}")
            
#             # Appel à `get_text` pour extraire le texte du PDF
#             try:
#                 text = get_text(file_path)
#             except Exception as e:
#                 print(f"Erreur lors de l'extraction du texte pour {file_path}: {e}")
#                 continue

#             # Traduction du texte si nécessaire
#             try:
#                 translated_text = detect_and_translate(text)
#             except Exception as e:
#                 print(f"Erreur lors de la traduction pour {file_path}: {e}")
#                 continue

#             # Segmentation du texte
#             segments = segment_text(translated_text, eps=0.4)

#             # Ajouter les segments au JSON
#             for segment in segments:
#                 current_data.append({
#                     "text": segment,
#                     "usable": 0  # À ajuster manuellement
#                 })

#     # Réécrire dans le fichier JSON après avoir traité tous les fichiers
#     try:
#         with open(output_json, 'w', encoding='utf-8') as f:
#             json.dump(current_data, f, indent=4, ensure_ascii=False)
#         print(f"Tous les segments ont été sauvegardés dans : {output_json}")
#     except Exception as e:
#         print(f"Erreur lors de l'écriture du fichier JSON : {e}")


# Charger le modèle spaCy
nlp = spacy.load("en_core_web_sm")  # Utiliser un modèle adapté à la langue du texte

def split_into_sentences(text):
    """
    Segmente le texte en phrases avec spaCy.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def chunk_text_with_overlap(sentences, chunk_size, overlap):
    """
    Découpe les phrases en chunks avec chevauchement.
    Args:
        sentences (list): Liste des phrases à découper.
        chunk_size (int): Taille de chaque chunk.
        overlap (int): Nombre de phrases qui se chevauchent entre les chunks.
    Returns:
        list: Liste des chunks avec chevauchement.
    """
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = sentences[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        if len(chunk) < chunk_size:
            break  # Fin de la segmentation si on atteint la fin du texte
    return chunks

def process_all_pdfs(directory, output_json, chunk_size=4, overlap=1):
    """
    Parcourt tous les fichiers PDF dans un répertoire et ajoute les chunks dans un fichier JSON,
    avec comme objectif de savoir si un chunk est exploitable ou non.
    """
    # Vérifier si le fichier JSON existe, sinon le créer avec une liste vide
    if not os.path.exists(output_json):
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)

    # Charger le fichier JSON existant une seule fois
    with open(output_json, 'r', encoding='utf-8') as f:
        current_data = json.load(f)

    for file_name in os.listdir(directory):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(directory, file_name)
            print(f"Traitement du fichier : {file_path}")
            
            # Appel à `get_text` pour extraire le texte du PDF
            try:
                text = get_text(file_path)
            except Exception as e:
                print(f"Erreur lors de l'extraction du texte pour {file_path}: {e}")
                continue

            # Traduction du texte si nécessaire
            try:
                translated_text = detect_and_translate(text)
            except Exception as e:
                print(f"Erreur lors de la traduction pour {file_path}: {e}")
                continue

            # Segmentation du texte en phrases
            sentences = split_into_sentences(translated_text)

            # Appliquer la segmentation avec chevauchement
            segments = chunk_text_with_overlap(sentences, chunk_size, overlap)

            # Ajouter les segments au JSON
            for segment in segments:
                current_data.append({
                    "text": segment,
                    "usable": 0  # À ajuster manuellement
                })

    # Réécrire dans le fichier JSON après avoir traité tous les fichiers
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=4, ensure_ascii=False)
        print(f"Tous les segments ont été sauvegardés dans : {output_json}")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier JSON : {e}")



if __name__ == '__main__':
    output_json_file = "data/training_data_usable_text.json"
    directory = "data/articles"
    process_all_pdfs(directory, output_json_file)
