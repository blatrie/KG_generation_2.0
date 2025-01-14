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
import random

# Charger le modèle spaCy
nlp = spacy.load("en_core_web_sm")  # Utiliser un modèle adapté à la langue du texte

def split_into_sentences(text):
    """
    Segmente le texte en phrases avec spaCy.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def chunk_text_with_overlap(sentences):
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
    chunk_size = random.choice([1, 2, 3])
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i+chunk_size]
        i += chunk_size
        chunks.append(" ".join(chunk))
        if len(chunk) < chunk_size:
            break  # Fin de la segmentation si on atteint la fin du texte
        chunk_size = random.choice([1, 2, 3])
    return chunks

def process_one_article(file_path, output_json):
    """
    Traite un seul fichier PDF et ajoute les triplets extraits dans un fichier JSON.
    Args:
        file_path (str): Chemin du fichier PDF à traiter.
        output_json (str): Chemin du fichier JSON où sauvegarder les triplets.
        chunk_size (int): Taille de chaque chunk.
        overlap (int): Nombre de phrases qui se chevauchent entre les chunks.
    """
    # Vérifier si le fichier JSON existe, sinon le créer avec une liste vide
    if not os.path.exists(output_json):
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)

    print(f"Traitement du fichier : {file_path}")

    # Appel à `get_text` pour extraire le texte du PDF
    try:
        text = get_text(file_path)
    except Exception as e:
        print(f"Erreur lors de l'extraction du texte pour {file_path}: {e}")
        return

    # Traduction du texte si nécessaire
    try:
        translated_text = detect_and_translate(text)
    except Exception as e:
        print(f"Erreur lors de la traduction pour {file_path}: {e}")
        return

    # Segmentation du texte en phrases
    sentences = split_into_sentences(translated_text)

    # Appliquer la segmentation avec chevauchement
    segments = chunk_text_with_overlap(sentences)

    # Ajouter les segments au JSON au fur et à mesure
    for segment in segments:
        try:
            triples = extract_triplets(segment)
            entry = {
                "text": segment,
                "triplets": triples
            }

            # Écrire l'entrée dans le fichier JSON directement
            with open(output_json, 'r+', encoding='utf-8') as f:
                data = json.load(f)  # Charger les données existantes
                data.append(entry)  # Ajouter la nouvelle entrée
                f.seek(0)  # Revenir au début du fichier
                json.dump(data, f, indent=4, ensure_ascii=False)  # Réécrire le fichier
        except Exception as e:
            print(f"Erreur lors de l'ajout d'un segment pour {file_path}: {e}")

    print(f"Les segments ont été traités et sauvegardés dans : {output_json}")

if __name__ == '__main__':
    output_json_file = "data/mrebel_training_data.json"
    file_path = "data/articles/Ballet-et-Petit-2023_compressed.pdf"  # Spécifiez le chemin de votre article

    process_one_article(file_path, output_json_file)
