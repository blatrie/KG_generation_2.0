import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import sys
sys.path.insert(0, '../pipeline')
from semantic_segmentation import segment_text
from text_selection import get_text
from llama import extract_triplets
from translation import detect_and_translate

def process_all_pdfs(directory, output_json):
    """
    Parcourt tous les fichiers PDF dans un répertoire et ajoute leurs triplets dans un fichier JSON.
    Les triplets sont ajoutés au fur et à mesure dans le fichier JSON.
    """
    # Vérifier si le fichier existe déjà, sinon le créer avec une liste vide
    if not os.path.exists(output_json):
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)

    for file_name in os.listdir(directory):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(directory, file_name)
            print(f"Traitement du fichier : {file_path}")
            
            # Appel à `extract_pdf_triples` pour chaque fichier
            text = get_text(file_path)
            segments = segment_text(text, threshold=0.5)

            for segment in segments:
                try:
                    segment_translated = detect_and_translate(segment)
                    triples = extract_triplets(segment_translated)
                    
                    # Charger le contenu actuel du fichier JSON
                    with open(output_json, 'r', encoding='utf-8') as f:
                        current_data = json.load(f)

                    # Ajouter les nouveaux résultats
                    current_data.append({
                        "text": segment_translated,
                        "triplets": triples
                    })

                    # Réécrire dans le fichier JSON
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump(current_data, f, indent=4, ensure_ascii=False)

                except Exception as e:
                    print(f"Erreur lors du traitement d'un segment : {e}")
                    continue

    print(f"Tous les triplets ont été sauvegardés dans : {output_json}")


if __name__ == '__main__':
    output_json_file = "data/all_mrebel_training_data.json"
    directory = "data/articles"
    process_all_pdfs(directory, output_json_file)
    # extract_pdf_triples("data/articles/01-Petit-181-216.pdf", output_json_file)