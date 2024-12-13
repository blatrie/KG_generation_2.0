import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import sys
sys.path.insert(0, '../pipeline')
from semantic_segmentation import segment_text
from text_selection import get_text
from llama import extract_triplets
from translation import detect_and_translate

def extract_pdf_triples(file, output_json):
    """
    Extrait les triplets d'un PDF, associe chaque texte segmenté aux triplets,
    et sauvegarde les résultats dans un fichier JSON.
    """
    text = get_text(file)  # Extraction du texte brut
    print(text)
    #text = text[:10000]
    segments = segment_text(text, threshold=0.5)  # Segmentation du texte
    results = []  # Liste pour accumuler les données

    for segment in segments:
        # Traduction si nécessaire
        segment_translated = detect_and_translate(segment)
        
        # Extraction des triplets
        triples = extract_triplets(segment_translated)

        # Formatage de l'entrée pour le fichier JSON
        results.append({
            "text": segment_translated,
            "triplets": [
                {
                    "head": triple['head'],
                    "head_type": triple['head_type'],
                    "type": triple['type'],
                    "tail": triple['tail'],
                    "tail_type": triple['tail_type']
                }
                for triple in triples
            ]
        })

    # Sauvegarde des résultats dans un fichier JSON
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Triplets sauvegardés dans le fichier : {output_json}")
    except Exception as e:
        print(f"Erreur lors de l'écriture dans le fichier JSON : {e}")


def process_all_pdfs(directory, output_json):
    """
    Parcourt tous les fichiers PDF dans un répertoire et ajoute leurs triplets dans un fichier JSON.
    """
    all_results = []

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
                    all_results.append({
                        "text": segment_translated,
                        "triplets": triples
                    })

                except:
                    print("pass")

    # Écriture dans le fichier JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"Tous les triplets ont été sauvegardés dans : {output_json}")


if __name__ == '__main__':
    output_json_file = "data/mrebel_training_data.json"

    directory = "data/articles"
    #process_all_pdfs(directory, output_json_file)

    extract_pdf_triples("data/articles/01-Petit-181-216.pdf", output_json_file)