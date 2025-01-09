import torch
from transformers import AutoModel

def merge_with_finetuned_all_mini(triples):
    """
    Fusionne les triplets redondants en utilisant un modèle fine-tuné pour détecter les similarités.
    
    Arguments:
    - triples : Liste de triplets, chaque triplet étant un dictionnaire de la forme :
                {head:..., head_type:..., type:..., tail:..., tail_type:...}
    
    Retourne:
    - Une liste de triplets fusionnés sans redondances.
    """
    # Charger le modèle finetuné
    model_path = "../datasets/finetuned_all_mini.pth"
    model = torch.load(model_path)

    # Si un GPU est disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Modèle finetuné chargé avec succès.")
    
    # Charger le tokenizer pour le modèle
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def compute_similarity(triplet1, triplet2):
        """
        Calcule la similarité entre deux triplets en utilisant le modèle fine-tuné.
        Retourne un score entre 0 et 1.
        """
        # Construire les représentations textuelles des triplets
        text1 = f"{triplet1['head']},{triplet1['type']},{triplet1['tail']}"
        text2 = f"{triplet2['head']},{triplet2['type']},{triplet2['tail']}"
        
        # Tokenisation et encodage
        inputs = tokenizer(text1, text2, return_tensors="pt", truncation=True, padding=True).to(device)
        
        # Calculer le score de similarité
        with torch.no_grad():
            outputs = model(**inputs)
            score = sigmoid(outputs.logits)  # Sigmoïde pour obtenir un score entre 0 et 1
        return score.item()
    
    # Fusion des triplets
    merged_triples = []
    for i, triplet in enumerate(triples):
        is_redundant = False
        for existing_triplet in merged_triples:
            similarity_score = compute_similarity(triplet, existing_triplet)
            if similarity_score > 0.8:  # Seuil pour considérer les triplets comme redondants
                is_redundant = True
                break
        if not is_redundant:
            merged_triples.append(triplet)
    
    return merged_triples

# Exemple d'utilisation
triples = [
    {"head": "Paris", "head_type": "City", "type": "is_capital_of", "tail": "France", "tail_type": "Country"},
    {"head": "Paris", "head_type": "City", "type": "is_capital_of", "tail": "France", "tail_type": "Country"},
    {"head": "Berlin", "head_type": "City", "type": "is_capital_of", "tail": "Germany", "tail_type": "Country"},
]

merged_triples = merge_with_finetuned_all_mini(triples)
print("Triplets fusionnés :", merged_triples)
