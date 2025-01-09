from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import spacy
from sklearn.cluster import DBSCAN

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

nlp = spacy.load("en_core_web_sm")  # Pour l'anglais

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def mean_pooling(model_output, attention_mask):
    """
    Applique la méthode de mean pooling pour obtenir une représentation de la phrase.
    Args:
        model_output: Sortie du modèle Transformer (outputs.last_hidden_state).
        attention_mask: Masque d'attention indiquant les tokens non masqués (valides).
    Returns:
        np.ndarray: Embedding moyen de la phrase.
    """
    # Multiplier les embeddings par le masque d'attention pour ignorer les tokens de padding
    token_embeddings = model_output.last_hidden_state  # (batch_size, num_tokens, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Somme pondérée par le masque d'attention
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
    
    # Normaliser par le nombre de tokens valides (non masqués)
    sum_mask = input_mask_expanded.sum(dim=1)
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings.detach().numpy()

def get_sentence_embedding(sentence):
    """
    Génère l'embedding d'une phrase en utilisant mean pooling.
    Args:
        sentence (str): La phrase à transformer en embedding.
    Returns:
        np.ndarray: Embedding de la phrase.
    """
    # Encoder la phrase avec le tokenizer
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    # Appliquer la méthode de mean pooling
    mean_embedding = mean_pooling(outputs, inputs['attention_mask'])
    
    return mean_embedding[0]  # Retourne l'embedding pour la première (et unique) phrase

def segment_text(text, eps=0.7, min_samples=1):
    """
    Segmente un texte en groupes sémantiques basés sur les similarités cosines.
    Args:
        text (str): Texte à segmenter.
        eps (float): Seuil pour le clustering DBSCAN.
        min_samples (int): Minimum de phrases par cluster.
    Returns:
        list: Liste des segments de texte.
    """
    # 1. Segmenter en phrases
    sentences = split_into_sentences(text)
    
    # 2. Obtenir les embeddings des phrases
    embeddings = [get_sentence_embedding(sentence) for sentence in sentences]
    
    # 3. Clustering avec DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    labels = clustering.labels_
    
    # 4. Grouper les phrases par cluster
    segments = []
    for cluster_id in set(labels):
        if cluster_id == -1:  # Ignorer les outliers
            continue
        cluster_sentences = [sentences[i] for i in range(len(sentences)) if labels[i] == cluster_id]
        segments.append(" ".join(cluster_sentences))
    
    return segments

# # Exemple d'utilisation
# text = """
# Paris is the capital of France. It is known for the Eiffel Tower. The city attracts millions of tourists each year.
# In 2023, Paris hosted the Olympic Games. The event was a great success.
# Meanwhile, the economy of France continues to grow steadily.
# """

# segments = segment_text(text)
# for i, segment in enumerate(segments):
#     print(f"Segment {i+1}: {segment}")

# from sklearn.metrics.pairwise import cosine_distances
# import matplotlib.pyplot as plt

# # Calcul des distances cosinus entre toutes les phrases
# sentences = split_into_sentences(text)
# embeddings = [get_sentence_embedding(sentence) for sentence in sentences]
# distances = cosine_distances(embeddings)
# plt.hist(distances.flatten(), bins=50)
# plt.title("Distribution des distances cosinus")
# plt.show()

# # Extraire uniquement les distances uniques (valeurs hors diagonale)
# distance_values = distances[np.triu_indices_from(distances, k=1)]

# eps = np.percentile(distance_values, 25)
# print(f"Valeur d'epsilon (précise) : {eps}")

# # Trier les distances
# sorted_distances = np.sort(distance_values)

# # Tracer la courbe des distances triées
# plt.plot(sorted_distances)
# plt.title("Progression des distances triées")
# plt.xlabel("Paires de phrases")
# plt.ylabel("Distance cosinus")
# plt.show()