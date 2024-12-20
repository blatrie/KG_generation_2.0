from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
#from llama_index.core.node_parser import SemanticSplitterNodeParser

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def split_into_sentences(text):
    # Utilisation d'une expression régulière pour scinder le texte en phrases
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [sentence.strip() for sentence in sentences if sentence]

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    # Utilisation de la moyenne des embeddings de tous les tokens comme représentation de la phrase
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings[0]

# Fonction pour segmenter le texte
def segment_text(text, threshold=0.5):
    sentences = split_into_sentences(text)
    embeddings = [get_sentence_embedding(sentence) for sentence in sentences]
    
    segments = []
    current_segment = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if similarity < threshold:  # Si la similarité est en dessous du seuil, début d'un nouveau segment
            segments.append(" ".join(current_segment))
            current_segment = [sentences[i]]
        else:
            current_segment.append(sentences[i])
    
    # Ajouter le dernier segment
    if current_segment:
        segments.append(" ".join(current_segment))
    
    return segments