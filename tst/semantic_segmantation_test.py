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

# Exemple d'utilisation
texte = """
Jannik Sinner is rapidly establishing himself as one of the brightest stars in professional tennis, captivating the attention of fans and experts alike with his exceptional skill, composure, and tireless dedication to the sport. Born on August 16, 2001, in the picturesque town of San Candido in northern Italy, Sinner grew up in a region more renowned for winter sports than tennis. His early years were spent honing his athletic prowess as a competitive skier, a background that has profoundly shaped his physical and mental attributes on the tennis court. The balance, agility, and precise coordination demanded by skiing are evident in Sinner's movement, allowing him to glide across the court with an ease that belies the intensity of his game.

Despite entering the tennis scene relatively late compared to some of his peers, Sinner’s meteoric rise has been nothing short of remarkable. His swift ascent through the ATP rankings has been fueled by a potent combination of natural talent and an unwavering commitment to improvement. Known for his powerful and consistent baseline game, he possesses a formidable forehand and a backhand that can penetrate even the toughest defenses. His ability to adapt his strategy and maintain mental clarity under pressure makes him a challenging opponent for seasoned veterans and fellow rising stars alike.

Sinner's impressive achievements include several ATP titles, which serve as a testament to his growing dominance in the sport. His victories are not just a reflection of his technical skills but also his exceptional work ethic and maturity, traits that are rare for someone so young. Whether competing on hard courts, clay, or grass, Sinner’s versatility ensures that he remains a threat on any surface.

Off the court, Jannik Sinner is known for his humble demeanor and focus on continuous growth, traits that have endeared him to fans worldwide. Many see him as a beacon of the future for tennis, carrying the torch for a new generation of players. As he continues to develop his game and challenge the sport’s elite, Sinner’s journey promises to be one of perseverance, talent, and a relentless pursuit of greatness.
"""

segments = segment_text(texte)
for i, segment in enumerate(segments):
    print(f"Segment {i+1}:\n{segment}\n")
