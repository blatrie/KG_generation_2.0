import sys
sys.path.insert(0, '../pipeline/')
sys.path.insert(0, '../finetuning/')
from KB_generation import fetch_all_relations
from text_selection import get_text
from translation import detect_and_translate
from generate_article_triples import split_into_sentences
from KB_generation import fetch_all_relations
import numpy as np
# import PyPDF2
import os
# import io
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import spacy
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
import nltk
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
# relations = fetch_all_relations()
model_path = "./finetuning_usable_text.pth"
# Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "BAAI/bge-small-en-v1.5"  # Le modèle préentraîné utilisé
tokenizer = AutoTokenizer.from_pretrained(model_name)



class UsableTextClassifier(nn.Module):
    def __init__(self, pretrained_model_name):
        super(UsableTextClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dense1 = nn.Linear(self.encoder.config.hidden_size, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 64)
        self.dense4 = nn.Linear(64, 32)
        self.dense5 = nn.Linear(32, 32)
        self.classifier = nn.Linear(32, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Représentation CLS
        x = self.dense1(pooled_output)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dense4(x)
        x = self.relu(x)
        x = self.dense5(x)
        x = self.relu(x)
        logits = self.classifier(x).squeeze(-1)
        return logits
    


# Charger le modèle sauvegardé
model = torch.load(model_path, map_location=device)
model.eval()



def extract_text_from_pdfs(directory):
    """
    Extracts text from all PDF files in the given directory.
    
    Args:
        directory (str): Path to the directory containing PDF files.
    
    Returns:
        list of str: List of texts extracted from each PDF.
    """
    texts = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)  # Crée le chemin complet
        text = get_text(file_path)
        texts.append(detect_and_translate(text))  # Passe le chemin complet à la fonction
    return texts


def chunk_text(sentences, chunk_size=3):
    """
    Découpe les phrases en chunks
    Args:
        sentences (list): Liste des phrases à découper.
        chunk_size (int): Taille de chaque chunk.
    Returns:
        list: Liste des chunks avec chevauchement.
    """
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i+chunk_size]
        i += chunk_size
        chunks.append(" ".join(chunk))
        if len(chunk) < chunk_size:
            break
    return chunks

def combine_chunks(chunks):
    """
    Combine une liste de chunks en une seule chaîne de caractères.
    Args:
        chunks (list): Liste des chunks (paragraphes).
    Returns:
        str: Texte combiné en une seule chaîne.
    """
    return " ".join(chunks)


def is_usable_text(text, model, tokenizer, threshold=0.5, device='cpu'):
    """
    Prédit si un texte est utilisable en fonction d'un seuil.
    
    Args:
        text (str): Texte à évaluer.
        model: Modèle pré-entraîné.
        tokenizer: Tokenizer associé au modèle.
        threshold (float): Seuil pour déterminer si le texte est utilisable.
        device (str): Appareil sur lequel exécuter le modèle (cpu ou cuda).
        
    Returns:
        bool: True si la probabilité est supérieure au seuil, sinon False.
    """
    # Préparer les données d'entrée
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.to(device)
    model.eval()
    
    # Effectuer l'inférence
    with torch.no_grad():
        logits = model(**inputs)
        print(f"Lgits : {logits}")
        prob = torch.sigmoid(logits).item()
    
    # Comparer à threshold
    print(f"prob : {prob}")
    return prob > threshold



def get_keywords(directory, n, min_grams=1, n_grams=3, model=None, tokenizer=None, threshold=0.5, device='cpu'):
    """
    Calculates the top-n important n-grams using TF-IDF for a directory of PDF files.
    
    Args:
        directory (str): Path to the directory containing PDF files.
        n (int): Number of top n-grams to return based on TF-IDF score.
        n_grams (int): The size of the n-grams (e.g., 3 for trigrams).
        model: Modèle pré-entraîné pour évaluer les chunks.
        tokenizer: Tokenizer associé au modèle.
        threshold (float): Seuil pour filtrer les chunks non pertinents.
        device (str): Appareil pour exécuter le modèle (cpu ou cuda).
    
    Returns:
        list of tuples: List of top-n n-grams and their TF-IDF scores.
    """
    texts = extract_text_from_pdfs(directory)
    if not texts:
        return []

    new_texts = []
    for article in texts:
        sentences = split_into_sentences(article)
        chunks = chunk_text(sentences)
        usable_chunks = []
        
        for chunk in chunks:
            if is_usable_text(chunk, model, tokenizer, threshold, device):
                usable_chunks.append(chunk)
        new_texts.append(combine_chunks(usable_chunks))

    if not texts or all(len(text.strip()) == 0 for text in texts):
        raise ValueError("Les textes fournis sont vides ou non valides.")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(min_grams, n_grams))
    tfidf_matrix = vectorizer.fit_transform(new_texts)  # Utiliser new_texts filtrés
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    ngram_scores = dict(zip(feature_names, tfidf_scores))
    top_n_ngrams = Counter(ngram_scores).most_common(n)

    return top_n_ngrams



def compute_tfidf_score(directory, n, min_grams=1, n_grams=3, model=model, tokenizer=tokenizer, usable_threshold=0.49, cos_sim_thresold=0.5, device='cpu'):
    relations = fetch_all_relations()
    triplets = np.array([[item['head'], item['relation'], item['tail']] for item in relations])
    # n+5 pour avoir de la marge, car il peut avoir 'science' et 'sciences' par exemple
    key_words = get_keywords(directory, n+5, min_grams, n_grams, model, tokenizer, usable_threshold, device)

    # garantir l'unicité
    unique_keywords = []
    for i in range(len(key_words)):
        word = lemmatizer.lemmatize(key_words[i][0])
        if word not in unique_keywords:
            unique_keywords.append(word)
    key_words = unique_keywords[:n]

    nodes = []
    for rel in triplets:
        for node in [rel['head'], rel['tail']]:
            word = lemmatizer.lemmatize(node)
            if word not in nodes:
                nodes.append(word)
    nodes = list(set(nodes))
    
    nb = 0
    for word in key_words:
        embedding1 = encoder.encode(word, convert_to_tensor=True)
        for node in nodes:
            embedding2 = encoder.encode(node, convert_to_tensor=True)
            similarity = encoder.similarity(embedding1, embedding2)
            if similarity > cos_sim_thresold:
                nb += 1
                break
    return nb/len(key_words)
    


print("")
print("")
# avec 0.6 de thresold le texte est jugé comme inutilisable
# avec min_grams = 2 on retrouve les www cairn, bordeaux ip, ... (avec thresold=0.5)
# print(get_keywords("../finetuning/data/articles", 10, min_grams=1, n_grams=3, model=model, tokenizer=tokenizer, threshold=0.5, device='cpu'))
# print(f"Le score tf-idf est : {compute_tfidf_score('../finetuning/data/articles', 10, n_grams=3, model=model, tokenizer=tokenizer)}")
# text = "www.bordeaux-inp.fr, Michael Jordan Smith and M. Adams for Get the rid of fire. www.apocalyse"
# print(is_usable_text(text, model, tokenizer, threshold=0.5, device='cpu'))