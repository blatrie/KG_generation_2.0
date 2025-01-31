"""
This script performs the extraction of important keywords from a set of PDF documents, evaluates their relevance based on a pre-trained model, 
and computes a score based on the match between extracted keywords and nodes in a knowledge graph.

Key Steps:
1. Extracts text from PDF files in a given directory.
2. Splits the text into sentences and further chunks them into manageable sections.
3. Evaluates the usability of each chunk using a pre-trained text classifier model.
4. Extracts the top-n important n-grams (keywords) from the usable text chunks using TF-IDF.
5. Computes the similarity between extracted keywords and nodes from a knowledge graph based on cosine similarity of their embeddings.

Usage:
1. Call the `compute_tfidf_score()` function with the path to the directory containing PDFs and a pre-trained model.
2. The function will return the proportion of keywords that match nodes in the knowledge graph.
"""

import sys
sys.path.insert(0, '../pipeline/')
from KB_generation import fetch_all_relations
from text_selection import get_text
from translation import detect_and_translate
from semantic_segmentation import split_into_sentences
from params import DEVICE
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('wordnet')

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
        pooled_output = outputs[1]  
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
    

def extract_text_from_pdfs(directory):
    """
    Extracts text from all PDF files in the given directory.
    
    Args:
        directory (str): Path to the directory containing PDF files.
    
    Returns:
        list of str: List of texts extracted from each PDF.
    """
    texts = []
    pdf_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pdf')]
    
    for file_path in pdf_files:
        text = get_text(file_path)
        texts.append(detect_and_translate(text))
    
    return texts


def chunk_text(sentences, chunk_size=3):
    """
    Splits sentences into chunks.
    
    Args:
        sentences (list): List of sentences to split.
        chunk_size (int): Size of each chunk.
    
    Returns:
        list: List of overlapping chunks.
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
    Combines a list of chunks into a single string.
    
    Args:
        chunks (list): List of chunks (paragraphs).
    
    Returns:
        str: Combined text as a single string.
    """
    return " ".join(chunks)


def is_usable_text(text, model, tokenizer, threshold=0.5, device='cpu'):
    """
    Predicts whether a text is usable based on a threshold.
    
    Args:
        text (str): Text to evaluate.
        model: Pre-trained model.
        tokenizer: Tokenizer associated with the model.
        threshold (float): Threshold to determine if the text is usable.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        
    Returns:
        bool: True if the probability exceeds the threshold, otherwise False.
    """

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
    
    with torch.no_grad():
        logits = model(**inputs)
        prob = torch.sigmoid(logits).item()
    
    return prob > threshold



def get_keywords(directory, n, min_grams=1, n_grams=3, model=None, tokenizer=None, threshold=0.5, device='cpu'):
    """
    Calculates the top-n important n-grams using TF-IDF for a directory of PDF files.
    
    Args:
        directory (str): Path to the directory containing PDF files.
        n (int): Number of top n-grams to return based on the TF-IDF score.
        n_grams (int): The size of the n-grams (e.g., 3 for trigrams).
        model: Pre-trained model used to evaluate text chunks.
        tokenizer: Tokenizer associated with the model.
        threshold (float): Threshold to filter out irrelevant text chunks.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
    
    Returns:
        list of tuples: List of the top-n n-grams and their corresponding TF-IDF scores.
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
    tfidf_matrix = vectorizer.fit_transform(new_texts)  
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    ngram_scores = dict(zip(feature_names, tfidf_scores))
    top_n_ngrams = Counter(ngram_scores).most_common(n)

    return top_n_ngrams



def compute_tfidf_score(directory, n, model, tokenizer, min_grams=1, n_grams=3, usable_threshold=0.49, cos_sim_thresold=0.75, device=DEVICE):
    """
    Computes a score based on the match between extracted keywords and nodes in a knowledge graph,
    using cosine similarity between their embeddings.

    Args:
        directory (str): Directory containing documents for keyword extraction.
        n (int): Number of keywords to keep after filtering.
        min_grams (int, optional): Minimum n-gram length for keyword extraction. Default is 1.
        n_grams (int, optional): Maximum n-gram length for keyword extraction. Default is 3.
        model: Model used for keyword extraction.
        tokenizer: Tokenizer used with the model for keyword extraction.
        usable_threshold (float, optional): Minimum threshold to retain a keyword based on its relevance. Default is 0.49.
        cos_sim_thresold (float, optional): Cosine similarity threshold to consider a keyword related to a node. Default is 0.75.
        device (str, optional): Device for computation (e.g., 'cpu', 'cuda'). Default is 'cpu'.

    Returns:
        float: Proportion of keywords that match at least one node in the knowledge graph.
    """
    relations = fetch_all_relations()
    triplets = np.array([[item['head'], item['relation'], item['tail']] for item in relations])
    
    # n+5 to have some leeway, because it can have ‘science’ and ‘sciences’ for example
    key_words = get_keywords(directory, n+5, min_grams, n_grams, model, tokenizer, usable_threshold, device)
    if len(key_words) == 0:
        print("Aucun mot clé trouvé")
        return

    # guaranteeing uniqueness
    unique_keywords = []
    for i in range(len(key_words)):
        word = lemmatizer.lemmatize(key_words[i][0])
        if word not in unique_keywords:
            unique_keywords.append(word)
    key_words = unique_keywords[:n]

    nodes = []
    for rel in triplets:
        for node in [rel[0], rel[2]]:
            if node not in nodes:
                nodes.append(node)

    nb = 0
    for word in key_words:
        embedding1 = encoder.encode(word, convert_to_tensor=True)
        for node in nodes:
            embedding2 = encoder.encode(node, convert_to_tensor=True)
            similarity = encoder.similarity(embedding1, embedding2)
            if similarity > cos_sim_thresold:
                print(f"word : {word} ----- node : {node}")
                nb += 1
                break
    return nb/len(key_words)

if __name__ == "__main__":
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    lemmatizer = WordNetLemmatizer()
    model_path = "../../models/finetuning_usable_text.pth"
    model_name = "BAAI/bge-small-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the saved model
    model = torch.load(model_path, weights_only=False, map_location=DEVICE)
    model.eval()

    print(compute_tfidf_score("../../articles_KGG/5_articles", 50, model, tokenizer))