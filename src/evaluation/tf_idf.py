import sys
sys.path.insert(0, '../pipeline/')
from text_selection import get_text
from translation import detect_and_translate
import PyPDF2
import os
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


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


def get_top_n_ngrams_tfidf(directory, n, n_grams=3):
    """
    Calculates the top-n important n-grams using TF-IDF for a directory of PDF files.
    
    Args:
        directory (str): Path to the directory containing PDF files.
        n (int): Number of top n-grams to return based on TF-IDF score.
        n_grams (int): The size of the n-grams (e.g., 3 for trigrams).
    
    Returns:
        list of tuples: List of top-n n-grams and their TF-IDF scores.
    """
    # Extract text from PDFs
    texts = extract_text_from_pdfs(directory)
    
    # Check if there is any text
    if not texts:
        return []
    
    # Calculate TF-IDF with n-grams
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, n_grams))
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Get feature names (n-grams)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum TF-IDF scores across all documents
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    ngram_scores = dict(zip(feature_names, tfidf_scores))
    
    # Get top-n n-grams
    top_n_ngrams = Counter(ngram_scores).most_common(n)
    
    return top_n_ngrams
print("")
print("")
print("")
print(get_top_n_ngrams_tfidf("../finetuning/data/articles", n=50, n_grams=3))