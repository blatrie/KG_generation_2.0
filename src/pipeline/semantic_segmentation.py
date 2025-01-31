"""
This module provides functions to segment a given text into smaller, more coherent parts based on semantic clustering. 
The process involves checking the quality of sentences, splitting the text into valid sentences, encoding them into embeddings, and applying clustering algorithms to group similar sentences. 
The segmentation process ensures that large texts are divided into smaller, more manageable chunks.

Key components:
1. **is_good_sentence**: A function to evaluate whether a sentence meets certain quality criteria such as length and the number of numbers it contains.
2. **split_into_sentences**: A function to split a given text into individual sentences, filtering out those that do not meet the quality criteria.
3. **segment_text**: The main function that segments the text into meaningful clusters using HDBSCAN (a density-based clustering algorithm) and sentence embeddings.

This module is useful for text preprocessing, particularly when dealing with large chunks of text that need to be divided into meaningful segments for further processing or analysis.
"""

import re
import hdbscan
from params import nlp, merge_model

def is_good_sentence(sentence):
    """
    Checks if a sentence meets certain quality criteria to be considered valid.
    - The sentence must be longer than 15 characters.
    - The sentence should not contain too many numbers (more than 5 by default).
    
    Args:
    - sentence : str : The sentence to evaluate.
    
    Returns:
    - bool : True if the sentence is valid, False otherwise.
    """
    # Criterion 1: The sentence must have more than 15 characters
    if len(sentence) <= 15:
        return False

    # Criterion 2: The sentence must not contain more than 5 numbers
    numbers = re.findall(r'\d+', sentence)  # Find all numbers in the sentence
    if len(numbers) > 5:  # Reject the sentence if there are more than 5 numbers
        return False
    
    return True


def split_into_sentences(text):
    """
    Splits a given text into individual sentences, filtering out those that do not meet the quality criteria.
    
    Args:
    - text : str : The input text to split into sentences.
    
    Returns:
    - list : A list of valid sentences (those that meet the criteria).
    """
    # Process the text with the NLP pipeline to break it into sentences
    doc = nlp(text)
    
    # Filter sentences based on the 'is_good_sentence' criteria
    return [sent.text.strip() for sent in doc.sents if is_good_sentence(sent.text.strip())]


def segment_text(text, min_cluster_size=2):
    """
    Segments the input text into meaningful segments based on sentence embeddings and clustering.
    - First, the text is split into individual sentences.
    - Then, the sentences are encoded into embeddings.
    - Finally, a clustering algorithm (HDBSCAN) is applied to group semantically similar sentences together.

    Args:
    - text : str : The text to segment into clusters.
    - min_cluster_size : int, optional : The minimum number of sentences required for a cluster (default is 2).
    
    Returns:
    - list : A list of text segments (clusters of sentences).
    """
    # Step 1: Split the text into sentences
    sentences = split_into_sentences(text)
    
    # Step 2: Obtain embeddings for the sentences
    embeddings = merge_model.encode(sentences)

    # Step 3: Apply clustering (HDBSCAN) to group similar sentences
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean').fit(embeddings)
    labels = clustering.labels_
    
    # Step 4: Group the sentences by their cluster labels
    segments = []
    for cluster_id in set(labels):
        cluster_sentences = [sentences[i] for i in range(len(sentences)) if labels[i] == cluster_id]
        
        # If the total length of the cluster exceeds a threshold (5000 characters), split it into two segments
        if sum([len(segment) for segment in cluster_sentences]) > 5000:
            segments.append(" ".join(cluster_sentences[:len(cluster_sentences)//2]))
            segments.append(" ".join(cluster_sentences[len(cluster_sentences)//2:]))
        else:
            segments.append(" ".join(cluster_sentences))
    
    return segments
