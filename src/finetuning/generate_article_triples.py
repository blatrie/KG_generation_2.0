"""
This script processes a single article (PDF file) to extract relational triplets 
and save them in a JSON format. This is useful for generating the training set required for finetuning MRebel (but it's not perfect). 
It performs the following tasks:

- Extracts text from a PDF file.
- Translates the text if necessary.
- Splits the text into sentences.
- Breaks the sentences into random-sized chunks.
- Extracts relational triplets from each chunk using Llama3.1.
- Saves the extracted triplets along with the corresponding text segments into a JSON file.

Usage:
Run this script with a specific PDF file to extract triplets and append them to an existing 
JSON file. If the JSON file does not exist, it will create a new one.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import sys
sys.path.insert(0, '../pipeline')
from semantic_segmentation import split_into_sentences
from text_selection import get_text
from llama import extract_triplets
from translation import detect_and_translate
import random

def chunk_text(sentences):
    """
    Splits the text into chunks of random sizes.

    This function divides a list of sentences into chunks, where each chunk 
    contains 1, 2, or 3 sentences chosen randomly. 

    Args:
        sentences (list): A list of sentences to be split into chunks.

    Returns:
        list: A list of chunks where each chunk is a string of sentences.
    """
    chunks = []
    chunk_size = random.choice([1, 2, 3])  # Randomly choose chunk size (1, 2, or 3 sentences)
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i + chunk_size]  
        i += chunk_size  
        chunks.append(" ".join(chunk))  # Join the sentences into a single string and add it to the chunks list
        if len(chunk) < chunk_size:
            break  # If the chunk is smaller than the selected size, end the process
        chunk_size = random.choice([1, 2, 3])  # Randomly change the chunk size for the next iteration
    return chunks

def process_one_article(file_path, output_json):
    """
    Processes a single PDF file, extracts text, translates if needed, segments it, 
    and extracts triplets for each segment, then stores the results in a JSON file.

    This function performs multiple steps: 
    - Extracting text from a PDF
    - Translating the text if necessary
    - Segmenting the text into chunks of different sizes
    - Extracting triplets from each chunk
    - Writing the results to a JSON file.

    Args:
        file_path (str): The path to the PDF file to process.
        output_json (str): The path to the JSON file where the triplets will be saved.
    """
    if not os.path.exists(output_json):
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)  

    print(f"Processing file: {file_path}")

    # Call `get_text` to extract the text from the PDF
    try:
        text = get_text(file_path)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return

    # Translate the extracted text if necessary
    try:
        translated_text = detect_and_translate(text)
    except Exception as e:
        print(f"Error translating text for {file_path}: {e}")
        return

    # Split the translated text into individual sentences
    sentences = split_into_sentences(translated_text)

    # Apply chunking on the segmented sentences
    segments = chunk_text(sentences)

    # Process each segment and extract triplets
    for segment in segments:
        try:
            # Extract triplets from the current segment (using Llama3.1)
            triples = extract_triplets(segment)
            entry = {
                "text": segment,
                "triplets": triples
            }

            # Write the extracted data to the JSON file
            with open(output_json, 'r+', encoding='utf-8') as f:
                data = json.load(f)  # Load the existing data
                data.append(entry)  # Append the new entry with the segment and triplets
                f.seek(0)  # Move the file pointer back to the start
                json.dump(data, f, indent=4, ensure_ascii=False)  # Write the updated data to the file
        except Exception as e:
            print(f"Error processing segment for {file_path}: {e}")

    print(f"Segments have been processed and saved to: {output_json}")

if __name__ == '__main__':
    output_json_file = "data/mrebel_training_data.json"
    file_path = "data/articles/Ballet-et-Petit-2023_compressed.pdf"  

    # Uncomment the following line to process the article
    #process_one_article(file_path, output_json_file) 
