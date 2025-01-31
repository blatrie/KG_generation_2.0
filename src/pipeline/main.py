"""
Main module for Knowledge Graph Generation.

This function serves as the entry point for generating a knowledge graph from uploaded PDF files.
It orchestrates the entire process by performing text extraction, translation, semantic segmentation,
knowledge base (KB) construction, triplet lemmatization, and pre-merge operations. The function also
integrates clustering and storing of the extracted knowledge into a graph database. 

Key Steps:
1. **Loading Pre-trained Models**: Loads the required pre-trained models for text processing and fine-tuning, such as the BGE small and MRebel models.
2. **File Upload**: Allows users to upload a directory of PDF files for processing. 
3. **Text Extraction**: For each uploaded PDF, text is extracted using the `get_text` function.
4. **Language Detection and Translation**: The function detects the language of the extracted text and translates it to English if needed using the `detect_and_translate` function.
5. **Semantic Segmentation**: The text is divided into semantic segments using the `segment_text` function to facilitate better knowledge extraction.
6. **Knowledge Base (KB) Construction**: The text segments are processed in batches, generating triplets (subject, relation, object) using the `get_kb` function. Each batch is merged with the existing knowledge base, and relationships are stored.
7. **Lemmatization**: Triplets are lemmatized to ensure that entity names are normalized, which is handled by the `lemmatize_triples` function.
8. **Pre-merge and Clustering**: The extracted triplets are merged and stored in a graph database using the `merge_with_finetuned_model` function and `store_kb_clustering` for clustering triplets and saving them into the database.
9. **Performance Tracking**: The execution time for each major step (translation, segmentation, model inference, etc.) is calculated and displayed for user feedback.

This function is designed to efficiently generate knowledge graphs from large sets of unstructured text, providing tools for natural language processing, clustering, and storing structured information in a graph database.

"""

from text_selection import get_text
from translation import detect_and_translate
from pre_merge import merge_with_finetuned_model, lemmatize_triples, TripletClassifier
from KB_generation import get_kb, KB, store_kb_clustering
from clustering_merge import initial_load
import time
import streamlit as st #  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
import nltk
import torch
from semantic_segmentation import segment_text
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from params import DEVICE

# Téléchargement des ressources NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')

def main() :
    """
    Main function for Knowledge Graph Generation.
    
    This function allows the user to upload a directory containing PDF files,
    extract text from the files, and generate a knowledge graph based on the extracted text.
    The generated graph is then stored and the execution time is displayed.
    """
    # Load the fine-tuned models
    #all_mini_finetuned = torch.load("./models/finetuned_all_mini.pth", weights_only=False) # if fine-tuned all-Mini is used
    bge_small_finetuned = torch.load("./models/finetuned_bge_small.pth", weights_only=False, map_location=DEVICE) # if fine-tuned bge-small is used

    mrebel_finetuned = AutoModelForSeq2SeqLM.from_pretrained("./models/finetuned_mrebel")
    tokenizer_mrebel_finetuned = AutoTokenizer.from_pretrained("./models/finetuned_mrebel")

    batch_size_save = 1000
    st.title("Knowledge Graph Generation")
    files = st.file_uploader("Upload a directory contaning PDF files", accept_multiple_files=True, type="pdf")

    kb = KB()
    if files != [] : 
        with st.status("Generating graph...", expanded=True) as status:
            batch_size = 15000
            clusters = initial_load([])

            for idx, file in enumerate(files):
                model_time = 0
                merge_time = 0
                start_time = time.time()
                st.write("Generating graph for : ", file.name)
                pourcentage_progress_bar = st.progress(0)
                text = get_text(file)

                # Translation in english if text not yet in english
                start_trad = time.time()
                text = detect_and_translate(text)
                end_trad = time.time()
                trad_time = end_trad - start_trad
                for i in range(0, len(text), batch_size):
                    if i+batch_size > len(text) :
                        text_part = text[i:]
                    else :
                        text_part = text[i:i+batch_size]
                    # Semantic segmentation module
                    start_sem = time.time()
                    text_segments = segment_text(text_part)
                    end_sem = time.time()
                    sem_time = end_sem - start_sem
                    for segment in text_segments:
                        if len(segment) > 2000:
                            kb, partial_model_time = get_kb(segment, mrebel_finetuned, tokenizer_mrebel_finetuned, verbose=False, kb=kb, pdf_name=file.name, use_finetuned_mrebel = False, span_length=512, max_length=1024)
                        else:
                            kb, partial_model_time = get_kb(segment, mrebel_finetuned, tokenizer_mrebel_finetuned, verbose=False, kb=kb, pdf_name=file.name, use_finetuned_mrebel = False)
                    # Lemmatization of triplets 
                    start_lem = time.time()      
                    kb.relations = lemmatize_triples(kb.relations)
                    end_lem = time.time()
                    lem_time = end_lem - start_lem

                    # Premerge module
                    start_premerge = time.time()
                    kb.relations = merge_with_finetuned_model(kb.relations, bge_small_finetuned, model_name="bge_small")
                    end_premerge = time.time()
                    premerge_time = end_premerge - start_premerge
                    if i % batch_size_save == 0 :
                        is_stored, partial_merge_time, clusters = store_kb_clustering(kb, clusters)
                        
                        kb = KB()
                    pourcentage_progress_bar.progress(int(i/len(text)*100))
                    model_time += partial_model_time
                    merge_time += partial_merge_time
                        
                is_stored, partial_merge_time, clusters = store_kb_clustering(kb, clusters)

                merge_time += partial_merge_time
                
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"c    ################# Translation Time : {trad_time:.4f} seconds #################")
                print(f"c    ################# Semantic Segmentation Time : {sem_time:.4f} seconds #################")
                print(f"c    ################# Generation Time : {model_time:.4f} seconds #################")
                print(f"c    ################# Lemmatization Time : {lem_time:.4f} seconds #################")
                print(f"c    ################# Premerge Time : {premerge_time:.4f} seconds #################")
                print(f"c    ################# Merge Time : {merge_time:.4f} seconds #################")
                print(f"c    ################# Total Time : {execution_time:.4f} seconds #################")
                print(f"c    #################  {idx} #################")
                pourcentage_progress_bar.progress(int(100))
                st.write(f"Total Time for {file.name}: {execution_time:.4f} seconds.")
                st.write(f"Translation Time for {file.name}: {trad_time:.4f} seconds.")
                st.write(f"Semantic Segmentation Time for {file.name}: {sem_time:.4f} seconds.")
                st.write(f"Model Time for {file.name}: {model_time:.4f} seconds.")
                st.write(f"Lemmatization Time for {file.name}: {lem_time:.4f} seconds.")
                st.write(f"Premerge Time for {file.name}: {premerge_time:.4f} seconds.")
                st.write(f"Merge Time for {file.name}: {merge_time:.4f} seconds.")
     
            st.success(f"graph generated.")
    

if __name__ == "__main__" :
    main()
    
    
