"""
This module provides functions to extract, process, and store knowledge from text data (Memgraph is used as graph database). 
The core functionality revolves around identifying relationships (triplets) between entities, such as "subject-relation-object," within a given text. 
It utilizes tokenization, model inference, and clustering techniques to process and organize the extracted knowledge.

Key Functionality:
- **extract_relations_from_model_output**: This function processes the model's output to extract triplets in the form of relationships between entities (subject, relation, and object). 
It parses the output and identifies specific parts of speech and structure to form valid triplets.
- **get_kb**: This function processes input text in spans, uses a pre-trained model to generate triplets, and stores them in a knowledge base (KB). The function also handles fine-tuning and tokenization.
- **store_kb_clustering**: This function stores the extracted knowledge (triplets) into a graph database (Memgraph), ensuring they are organized based on clustering algorithms to identify relationships between different entities.
- **clear_num and clear_str**: These helper functions are used to clean up text data, removing unnecessary characters, and normalizing numbers to ensure the extracted knowledge is formatted properly.

This module is particularly useful for applications in natural language processing (NLP) and knowledge graph generation, where the goal is to automatically extract structured knowledge from unstructured text.
"""

import math
import torch
from neo4j import GraphDatabase
import re
from params import tokenizer, rdf_model, merge_model, DEVICE
import time
from clustering_merge import batch_merge_triplets

# knowledge base class for meta data collection
class KB():
    def __init__(self):
        self.relations = []
        self.pdf_name = ""

    def add_relation(self, r):
        """
        Adds a relation to the knowledge base.
        """
        self.relations.append(r)

    def print(self):
        """
        Prints all the relations stored in the knowledge base.
        """
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")
        

def extract_relations_from_model_output(text):
    """
    Extracts relations in the form of triplets from the output of a model.
    
    Args:
        text (str): The text to process to extract relations as triplets.
    
    Returns:
        list: A list of dictionaries representing the triplets in the form of {'head', 'head_type', 'type', 'tail', 'tail_type'}.
    """
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '','','','',''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__en__", "").split():
        if token == "<triplet>" or token == "<relation>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                relation = ''
            subject = ''
        elif token.startswith("<") and token.endswith(">"):
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                object_ = ''
                subject_type = token[1:-1]
            else:
                current = 'o'
                object_type = token[1:-1]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
    return triplets


def get_kb(text,  rdf_model_get_kb, tokenizer_get_kb, span_length=512, max_length=512, verbose=False, kb=KB(), pdf_name="", use_finetuned_mrebel=False):
    """
    Extracts knowledge base (KB) from a given text by generating relations through model inference. It divides the text into spans and processes each span independently to extract relations.
    
    Args:
        text (str): The text to process for relation extraction.
        rdf_model_get_kb (model): The model used for relation extraction if fine-tuned MRebel is used.
        tokenizer_get_kb (tokenizer): The tokenizer used to process the text if fine-tuned MRebel is used.
        span_length (int): The length of each span to divide the text into.
        max_length (int): The maximum length of input text for the model.
        verbose (bool): Whether to print detailed processing steps.
        kb (KB): The knowledge base object where extracted relations will be stored.
        pdf_name (str): The name of the PDF file, if applicable.
        use_finetuned_mrebel (bool): Whether to use a fine-tuned model.
    
    Returns:
        KB: The knowledge base object populated with extracted relations.
        float: The time taken to process the model inference.
    """
    if use_finetuned_mrebel:
        rdf_model_get_kb = rdf_model_get_kb.to(DEVICE)
        tokenizer_get_kb = tokenizer_get_kb
    else:
        tokenizer_get_kb = tokenizer
        rdf_model_get_kb = rdf_model

    spans_boundaries = []
    tensor_ids = []
    tensor_masks = []
    
    # tokenize whole text
    inputs = tokenizer_get_kb([text], max_length=max_length, padding=True, truncation=True,  return_tensors = 'pt')

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) /
                        max(num_spans - 1, 1))

    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                start + span_length * (i + 1)])
        start -= overlap
    if verbose:
        print(f"Span boundaries are {spans_boundaries}")

    # transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]

    inputs = {
        "input_ids": torch.stack(tensor_ids).to(DEVICE),
        "attention_mask": torch.stack(tensor_masks).to(DEVICE)
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
      "max_length": max_length,
      "length_penalty": 0,
      "num_beams": 3,
      "num_return_sequences": num_return_sequences,
      "forced_bos_token_id": None,
    }
    
    partial_model_time = 0
    start_time = time.time()
    generated_tokens = rdf_model_get_kb.generate(
      inputs["input_ids"].to(rdf_model_get_kb.device),
      attention_mask=inputs["attention_mask"].to(rdf_model_get_kb.device),
      decoder_start_token_id = tokenizer_get_kb.convert_tokens_to_ids("tp_XX"),
      **gen_kwargs,
    )

    del inputs, tensor_ids, tensor_masks
    torch.cuda.empty_cache()

    # decode relations
    decoded_preds = tokenizer_get_kb.batch_decode(generated_tokens, skip_special_tokens=False)
    torch.cuda.empty_cache()

    # extract relations
    partial_model_time = time.time() - start_time
    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                "spans": [spans_boundaries[current_span_index]]
            }
            relation["fname"] = pdf_name
            kb.add_relation(relation)
        i += 1

    return kb, partial_model_time


def clear_num(text):
    """
    Cleans a text by removing or standardizing numeric values.
    
    Args:
        text (str): The input text to clean.
    
    Returns:
        str: The cleaned text with numeric values processed.
    """
    result = []
    for word in text.split(" "):
        try :
            int_val = int(word)
            result.append(str(int_val))
        except ValueError :
            clean_word = [l for l in word if not l.isdigit()]
            if len(clean_word) > 1:
                result.append("".join(clean_word))

    return " ".join(result) 


def clear_str(word):
    """
    Cleans a string by removing unwanted characters and handling repeated words.
    
    Args:
        word (str): The string to clean.
    
    Returns:
        str: The cleaned string with unwanted characters removed.
    """
    # remove all caractere like : ',|- and replace them by space
    word = re.sub(r'[\',\|\-]', ' ', word)

    # if their are repetition of a word like : "the the" we remove the second "the" until there is no more repetition
    while re.search(r'(\w+) \1', word) :
        word = re.sub(r'(\w+) \1', r'\1', word)    
    # if a word is ending with numbers without space like : "the2" we remove the numbers
    word = clear_num(word)
    # delete double space
    word = re.sub(r' +', ' ', word)
    
    return word


def store_kb_clustering(kb, clusters, model = merge_model):
    """
    Stores the knowledge base by associating each triplet with the nearest cluster and saving it to the database.
    
    Args:
        kb (KnowledgeBase): The knowledge base object containing the triplets to store.
        clusters (dict): The current clusters of triplets.
        model (model): The model used to generate embeddings for clustering.
    
    Returns:
        bool: True if the KB was successfully stored, False otherwise.
        float: Time spent on merging the triplets.
        dict: The updated clusters after merging.
    """
    print("c    storing...")
    # Define correct URI and AUTH arguments (no AUTH by default)
    URI = "bolt://localhost:7687"
    AUTH = ("", "")
    updated_clusters = clusters
    with GraphDatabase.driver(URI, auth=AUTH) as client:
        # Check the connection
        client.verify_connectivity()
        print("c    database connection verified.")

        # Initialize variables
        partial_merge_time = 0
        history = []

        # Batch process the triplets
        batch_size = 100  # Adjust as needed for performance
        triplets = kb.relations
        for i in range(0, len(triplets), batch_size):
            # Process a batch of triplets
            batch = triplets[i:i + batch_size]

            # Merge the batch into the nearest clusters
            start_time = time.time()
            updated_clusters, new_triplets = batch_merge_triplets(batch, clusters, model)
            # print()
            # print("new_triplets" , new_triplets)
            # print()
            partial_merge_time += time.time() - start_time

            # Store new triplets in the database
            for triplet in new_triplets:
                head = triplet['head']
                head_type = triplet['head_type']
                relation_type = triplet['type']
                tail = triplet['tail']
                tail_type = triplet['tail_type']
                fname = triplet['fname']

                head = clear_str(head)
                tail = clear_str(tail)
                head_type = clear_str(head_type)
                tail_type = clear_str(tail_type)
                relation_type = clear_str(relation_type)
                fname = clear_str(fname)

                if head != "" and tail != "" and relation_type != "" and head != tail:
                    # Check if head node exists in the database
                    query = f"MATCH (n:`{head_type}`) WHERE n.name = '{head}' RETURN n"
                    with client.session() as session:
                        result = session.run(query)
                        if not result.single():
                            # Create head node
                            query = f"CREATE (n:`{head_type}` {{name: '{head}', fname: '{fname}', head_type: '{head_type}'}})"
                            session.run(query)
                            history.append(query)

                    # Check if tail node exists in the database
                    query = f"MATCH (n:`{tail_type}`) WHERE n.name = '{tail}' RETURN n"
                    with client.session() as session:
                        result = session.run(query)
                        if not result.single():
                            # Create tail node
                            query = f"CREATE (n:`{tail_type}` {{name: '{tail}', fname: '{fname}', tail_type: '{tail_type}'}})"
                            session.run(query)
                            history.append(query)

                    # Check if relation exists in the database
                    query = f"MATCH (n:`{head_type}`)-[r:`{relation_type}`]->(m:`{tail_type}`) WHERE n.name = '{head}' AND m.name = '{tail}' RETURN r"
                    with client.session() as session:
                        result = session.run(query)
                        if not result.single():
                            # Create relation
                            query = f"MATCH (n:`{head_type}`), (m:`{tail_type}`) WHERE n.name = '{head}' AND m.name = '{tail}' CREATE (n)-[r:`{relation_type}`]->(m)"
                            session.run(query)
                            history.append(query)
                else:
                    print("c    Invalid triplet skipped: ", head, relation_type, tail)

        print("c    stored.")

    return True, partial_merge_time, updated_clusters


def fetch_all_relations():
    """
    Fetch all relations from the database and return them as a list of dictionaries.

    Returns:
        list: A list of relations, where each relation is represented as a dictionary.
    """
    URI = "bolt://localhost:7687"
    AUTH = ("", "")  

    relations = []
    with GraphDatabase.driver(URI, auth=AUTH) as client:
        with client.session() as session:
            query = """
            MATCH (n)-[r]->(m)
            RETURN n.name AS head, labels(n) AS head_type, n.fname AS head_fname, 
                   type(r) AS relation, 
                   m.name AS tail, labels(m) AS tail_type, m.fname AS tail_fname
            """
            result = session.run(query)
            for record in result:
                relations.append({
                    "head": record["head"],
                    "head_type": record["head_type"],
                    "head_fname": record["head_fname"],
                    "relation": record["relation"],
                    "tail": record["tail"],
                    "tail_type": record["tail_type"],
                    "tail_fname": record["tail_fname"]
                })

    return relations