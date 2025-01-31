"""
This module defines a neural network-based model and functions for processing and merging triplets in a knowledge graph. It's a premerge module, before storing triplets in Memgraph.
The primary model, `TripletClassifier`, uses a pretrained transformer-based model to encode triplet information and classify their relevance. 
Additionally, there are utility functions for lemmatizing triplet values and merging redundant triplets based on their similarity.

Key components:
1. **TripletClassifier**: A PyTorch-based classifier that fine-tunes a pretrained transformer model to classify triplets and identify meaningful relationships. It passes the encoded triplets through multiple dense layers before applying a final classification layer.
2. **lemmatize_triples**: A utility function that lemmatizes the `head`, `type`, and `tail` values of each triplet using the WordNet Lemmatizer.
3. **merge_with_finetuned_model**: This function merges redundant triplets based on their semantic similarity by leveraging a fine-tuned model to detect similarities between triplets.

This module is ideal for knowledge graph creation (premerge module) and optimization tasks, such as filtering out redundant information and making the data more meaningful by processing the triplets through advanced machine learning techniques.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from params import DEVICE
import torch.nn as nn
from nltk.stem import WordNetLemmatizer

class TripletClassifier(nn.Module):
    def __init__(self, pretrained_model_name):
        """
        A neural network model that classifies triplets by encoding them with a pretrained transformer model.
        The model consists of a transformer encoder followed by several dense layers and a final classification layer.
        
        Args:
        - pretrained_model_name : str : The name of the pretrained model to be used for encoding the triplets.
        """
        super(TripletClassifier, self).__init__()
        # Load pretrained model
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

        # Define intermediate dense layers and activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dense1 = nn.Linear(self.encoder.config.hidden_size, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 64)
        self.dense4 = nn.Linear(64, 32)
        self.dense5 = nn.Linear(32, 32)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Final classification layer
        self.classifier = nn.Linear(32, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass through the model.
        
        Args:
        - input_ids : torch.Tensor : Tensor of token IDs.
        - attention_mask : torch.Tensor : Tensor of attention masks.
        - token_type_ids : torch.Tensor, optional : Tensor of token type IDs.
        
        Returns:
        - logits : torch.Tensor : The output classification logits.
        """
        # Pass through the transformer encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token representation

        # Pass through the dense layers with ReLU activations
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

        # Final classification
        logits = self.classifier(x)
        logits = logits.squeeze(-1)  # Squeeze to get a 1D vector
        
        return logits

def lemmatize_triples(triples):
    """
    Lemmatize the 'head', 'type', and 'tail' elements of each triplet.
    
    Args:
    - triples : list of dicts : Each dictionary represents a triplet with 'head', 'type', and 'tail' keys.
    
    Returns:
    - list of dicts : The lemmatized triplets.
    """
    lemmatizer = WordNetLemmatizer()

    # Lemmatizing each component of the triplets
    for relation in triples:
        relation['head'] = lemmatizer.lemmatize(relation['head'])
        relation['type'] = lemmatizer.lemmatize(relation['type'])
        relation['tail'] = lemmatizer.lemmatize(relation['tail'])

    return triples

def merge_with_finetuned_model(triples, model, model_name="all_mini"):
    """
    Merge redundant triplets based on their semantic similarity using a fine-tuned model.
    
    Args:
    - triples : list of dicts : Each dictionary represents a triplet to be compared.
    - model : nn.Module : The model used to compute semantic similarity.
    - model_name : str, optional : The model name to select the corresponding tokenizer.
    
    Returns:
    - list of dicts : The merged triplets, excluding redundant ones.
    """
    # Move the model to the appropriate device
    model.to(DEVICE)
    
    # Select the appropriate tokenizer based on the model
    if model_name == "all_mini":
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    elif model_name == "bge_small":
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model.eval()
    print("Fine-tuned model and tokenizer loaded successfully.")
    
    def compute_similarity(triplet1, triplet2):
        """
        Compute the semantic similarity between two triplets using the fine-tuned model.
        
        Args:
        - triplet1 : dict : The first triplet.
        - triplet2 : dict : The second triplet.
        
        Returns:
        - score : float : Similarity score between 0 and 1.
        """
        # Represent the triplets as text
        text1 = f"{triplet1['head']},{triplet1['type']},{triplet1['tail']}"
        text2 = f"{triplet2['head']},{triplet2['type']},{triplet2['tail']}"
        
        # Tokenize and encode the text
        inputs = tokenizer(text1, text2, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        
        # Compute the similarity score
        with torch.no_grad():
            outputs = model(**inputs)
            score = torch.sigmoid(outputs)  # Apply sigmoid to get a similarity score between 0 and 1
        return score.item()
    
    # Merge the triplets based on similarity threshold
    merged_triples = []
    for _, triplet in enumerate(triples):
        is_redundant = False
        for existing_triplet in merged_triples:
            similarity_score = compute_similarity(triplet, existing_triplet)
            if similarity_score > 0.8:  # Threshold for considering triplets as redundant
                is_redundant = True
                break
        if not is_redundant:
            merged_triples.append(triplet)
    
    return merged_triples


