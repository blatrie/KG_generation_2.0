"""
This script computes knowledge graph embedding metrics using PyKEEN.
It fetches relational triplets, splits them into training and test sets,
trains a TransE model, and evaluates its performance.

Usage:
Run this script to compute and display metrics such as MRR, Hits@10, and Mean Rank.
"""

import sys
sys.path.insert(0, '../pipeline')
from KB_generation import fetch_all_relations
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from sklearn.model_selection import train_test_split

def compute_pykeen_metrics():
    """
    Compute knowledge graph embedding metrics using PyKEEN.
    
    - Fetches relational triplets.
    - Splits data into training and test sets.
    - Trains a TransE model.
    - Evaluates the model and prints key metrics (MRR, Hits@10 and Mean Rank).
    """
    relations = fetch_all_relations()

    # Convert relations to a NumPy array of triplets (head, relation, tail)
    triplets = np.array([[item['head'], item['relation'], item['tail']] for item in relations])

    # Split data into training (60%) and test (40%) sets
    train_triplets, test_triplets = train_test_split(triplets, test_size=0.4, random_state=42)

    # Create TriplesFactory for training and testing
    train_tf = TriplesFactory.from_labeled_triples(train_triplets)
    test_tf = TriplesFactory.from_labeled_triples(test_triplets)

    # Train a TransE model using PyKEEN
    results = pipeline(
        model='TransE',
        training=train_tf,
        testing=test_tf,
        training_loop='slcwa',
        training_kwargs=dict(num_epochs=80)
    )

    # Extract evaluation metrics
    mrr_score = results.metric_results.get_metric('mrr')
    hits_at_10 = results.metric_results.get_metric('hits@10')
    mean_rank = results.metric_results.get_metric('mean_rank')

    # Display results
    print(f"MRR: {mrr_score:.3f}")
    print(f"Hits@10: {hits_at_10:.3f}")
    print(f"Mean Rank: {mean_rank:.2f}")

if __name__ == "__main__":
    compute_pykeen_metrics()
