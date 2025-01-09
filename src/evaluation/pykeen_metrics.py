import torchvision
from pykeen.pipeline import pipeline
import numpy as np
from pykeen.triples import TriplesFactory


# Définissez les triplets d'entraînement et de test
# train_triples = [("Alice", "FRIEND", "Bob"), ("Bob", "WORKS_AT", "Company")]
# test_triples = [("Alice", "FRIEND", "Charlie"), ("Bob", "LIVES_IN", "City")]

# # Exécutez un pipeline avec un modèle, par exemple TransE
# result = pipeline(
#     training=train_triples,
#     testing=test_triples,
#     model="TransE",
#     training_kwargs=dict(num_epochs=5),
# )

# triplets = np.array([
#     ['A', 'knows', 'B'],
#     ['A', 'likes', 'Coffee'],
#     ['B', 'likes', 'Tea'],
# ])

# tf = TriplesFactory.from_labeled_triples(triplets)

# results = pipeline(
#     model='TransE',
#     training=tf,
#     testing=tf,
#     training_loop='slcwa',
#     training_kwargs=dict(num_epochs=5)
# )

# mr_score = results.metric_results.get_metric('mean_rank')
# print(f"Mean Rank (MR): {mr_score:.2f}")

modified_triplets = np.array([
    ['A', 'knows', 'B'],
    ['B', 'knows', 'C'],
    ['C', 'knows', 'A'],
    ['A', 'likes', 'Coffee'],
    ['B', 'likes', 'Tea'],
    ['C', 'likes', 'Juice'],
    ['A', 'visits', 'Paris'],
    ['B', 'visits', 'London'],
    ['C', 'visits', 'Berlin'],
    ['Paris', 'located_in', 'France'],
    ['London', 'located_in', 'England'],
    ['Berlin', 'located_in', 'Germany'],
])

modified_tf = TriplesFactory.from_labeled_triples(modified_triplets)
print(modified_tf)