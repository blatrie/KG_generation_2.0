import sys
sys.path.insert(0, '../pipeline')
from KB_generation import fetch_all_relations
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

relations = fetch_all_relations()

# Définitions des triplets
triplets = np.array([[item['head'], item['relation'], item['tail']] for item in relations])

# Diviser les données
tf = TriplesFactory.from_labeled_triples(triplets)

# Entraîner et tester
results = pipeline(
    model='TransE',
    training=tf,
    testing=tf,
    training_loop='slcwa',
    training_kwargs=dict(num_epochs=10)
)

# Extraction des métriques
mrr_score = results.metric_results.get_metric('mrr')
hits_at_10 = results.metric_results.get_metric('hits@10')
mean_rank = results.metric_results.get_metric('mean_rank')

print(f"MRR: {mrr_score:.3f}")
print(f"Hits@10: {hits_at_10:.3f}")
print(f"Mean Rank: {mean_rank:.2f}")