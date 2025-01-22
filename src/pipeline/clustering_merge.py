from typing import List, Dict, Tuple, Set
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from params import merge_model
fields = ['head', 'type', 'tail']



def is_good_triplet(triplet: dict) -> bool:
    # Destructure the triplet for cleaner access
    head, tail, triplet_type = triplet['head'].replace(".", ""), triplet['tail'].replace(".", ""), triplet['type'].replace(".", "")

    # Check conditions
    is_not_empty = head and tail and triplet_type
    is_not_long = len(head.split()) <= 3 and len(tail.split()) <= 3 and len(triplet_type.split()) <= 2
    is_not_same_tail_head = len(set([head, tail, triplet_type])) == 3

    return is_not_long and is_not_empty and is_not_same_tail_head

def preprocess_triplets_list(triplets: List[dict]) -> List[dict]:
    kb_triplets_unique = [eval(triplet) for triplet in set(map(str, triplets))]
    filtered_triplets = [
        triplet for triplet in kb_triplets_unique
        if is_good_triplet(triplet)
    ]
    return filtered_triplets




def initial_load(triplets: List[dict], model=merge_model) -> Dict[int, List[dict]]:
    if not triplets:
        return {}
    # Step 1: Prepare Triplet Data
    procecessed_triplets = preprocess_triplets_list(triplets)
    print(len(procecessed_triplets))

        
    # Encode the elements for each field
    # embeddings = [merge_model.encode([ trp[fields[0]] + trp[fields[1]] + trp[fields[2]] ]) for trp in triplets]
    embeddings = [model.encode([ trp ]) for trp in triplets]

    # embeddings = merge_model.encode(triplets)
    # Step 3: Cluster Using DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')  # Tune `eps` and `min_samples`
    cluster_labels = dbscan.fit_predict(np.array(embeddings).squeeze(1))

    # Step 4: Assign Clusters
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(triplets[idx])

    return clusters

# Step 5: Intra-Cluster Merging
def merge_within_cluster(cluster_triplets: List[dict], model=merge_model, threshold=0.75) -> List[dict]:
    merged_triplets = []
    while cluster_triplets:
        current = cluster_triplets.pop(0)
        current_embedding = model.encode([" ".join([current[field] for field in fields])])[0]
        # current_embedding = model.encode([ current])[0]

        similar_found = False
        for idx, triplet in enumerate(merged_triplets):
            triplet_embedding = model.encode([" ".join([triplet[field] for field in fields])])[0]
            similarity = cosine_similarity([current_embedding], [triplet_embedding]).flatten()[0]

            if similarity > threshold:
                similar_found = True
                break

        if not similar_found:
            merged_triplets.append(current)

    return merged_triplets 

def merge_within_clusters(clusters: Dict[int, List[dict]], model=merge_model) -> Dict[int, List[dict]]:
    return {label: merge_within_cluster(cluster, model) for label, cluster in clusters.items() if label != -1}



# Step 5: Insertion Logic
def insert_new_triplet(new_triplet: dict, clusters: Dict[int, List[dict]], 
                      model=merge_model, threshold=0.75) -> Tuple[Dict[int, List[dict]], bool]:
    # new_embedding = model.encode([ " ".join([new_triplet[fields[0]], new_triplet[fields[1]], new_triplet[fields[2]]]) ])[0]
    # print("new_triplet", new_triplet)
    new_embedding = model.encode([ new_triplet])[0]
    # print(np.array(new_embedding).shape)
    for cluster_id, cluster_triplets in clusters.items():
        cluster_embeddings = model.encode(cluster_triplets)
        # print(np.array(cluster_embeddings).shape)
        similarities = cosine_similarity([new_embedding], cluster_embeddings).flatten()
        if np.max(similarities) > threshold:
            # Similar triplet found, handle merge or discard
            return clusters, False # not effectively added
    
    # No similar triplet found, create a new cluster
    new_cluster_id = max(clusters.keys()) + 1
    clusters[new_cluster_id] = [new_triplet]
    return clusters, True # effectively added

def batch_merge_triplets(new_triplets: List[dict], clusters: Dict[int, List[dict]], 
                        model=merge_model, threshold=0.75) -> Tuple[Dict[int, List[dict]], List[dict]]:
    # print("cluster before adding", clusters)
    added_triplets = []
    new_triplets = preprocess_triplets_list(new_triplets)
    updated_clusters = clusters
    for new_triplet in new_triplets:
        updated_clusters,  effectively_added = insert_new_triplet(new_triplet, clusters, model, threshold)
        if effectively_added :
            added_triplets.append(new_triplet)
    return merge_within_clusters(updated_clusters), added_triplets
