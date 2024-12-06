from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
import json

# Charger l'ontologie Schema.org
SCHEMA = Namespace("https://schema.org/")

# Initialiser un graphe RDF
knowledge_graph = Graph()
knowledge_graph.bind("schema", SCHEMA)

def validate_triplet_with_ontology(head, relation, tail, ontology=SCHEMA):
    """
    Valide un triplet en s'assurant que la relation et les entités sont conformes à l'ontologie.
    """
    # Dans cet exemple, on suppose que la relation appartient à l'ontologie Schema.org
    relation_uri = ontology[relation]
    
    # Vérifier si la relation existe dans l'ontologie
    if (relation_uri, RDF.type, RDF.Property) in knowledge_graph or (relation_uri, RDF.type, RDFS.Class) in knowledge_graph:
        return True
    else:
        print(f"Relation non trouvée dans l'ontologie : {relation}")
        return False

def add_triplet_to_graph(graph, head, relation, tail, head_type=None, tail_type=None):
    """
    Ajoute un triplet validé au graphe RDF.
    """
    head_uri = URIRef(f"https://example.org/{head.replace(' ', '_')}")
    tail_uri = URIRef(f"https://example.org/{tail.replace(' ', '_')}") if tail_type else Literal(tail, datatype=XSD.string)
    relation_uri = SCHEMA[relation]
    
    # Ajouter les entités et leur type (si fourni)
    graph.add((head_uri, RDF.type, SCHEMA[head_type] if head_type else RDFS.Resource))
    graph.add((tail_uri, RDF.type, SCHEMA[tail_type] if tail_type else RDFS.Resource))
    
    # Ajouter la relation
    graph.add((head_uri, relation_uri, tail_uri))

# Exemple : textes multilingues et extraction de triplets
texts = [
    {
        "language": "en",
        "text": "Paris is the capital of France.",
        "triplets": [
            {"head": "Paris", "head_type": "City", "type": "isCapitalOf", "tail": "France", "tail_type": "Country"}
        ]
    },
    {
        "language": "fr",
        "text": "Berlin est la capitale de l'Allemagne.",
        "triplets": [
            {"head": "Berlin", "head_type": "City", "type": "isCapitalOf", "tail": "Germany", "tail_type": "Country"}
        ]
    }
]

# Pipeline
for item in texts:
    print(f"Processing text: {item['text']}")

    for triplet in item['triplets']:
        head, relation, tail = triplet['head'], triplet['type'], triplet['tail']
        head_type, tail_type = triplet.get('head_type'), triplet.get('tail_type')
        
        # Validation avec l'ontologie
        if validate_triplet_with_ontology(head, relation, tail):
            # Ajout au graphe RDF
            add_triplet_to_graph(knowledge_graph, head, relation, tail, head_type, tail_type)

# Sauvegarder le graphe RDF dans un fichier Turtle
output_file = "knowledge_graph.ttl"
knowledge_graph.serialize(destination=output_file, format="turtle")
print(f"Graphe sauvegardé dans {output_file}")
