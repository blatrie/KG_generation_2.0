from SPARQLWrapper import SPARQLWrapper, JSON
import json

# Configuration de l'endpoint SPARQL de DBpedia
sparql = SPARQLWrapper("http://dbpedia.org/sparql")

# Fonction pour exécuter une requête SPARQL
def execute_sparql_query(query):
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]

# Requête SPARQL pour extraire des données économiques (pays et PIB nominal)
query = """
SELECT ?country ?gdpNominal
WHERE {
  ?country rdf:type dbo:Country .
  ?country dbo:gdpNominal ?gdpNominal .
}
LIMIT 100
"""

# Exécution de la requête
sparql_results = execute_sparql_query(query)

# Fonction pour convertir les résultats SPARQL en triplets RDF
def convert_to_rdf_triplets(results):
    triplets = []
    for result in results:
        try:
            # Extraction des valeurs
            country = result["country"]["value"].split("/")[-1]  # Extraire le nom du pays
            gdp = result["gdpNominal"]["value"]

            # Création du triplet RDF
            triplets.append({
                "text": f"The GDP of {country} is {gdp} USD.",
                "triplets": [
                    [country, "hasGDP", gdp]
                ]
            })
        except KeyError:
            continue  # Si une donnée manque, on l'ignore
    return triplets

# Conversion des résultats en format JSON pour MRebel
dataset = convert_to_rdf_triplets(sparql_results)

# Sauvegarde des données dans un fichier JSON
output_file = "mrebel_training_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset d'entraînement créé et sauvegardé dans {output_file}")
