import spacy
import time

def validate_entities_in_text(triplets, text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    extracted_entities = {ent.text for ent in doc.ents}
    
    validated_triplets = []
    for triplet in triplets:
        if triplet["head"] in extracted_entities and triplet["tail"] in extracted_entities:
            validated_triplets.append(triplet)
    
    coverage = len(validated_triplets) / len(triplets) if triplets else 0
    return validated_triplets, coverage



nlp = spacy.load("en_core_web_sm")

# Obtenir les labels des entités nommées
ner_labels = nlp.get_pipe("ner").labels

print("Classes d'entités nommées (NER) :")
print(ner_labels)
print("------------------------------------------")
print("")


text = "Jannik Sinner is a rising star in professional tennis, known for his exceptional talent, composure, and relentless work ethic. Born on August 16, 2001, in San Candido, Italy, Sinner has quickly climbed the ATP rankings, earning a reputation as one of the sport's most promising young players. His powerful baseline game, combined with remarkable agility and precision, makes him a formidable opponent on any surface. A former skier, Sinner’s athletic background contributes to his strong footwork and mental toughness on the court. With several ATP titles already under his belt, he continues to impress fans and critics alike, embodying the future of tennis with his dedication and skill."

print()
print(text)
print()

start_time = time.time()
nlp = spacy.load("en_core_web_sm") 
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
end_time = time.time()
execution_time = end_time - start_time
print(f"Temps d'exécution (en_core_web_sm) : {execution_time:.5f} secondes")

start_time = time.time()
nlp = spacy.load("en_core_web_md") 
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
end_time = time.time()
execution_time = end_time - start_time
print(f"Temps d'exécution (en_core_web_md) : {execution_time:.5f} secondes")

start_time = time.time()
entities = []
nlp = spacy.load("en_core_web_lg") 
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
    entities.append(ent.text)
end_time = time.time()
execution_time = end_time - start_time
print(f"Temps d'exécution (en_core_web_lg) : {execution_time:.5f} secondes")
print("------------------------------------------")
print("")


from collections import Counter
import re

def extract_keywords(text, n=5):
    # Liste des mots vides (stop words) que vous souhaitez ignorer
    stop_words = set([
    "the", "is", "in", "and", "of", "to", "for", "on", "with", "a", "an", "by", "that", "it", "as", "at", "from", "but", "or", 
    "be", "are", "were", "was", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "will", "just", 
    "don", "nor", "not", "under", "own", "same", "she", "all", "such", "until", "below", "are", "we", "your", "yours", "yourselves", 
    "he", "through", "don’t", "should", "our", "ours", "ourselves", "you", "which", "who", "whom", "this", "that", "these", "those", 
    "am", "isn’t", "wasn’t", "weren’t", "being", "have", "has", "had", "having", "you’ve", "hasn’t", "haven’t", "isn’t", "doesn’t", 
    "didn’t", "wasn’t", "weren’t", "won’t", "wouldn’t", "can", "can’t", "could", "couldn’t", "shouldn’t", "might", "mightn’t", 
    "must", "mustn’t", "needs", "needed", "doing", "each", "every", "few", "less", "least", "more", "most", "other", "another", 
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don’t", 
    "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain’t", "ain", "aren’t", "couldn’t", "didn’t", "doesn’t", "hadn’t",
    "hasn’t", "haven’t", "isn’t", "ma", "mightn’t", "mustn’t", "needn’t", "shan’t", "shouldn’t", "wasn’t", "weren’t", "won’t", 
    "wouldn’t", "alike"
])

    
    # Nettoyer le texte
    words = re.findall(r'\w+', text.lower())
    
    # Filtrer les mots vides
    filtered_words = [word for word in words if word not in stop_words]
    
    # Compter la fréquence des mots restants
    word_counts = Counter(filtered_words)
    
    # Trier les mots par fréquence croissante (prendre les moins fréquents)
    least_frequent_keywords = word_counts.most_common()[-n:]
    
    # Retourner les mots clés avec leurs fréquences
    return least_frequent_keywords

# Exemple
start_time = time.time()
keywords = extract_keywords(text, n=5)
end_time = time.time()
execution_time = end_time - start_time
print(keywords) 
print(f"Temps d'exécution (fréquence simple) : {execution_time:.5f} secondes")

key_parts = []
for w in keywords:
    key_parts.append(w[0])
for e in entities:
    key_parts.append(e)
key_parts = list(set(key_parts))
print(key_parts)
print("------------------------------------------")
print("")


from rake_nltk import Rake
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def extract_keywords_rake(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()  # Liste de phrases clés triées

# Exemple
start_time = time.time()
keywords = extract_keywords_rake(text)
end_time = time.time()
execution_time = end_time - start_time
print("------------------------------------------")
print("")
print(keywords) 
print(f"Temps d'exécution (Rake) : {execution_time:.5f} secondes") 



from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_tfidf(text, n=5):
    # Initialiser TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    # Adapter le texte
    tfidf_matrix = vectorizer.fit_transform([text])
    # Associer les mots à leurs scores
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    # Trier par score
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_scores[:n]]

# Exemple
start_time = time.time()
keywords = extract_keywords_tfidf(text, n=10)
end_time = time.time()
execution_time = end_time - start_time
print("------------------------------------------")
print("")
print(keywords) 
print(f"Temps d'exécution (TF-IDF) : {execution_time:.5f} secondes") 



def extract_keywords_spacy(text, n=5):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # Extraire les noms et adjectifs
    keywords = [token.text for token in doc if token.pos_ in ('NOUN', 'ADJ')]
    return keywords[:n]

# Exemple
start_time = time.time()
keywords = extract_keywords_spacy(text, 10) 
end_time = time.time()
execution_time = end_time - start_time
print("------------------------------------------")
print("")
print(keywords) 
print(f"Temps d'exécution (spacy) : {execution_time:.5f} secondes") 



from keybert import KeyBERT

def extract_keywords_keybert(text, n=5):
    model = KeyBERT()
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=n)
    return [kw[0] for kw in keywords]

# Exemple
start_time = time.time()
keywords = extract_keywords_keybert(text, 10) 
end_time = time.time()
execution_time = end_time - start_time
print("------------------------------------------")
print("")
print(keywords) 
print(f"Temps d'exécution (keybert) : {execution_time:.5f} secondes") 




