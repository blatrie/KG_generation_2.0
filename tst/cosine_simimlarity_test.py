from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

word1 = "Adam likes animals"
word2 = "Adam has a dog named Milou"

# Obtenir les embeddings des mots
embedding1 = model.encode(word1, convert_to_tensor=True)
embedding2 = model.encode(word2, convert_to_tensor=True)

# Calcul de la similarité cosinus
similarity = cosine_similarity([embedding1.cpu().numpy()], [embedding2.cpu().numpy()])[0][0]

print(f"La similarité cosinus entre '{word1}' et '{word2}' est : {similarity:.4f}")
