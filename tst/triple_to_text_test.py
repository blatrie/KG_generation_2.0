from transformers import T5ForConditionalGeneration, T5Tokenizer

# Charger le modèle et le tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def triplet_to_text(triplets):
    try:
        # Préparation du texte d'entrée
        input_text = " ; ".join([f"{sujet} {predicat} {objet}" for (sujet, predicat, objet) in triplets])
        input_text = "Convert to text: " + input_text

        # Encodage
        inputs = tokenizer(input_text, return_tensors="pt")

        # Génération de texte avec des paramètres ajustés
        output = model.generate(inputs.input_ids, max_length=50, num_beams=5)#, early_stopping=True)
        
        # Décodage
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Debugging
        print(f"Texte généré : {generated_text}")
        return generated_text

    except Exception as e:
        print(f"Erreur : {e}")
        return False


# Exemple d'utilisation
triplets = [("Paris", "capitale_de", "France"), ("Albert Einstein", "a_inventé", "la relativité")]
texte = triplet_to_text(triplets)
print(texte)
