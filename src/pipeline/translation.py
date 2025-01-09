from langdetect import detect
from deep_translator import GoogleTranslator

def detect_and_translate(text, max_length=4000):
    """
    Détecte la langue de 'text' et le traduit en anglais si nécessaire.
    Si le texte est trop long, il est divisé en morceaux plus petits pour éviter les erreurs.
    """
    try:
        lang = detect(text[:max_length]) if len(text) > max_length else detect(text)  # Détection de la langue

        if lang != 'en':  # Si la langue n'est pas l'anglais
            translated_text = []

            # Diviser le texte en morceaux de taille max_length
            for i in range(0, len(text), max_length):
                chunk = text[i:i+max_length]
                translated_chunk = GoogleTranslator(source=lang, target='en').translate(chunk)
                translated_text.append(translated_chunk)

            # Joindre les morceaux traduits pour obtenir le texte complet traduit
            return ''.join(translated_text)

        return text  # Si déjà en anglais, retourner le texte original

    except Exception as e:
        print(f"Erreur lors de la détection/traduction de la langue pour le texte : {text}")
        print(f"Erreur: {e}")
        return text  # Retourner le texte original en cas d'erreur

    