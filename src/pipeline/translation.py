from langdetect import detect
from deep_translator import GoogleTranslator

def detect_and_translate(text, max_length=1000):
    """
    Détecte la langue de 'text' et le traduit en anglais si nécessaire.
    """
    try:
        lang = detect(text[:max_length]) if len(text) > max_length else detect(text) # Détection de la langue
        if lang != 'en':  # Si la langue n'est pas l'anglais
            translated = GoogleTranslator(source=lang, target='en').translate(text)
            return translated
        return text
    except:
        print(f"Erreur lors de la détection/traduction de la langue pour le texte : {text}")
        return text  # Retourne le texte original en cas d'erreur
    
