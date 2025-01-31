"""
This module provides a function to detect the language of a given text and translate it to English if necessary. 
It utilizes the `langdetect` library to detect the language and the `deep_translator` library for translation. 
The function ensures that long texts are split into smaller chunks to avoid errors during translation.
"""

from langdetect import detect
from deep_translator import GoogleTranslator

def detect_and_translate(text, max_length=4000):
    """
    Detects the language of the input text and translates it to English if it's not already in English.
    If the text exceeds the specified `max_length`, it is split into smaller chunks to avoid errors during translation.
    
    Args:
        text (str): The text to be detected and translated.
        max_length (int, optional): The maximum length of the text chunk to be translated at a time. Default is 4000 characters.
    
    Returns:
        str: The translated text in English if the original text is not in English, or the original text if it is already in English.
        
    If an error occurs during the detection or translation process, the original text is returned.
    """
    try:
        # Detect the language of the input text
        lang = detect(text[:max_length]) if len(text) > max_length else detect(text)

        # If the detected language is not English, translate it
        if lang != 'en':
            translated_text = []

            # Split the text into chunks if it exceeds max_length and translate each chunk
            for i in range(0, len(text), max_length):
                chunk = text[i:i+max_length]
                translated_chunk = GoogleTranslator(source=lang, target='en').translate(chunk)
                translated_text.append(translated_chunk)

            # Join the translated chunks and return the full translated text
            return ''.join(translated_text)

        return text  # Return the original text if it's already in English

    except Exception as e:
        # Print error message if any exception occurs
        print(f"Erreur lors de la détection/traduction de la langue pour le texte : {text}")
        print(f"Erreur: {e}")
        return text  # Return the original text in case of an error

    