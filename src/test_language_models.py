from transformers import pipeline
import torch
from config import TRANSLATION_MODELS

device = 0 if torch.cuda.is_available() else -1

def validate_language_codes(language_codes):
    """
    Validates if the provided language codes correspond to valid Helsinki-NLP models on Hugging Face.
    Translates a test phrase for each valid language to verify correctness.
    """
    base_phrase = "this is an example sentence"
    unsupported_languages = []
    valid_languages = []

    for lang, code in language_codes.items():
        try:
            # Initialize the translation pipeline
            translator = pipeline("translation", model=TRANSLATION_MODELS[lang], device=device)
            
            # Translate the test phrase
            translated_phrase = translator(base_phrase, max_length=40)[0]["translation_text"]
            print(f"Language '{lang}' ({code}): Model '{model_name}' is valid.")
            print(f"  Test phrase translation: '{translated_phrase}'\n")
            valid_languages.append((lang, translated_phrase))
        except Exception as e:
            unsupported_languages.append((lang, model_name))
            print(f"@@ Error: Language '{lang}' ({code}) @@")

    if unsupported_languages:
        print("\nThe following languages are not supported:")
        for lang, model_name in unsupported_languages:
            print(f" - {lang}: Model '{model_name}' is unavailable.")
    else:
        print("\nAll languages in config.py are supported.")

    print("\nValid language translations:")
    for lang, translated_phrase in valid_languages:
        print(f" - {lang}: {translated_phrase}")

# Call this function with LANGUAGE_CODES from your config
validate_language_codes(TRANSLATION_MODELS)