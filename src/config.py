LABEL_MAP = {
    0: "Entailment",
    1: "Neutral",
    2: "Contradiction"
}

TRANSLATION_MODELS = {
    "spanish": "Helsinki-NLP/opus-mt-en-es",
    "german": "Helsinki-NLP/opus-mt-en-de",
    # Add more languages as needed...
}

# Mapping language names to deep-translator codes
LANGUAGE_CODES = {
    "english": "en",
    "spanish": "es",
    "german": "de",
    # Add other languages here as needed...
}

# For specific keywords that often get mistranslated
HARDCODE_TRANSLATIONS = {
    "sé": "know", #sé often gets reverse-translated as "HE"
    "sé.": "know",
    "sé,": "know"
}