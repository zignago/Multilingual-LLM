LABEL_MAP = {
    0: "Entailment",
    1: "Neutral",
    2: "Contradiction"
}

LABEL_TRANSLATION = {
    "french": {
        "Entailment": "Implication",
        "Neutral": "Neutre",
        "Contradiction": "Contradiction" 
    },
    "german": {
        "Entailment": "Erforderung",
        "Neutral": "Neutral",
        "Contradiction": "Widerspruch" 
    },
    "italian": {
        "Entailment": "Implicazione",
        "Neutral": "Neutrale",
        "Contradiction": "Contraddizione" 
    },
    "spanish": {
        "Entailment": "Vinculación",
        "Neutral": "Neutral",
        "Contradiction": "Contradicción" 
    }
}

TRANSLATION_MODELS = {
    "french": "Helsinki-NLP/opus-mt-en-fr",
    "german": "Helsinki-NLP/opus-mt-en-de",
    "italian": "Helsinki-NLP/opus-mt-en-it",
    "spanish": "Helsinki-NLP/opus-mt-en-es"
    # Add more languages as needed...
}

# Mapping language names to deep-translator codes
LANGUAGE_CODES = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "spanish": "es"
    # Add other languages here as needed...
}

PROMPT_COMPONENTS = {
    "english": {
        "Premise": "Premise",
        "Hypothesis": "Hypothesis",
        "Label": "Label"
    },
    "french": {
        "Premise": "Premise",
        "Hypothesis": "Hypothèse",
        "Label": "Étiquette"
    },
    "german": {
        "Premise": "Prämisse",
        "Hypothesis": "Hypothese",
        "Label": "Etikett"
    },
    "italian": {
        "Premise": "Premere",
        "Hypothesis": "Ipotesi",
        "Label": "Etichetta"
    },
    "spanish": {
        "Premise": "Premisa",
        "Hypothesis": "Hipótesis",
        "Label": "Etiqueta"
    },
    # Add other languages as needed
}

# For specific keywords that often get mistranslated
HARDCODE_TRANSLATIONS = {
    "sé": "know", #sé often gets reverse-translated as "HE"
    "sé.": "know",
    "sé,": "know"
}