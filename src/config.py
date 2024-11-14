LABEL_MAP = {
    0: "Entailment",
    1: "Neutral",
    2: "Contradiction"
}

LABEL_TRANSLATION = {
    "italian": {
        "Entailment": "Implicazione",
        "Neutral": "Neutrale",
        "Contradiction": "Contraddizione" 
    },
    "german": {
        "Entailment": "Erforderung",
        "Neutral": "Neutral",
        "Contradiction": "Widerspruch" 
    },
    "spanish": {
        "Entailment": "Vinculación",
        "Neutral": "Neutral",
        "Contradiction": "Contradicción" 
    }
}

TRANSLATION_MODELS = {
    "spanish": "Helsinki-NLP/opus-mt-en-es",
    "german": "Helsinki-NLP/opus-mt-en-de",
    "italian": "Helsinki-NLP/opus-mt-en-it"
    # Add more languages as needed...
}

# Mapping language names to deep-translator codes
LANGUAGE_CODES = {
    "english": "en",
    "spanish": "es",
    "german": "de",
    "italian": "it"
    # Add other languages here as needed...
}

PROMPT_COMPONENTS = {
    "english": {
        "Premise": "Premise",
        "Hypothesis": "Hypothesis",
        "Label": "Label"
    },
    "italian": {
        "Premise": "Premere",
        "Hypothesis": "Ipotesi",
        "Label": "Etichetta"
    },
    "german": {
        "Premise": "Prämisse",
        "Hypothesis": "Hypothese",
        "Label": "Etikett"
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