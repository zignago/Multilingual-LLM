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
    "russian": {
        "Entailment": "Вследствие",
        "Neutral": "Нейтральный",
        "Contradiction": "Противоречие" 
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
    "russian": "Helsinki-NLP/opus-mt-en-ru",
    "spanish": "Helsinki-NLP/opus-mt-en-es"
}

LANGUAGE_CODES = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "russian": "ru",
    "spanish": "es"
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
    "russian": {
        "Premise": "Презюмировать",
        "Hypothesis": "Гипотеза",
        "Label": "Лабель"
    },
    "spanish": {
        "Premise": "Premisa",
        "Hypothesis": "Hipótesis",
        "Label": "Etiqueta"
    },
}

# For specific keywords that often get mistranslated
HARDCODE_TRANSLATIONS = {
    "sé": "know", #sé often gets reverse-translated as "HE"
    "sé.": "know",
    "sé,": "know"
}

# Models that have been tested and verified to work with this tool
# Info on GPT models: https://platform.openai.com/docs/models
SUPPORTED_LLM_MODELS = [
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125"
]