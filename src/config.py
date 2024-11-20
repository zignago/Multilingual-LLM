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
SUPPORTED_LLM_MODELS = {
    "gpt": [
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
    ],
    "llama": [
        "llama-2-7b", 
        "llama-2-13b", 
        "llama-2-70b"
    ]
}


# LLAMA Models
# llama3.2-11b-vision
# llama3.2-1b
# llama3.2-3b
# llama3.2-90b
# llama3.1-405b
# llama3.1-70b
# llama3.1-8b
# llama3-70b
# llama3-8b
# gemma2-27b
# gemma2-9b
# mixtral-8x22b
# mixtral-8x22b-instruct
# mixtral-8x7b-instruct
# mistral-7b
# mistral-7b-instruct
# llama-7b-32k
# llama2-13b
# llama2-70b
# llama2-7b
# Nous-Hermes-2-Mixtral-8x7B-DPO
# Nous-Hermes-2-Yi-34B
# Qwen1.5-0.5B-Chat
# Qwen1.5-1.8B-Chat
# Qwen1.5-110B-Chat
# Qwen1.5-14B-Chat
# Qwen1.5-32B-Chat
# Qwen1.5-4B-Chat
# Qwen1.5-72B-Chat
# Qwen1.5-7B-Chat
# Qwen2-72B-Instruct