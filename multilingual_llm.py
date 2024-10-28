import os
from datasets import Dataset, load_dataset
from transformers import pipeline
import openai
import torch
import re
import argparse


# TDOO:
# - test other languages
# - Write an english pipeline to run at the same time as translated pipeline to be compared (keep track of sample number for each response)
# - Separate functionality into different files
# - Write evaluation functionality
# - Write detailed testing for everything I can think of (mostly prompt engineering)
# - Have some functionality to ensure translation is satisfactory
# - Integrate other gpt models
# - Integrate Llama models


openai.api_key = os.getenv("OPENAI_API_KEY")

# Label mapping for translation
LABEL_MAP = {
    0: "Entailment",
    1: "Neutral",
    2: "Contradiction"
}

# Language-specific models dictionary
TRANSLATION_MODELS = {
    "spanish": "Helsinki-NLP/opus-mt-en-es",
    "german": "Helsinki-NLP/opus-mt-en-de",
    "french": "Helsinki-NLP/opus-mt-en-fr",
    "italian": "Helsinki-NLP/opus-mt-en-it",
    "portuguese": "Helsinki-NLP/opus-mt-en-pt",
    "dutch": "Helsinki-NLP/opus-mt-en-nl",
    "russian": "Helsinki-NLP/opus-mt-en-ru",
    "chinese": "Helsinki-NLP/opus-mt-en-zh",
    "japanese": "Helsinki-NLP/opus-mt-en-jp",
    "korean": "Helsinki-NLP/opus-mt-en-ko",
    "arabic": "Helsinki-NLP/opus-mt-en-ar",
    "swedish": "Helsinki-NLP/opus-mt-en-sv",
    "norwegian": "Helsinki-NLP/opus-mt-en-no",
    "finnish": "Helsinki-NLP/opus-mt-en-fi",
    "danish": "Helsinki-NLP/opus-mt-en-da",
    "polish": "Helsinki-NLP/opus-mt-en-pl",
    "turkish": "Helsinki-NLP/opus-mt-en-tr",
    "czech": "Helsinki-NLP/opus-mt-en-cs",
    "hungarian": "Helsinki-NLP/opus-mt-en-hu",
    "bulgarian": "Helsinki-NLP/opus-mt-en-bg",
    "greek": "Helsinki-NLP/opus-mt-en-el",
    "romanian": "Helsinki-NLP/opus-mt-en-ro",
    "hindi": "Helsinki-NLP/opus-mt-en-hi",
    "indonesian": "Helsinki-NLP/opus-mt-en-id",
    "vietnamese": "Helsinki-NLP/opus-mt-en-vi",
    "thai": "Helsinki-NLP/opus-mt-en-th",
    "hebrew": "Helsinki-NLP/opus-mt-en-he",
    "ukrainian": "Helsinki-NLP/opus-mt-en-uk",
    "bengali": "Helsinki-NLP/opus-mt-en-bn",
}

# Check if GPU is available and set device accordingly
device = 0 if torch.cuda.is_available() else -1

def get_translator(language):
    try:
        model_name = TRANSLATION_MODELS[language]
    except KeyError:
        raise ValueError(f"Unsupported language '{language}'. Choose from: {', '.join(TRANSLATION_MODELS.keys())}")
    return pipeline("translation", model=model_name, device=device, batch_size=8)

def translate_batch(dataset, translator):
    dataset = dataset.map(
        lambda examples: {
            "translated_premise": [item["translation_text"] for item in translator(examples["premise"])],
            "translated_hypothesis": [item["translation_text"] for item in translator(examples["hypothesis"])],
        },
        batched=True,
    )
    return dataset

def generate_prompts(row, limit, language="english"):
    if language == "english":
        prompt_template = (
            f"Identify the top {limit} keywords relevant to understanding the relationship between the premise and hypothesis.\n"
            "Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label}\n\n"
            'Return keywords in array format like ["a", "b", "c"]. Only include single words, no phrases.'
        )
    else:
        prompt_template = (
            f"Identify the top {limit} keywords relevant to understanding the relationship.\n"
            "Premise: {premise}\nHypothesis: {hypothesis}\nLabel: {label}\n\n"
            'Return keywords in array format like ["a", "b", "c"]. Only include single words, no phrases.'
        )
    return prompt_template.format(
        premise=row["translated_premise"] if language != "english" else row["premise"],
        hypothesis=row["translated_hypothesis"] if language != "english" else row["hypothesis"],
        label=LABEL_MAP[row["label"]],
    )

def get_keywords_from_llm_batch(prompts, LLM_model, limit):
    keywords_batch = []
    for prompt in prompts:
        response = openai.chat.completions.create(
            model=LLM_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0
        )
        response_text = response.choices[0].message.content
        print(f"LLM Response: {response_text}")  # Debugging output

        keywords = re.findall(r'"(.*?)"', response_text)
        if not keywords:
            keywords = [word.strip() for word in response_text.strip("[]").split(",")]
        
        # Truncate keywords to specified limit
        keywords_batch.append(keywords[:limit])
    return keywords_batch

def main(language, limit, LLM_model):
    translator = get_translator(language)
    dataset = load_dataset("glue", "mnli", split="validation_matched").select(range(20))
    translated_dataset = translate_batch(dataset, translator)

    translated_prompts = [generate_prompts(row, limit, language) for row in translated_dataset]
    english_prompts = [generate_prompts(row, limit, "english") for row in dataset]

    translated_keywords_batch = get_keywords_from_llm_batch(translated_prompts, LLM_model, limit)
    english_keywords_batch = get_keywords_from_llm_batch(english_prompts, LLM_model, limit)

    for i, (translated_keywords, english_keywords) in enumerate(zip(translated_keywords_batch, english_keywords_batch), 1):
        print(f"Sample {i} Keywords ({language.capitalize()}): {translated_keywords}")
        print(f"Sample {i} Keywords (English): {english_keywords}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual LLM Keyword Extraction")
    parser.add_argument("--language", type=str, default="german", help="Specify the target language.")
    parser.add_argument("--limit", type=int, default=3, help="Limit to top n keywords.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Specify the GPT model.")
    args = parser.parse_args()
    
    main(args.language.lower(), args.limit, args.model)