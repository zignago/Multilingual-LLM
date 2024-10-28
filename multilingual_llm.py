import os
import json
import time
from datasets import load_dataset
from transformers import pipeline
import openai
import torch
import re
import argparse

# TODO features:
# - Differentiate Chinese simplified vs traditional
# - Separate functionality into different files
# - Write evaluation functionality
# - Integrate Llama models

# TODO Testing:
# - raise errors for garbage inputs (invalid/unsupported language, invalid llm model, limit too large/small/negative)
# - test output with other languages
# - Write detailed testing for everything I can think of (mostly prompt engineering)
# - Have some functionality to ensure translation is satisfactory
# - Test consistency across runs

openai.api_key = os.getenv("OPENAI_API_KEY")

# Label mapping for translation
LABEL_MAP = {
    0: "Entailment",
    1: "Neutral",
    2: "Contradiction"
}

# Language-specific models dictionary
TRANSLATION_MODELS = {
    "spanish": "Helsinki-NLP/opus-mt-en-es", #
    "german": "Helsinki-NLP/opus-mt-en-de", #
    "french": "Helsinki-NLP/opus-mt-en-fr", #
    "italian": "Helsinki-NLP/opus-mt-en-it", #
    "portuguese": "Helsinki-NLP/opus-mt-en-pt",
    "dutch": "Helsinki-NLP/opus-mt-en-nl",
    "russian": "Helsinki-NLP/opus-mt-en-ru",
    "chinese": "Helsinki-NLP/opus-mt-en-zh", #
    "arabic": "Helsinki-NLP/opus-mt-en-ar",
    "swedish": "Helsinki-NLP/opus-mt-en-sv",
    "norwegian": "Helsinki-NLP/opus-mt-en-no",
    "finnish": "Helsinki-NLP/opus-mt-en-fi",
    "danish": "Helsinki-NLP/opus-mt-en-da",
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
    prompt_template = (
        f"Identify the top {limit} keywords relevant to understanding the relationship between the premise and hypothesis.\n"
        "ONLY consider words between \'<<<\' and \'>>>\' as potential keywords.\n"
        "Premise: <<<{premise}>>>\nHypothesis: <<<{hypothesis}>>>\nLabel: {label}\n\n"
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
        
        keywords_batch.append(keywords[:limit])  # Enforce the limit
    return keywords_batch

def main(languages, limit, LLM_model, output_file=None):
    dataset = load_dataset("glue", "mnli", split="validation_matched").select(range(20))

    # Run the English pipeline first
    english_prompts = [generate_prompts(row, limit, "english") for row in dataset]
    english_keywords_batch = get_keywords_from_llm_batch(english_prompts, LLM_model, limit)

    # Initialize output data structure with metadata
    output_data = {
        "metadata": {
            "model": LLM_model,
            "languages": languages,
            "limit": limit
        },
        "results": []
    }

    # Populate results for each dataset item starting with English keywords
    for idx, english_keywords in enumerate(english_keywords_batch, 1):
        output_data["results"].append({
            "idx": idx,
            "english": english_keywords
        })

    # Run the pipeline for each specified language
    for language in languages:
        translator = get_translator(language)
        translated_dataset = translate_batch(dataset, translator)
        
        # Generate prompts and get keywords for each language
        translated_prompts = [generate_prompts(row, limit, language) for row in translated_dataset]
        translated_keywords_batch = get_keywords_from_llm_batch(translated_prompts, LLM_model, limit)

        # Add translated keywords to output data
        for idx, translated_keywords in enumerate(translated_keywords_batch):
            output_data["results"][idx][language] = translated_keywords

    # Set default output filename with timestamp if no custom filename is provided
    if not output_file:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"outputs/keywords_output_{timestamp}.json"
    else:
        output_file = f"outputs/{output_file}"

    # Ensure the 'outputs' directory exists
    os.makedirs("outputs", exist_ok=True)

    # Write the output data to a JSON file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual LLM Keyword Extraction")
    parser.add_argument("--languages", nargs="+", default=["german"], help="Specify one or more target languages.")
    parser.add_argument("--limit", type=int, default=3, help="Limit to top n keywords.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Specify the GPT model.\nDefault: gpt-3.5-turbo\nSupported models:https://platform.openai.com/docs/models")
    parser.add_argument("--output", type=str, help="Optional custom output filename (without path) for JSON results.")
    args = parser.parse_args()
    
    main(args.languages, args.limit, args.model, args.output)