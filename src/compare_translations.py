# File to compare the quality of the translators by running them on phrases in the dataset
import json
import os
from datasets import load_dataset
from transformers import pipeline
from deep_translator import GoogleTranslator
import torch
import openai
from src.config import LABEL_MAP, TRANSLATION_MODELS, LANGUAGE_CODES

openai.api_key = os.getenv("OPENAI_API_KEY")

device = 0 if torch.cuda.is_available() else -1

# Define forward and reverse translation models
TRANSLATION_MODELS = {
    "spanish": "Helsinki-NLP/opus-mt-en-es",
    "spanish-reverse": "Helsinki-NLP/opus-mt-es-en",
    "german": "Helsinki-NLP/opus-mt-en-de",
    "german-reverse": "Helsinki-NLP/opus-mt-de-en",
    # Add more languages and their reverse models as needed
}

LANGUAGE_CODES = {
    "spanish": "es",
    "german": "de",
    # Add other languages here as needed
}

def get_helsinki_translator(language, reverse=False):
    """Sets up the Helsinki translation pipeline for the specified language and direction."""
    model_key = f"{language}-reverse" if reverse else language
    model_name = TRANSLATION_MODELS.get(model_key)
    if model_name is None:
        raise ValueError(f"No Helsinki translation model available for '{language}' with reverse={reverse}")
    return pipeline("translation", model=model_name, device=device, batch_size=8)

def forward_and_reverse_translate(premise, language):
    # Helsinki forward and reverse translation
    helsinki_translator = get_helsinki_translator(language)
    helsinki_translation = helsinki_translator(premise)[0]['translation_text']
    
    helsinki_reverse_translator = get_helsinki_translator(language, reverse=True)
    helsinki_reverse_translation = helsinki_reverse_translator(helsinki_translation)[0]['translation_text']

    # Deep Translator forward and reverse translation
    deep_translator = GoogleTranslator(source="en", target=LANGUAGE_CODES[language])
    deep_translation = deep_translator.translate(premise)
    deep_reverse_translator = GoogleTranslator(source=LANGUAGE_CODES[language], target="en")
    deep_reverse_translation = deep_reverse_translator.translate(deep_translation)

    # GPT-3.5 and GPT-4o reverse translations
    gpt_35_reverse_translation = gpt_translate_back(helsinki_translation, "gpt-3.5-turbo")
    gpt_4o_reverse_translation = gpt_translate_back(helsinki_translation, "gpt-4-turbo")

    return {
        "original": premise,
        "helsinki_reverse": helsinki_reverse_translation,
        "deep_reverse": deep_reverse_translation,
        "GPT 3.5 reverse": gpt_35_reverse_translation,
        "GPT 4o reverse": gpt_4o_reverse_translation,
    }

def gpt_translate_back(text, model):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Translate the following text back to English."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()

def main(language, output_file="translation_comparison.json"):
    # Load dataset and select first 10 premises for simplicity
    dataset = load_dataset("glue", "mnli", split="validation_matched").select(range(35))
    premises = [item["premise"] for item in dataset]

    # Store the results
    results = []

    for premise in premises:
        translations = forward_and_reverse_translate(premise, language)

        # Store result
        results.append(translations)

        # Output to terminal if translations differ
        if translations["helsinki_reverse"] != translations["deep_reverse"] or translations["helsinki_reverse"] != translations["GPT 3.5 reverse"]:
            print(f"Original: {translations['original']}")
            print(f"Helsinki Reverse: {translations['helsinki_reverse']}")
            print(f"Deep Reverse: {translations['deep_reverse']}")
            print(f"GPT 3.5 Reverse: {translations['GPT 3.5 reverse']}")
            print(f"GPT 4o Reverse: {translations['GPT 4o reverse']}")
            print("-" * 50)

    # Save results to JSON
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Translation comparison saved to {output_path}")

if __name__ == "__main__":
    import argparse

    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Compare translations of Helsinki and Deep Translator with GPT reversals.")
    parser.add_argument("--language", type=str, default="german", choices=LANGUAGE_CODES.keys(),
                        help="Specify the target language for translation.")
    parser.add_argument("--output", type=str, default="translation_comparison.json", 
                        help="Specify the output JSON filename.")
    args = parser.parse_args()

    main(args.language, args.output)