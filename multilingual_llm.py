import os
import json
import time
from datasets import load_dataset
from transformers import pipeline
import openai
import torch
import re
import argparse
from deep_translator import GoogleTranslator
from collections import Counter

openai.api_key = os.getenv("OPENAI_API_KEY")

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

device = 0 if torch.cuda.is_available() else -1

def get_translator(language):
    """Retrieves the translation pipeline for the specified language, if supported."""
    if language == "english":
        return None  # Skip translation for English

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

def translate_static_prompt_parts(translator, limit):
    """Translates only the static instruction parts of the prompt for a given language."""
    # Define the static instruction template without dynamic content
    static_instruction = (
        f"Identify the top {limit} keywords relevant to understanding the relationship between the premise and hypothesis. "
        "Only include words from the premise and hypothesis. Do not include any other words.\n"
        "Premise: [Premise]\nHypothesis: [Hypothesis]\nLabel: [Label]\n\n"
        'Return keywords in array format like ["a", "b", "c"]. Only include single words, no phrases.'
    )
    
    # Translate the static instructions only
    translated_instruction = translator(static_instruction)[0]['translation_text']
    
    # Return the translated instruction with placeholders in brackets
    return translated_instruction


def generate_prompts(row, translated_instruction, language="english"):
    """Generates a single prompt by replacing placeholders with specific row data."""
    # Replace placeholders with actual premise, hypothesis, and label
    return translated_instruction.replace("[Premise]", row["translated_premise"] if language != "english" else row["premise"])\
                                 .replace("[Hypothesis]", row["translated_hypothesis"] if language != "english" else row["hypothesis"])\
                                 .replace("[Label]", LABEL_MAP[row["label"]])

def filter_keywords(keywords, premise, hypothesis):
    """Filter out any keywords that are not in the premise or hypothesis."""
    allowed_words = set(premise.split() + hypothesis.split())
    return [word for word in keywords if word in allowed_words]

def enforce_limit(keywords, limit, premise, hypothesis):
    """Ensures that the keyword list meets the required limit by adding words from the premise or hypothesis if necessary."""
    if len(keywords) < limit:
        # Extract words from premise and hypothesis as additional options
        extra_words = (premise + " " + hypothesis).split()
        extra_words = [word for word in extra_words if word not in keywords]
        # Add words until reaching the limit
        keywords.extend(extra_words[:limit - len(keywords)])
    return keywords[:limit]

def get_common_keywords(all_keywords, limit):
    """Selects the top 'limit' most common keywords from a list of keyword lists."""
    flat_keywords = [keyword for keywords in all_keywords for keyword in keywords]
    keyword_counts = Counter(flat_keywords)
    most_common_keywords = [keyword for keyword, _ in keyword_counts.most_common(limit)]
    return most_common_keywords

def get_keywords_from_llm_multiple(prompts, LLM_model, limit, premise_hypothesis_pairs, x):
    """Runs the keyword extraction 'x' times for each prompt and returns the most common 'limit' keywords."""
    keywords_batch = []
    for idx, prompt in enumerate(prompts):
        all_run_keywords = []  # Collect keywords from each run

        # Run the keyword extraction 'x' times
        for run in range(x):
            response = openai.chat.completions.create(
                model=LLM_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0
            )
            response_text = response.choices[0].message.content
            
            # Extract keywords and filter by premise/hypothesis words only
            keywords = re.findall(r'"(.*?)"', response_text)
            if not keywords:
                keywords = [word.strip() for word in response_text.strip("[]").split(",")]
            
            # Filter to only include words from the premise and hypothesis
            premise, hypothesis = premise_hypothesis_pairs[idx]
            filtered_keywords = filter_keywords(keywords, premise, hypothesis)
            
            # Enforce the limit and add to the collection
            limited_keywords = enforce_limit(filtered_keywords, limit, premise, hypothesis)
            all_run_keywords.append(limited_keywords)

            # Print the generated keywords for this run
            print(f"Run {run + 1} for prompt {idx + 1}: {limited_keywords}")

        # After 'x' runs, get the most common keywords
        common_keywords = get_common_keywords(all_run_keywords, limit)
        keywords_batch.append(common_keywords)
    
    return keywords_batch

def reverse_translate_keywords_deeptranslator(keywords, source_language):
    """Translates keywords back to English using deep-translator's Google Translator."""
    # Get the correct language code for deep-translator
    source_code = LANGUAGE_CODES.get(source_language)
    
    if not source_code:
        raise ValueError(f"Unsupported language '{source_language}'. Available languages: {list(LANGUAGE_CODES.keys())}")

    reverse_translated_keywords = []
    
    for keyword in keywords:
        try:
            # Translate each keyword back to English
            translated_word = GoogleTranslator(source=source_code, target="en").translate(keyword)
            reverse_translated_keywords.append(translated_word.strip())
        except Exception as e:
            # Handle translation errors, add original word if translation fails
            print(f"Error translating '{keyword}': {e}")
            reverse_translated_keywords.append(keyword)
    
    return reverse_translated_keywords

def get_keywords_with_reverse_translation(prompts, LLM_model, limit, premise_hypothesis_pairs, x, language_name):
    """Runs the keyword extraction 'x' times for each prompt and returns the most common 'limit' keywords, along with reverse translation."""
    keywords_batch = []
    reverse_translations_batch = []

    for idx, prompt in enumerate(prompts):
        all_run_keywords = []  # Collect keywords from each run

        # Run the keyword extraction 'x' times
        for run in range(x):
            response = openai.chat.completions.create(
                model=LLM_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0
            )
            response_text = response.choices[0].message.content
            
            # Extract keywords and filter by premise/hypothesis words only
            keywords = re.findall(r'"(.*?)"', response_text)
            if not keywords:
                keywords = [word.strip() for word in response_text.strip("[]").split(",")]
            
            # Filter to only include words from the premise and hypothesis
            premise, hypothesis = premise_hypothesis_pairs[idx]
            filtered_keywords = filter_keywords(keywords, premise, hypothesis)
            
            # Enforce the limit and add to the collection
            limited_keywords = enforce_limit(filtered_keywords, limit, premise, hypothesis)
            all_run_keywords.append(limited_keywords)

            # Print the generated keywords for this run
            print(f"Run {run + 1} for prompt {idx + 1}: {limited_keywords}")

        # After 'x' runs, get the most common keywords
        common_keywords = get_common_keywords(all_run_keywords, limit)
        keywords_batch.append(common_keywords)

        # Reverse translate the keywords back to English if language is not English
        if language_name != "english":
            reverse_translated_keywords = reverse_translate_keywords_deeptranslator(common_keywords, language_name)
            reverse_translations_batch.append(reverse_translated_keywords)
        else:
            reverse_translations_batch.append(common_keywords)  # Keep original for English
    
    return keywords_batch, reverse_translations_batch

def main(languages, limit, LLM_model, x=5, output_file=None):
    dataset = load_dataset("glue", "mnli", split="validation_matched").select(range(3))

    # Generate the static prompt template for English and other languages
    english_instruction = translate_static_prompt_parts(lambda x: [{"translation_text": x}], limit)
    english_prompts = [generate_prompts(row, english_instruction, "english") for row in dataset]
    premise_hypothesis_pairs = [(row["premise"], row["hypothesis"]) for row in dataset]
    
    # Get keywords with post-filtering, running 'x' times for consistency
    english_keywords_batch = get_keywords_from_llm_multiple(english_prompts, LLM_model, limit, premise_hypothesis_pairs, x)

    # Initialize output data structure with metadata
    output_data = {
        "metadata": {
            "model": LLM_model,
            "languages": ["english"] + languages,
            "limit": limit,
            "runs_per_language": x
        },
        "results": []
    }

    for idx, english_keywords in enumerate(english_keywords_batch, 1):
        output_data["results"].append({
            "idx": idx,
            "english": english_keywords,
            "prompts": {
                "english": english_prompts[idx-1]
            },
            "translations": {
                "english": {
                    "premise": dataset[idx-1]["premise"],
                    "hypothesis": dataset[idx-1]["hypothesis"]
                }
            }
        })

    # Process each language by translating instructions only once and formatting each prompt dynamically
    for language in languages:
        translator = get_translator(language)
        
        # Pass only the language name for reverse translation
        translated_instruction = translate_static_prompt_parts(translator, limit)
        
        translated_dataset = translate_batch(dataset, translator)
        
        translated_prompts = [generate_prompts(row, translated_instruction, language) for row in translated_dataset]
        translated_premise_hypothesis_pairs = [(row["translated_premise"], row["translated_hypothesis"]) for row in translated_dataset]
        
        # Run multiple times and get the most common keywords, along with reverse translation if applicable
        translated_keywords_batch, reverse_translated_batch = get_keywords_with_reverse_translation(
            translated_prompts, LLM_model, limit, translated_premise_hypothesis_pairs, x, language  # Pass language name here
        )

        for idx, (translated_keywords, reverse_translated) in enumerate(zip(translated_keywords_batch, reverse_translated_batch)):
            output_data["results"][idx][language] = translated_keywords
            output_data["results"][idx]["prompts"][language] = translated_prompts[idx]
            output_data["results"][idx]["translations"][language] = {
                "premise": translated_dataset[idx]["translated_premise"],
                "hypothesis": translated_dataset[idx]["translated_hypothesis"]
            }
            if language != "english":  # Only add reverse translation if not English
                output_data["results"][idx][f"{language}-to-english"] = reverse_translated

    # Set default output filename with timestamp if no custom filename is provided
    if not output_file:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"outputs/keywords_output_{timestamp}.json"
    else:
        output_file = f"outputs/{output_file}"

    os.makedirs("outputs", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual LLM Keyword Extraction")
    parser.add_argument("--languages", nargs="+", default=["german"], help="Specify one or more target languages.")
    parser.add_argument("--limit", type=int, default=3, help="Limit to top n keywords.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Specify the GPT model.")
    parser.add_argument("--x", type=int, default=1, help="Specify the number of times to run each query.")
    parser.add_argument("--output", type=str, help="Optional custom output filename (without path) for JSON results.")
    args = parser.parse_args()
    
    main(args.languages, args.limit, args.model, args.x, args.output)