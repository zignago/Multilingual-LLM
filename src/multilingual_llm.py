import os
import json
import time
from datetime import datetime
from datasets import load_dataset
from transformers import pipeline
import openai
import torch
import re
import string
import unicodedata
from src.llama_handler import LlamaHandler
from deep_translator import GoogleTranslator
from collections import Counter
from src.config import (
    LABEL_MAP, TRANSLATION_MODELS, LANGUAGE_CODES, SUPPORTED_LLM_MODELS
)

device = 0 if torch.cuda.is_available() else -1

openai.api_key = os.getenv("OPENAI_API_KEY")

llama_handler = LlamaHandler()

def determine_model_type(model_name):
    """Determines whether the model is GPT-based or LLaMA-based."""
    for model_type, models in SUPPORTED_LLM_MODELS.items():
        if model_name in models:
            return model_type
    raise ValueError(f"Model {model_name} is not supported. Supported models: {SUPPORTED_LLM_MODELS}")

def get_translator(language):
    """
    Retrieves the translation pipeline for the specified language, if supported.
    Dynamically constructs the Opus-MT model path using TRANSLATION_MODELS.
    """
    if language == "english":
        return None  # Skip translation for English

    # Ensure the language exists in TRANSLATION_MODELS
    if language not in TRANSLATION_MODELS or not TRANSLATION_MODELS[language]:
        raise ValueError(f"Unsupported language '{language}'. Choose from: {', '.join(TRANSLATION_MODELS.keys())}")

    # Return the translation pipeline
    return pipeline("translation", model=TRANSLATION_MODELS[language], device=device, batch_size=8)

def translate_static_prompt_parts(translator, limit, language="english"):
    """
    Translates only the static instruction parts of the prompt for a given language.
    Uses hardcoded translations for premise, hypothesis, and label to avoid errors.
    """
    # Define the static instruction template with placeholders
    static_instruction = (
        f"Identify the most important {limit} keywords for understanding the relationship between the premise and hypothesis. "
        "Include only words from the premise and hypothesis. Do not include any other words. "
        "Do not include punctuation or commas ('.' or ',') in keywords.\n"
        "Premise: [Premise]\nHypothesis: [Hypothesis]\nLabel: [Label].\n\n"
        'Return keywords in array format like ["a", "b", "c"]. Include only single words, no phrases.'
    )

    # Translate the static instruction using the translator
    if translator:
        translated_instruction = translator(static_instruction)[0]["translation_text"]
    else:
        translated_instruction = static_instruction

    return translated_instruction

def mark_keywords_in_prompt(prompt, keywords):
    """
    Add markers around the specified keywords in the prompt.
    :param prompt: The original prompt as a string.
    :param keywords: A list of keywords to mark.
    :return: The modified prompt with markers.
    """
    for keyword in keywords:
        # Use regex to match the whole word and wrap it in markers
        prompt = re.sub(rf'\b{re.escape(keyword)}\b', rf'[{keyword}]', prompt, flags=re.IGNORECASE)
    return prompt

def generate_prompts(row, instruction):
    """
    Generates a single prompt by replacing placeholders with specific row data.
    Ensures that english premise, hypothesis, and label are correctly populated.
    """
    # Replace placeholders in the instruction
    prompt = instruction.replace("[Premise]", row["premise"].lower())
    prompt = prompt.replace("[Hypothesis]", row["hypothesis"].lower())
    prompt = prompt.replace("[Label]", LABEL_MAP[row["label"]] )

    # Check for unresolved English placeholders
    if "[Premise]" in prompt or "[Hypothesis]" in prompt or "[Label]" in prompt:
        raise ValueError(f"Unresolved placeholders in prompt: {prompt}")

    return prompt

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

def run_gpt(prompt, model_name):
    max_tokens_param = "max_tokens"

    # Adjust max_tokens to leave room for prompt
    request_params = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        max_tokens_param: 500,  # Adjust based on model capabilities
    }

    # print(f"@@@PROMPT MESSAGE@@@: {request_params}\n")

    response = openai.chat.completions.create(**request_params)
    return response.choices[0].message.content

def run_llama(prompt, model_name, label):
    """
    Interact with Llama API via LlamaHandler.
    :param prompt: The input prompt to the model.
    :param model_name: The model name to use.
    :param label: The label for the relationship (e.g., entailment, contradiction, neutral).
    :return: The model's response.
    """
    full_prompt = f"{prompt}\nLabel: {label}"
    response = llama_handler.run_prompt(full_prompt, model_name=model_name)
    return response

def clean_keyword(keyword):
    """
    Clean a keyword by removing punctuation, normalizing accents, and converting to lowercase.
    """
    keyword = unicodedata.normalize("NFKC", keyword)  # Normalize text
    keyword = ''.join(char for char in keyword if char not in string.punctuation)  # Remove punctuation
    return keyword.strip().lower()  # Strip whitespace and convert to lowercase

from difflib import SequenceMatcher

def closest_match_keywords(word, english_premise, english_hypothesis, translated_premise, translated_hypothesis):
    """
    Compare a reverse-translated keyword to the English premise and hypothesis
    to find the closest match, considering section (premise or hypothesis) and positional alignment.
    :param word: The reverse-translated keyword.
    :param source_language: The source language of the reverse translation.
    :param english_premise: The original English premise.
    :param english_hypothesis: The original English hypothesis.
    :param translated_premise: The translated premise.
    :param translated_hypothesis: The translated hypothesis.
    :return: The closest matching word from the premise or hypothesis.
    """
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # Tokenize sentences
    premise_words = english_premise.lower().split()
    hypothesis_words = english_hypothesis.lower().split()
    translated_premise_words = translated_premise.lower().split()
    translated_hypothesis_words = translated_hypothesis.lower().split()

    # Check section: translated premise or hypothesis
    if word in translated_premise_words:
        section_words = premise_words
        translated_section_words = translated_premise_words
    elif word in translated_hypothesis_words:
        section_words = hypothesis_words
        translated_section_words = translated_hypothesis_words
    else:
        # If the word cannot be found, treat it as belonging to both
        section_words = premise_words + hypothesis_words
        translated_section_words = translated_premise_words + translated_hypothesis_words

    # Step 1: Direct Match
    if word in section_words:
        return word

    # Step 2: Semantic Similarity
    semantic_scores = [(w, similarity(word, w)) for w in section_words]
    best_match, best_score = max(semantic_scores, key=lambda x: x[1])

    # If score is high enough, return the best semantic match
    if best_score > 0.8:
        return best_match

    # Step 3: Positional Heuristic (when semantic similarity isn't decisive)
    if word in translated_section_words:
        word_index = translated_section_words.index(word)
        closest_index = min(range(len(section_words)), key=lambda i: abs(i - word_index))
        return section_words[closest_index]

    # Fallback: Return the closest match by semantic similarity
    return best_match

def reverse_translate_keywords_deeptranslator(keywords, source_language, english_premise, english_hypothesis, translated_premise, translated_hypothesis):
    """Translates keywords back to English using deep-translator's Google Translator, preserving order."""
    source_code = LANGUAGE_CODES.get(source_language)
    if not source_code:
        raise ValueError(f"Unsupported language '{source_language}'. Available languages: {list(LANGUAGE_CODES.keys())}")

    reverse_translated_keywords = []
    for keyword in keywords:
        try:
            translated_word = GoogleTranslator(source=source_code, target="en").translate(keyword)
            translated_word = clean_keyword(translated_word)

            # Compare to words in source premise/hypothesis
            closest_match = closest_match_keywords(
                translated_word, 
                english_premise, 
                english_hypothesis, 
                translated_premise, 
                translated_hypothesis
            )

            reverse_translated_keywords.append(closest_match)
        except Exception as e:
            print(f"Error translating '{keyword}': {e}")
            reverse_translated_keywords.append(clean_keyword(keyword))

    return reverse_translated_keywords  # Order is preserved

def split_and_translate(text, translator, max_length=500):
    """
    Splits a long text into smaller chunks for translation and translates each chunk.
    """
    if len(text) <= max_length:
        return translator(text)[0]["translation_text"]

    # Split text into manageable parts
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    translated_chunks = [translator(chunk)[0]["translation_text"] for chunk in chunks]
    return " ".join(translated_chunks)

def parse_keywords_from_response(response_text, premise, hypothesis):
    """
    Extracts keywords from the model's response and validates them
    against the premise and hypothesis.
    """
    # Attempt to extract a JSON-like array from the response
    match = re.search(r'\["(.*?)"\]', response_text)
    if not match:
        # If no valid array found, treat as a simple comma-separated list
        keywords = [word.strip() for word in response_text.strip("[]").split(",")]
    else:
        keywords = match.group(1).split('", "')

    # Normalize keywords (remove punctuation, lowercase)
    keywords = [clean_keyword(word) for word in keywords]

    # Validate keywords against premise and hypothesis
    allowed_words = set(premise.lower().split() + hypothesis.lower().split())
    return [keyword for keyword in keywords if keyword in allowed_words]

def get_keywords_with_reverse_translation(prompts, model_name, limit, premise_hypothesis_pairs, x, language_name):
    """
    Runs the keyword extraction 'x' times for each prompt and returns the most common 'limit' keywords,
    along with reverse translation if applicable.
    """

    keywords_batch = []
    reverse_translations_batch = []

    # Initialize translator for the target language
    translator = None if language_name == "english" else get_translator(language_name)

    for idx, prompt in enumerate(prompts):
        all_run_keywords = []

        # # Translate the prompt if necessary
        # translated_prompt = translator(prompt)[0]["translation_text"] if translator else prompt

        # # Translate premise and hypothesis for validation
        premise, hypothesis = premise_hypothesis_pairs[idx][:2]
        translated_premise = translator(premise)[0]["translation_text"] if translator else premise
        translated_hypothesis = translator(hypothesis)[0]["translation_text"] if translator else hypothesis

        # Run the keyword extraction 'x' times
        for run in range(x):
            model_type = determine_model_type(model_name)
            if model_type == "gpt":
                response_text = run_gpt(prompt, model_name)
            elif model_type == "llama":
                # Assuming label is embedded in the prompt
                response_text = run_llama(prompt, model_name, premise_hypothesis_pairs[idx][2])

            # Parse and validate keywords from the response
            keywords = parse_keywords_from_response(response_text, translated_premise, translated_hypothesis)

            # Enforce limit and add to collection
            limited_keywords = enforce_limit(keywords, limit, translated_premise, translated_hypothesis)
            all_run_keywords.append(limited_keywords)

            print(f"Run {run + 1} for prompt {idx + 1}: {limited_keywords}")

        # Determine the most common keywords
        common_keywords = get_common_keywords(all_run_keywords, limit)
        keywords_batch.append(common_keywords)

        # Reverse translate keywords back to English if language is not English
        if language_name != "english":
            reverse_translated_keywords = reverse_translate_keywords_deeptranslator(common_keywords, language_name, premise, hypothesis, translated_premise, translated_hypothesis)
            reverse_translations_batch.append(reverse_translated_keywords)
        else:
            reverse_translations_batch.append(common_keywords)

    return keywords_batch, reverse_translations_batch


def main(languages, limit, model_name, subset=20, repeats=5, output_file=None):
    """
    Main function for multilingual keyword extraction, marking, and reverse translation.
    """
    # Load dataset and select a subset
    dataset = load_dataset("glue", "mnli", split="validation_matched").select(range(subset))

    # Generate raw English prompts and premise-hypothesis pairs
    english_instruction = translate_static_prompt_parts(None, limit)
    english_prompts = [generate_prompts(row, english_instruction) for row in dataset]
    premise_hypothesis_pairs = [(row["premise"], row["hypothesis"], LABEL_MAP[row["label"]]) for row in dataset]

    # Extract keywords for raw English prompts
    english_keywords_batch, _ = get_keywords_with_reverse_translation(
        english_prompts, model_name, limit, premise_hypothesis_pairs, repeats, "english"
    )

    # Create marked prompts for translation
    marked_english_prompts = [
        mark_keywords_in_prompt(english_prompts[idx], english_keywords_batch[idx])
        for idx in range(len(english_prompts))
    ]

    # Log marked English prompts
    print("\nMarked English Prompts with Keywords:")
    for idx, prompt in enumerate(marked_english_prompts, 1):
        print(f"Prompt {idx}: {prompt}")

    # Output structure
    output_data = {
        "metadata": {
            "model": model_name,
            "languages": ["english"] + languages,
            "limit": limit,
            "runs_per_language": repeats,
            "run_date": datetime.now().strftime("%m-%d-%Y, %H:%M:%S"),
        },
        "results": [],
    }

    # Save raw English results
    for idx, english_keywords in enumerate(english_keywords_batch, 1):
        output_data["results"].append({
            "idx": idx,
            "prompts": {"english": english_prompts[idx - 1]},
            "translations": {
                "english": {
                    "premise": dataset[idx - 1]["premise"],
                    "hypothesis": dataset[idx - 1]["hypothesis"],
                }
            },
            "english": english_keywords,
        })

    # Translate prompts for each target language
    for language in languages:
        translator = get_translator(language)
        translated_prompts_marked = [
            split_and_translate(marked_prompt, translator) for marked_prompt in marked_english_prompts
        ]

        # Log translated marked prompts
        print(f"\nTranslated Marked Prompts for {language.capitalize()}:")
        for idx, translated_prompt in enumerate(translated_prompts_marked, 1):
            print(f"Prompt {idx} (Marked): {translated_prompt}\n")

        translated_prompts_unmarked = [
            split_and_translate(unmarked_prompt, translator) for unmarked_prompt in english_prompts
        ]

        print(f"\nQueried {language.capitalize()} Prompts:")
        for idx, translated_prompt in enumerate(translated_prompts_unmarked, 1):
            print(f"Prompt {idx}: {translated_prompt}\n")

        # Run translated prompts through the model
        translated_keywords_batch, reverse_translations_batch = get_keywords_with_reverse_translation(
            translated_prompts_unmarked, model_name, limit, premise_hypothesis_pairs, repeats, language
        )

        # Save translated results
        for idx, (translated_keywords, reverse_translated) in enumerate(zip(translated_keywords_batch, reverse_translations_batch)):
            output_data["results"][idx][language] = translated_keywords
            output_data["results"][idx]["prompts"][language] = translated_prompts_unmarked[idx]
            output_data["results"][idx]["translations"][language] = {
                    "premise": translator(dataset[idx]["premise"])[0]["translation_text"],
                    "hypothesis": translator(dataset[idx]["hypothesis"])[0]["translation_text"],
                }
            if language != "english":
                output_data["results"][idx][f"{language}-to-english"] = reverse_translated

    # Save results to file
    if not output_file:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"keywords_output_{timestamp}.json"

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, output_file)

    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file_path}".replace("../", ""))

    return output_file_path