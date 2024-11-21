import os
import json
import time
from datetime import datetime
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import openai
import torch
import re
import string
import unicodedata
from src.llama_handler import LlamaHandler
from src.iou_evaluation import compare_with_components
from deep_translator import GoogleTranslator
from collections import Counter
from src.config import (
    LABEL_MAP, TRANSLATION_MODELS, LANGUAGE_CODES, HARDCODE_TRANSLATIONS,
    PROMPT_COMPONENTS, LABEL_TRANSLATION, SUPPORTED_LLM_MODELS
)

device = 0 if torch.cuda.is_available() else -1

# Initialize LLaMA handler
llama_handler = LlamaHandler()

openai.api_key = os.getenv("OPENAI_API_KEY")

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

def translate_static_prompt_parts(translator, limit, language="english"):
    """
    Translates only the static instruction parts of the prompt for a given language.
    Uses hardcoded translations for premise, hypothesis, and label to avoid errors.
    """
    # Define the static instruction template with placeholders
    static_instruction = (
        f"Identify the top {limit} keywords relevant to understanding the relationship between the premise and hypothesis. "
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

# Works for full prompt, just leaves parts untranslated
def generate_prompts(row, translated_instruction, language="english"):
    """
    Generates a single prompt by replacing placeholders with specific row data.
    Ensures that premise, hypothesis, and label are correctly populated.
    """
    # Replace placeholders with actual premise, hypothesis, and label
    premise = row["translated_premise"] if language != "english" else row["premise"]
    hypothesis = row["translated_hypothesis"] if language != "english" else row["hypothesis"]

    label = LABEL_MAP[row["label"]]
    if language != "english":
        label = LABEL_TRANSLATION[language][label]

    # Output for debugging placeholders
    # print(f"pre-replacement prompt: {translated_instruction}")

    # Replace placeholders in the instruction
    prompt = translated_instruction.replace("[Premise]", premise)
    prompt = prompt.replace("[Hypothesis]", hypothesis)
    prompt = prompt.replace("[Label]", label)

    # Make sure that placeholders in the instruction are resolved, even if they are translated (i.e., "[Premere]", "[Hypothese]", etc.)
    if language != "english":
        for component in PROMPT_COMPONENTS[language]:
            placeholder = f"[{PROMPT_COMPONENTS[language][component]}]"
            if placeholder in prompt:
                if component == "Premise":
                    prompt = prompt.replace(placeholder, premise)
                elif component == "Hypothesis":
                    prompt = prompt.replace(placeholder, hypothesis)
                elif component == "Label":
                    prompt = prompt.replace(placeholder, label)

    # Check for unresolved English placeholders
    if "[Premise]" in prompt or "[Hypothesis]" in prompt or "[Label]" in prompt:
        raise ValueError(f"Unresolved placeholders in prompt: {prompt}")
    
    # Check for unresolved Translated placeholders
    if f"[{PROMPT_COMPONENTS[language]["Premise"]}]" in prompt or f"[{PROMPT_COMPONENTS[language]["Hypothesis"]}]" in prompt or f"[{PROMPT_COMPONENTS[language]["Label"]}]" in prompt:
        raise ValueError(f"Unresolved placeholders in prompt: {prompt}")

    return prompt

def filter_keywords(keywords, premise, hypothesis, language="english"):
    """
    Filter out any keywords that are not in the premise or hypothesis.
    Also accounts for multi-word translations and compound words.
    """
    allowed_words = set(premise.split() + hypothesis.split())
    filtered_keywords = []

    for keyword in keywords:
        if keyword in allowed_words:
            filtered_keywords.append(keyword)
        elif compare_with_components(keyword, allowed_words, language):
            filtered_keywords.append(keyword)

    return filtered_keywords

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

def determine_model_type(model_name):
    """Determines whether the model is GPT-based or LLaMA-based."""
    for model_type, models in SUPPORTED_LLM_MODELS.items():
        if model_name in models:
            return model_type
    raise ValueError(f"Model {model_name} is not supported. Supported models: {SUPPORTED_LLM_MODELS}")

def refine_prompt_for_o1_preview(prompt):
    return f"Task: {prompt.strip()}\n\nOutput: Keywords: [\"a\", \"b\", \"c\", \"d\"]"

def run_gpt(prompt, model_name):
    """Runs GPT-based models using OpenAI API."""
    is_o1_preview = "o1-preview" in model_name
    if is_o1_preview:
        prompt = refine_prompt_for_o1_preview(prompt)

    max_tokens_param = "max_tokens" if not is_o1_preview else "max_completion_tokens"
    request_params = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        max_tokens_param: 50,
    }
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

def reverse_translate_keywords_deeptranslator(keywords, source_language):
    """Translates keywords back to English using deep-translator's Google Translator, preserving order."""
    source_code = LANGUAGE_CODES.get(source_language)
    if not source_code:
        raise ValueError(f"Unsupported language '{source_language}'. Available languages: {list(LANGUAGE_CODES.keys())}")

    reverse_translated_keywords = []
    for keyword in keywords:
        try:
            if keyword in HARDCODE_TRANSLATIONS:
                reverse_translated_keywords.append(HARDCODE_TRANSLATIONS[keyword])
            else:
                translated_word = GoogleTranslator(source=source_code, target="en").translate(keyword)
                reverse_translated_keywords.append(clean_keyword(translated_word))
        except Exception as e:
            print(f"Error translating '{keyword}': {e}")
            reverse_translated_keywords.append(clean_keyword(keyword))

    return reverse_translated_keywords  # Order is preserved

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

    for idx, prompt in enumerate(prompts):
        all_run_keywords = []

        # Run the keyword extraction 'x' times
        for run in range(x):
            model_type = determine_model_type(model_name)
            if model_type == "gpt":
                response_text = run_gpt(prompt, model_name)
            elif model_type == "llama":
                # Assuming label is embedded in the prompt
                response_text = run_llama(prompt, model_name, premise_hypothesis_pairs[idx][2])

            # Parse and validate keywords from the response
            premise, hypothesis = premise_hypothesis_pairs[idx][:2]
            keywords = parse_keywords_from_response(response_text, premise, hypothesis)

            # Enforce limit and add to collection
            limited_keywords = enforce_limit(keywords, limit, premise, hypothesis)
            all_run_keywords.append(limited_keywords)

            print(f"Run {run + 1} for prompt {idx + 1}: {limited_keywords}")

        # Determine the most common keywords
        common_keywords = get_common_keywords(all_run_keywords, limit)
        keywords_batch.append(common_keywords)

        # Reverse translate keywords back to English if language is not English
        if language_name != "english":
            reverse_translated_keywords = reverse_translate_keywords_deeptranslator(common_keywords, language_name)
            # Maintain order during reverse translation
            reverse_translated_keywords_ordered = [
                reverse_translated_keywords[common_keywords.index(k)] if k in common_keywords else k for k in common_keywords
            ]
            reverse_translations_batch.append(reverse_translated_keywords_ordered)
        else:
            reverse_translations_batch.append(common_keywords)

    return keywords_batch, reverse_translations_batch

def main(languages, limit, model_name, subset=20, repeats=5, output_file=None):
    """
    Main function for multilingual keyword extraction and reverse translation.
    """
    # Load dataset and select a subset
    dataset = load_dataset("glue", "mnli", split="validation_matched").select(range(subset))

    # Generate prompts and premise-hypothesis pairs for English
    english_instruction = translate_static_prompt_parts(None, limit)
    english_prompts = [generate_prompts(row, english_instruction, "english") for row in dataset]
    premise_hypothesis_pairs = [(row["premise"], row["hypothesis"], LABEL_MAP[row["label"]]) for row in dataset]

    # Run the English prompts through the model
    english_keywords_batch, _ = get_keywords_with_reverse_translation(
        english_prompts, model_name, limit, premise_hypothesis_pairs, repeats, "english"
    )

    # Initialize output data structure
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

    # Save English results
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

    # Process each additional language
    for language in languages:
        translator = get_translator(language)
        translated_instruction = translate_static_prompt_parts(translator, limit, language)
        translated_dataset = translate_batch(dataset, translator)

        translated_prompts = [
            generate_prompts(row, translated_instruction, language) for row in translated_dataset
        ]
        translated_premise_hypothesis_pairs = [
            (row["translated_premise"], row["translated_hypothesis"], LABEL_TRANSLATION[language][LABEL_MAP[row["label"]]])
            for row in translated_dataset
        ]

        # Get keywords and reverse translations
        translated_keywords_batch, reverse_translated_batch = get_keywords_with_reverse_translation(
            translated_prompts, model_name, limit, translated_premise_hypothesis_pairs, repeats, language
        )

        # Save translated results
        for idx, (translated_keywords, reverse_translated) in enumerate(zip(translated_keywords_batch, reverse_translated_batch)):
            output_data["results"][idx][language] = translated_keywords
            output_data["results"][idx]["prompts"][language] = translated_prompts[idx]
            output_data["results"][idx]["translations"][language] = {
                "premise": translated_dataset[idx]["translated_premise"],
                "hypothesis": translated_dataset[idx]["translated_hypothesis"],
            }
            if language != "english":  # Add reverse translations only for non-English
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
    print(f"Results saved to {output_file_path}")

    return output_file_path