import os
import json
import time
from datasets import load_dataset
from transformers import pipeline
import openai
import torch
import re
import string
from datetime import datetime
import unicodedata
from deep_translator import GoogleTranslator
from collections import Counter
from src.iou_evaluation import compare_with_components
from src.config import LABEL_MAP, TRANSLATION_MODELS, LANGUAGE_CODES, HARDCODE_TRANSLATIONS, PROMPT_COMPONENTS, LABEL_TRANSLATION

openai.api_key = os.getenv("OPENAI_API_KEY")

#TODO
# Hardcode spanish se to "Know" for reverse translation
# Hardcode to strip punctuation from keywords

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

# def translate_static_prompt_parts(translator, limit, language="english"):
#     """
#     Translates only the static instruction parts of the prompt for a given language.
#     Uses hardcoded translations for premise, hypothesis, and label to avoid errors.
#     """
#     if language not in PROMPT_COMPONENTS:
#         raise ValueError(f"Language '{language}' is not supported in PROMPT_COMPONENTS.")

#     # Check all required keys exist
#     for key in ["premise", "hypothesis", "label"]:
#         if key not in PROMPT_COMPONENTS[language]:
#             raise KeyError(f"Missing '{key}' in PROMPT_COMPONENTS for language '{language}'.")

#     # Define the static instruction template
#     static_instruction = (
#         f"Identify the top {limit} keywords relevant to understanding the relationship between the premise and hypothesis. "
#         "Include only words from the premise and hypothesis. Do not include any other words. "
#         "Do not include punctuation or commas ('.' or ',') in keywords.\n"
#         "Premise: [Premise]\nHypothesis: [Hypothesis]\nLabel: [Label]\n\n"
#         'Return keywords in array format like ["a", "b", "c"]. Include only single words, no phrases.'
#     )

#     # Replace placeholders with translations
#     static_instruction = static_instruction.replace("[Premise]", PROMPT_COMPONENTS[language]["premise"])
#     static_instruction = static_instruction.replace("[Hypothesis]", PROMPT_COMPONENTS[language]["hypothesis"])
#     static_instruction = static_instruction.replace("[Label]", PROMPT_COMPONENTS[language]["label"])

#     # Translate the instruction if a translator is provided
#     if translator:
#         static_instruction = translator(static_instruction)[0]['translation_text']

#     return static_instruction

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

def clean_keyword(keyword):
    """
    Clean a keyword by removing punctuation, normalizing accents, and converting to lowercase.
    """
    keyword = unicodedata.normalize("NFKC", keyword)  # Normalize text
    keyword = ''.join(char for char in keyword if char not in string.punctuation)  # Remove punctuation
    return keyword.strip().lower()  # Strip whitespace and convert to lowercase

def reverse_translate_keywords_deeptranslator(keywords, source_language):
    """Translates keywords back to English using deep-translator's Google Translator."""
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
                translated_word = clean_keyword(translated_word)
                reverse_translated_keywords.extend(translated_word.split())  # Split multi-word translations
        except Exception as e:
            print(f"Error translating '{keyword}': {e}")
            reverse_translated_keywords.append(clean_keyword(keyword))

    return list(set(reverse_translated_keywords))  # Deduplicate keywords


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

            # Terminal output for debugging showing the prompt sent to the LLM and the full response received
            # print(f"\n\nprompt: {prompt}\n\n")
            # print(f"response: {response}")

            response_text = response.choices[0].message.content
            
            # Extract keywords and normalize
            keywords = re.findall(r'"(.*?)"', response_text)
            if not keywords:
                keywords = [word.strip() for word in response_text.strip("[]").split(",")]
            keywords = [clean_keyword(word) for word in keywords]  # Normalize keywords
            
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

def main(languages, limit, LLM_model, subset=20, repeats=5, output_file=None):
    dataset = load_dataset("glue", "mnli", split="validation_matched").select(range(subset))

    # Generate the static prompt template for English and other languages
    english_instruction = translate_static_prompt_parts(lambda x: [{"translation_text": x}], limit)
    english_prompts = [generate_prompts(row, english_instruction, "english") for row in dataset]
    premise_hypothesis_pairs = [(row["premise"], row["hypothesis"]) for row in dataset]
    
    # Get keywords with post-filtering, running 'x' times for consistency
    english_keywords_batch = get_keywords_from_llm_multiple(english_prompts, LLM_model, limit, premise_hypothesis_pairs, repeats)

    # Initialize output data structure with metadata
    output_data = {
        "metadata": {
            "model": LLM_model,
            "languages": ["english"] + languages,
            "limit": limit,
            "runs_per_language": repeats,
            "run_date": datetime.now().strftime("%m-%d-%Y, %H:%M:%S")
        },
        "results": []
    }

    for idx, english_keywords in enumerate(english_keywords_batch, 1):
        output_data["results"].append({
            "idx": idx,
            "prompts": {
                "english": english_prompts[idx-1]
            },
            "translations": {
                "english": {
                    "premise": dataset[idx-1]["premise"],
                    "hypothesis": dataset[idx-1]["hypothesis"]
                }
            },
            "english": english_keywords
        })

    # Process each language by translating instructions only once and formatting each prompt dynamically
    for language in languages:
        translator = get_translator(language)
        
        # Translate static instruction and dataset
        translated_instruction = translate_static_prompt_parts(translator, limit, language)
        translated_dataset = translate_batch(dataset, translator)
        
        # Generate and translate prompts
        translated_prompts = [
            generate_prompts(row, translated_instruction, language)
            for row in translated_dataset
        ]
        
        translated_premise_hypothesis_pairs = [
            (row["translated_premise"], row["translated_hypothesis"])
            for row in translated_dataset
        ]
        
        # Run multiple times and get the most common keywords, along with reverse translation if applicable
        translated_keywords_batch, reverse_translated_batch = get_keywords_with_reverse_translation(
            translated_prompts, LLM_model, limit, translated_premise_hypothesis_pairs, repeats, language
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
        output_file = f"keywords_output_{timestamp}.json"
    else:
        output_file = output_file

    # Ensure the outputs directory is created in the project root
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full path for the output file
    output_file_path = os.path.join(output_dir, output_file)

    # Write the JSON file
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to outputs/{output_file}")

    return output_file_path
