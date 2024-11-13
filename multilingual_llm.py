import os
import json
import time
from datasets import load_dataset
import openai
import torch
import re
import argparse
from deep_translator import GoogleTranslator  # Import deep_translator

openai.api_key = os.getenv("OPENAI_API_KEY")

# Import configurations from config.py
from config import LABEL_MAP, LANGUAGE_CODES

device = 0 if torch.cuda.is_available() else -1

def get_translator(language):
    """Retrieves the GoogleTranslator for the specified language."""
    if language == "english":
        return None  # Skip translation for English
    try:
        target_lang_code = LANGUAGE_CODES[language]
    except KeyError:
        raise ValueError(f"Unsupported language '{language}'. Choose from: {', '.join(LANGUAGE_CODES.keys())}")
    
    return GoogleTranslator(source="en", target=target_lang_code)

def translate_text(text, translator):
    """Translate a single piece of text using deep_translator."""
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"Error translating '{text}': {e}")
        return text  # Return the original text in case of error

def translate_batch(dataset, language):
    """Batch translate premises and hypotheses in the dataset."""
    translator = get_translator(language)
    if translator:
        dataset = dataset.map(
            lambda examples: {
                "translated_premise": [translate_text(text, translator) for text in examples["premise"]],
                "translated_hypothesis": [translate_text(text, translator) for text in examples["hypothesis"]],
            },
            batched=True,
        )
    return dataset

def translate_static_prompt_parts(translator, limit):
    """Translates only the static instruction parts of the prompt for a given language."""
    static_instruction = (
        f"Identify the top {limit} keywords relevant to understanding the relationship between the premise and hypothesis. "
        "Only include words from the premise and hypothesis. Do not include any other words.\n"
        "Premise: [Premise]\nHypothesis: [Hypothesis]\nLabel: [Label]\n\n"
        'Return keywords in array format like ["a", "b", "c"]. Only include single words, no phrases.'
    )
    return translate_text(static_instruction, translator)

def generate_prompts(row, translated_instruction, language="english"):
    """Generates a single prompt by replacing placeholders with specific row data."""
    return translated_instruction.replace("[Premise]", row["translated_premise"] if language != "english" else row["premise"]) \
                                 .replace("[Hypothesis]", row["translated_hypothesis"] if language != "english" else row["hypothesis"]) \
                                 .replace("[Label]", LABEL_MAP[row["label"]])

def filter_keywords(keywords, premise, hypothesis):
    """Filter out any keywords that are not in the premise or hypothesis."""
    allowed_words = set(premise.split() + hypothesis.split())
    return [word for word in keywords if word in allowed_words]

def enforce_limit(keywords, limit, premise, hypothesis):
    """Ensures that the keyword list meets the required limit by adding words from the premise or hypothesis if necessary."""
    if len(keywords) < limit:
        extra_words = (premise + " " + hypothesis).split()
        extra_words = [word for word in extra_words if word not in keywords]
        keywords.extend(extra_words[:limit - len(keywords)])
    return keywords[:limit]

from collections import Counter

def get_common_keywords(all_keywords, limit):
    """Selects the top 'limit' most common keywords from a list of keyword lists."""
    flat_keywords = [keyword for keywords in all_keywords for keyword in keywords]
    keyword_counts = Counter(flat_keywords)
    return [keyword for keyword, _ in keyword_counts.most_common(limit)]

def get_keywords_from_llm_multiple(prompts, LLM_model, limit, premise_hypothesis_pairs, x):
    """Runs the keyword extraction 'x' times for each prompt and returns the most common 'limit' keywords."""
    keywords_batch = []
    for idx, prompt in enumerate(prompts):
        all_run_keywords = []
        for run in range(x):
            response = openai.chat.completions.create(
                model=LLM_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0
            )
            response_text = response.choices[0].message.content
            keywords = re.findall(r'"(.*?)"', response_text) or response_text.strip("[]").split(",")
            premise, hypothesis = premise_hypothesis_pairs[idx]
            filtered_keywords = filter_keywords(keywords, premise, hypothesis)
            limited_keywords = enforce_limit(filtered_keywords, limit, premise, hypothesis)
            all_run_keywords.append(limited_keywords)
            print(f"Run {run + 1} for prompt {idx + 1}: {limited_keywords}")
        common_keywords = get_common_keywords(all_run_keywords, limit)
        keywords_batch.append(common_keywords)
    return keywords_batch

def reverse_translate_keywords_deeptranslator(keywords, source_language):
    """Translates keywords back to English using GoogleTranslator."""
    source_code = LANGUAGE_CODES.get(source_language)
    reverse_translated_keywords = []
    for keyword in keywords:
        try:
            translated_word = GoogleTranslator(source=source_code, target="en").translate(keyword)
            reverse_translated_keywords.append(translated_word.strip())
        except Exception as e:
            print(f"Error translating '{keyword}': {e}")
            reverse_translated_keywords.append(keyword)
    return reverse_translated_keywords

def main(languages, limit, LLM_model, subset, x, output_file=None):
    dataset = load_dataset("glue", "mnli", split="validation_matched").select(range(subset))
    english_instruction = translate_static_prompt_parts(lambda x: [{"translation_text": x}], limit)
    english_prompts = [generate_prompts(row, english_instruction, "english") for row in dataset]
    premise_hypothesis_pairs = [(row["premise"], row["hypothesis"]) for row in dataset]
    
    english_keywords_batch = get_keywords_from_llm_multiple(english_prompts, LLM_model, limit, premise_hypothesis_pairs, x)
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

    for language in languages:
        translator = get_translator(language)
        translated_instruction = translate_static_prompt_parts(translator, limit)
        translated_dataset = translate_batch(dataset, language)
        translated_prompts = [generate_prompts(row, translated_instruction, language) for row in translated_dataset]
        translated_premise_hypothesis_pairs = [(row["translated_premise"], row["translated_hypothesis"]) for row in translated_dataset]
        translated_keywords_batch = get_keywords_from_llm_multiple(
            translated_prompts, LLM_model, limit, translated_premise_hypothesis_pairs, x
        )

        for idx, translated_keywords in enumerate(translated_keywords_batch):
            reverse_translated = reverse_translate_keywords_deeptranslator(translated_keywords, language)
            output_data["results"][idx][language] = translated_keywords
            output_data["results"][idx]["prompts"][language] = translated_prompts[idx]
            output_data["results"][idx]["translations"][language] = {
                "premise": translated_dataset[idx]["translated_premise"],
                "hypothesis": translated_dataset[idx]["translated_hypothesis"]
            }
            if language != "english":
                output_data["results"][idx][f"{language}-to-english"] = reverse_translated

    if not output_file:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"outputs/keywords_output_{timestamp}.json"
    else:
        output_file = f"outputs/{output_file}"

    os.makedirs("outputs", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")
