import os
from datasets import load_dataset
from transformers import pipeline
import openai
import torch
import re
import argparse


# TDOO:
# - Fix "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset"
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

# Initialize the translator based on the chosen language
def get_translator(language):
    try:
        model_name = TRANSLATION_MODELS[language]
    except KeyError:
        raise ValueError(f"Unsupported language '{language}'. Please choose a supported language from: {', '.join(TRANSLATION_MODELS.keys())}")
    
    return pipeline("translation", model=model_name, device=device)

# Function to translate labels once and store them based on the selected language
def get_translated_labels(language):
    translator = get_translator(language)
    return [
        translator(LABEL_MAP[0])[0]['translation_text'], 
        translator(LABEL_MAP[1])[0]['translation_text'], 
        translator(LABEL_MAP[2])[0]['translation_text']
    ]

# Translate the prompt template into the target language
def translate_prompt_template(translator, limit):
    # Use unique placeholder tokens to avoid conflicts during translation
    if limit > 1:
        response_limit = f"a list of the top {limit} most relevant keywords"
    elif limit == 1:
        response_limit = "the single most relevant keyword"
    else:
        response_limit = "a list of the most relevant keywords"

    prompt_template_en = (
        "Identify the most important keywords relevant to understanding the relationship between the premise and hypothesis. "
        "The label is provided to indicate their relationship.\n"
        "Premise: {premise_placeholder}\n"
        "Hypothesis: {hypothesis_placeholder}\n"
        "Label: {label_placeholder}\n\n"
        f"Provide {response_limit} in array format. Use only words from the premise and hypothesis, not from the label."
        "Your response should list the words in an array format (for example, [\"a\", \"b\", \"c\"])"
        "Be sure to only respond with words and not phrases composed of multiple words."
    )
    translated_template = translator(prompt_template_en)[0]['translation_text']
    
    # Replace placeholders with the correct tokens after translation
    translated_template = translated_template.replace("premise_placeholder", "premise")\
                                             .replace("hypothesis_placeholder", "hypothesis")\
                                             .replace("label_placeholder", "label")
    print(translated_template)
    return translated_template

# Batch translation of premises, hypotheses, and labels
def translate_batch(translator, premises, hypotheses, labels):
    translated_premises = translator(premises)
    translated_hypotheses = translator(hypotheses)
    translated_labels = [translator(LABEL_MAP[label])[0]['translation_text'] for label in labels]
    return (
        [item['translation_text'] for item in translated_premises],
        [item['translation_text'] for item in translated_hypotheses],
        translated_labels
    )

# Function to remove translated label words and duplicates from the given keywords array
# def clean_keywords(keywords, banned_words):
#     unique_keywords = []
#     seen_words = set()

#     for word in keywords:
#         # Check if the word is not banned and hasn't been seen in this array yet
#         if word not in banned_words and word not in seen_words:
#             unique_keywords.append(word)
#             seen_words.add(word)  # Mark this word as seen

#     return unique_keywords

# Generate prompts for each data point in batch using the translated prompt template
def generate_prompts(translated_template, translated_premises, translated_hypotheses, translated_labels):
    prompts = []
    for premise, hypothesis, label in zip(translated_premises, translated_hypotheses, translated_labels):
        prompt = translated_template.format(premise=premise, hypothesis=hypothesis, label=label)
        prompts.append(prompt)
    return prompts

def get_keywords_from_llm_batch(prompts, LLM_model):
    keywords_batch = []
    for prompt in prompts:
        response = openai.chat.completions.create(
            model=LLM_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0
        )
        response_text = response.choices[0].message.content
        
        # Print the response for debugging
        print(f"LLM Response: {response_text}")
        
        # Try to find keywords in a list format
        keywords = re.findall(r'"(.*?)"', response_text)
        
        # If the regex extraction fails, try splitting by commas as a fallback
        if not keywords:
            # Attempt to parse manually if response doesn't contain quotation marks
            keywords = [word.strip() for word in response_text.strip("[]").split(",")]
        
        keywords_batch.append(keywords)
    return keywords_batch


# Main function to get and translate data, and query LLM in batch
def main(language, limit, LLM_model):
    print(f"\nRunning reasoning analysis in {language}.\nLLM model: {LLM_model}\n")

    # Initialize the translator and translated labels
    translator = get_translator(language)
    banned_words = get_translated_labels(language)

    # Translate the prompt template into the target language
    translated_template = translate_prompt_template(translator, limit)

    # Load the dataset and extract only the first 20 items
    dataset = dataset = load_dataset('glue', 'mnli', split='validation_matched')
    dataset = dataset.select(range(20))  # Use only the first 20 items

    # Prepare data lists
    premises = [item['premise'] for item in dataset]
    hypotheses = [item['hypothesis'] for item in dataset]
    labels = [item['label'] for item in dataset]

    # Ensure no duplicate entries
    data_ids = set()  # A set to track unique dataset entries
    unique_premises, unique_hypotheses, unique_labels = [], [], []

    for i, (premise, hypothesis, label) in enumerate(zip(premises, hypotheses, labels)):
        entry_id = f"{premise}-{hypothesis}-{label}"
        if entry_id not in data_ids:
            data_ids.add(entry_id)
            unique_premises.append(premise)
            unique_hypotheses.append(hypothesis)
            unique_labels.append(label)

    # Batch translate text and labels
    translated_premises, translated_hypotheses, translated_labels = translate_batch(
        translator, unique_premises, unique_hypotheses, unique_labels
    )

    # Generate batch prompts
    prompts = generate_prompts(translated_template, translated_premises, translated_hypotheses, translated_labels)

    # Get keywords in batch from the LLM
    keywords_batch = get_keywords_from_llm_batch(prompts, LLM_model)

    # Clean keywords by removing any label terms
    # cleaned_keywords_batch = [clean_keywords(keywords, banned_words) for keywords in keywords_batch]

    # Display cleaned keywords for each item
    for i, cleaned_keywords in enumerate(keywords_batch):
        print(f"Sample {i + 1} Keywords: {cleaned_keywords}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual LLM Keyword Extraction")
    parser.add_argument("--language", type=str, default="spanish", help="Specify the target language for translation.")
    parser.add_argument("--limit", type=int, default=0, help="Specify the maximum number of responses (limit to top n most important words). Default: no limit")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Specify the GPT model to test. Default: gpt-3.5-turbo")
    args = parser.parse_args()

    # Run the main function with the specified language
    main(args.language.lower(), args.limit, args.model)
