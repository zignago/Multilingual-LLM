from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline

def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score between a reference (human translation) and hypothesis (model translation)."""
    reference_tokens = [reference.split()]  # Reference should be a list of tokens
    hypothesis_tokens = hypothesis.split()   # Hypothesis should be tokenized as well
    return sentence_bleu(reference_tokens, hypothesis_tokens)

def validate_translation_quality(dataset, translator, human_references):
    """Calculate BLEU scores for each premise and hypothesis in the dataset."""
    bleu_scores = []
    for idx, row in enumerate(dataset):
        # Translate premise and hypothesis using the translator
        translated_premise = translator(row["premise"])[0]['translation_text']
        translated_hypothesis = translator(row["hypothesis"])[0]['translation_text']

        # Calculate BLEU scores for both premise and hypothesis
        premise_bleu = calculate_bleu(human_references[idx]["premise"], translated_premise)
        hypothesis_bleu = calculate_bleu(human_references[idx]["hypothesis"], translated_hypothesis)

        # Store scores
        bleu_scores.append({
            "idx": idx,
            "premise_bleu": premise_bleu,
            "hypothesis_bleu": hypothesis_bleu
        })

    # Calculate average BLEU scores for an overall quality assessment
    avg_premise_bleu = sum(score["premise_bleu"] for score in bleu_scores) / len(bleu_scores)
    avg_hypothesis_bleu = sum(score["hypothesis_bleu"] for score in bleu_scores) / len(bleu_scores)
    
    print(f"Average Premise BLEU: {avg_premise_bleu:.4f}")
    print(f"Average Hypothesis BLEU: {avg_hypothesis_bleu:.4f}")
    return bleu_scores

# Usage within main translation function
def main_translation_with_bleu(languages, human_references):
    dataset = load_dataset("glue", "mnli", split="validation_matched").select(range(20))

    # For each specified language, calculate translations and BLEU scores
    for language in languages:
        translator = get_translator(language)
        print(f"Evaluating translation quality for language: {language}")
        bleu_scores = validate_translation_quality(dataset, translator, human_references)
        
        # Print or store BLEU scores for each translation
        for score in bleu_scores:
            print(f"Sample {score['idx']} - Premise BLEU: {score['premise_bleu']:.4f}, Hypothesis BLEU: {score['hypothesis_bleu']:.4f}")
