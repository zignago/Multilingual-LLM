# Inplementing Intersection Over Union (IoU) Scoring using Jaccard Index

import json
import re
import time
from pathlib import Path
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

# Load a pre-trained embedding model for semantic similarity
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def normalize_case(keywords):
    """Convert all keywords to lowercase for case-insensitive comparison."""
    return [keyword.lower() for keyword in keywords]

def calculate_string_similarity(word1, word2):
    """Calculate similarity between two words using Levenshtein Distance or similar string comparison."""
    return SequenceMatcher(None, word1, word2).ratio()

def calculate_vector_similarity(word1, word2):
    """Calculate cosine similarity between two words using embedding vectors."""
    embeddings = embedding_model.encode([word1, word2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return similarity[0][0]

def are_words_comparable(word1, word2, threshold=0.7):
    """Determine if two words are semantically comparable."""
    word1, word2 = word1.lower(), word2.lower()  # Normalize case

    # High string similarity is considered comparable
    if calculate_string_similarity(word1, word2) > threshold:
        return True

    # High semantic similarity based on vector comparison
    if calculate_vector_similarity(word1, word2) > threshold:
        return True

    return False

def calculate_iou_advanced(set1, set2, threshold=0.7, language="english"):
    """
    Calculate Intersection over Union (IoU) with semantic similarity consideration.
    Account for multi-word translations and compound words.
    """
    set1, set2 = normalize_case(set1), normalize_case(set2)
    intersection = 0

    for word1 in set1:
        for word2 in set2:
            if are_words_comparable(word1, word2, threshold) or compare_with_components(word1, set2, language, threshold):
                intersection += 1
                break

    union = len(set(set1)) + len(set(set2)) - intersection
    if union == 0:
        return 0  # Avoid division by zero
    return intersection / union





import nltk
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load pre-trained embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def split_compound_word(word, language="german"):
    """
    Decompose a compound word into its components.
    For German, use heuristics or a dictionary-based approach.
    """
    if language == "german":
        # Simplified heuristic for splitting German compound words
        # In practice, you might use a library like `germanet` or `compoundsplit`
        components = re.findall('[A-Z][^A-Z]*', word)
        return components if components else [word]
    return [word]

def compare_with_components(word, keywords, language="english", threshold=0.7):
    """
    Check if any component of the word matches the keywords semantically or directly.
    """
    components = split_compound_word(word, language)
    for component in components:
        for keyword in keywords:
            if are_words_comparable(component, keyword, threshold):
                return True
    return False

def are_words_comparable(word1, word2, threshold=0.7):
    """Determine if two words are semantically comparable."""
    word1, word2 = word1.lower(), word2.lower()  # Normalize case

    # Check high string similarity
    if calculate_string_similarity(word1, word2) > threshold:
        return True

    # Check high semantic similarity based on embedding vectors
    embeddings = embedding_model.encode([word1, word2])
    if cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] > threshold:
        return True

    return False







def compute_iou_from_json_advanced(json_file_path, threshold=0.7):
    """Compute IoU for reverse-translated keywords compared to original English keywords, per language."""
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    iou_scores = {}
    per_language_iou = {}
    total_iou = []
    
    for result in data["results"]:
        english_keywords = result.get("english", [])

        # Loop through each language and compute IoU for reverse translation
        for key, value in result.items():
            if key.endswith("-to-english"):  # Check for reverse-translated keywords
                language = key.replace("-to-english", "")
                reverse_translated_keywords = value
                iou = calculate_iou_advanced(english_keywords, reverse_translated_keywords, threshold)

                # Add IoU for the specific language
                if language not in per_language_iou:
                    per_language_iou[language] = []
                per_language_iou[language].append(iou)

                # Append to the total IoU list for overall average
                total_iou.append(iou)

                # Save individual scores
                if result["idx"] not in iou_scores:
                    iou_scores[result["idx"]] = {}
                iou_scores[result["idx"]][language] = iou

    # Compute average IoU per language
    language_averages = {lang: sum(scores) / len(scores) for lang, scores in per_language_iou.items()}

    # Compute overall average IoU
    overall_average_iou = sum(total_iou) / len(total_iou) if total_iou else 0

    # Output results
    return iou_scores, language_averages, overall_average_iou

def save_iou_results(iou_scores, language_averages, overall_average_iou, output_path):
    """Save IoU results to a JSON file."""
    results = {
        "per_language_iou": language_averages,
        "overall_average_iou": overall_average_iou,
        "detailed_iou_scores": iou_scores
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"IoU results saved to {output_path}")

def jaccard(input_json_path):
    # Ensure input file exists
    if not Path(input_json_path).exists():
        print(f"Input file {input_json_path} does not exist.")
        return
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_iou_path = f"outputs/iou_output_{timestamp}.json"

    # Compute IoU and save results
    iou_scores, language_averages, overall_average_iou = compute_iou_from_json_advanced(input_json_path, threshold=0.7)
    save_iou_results(iou_scores, language_averages, overall_average_iou, output_iou_path)

    # Print per-language averages and overall average
    print("Per-Language IoU Averages:")
    for lang, avg in language_averages.items():
        print(f"{lang}: {avg:.4f}")
    print(f"Overall Average IoU: {overall_average_iou:.4f}")
