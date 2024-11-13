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

def calculate_iou_advanced(set1, set2, threshold=0.7):
    """
    Calculate Intersection over Union (IoU) with semantic similarity consideration.
    Two words are considered intersecting if they are semantically comparable.
    """
    set1, set2 = normalize_case(set1), normalize_case(set2)
    intersection = 0

    # Check comparability for each word in set1 with words in set2
    for word1 in set1:
        for word2 in set2:
            if are_words_comparable(word1, word2, threshold):
                intersection += 1
                break

    union = len(set(set1)) + len(set(set2)) - intersection
    if union == 0:
        return 0  # Avoid division by zero
    return intersection / union

def compute_iou_from_json_advanced(json_file_path, threshold=0.7):
    """Compute IoU for reverse-translated keywords compared to original English keywords."""
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    iou_scores = []
    for result in data["results"]:
        english_keywords = result.get("english", [])
        
        # Loop through each language and compute IoU for reverse translation
        for key, value in result.items():
            if key.endswith("-to-english"):  # Check for reverse-translated keywords
                reverse_translated_keywords = value
                iou = calculate_iou_advanced(english_keywords, reverse_translated_keywords, threshold)
                iou_scores.append({
                    "idx": result["idx"],
                    "language": key.replace("-to-english", ""),
                    "iou": iou
                })
    
    # Compute average IoU
    if iou_scores:
        average_iou = sum(score["iou"] for score in iou_scores) / len(iou_scores)
    else:
        average_iou = 0

    # Output results
    return iou_scores, average_iou

def save_iou_results(iou_scores, average_iou, output_path):
    """Save IoU results to a JSON file."""
    results = {
        "iou_scores": iou_scores,
        "average_iou": average_iou
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
    iou_scores, average_iou = compute_iou_from_json_advanced(input_json_path, threshold=0.7)
    save_iou_results(iou_scores, average_iou, output_iou_path)

    # Print average IoU
    print(f"Average IoU: {average_iou:.4f}")
