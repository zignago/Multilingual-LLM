# Inplementing Intersection Over Union (IoU) Scoring using Jaccard Index

import json
import time
from pathlib import Path
import re


def calculate_iou(set1, set2):
    """Calculate Intersection over Union (IoU) between two sets."""
    intersection = len(set(set1) & set(set2))
    union = len(set(set1) | set(set2))
    if union == 0:
        return 0  # Avoid division by zero
    return intersection / union

def compute_iou_from_json(json_file_path):
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
                iou = calculate_iou(english_keywords, reverse_translated_keywords)
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

def extract_numbers(file_path):
    """Extract numbers from the given file path string."""
    match = re.search(r'_(\d+)\.json$', file_path)
    if match:
        return match.group(1)
    return None

def jaccard(input_json_path):
    # Ensure input file exists
    if not Path(input_json_path).exists():
        print(f"Input file {input_json_path} does not exist.")
        return
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_iou_path = f"outputs/iou_output_{timestamp}.json"

    # Compute IoU and save results
    iou_scores, average_iou = compute_iou_from_json(input_json_path)
    save_iou_results(iou_scores, average_iou, output_iou_path)

    # Print average IoU
    print(f"Average IoU: {average_iou:.4f}")
