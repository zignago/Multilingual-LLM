import json
from glob import glob
import shutil
import os
import time
from datetime import datetime

DIRECTORY_NAME = "LANGUAGES_llama"
MODEL_NAME = "llama3.1-8b"

def output_handler(iou_results, rank_results, keywords_json_path):

    # Generate timestamp and output directory
    timestamp = time.strftime("%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"outputs/{timestamp}_combined")
    os.makedirs(output_dir, exist_ok=True)

    # Save IoU results to JSON
    output_iou_path = os.path.join(output_dir, f"iou_output_{timestamp}.json")
    with open(output_iou_path, "w", encoding="utf-8") as file:
        json.dump(iou_results, file, ensure_ascii=False, indent=4)
    print(f"IoU results saved to {output_iou_path}")

    # Save rank correlation results to JSON
    output_rank_path = os.path.join(output_dir, f"rank_output_{timestamp}.json")
    with open(output_rank_path, "w", encoding="utf-8") as file:
        json.dump(rank_results, file, ensure_ascii=False, indent=4)
    print(f"Rank correlation results saved to {output_rank_path}")

    # Move keywords JSON into the output directory
    output_keywords_path = os.path.join(output_dir, os.path.basename(keywords_json_path))
    try:
        shutil.move(keywords_json_path, output_keywords_path)
        print(f"File moved successfully to {output_keywords_path}")
    except FileNotFoundError:
        print(f"Source file {keywords_json_path} not found.")
    except Exception as e:
        print(f"Error occurred: {e}")

# Paths to all JSON files
file_paths = glob(f"/Users/gianz/Desktop/Multilingual-LLM/outputs/{DIRECTORY_NAME}/*.json")  # Adjust the path to where your JSON files are located

# Initialize the combined structure
combined_json = {
    "metadata": {
        "model": MODEL_NAME,
        "languages": ["english"],
        "limit": 3,
        "runs_per_language": 1,
        "run_date": datetime.now().strftime("%m-%d-%Y, %H:%M:%S")
    },
    "results": []
}

# Collect all languages and results
language_set = set(["english"])
results_dict = {}

for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as f:  # Specify UTF-8 encoding
        data = json.load(f)
        # Add new languages to metadata
        languages = data["metadata"]["languages"]
        language_set.update(languages)
        
        # Process results
        for result in data["results"]:
            idx = result["idx"]
            if idx not in results_dict:
                # Initialize result if not already added
                results_dict[idx] = {
                    "idx": idx,
                    "prompts": {"english": result["prompts"]["english"]},
                    "translations": {"english": result["translations"]["english"]},
                    "english": result.get("english", [])
                }
            # Add other language data
            for lang in languages:
                if lang != "english":  # Skip English as it is already included
                    results_dict[idx]["prompts"][lang] = result["prompts"].get(lang, "")
                    results_dict[idx]["translations"][lang] = result["translations"].get(lang, {})
                    results_dict[idx][lang] = result.get(lang, [])
                    results_dict[idx][f"{lang}-to-english"] = result.get(f"{lang}-to-english", [])

# Update combined metadata and results
combined_json["metadata"]["languages"] = list(language_set)
combined_json["results"] = list(results_dict.values())

# Save the combined JSON file
output_path = "combined_keywords_gpt_output.json"  # Update the path as needed
with open(output_path, 'w', encoding='utf-8') as f:  # Save file with UTF-8 encoding
    json.dump(combined_json, f, indent=4, ensure_ascii=False)  # Ensure non-ASCII characters are not escaped

print(f"Combined JSON saved to {output_path}")

from iou_evaluation import jaccard
from rank_correlation import compute_rank_correlation

output_handler(jaccard(output_path), compute_rank_correlation(output_path), output_path)