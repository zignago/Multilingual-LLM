import argparse
import os
import time
import json
import shutil
from src.config import SUPPORTED_LLM_MODELS

def output_handler(iou_results, rank_results, keywords_json_path, model):

    # Generate timestamp and output directory
    timestamp = time.strftime("%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"outputs/{timestamp}_{model}")
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

def main():
    models_formatted_output = "\n".join(
        [f"\n{category.upper()}:\n   - " + "\n   - ".join(models) for category, models in SUPPORTED_LLM_MODELS.items()]
    )

    parser = argparse.ArgumentParser(description="Multilingual LLM Keyword Extraction")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help=f"Specify the LLM model to run. Current supported models:\n{models_formatted_output}\n")
    parser.add_argument("--languages", nargs="+", default=["german"], help="Specify one or more target languages.")
    parser.add_argument("--limit", type=int, default=3, help="Limit to top n keywords.")
    parser.add_argument("--subset", type=int, default=20, help="Amount of entries in the dataset to use. Default=20")
    parser.add_argument("--iterations", type=int, default=1, help="Specify the number of times to run each query.")
    parser.add_argument("--output", type=str, help="Optional custom output filename (without path) for JSON results.")
    args = parser.parse_args()
    
    if not any(args.model in models for models in SUPPORTED_LLM_MODELS.values()):
        print(f"ERROR -- unsupported model '{args.model}'\nCurrent supported models:\n{models_formatted_output}\n")
        return 
    
    print(f"Running with the following parameters:\n"
        f" - Model: {args.model}\n"
        f" - Languages: {', '.join(args.languages)}\n"
        f" - Keyword limit: {args.limit}\n"
        f" - Subset size: {args.subset}\n"
        f" - Prompt iterations: {args.iterations}\n"
        f" - Output file: {args.output if args.output else 'Not specified'}\n")
      
    # Import here to prevent long loads before input validation
    from src.multilingual_llm import main as multilingual_llm
    from src.iou_evaluation import jaccard
    from src.rank_correlation import compute_rank_correlation

    keywords_json_path = multilingual_llm(args.languages, args.limit, args.model, args.subset, args.iterations, args.output)

    output_handler(jaccard(keywords_json_path), compute_rank_correlation(keywords_json_path), keywords_json_path, args.model)
    
if __name__ == "__main__":
    main()