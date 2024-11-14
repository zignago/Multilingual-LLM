import argparse
from multilingual_llm import main
from iou_evaluation import jaccard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual LLM Keyword Extraction")
    parser.add_argument("--languages", nargs="+", default=["german"], help="Specify one or more target languages.")
    parser.add_argument("--limit", type=int, default=3, help="Limit to top n keywords.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Specify the GPT model.")
    parser.add_argument("--subset", type=int, default=20, help="Amount of entries in the dataset to use. Default=20")
    parser.add_argument("--repeat", type=int, default=1, help="Specify the number of times to run each query.")
    parser.add_argument("--output", type=str, help="Optional custom output filename (without path) for JSON results.")
    args = parser.parse_args()
    
    jaccard(main(args.languages, args.limit, args.model, args.subset, args.repeat, args.output))
    
    # Debugging IoU using a specific JSON output:
    # jaccard("outputs/keywords_output_20241112-234949.json")