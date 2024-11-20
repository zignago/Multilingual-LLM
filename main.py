import argparse
from src.config import SUPPORTED_LLM_MODELS

def main():
    models_formatted_output = "\n".join(
        [f"\n{category.upper()}:\n   - " + "\n   - ".join(models) for category, models in SUPPORTED_LLM_MODELS.items()]
    )

    parser = argparse.ArgumentParser(description="Multilingual LLM Keyword Extraction")
    parser.add_argument("--languages", nargs="+", default=["german"], help="Specify one or more target languages.")
    parser.add_argument("--limit", type=int, default=3, help="Limit to top n keywords.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help=f"Specify the LLM model to run. Current supported models:\n{models_formatted_output}\n")
    parser.add_argument("--subset", type=int, default=20, help="Amount of entries in the dataset to use. Default=20")
    parser.add_argument("--repeat", type=int, default=1, help="Specify the number of times to run each query.")
    parser.add_argument("--output", type=str, help="Optional custom output filename (without path) for JSON results.")
    args = parser.parse_args()
    
    if args.model not in SUPPORTED_LLM_MODELS:
        print(f"ERROR -- unsupported model '{args.model}'\nCurrent supported models:\n{models_formatted_output}\n")
        return 
    
    print(f"Running with the following parameters:\n"
      f" - Languages: {', '.join(args.languages)}\n"
      f" - Limit: {args.limit}\n"
      f" - Model: {args.model}\n"
      f" - Subset size: {args.subset}\n"
      f" - Repeat: {args.repeat}\n"
      f" - Output file: {args.output if args.output else 'Not specified'}\n")
      
    from src.multilingual_llm import main
    from src.iou_evaluation import jaccard

    jaccard(main(args.languages, args.limit, args.model, args.subset, args.repeat, args.output))

if __name__ == "__main__":
    main()