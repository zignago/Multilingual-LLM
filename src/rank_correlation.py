import json
from scipy.stats import kendalltau, spearmanr
from collections import defaultdict

def compute_rank_correlation(input_json_file):
    """
    Computes rank correlation metrics (Kendall's Tau and Spearman's Rank) for each language in the provided JSON file.
    Results are saved in a new JSON file with a timestamped filename and include per-language and overall averages.

    :param input_json_file: Path to the JSON file containing LLM output.
    :return: None
    """
    # Load the input JSON file
    with open(input_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data["results"]
    metadata = data["metadata"]
    
    rank_correlation_results = {
        "metadata": metadata,
        "per_language_correlation": {},
        "overall_average_correlation": {},
        "rank_correlation": []
    }

    # Initialize accumulators for average calculation
    kendall_sums = defaultdict(float)
    spearman_sums = defaultdict(float)
    counts = defaultdict(int)

    for entry in results:
        idx = entry["idx"]
        correlations = {"idx": idx, "rank_correlation_metrics": {}}

        for language in metadata["languages"]:
            if language == "english":
                continue  # Skip English base case
            
            english_keywords = entry["english"]
            translated_keywords = entry.get(f"{language}-to-english", [])

            if not translated_keywords:
                correlations["rank_correlation_metrics"][language] = {
                    "error": "Missing reverse-translated keywords."
                }
                continue

            try:
                tau, _ = kendalltau(english_keywords, translated_keywords)
                spearman, _ = spearmanr(english_keywords, translated_keywords)

                correlations["rank_correlation_metrics"][language] = {
                    "kendall_tau": tau,
                    "spearman_rank": spearman
                }

                # Accumulate results for averages
                kendall_sums[language] += tau if tau is not None else 0
                spearman_sums[language] += spearman if spearman is not None else 0
                counts[language] += 1

            except Exception as e:
                correlations["rank_correlation_metrics"][language] = {
                    "error": f"Failed to compute metrics: {e}"
                }

        rank_correlation_results["rank_correlation"].append(correlations)

    # Compute per-language and overall averages
    overall_kendall_sum = 0
    overall_spearman_sum = 0
    overall_count = 0

    for language in kendall_sums:
        if counts[language] > 0:
            avg_kendall = kendall_sums[language] / counts[language]
            avg_spearman = spearman_sums[language] / counts[language]

            rank_correlation_results["per_language_correlation"][language] = {
                "average_kendall_tau": avg_kendall,
                "average_spearman_rank": avg_spearman
            }

            overall_kendall_sum += kendall_sums[language]
            overall_spearman_sum += spearman_sums[language]
            overall_count += counts[language]

    if overall_count > 0:
        rank_correlation_results["overall_average_correlation"] = {
            "kendall_tau": overall_kendall_sum / overall_count,
            "spearman_rank": overall_spearman_sum / overall_count
        }

    # Print results to the console
    print("Rank Correlation Per-Language Averages:")
    for language, averages in rank_correlation_results["per_language_correlation"].items():
        print(f"{language}: Kendall's Tau = {averages['average_kendall_tau']:.4f}, Spearman's Rank = {averages['average_spearman_rank']:.4f}")
    print("\nOverall Average Rank Correlation:")
    print(f"Kendall's Tau = {rank_correlation_results['overall_average_correlation']['kendall_tau']:.4f}, "
          f"Spearman's Rank = {rank_correlation_results['overall_average_correlation']['spearman_rank']:.4f}")

    return rank_correlation_results
