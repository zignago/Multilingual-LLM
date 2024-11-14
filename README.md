# Multilingual LLM Keyword Extraction and IoU Evaluation

## Overview
This project enables users to analyze the multilingual capabilities of large language models (LLMs) by extracting keywords from textual inputs across multiple languages and comparing these keywords to their reverse translations in English. The program calculates the Intersection over Union (IoU) score to evaluate the semantic alignment of keywords between English and other languages, accounting for semantic and linguistic nuances.

## Purpose
The primary goal of this project is to:
1. Evaluate the multilingual reasoning capabilities of LLMs.
2. Assess the semantic accuracy of keywords extracted from translated and reverse-translated text.
3. Provide users with insights into how well translated keywords align with their original English counterparts.

The project leverages:
- Large Language Models (e.g., OpenAI GPT-3.5-turbo).
- Translation APIs (Deep Translator).
- Semantic similarity models (SentenceTransformers).
- Jaccard index for IoU calculations.

## Features
- **Keyword Extraction**: Extracts keywords from premises and hypotheses in English and multiple target languages.
- **Translation Support**: Translates prompts, premises, and hypotheses into supported languages and reverse-translates extracted keywords back to English.
- **IoU Calculation**: Computes Intersection over Union scores to assess semantic similarity.
- **Customizable Runs**: Allows users to specify the number of runs for repeated LLM keyword extractions and aggregates the most common keywords.

## Supported Models
The project supports:
- **Large Language Models**: GPT-3.5-turbo (default). Users can specify other LLM models in the configuration.
- **Semantic Similarity Models**: SentenceTransformers ("sentence-transformers/all-MiniLM-L6-v2").
- **Translation Models**: Deep Translator (Google Translate).

## Supported Languages
The program supports the following languages:
- **English** (default)
- **Spanish**
- **German**

More languages can be added by extending the `LANGUAGE_CODES` and `HARDCODE_TRANSLATIONS` dictionaries.

## Dataset
The program uses the **GLUE MNLI dataset** for textual inputs. This dataset provides:
- **Premises and Hypotheses**: Text samples with entailment relationships.
- **Labels**: Indicating relationships (e.g., Entailment, Neutral, Contradiction).

The dataset is preprocessed and translated into target languages to facilitate keyword extraction and IoU evaluation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zignago/Multilingual-LLM.git
   cd Multilingual-LLM
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Program
To run the main program:
```bash
python main.py --languages spanish german --limit 5 --model gpt-3.5-turbo --repeat 5 --output output.json
```
#### Arguments:
- `--languages`: Specify target languages (e.g., `spanish`, `german`).
- `--limit`: Number of keywords to extract per prompt.
- `--model`: LLM to use (default: `gpt-3.5-turbo`).
- `--repeat`: Number of runs to perform for each query (default: `1`).
- `--output`: Filename for the JSON output (optional).

### IoU Evaluation
To calculate IoU scores:
```bash
python iou_evaluation.py --input outputs/output.json
```
This generates IoU scores for each language and the overall average.

## Output Format
The program outputs a JSON file with the following structure:
```json
{
    "metadata": {
        "model": "gpt-3.5-turbo",
        "languages": ["english", "spanish", "german"],
        "limit": 5,
        "runs_per_language": 5
    },
    "results": [
        {
            "idx": 1,
            "english": ["keyword1", "keyword2", "keyword3"],
            "spanish": ["palabra1", "palabra2", "palabra3"],
            "spanish-to-english": ["translated1", "translated2", "translated3"],
            "german": ["wort1", "wort2", "wort3"],
            "german-to-english": ["translated1", "translated2", "translated3"]
        }
    ]
}
```

### IoU Results
IoU results are saved as a separate JSON file, including per-language IoU scores and the overall average:
```json
{
    "iou_scores": [
        {"idx": 1, "language": "spanish", "iou": 0.85},
        {"idx": 1, "language": "german", "iou": 0.75}
    ],
    "average_iou": 0.80
}
```

## Key Functions
- **`normalize_keywords`**: Ensures keywords are lowercase and free from punctuation.
- **`reverse_translate_keywords_deeptranslator`**: Translates keywords back to English using Deep Translator.
- **`calculate_iou_advanced`**: Computes IoU scores with semantic similarity considerations.
- **`filter_keywords`**: Matches keywords to the source premise and hypothesis.

## Limitations
- **Translation Errors**: Inconsistent translations may affect IoU scores.
- **Compound Words**: Decomposing compound words (e.g., "Lieblingsrestaurant") is language-dependent and may require additional tools for certain languages.

## Future Improvements
- Extend support for more languages.
- Improve compound word decomposition.
- Add contextual evaluation of keywords to better handle multi-word translations.

## Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

