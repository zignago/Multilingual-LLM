import os

supported_languages = [
    'afrikaans', 
    'albanian', 
    'arabic', 
    'armenian',
    'azerbaijani',
    'basque',
    'bengali',
    'bulgarian',
    'catalan',
    'cebuano',
    'chichewa',
    'chinese',
    'czech',
    'danish',
    'dutch',
    'esperanto',
    'estonian',
    'ewe',
    'filipino',
    'finnish',
    'french',
    'galician',
    'german',
    'greek',
    'haitian',
    'hausa',
    'hebrew',
    'hindi',
    'hungarian',
    'icelandic',
    'igbo',
    'ilocano',
    'indonesian',
    'irish',
    'italian',
    'japanese',
    'khmer',
    'kinyarwanda',
    'korean',
    'latvian',
    'lingala',
    'lithuanian',
    'luganda',
    'macedonian',
    'malagasy',
    'malayalam',
    'maltese',
    'marathi',
    'mizo',
    'oromo',
    'portuguese',
    'romanian',
    'russian',
    'samoan',
    'sepedi',
    'sesotho',
    'shona',
    'slovak',
    'spanish',
    'swahili',
    'swedish',
    'tigrinya',
    'tsonga',
    'turkish',
    'ukrainian',
    'urdu',
    'vietnamese',
    'xhosa'
]

supported_models = [
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125"
    "llama3.2-11b-vision",
    "llama3.2-1b",
    "llama3.2-3b",
    "llama3.2-90b",
    "llama3.1-405b",
    "llama3.1-70b",
    "llama3.1-8b",
    "llama3-70b",
    "llama3-8b",
    "gemma2-27b",
    "gemma2-9b",
    "mixtral-8x22b-instruct",
    "mixtral-8x7b-instruct",
    "mistral-7b-instruct",
    "llama2-13b", 
    "llama2-70b",
    "llama2-7b",
    "gemini-1.5-pro",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash"
]

def run_command(languages, limit, model, subset, iterations=1):

    command = f"python main.py --languages {languages} --limit {limit} --model {model} --subset {subset} --iterations {iterations}"

    print(f"Running command")
    exit_code = os.system(command)
    if exit_code != 0:
        print(f"Command failed with exit code {exit_code}")
    else:
        print(f"Command completed successfully\n")



# LANGUAGE
# Small subset run with all supported languages and 1 gpt model, 1 llama model, and 1 gemini model to compare all languages
# Attempting to get 3 models that are roughly similar in sophistication
# gemini-1.5-flash-8b
# GPT-4o-mini
# llama3.1-8b -- worse than the others but similar cheapness
# for model in ["gpt-4o-mini", "llama3.1-8b", "gemini-1.5-flash-8b"]:
#     # run_command(languages=' '.join(supported_languages), limit=3, model=model, subset=1, iterations=1)
#     # Around $0.50 with subset 100
#     for language in supported_languages:
#         run_command(languages=language, limit=3, model=model, subset=30, iterations=1)
for language in supported_languages:
    run_command(languages=language, limit=3, model="gemini-1.5-flash-8b", subset=30, iterations=1)

# LIMIT
run_command(languages='spanish', limit=1, model="gpt-4o-mini", subset=30, iterations=1)
run_command(languages='spanish', limit=2, model="gpt-4o-mini", subset=30, iterations=1)
run_command(languages='spanish', limit=3, model="gpt-4o-mini", subset=30, iterations=1)
run_command(languages='spanish', limit=4, model="gpt-4o-mini", subset=30, iterations=1)
run_command(languages='spanish', limit=5, model="gpt-4o-mini", subset=30, iterations=1)
run_command(languages='spanish', limit=6, model="gpt-4o-mini", subset=30, iterations=1)
run_command(languages='spanish', limit=7, model="gpt-4o-mini", subset=30, iterations=1)
run_command(languages='spanish', limit=8, model="gpt-4o-mini", subset=30, iterations=1)
run_command(languages='spanish', limit=9, model="gpt-4o-mini", subset=30, iterations=1)
run_command(languages='spanish', limit=10, model="gpt-4o-mini", subset=30, iterations=1)

# ITERATION
# Medium subset run with no iteration then same run with iterations to assess iteration impact
run_command(languages='spanish', limit=3, model="gpt-4o-mini", subset=30, iterations=1)
run_command(languages='spanish', limit=3, model="gpt-4o-mini", subset=30, iterations=2)
run_command(languages='spanish', limit=3, model="gpt-4o-mini", subset=30, iterations=3)
run_command(languages='spanish', limit=3, model="gpt-4o-mini", subset=30, iterations=4)
# Run more if results look interesting
# run_command(languages='spanish', limit=3, model="gpt-4o-mini", subset=50, iterations=5)
# run_command(languages='spanish', limit=3, model="gpt-4o-mini", subset=50, iterations=6)
# run_command(languages='spanish', limit=3, model="gpt-4o-mini", subset=50, iterations=7)
# run_command(languages='spanish', limit=3, model="gpt-4o-mini", subset=50, iterations=8)

# SUBSET
#Fibonacci sequence
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=1, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=2, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=3, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=4, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=5, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=6, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=7, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=8, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=9, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=10, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=15, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=20, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=25, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=50, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=100, iterations=1)
# Continue if the results look interesting
# run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=500, iterations=1)
# run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=1000, iterations=1)
# run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=2000, iterations=1)

# VARIATION
# Running the same command over and over again to gauge variability of results
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=10, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=10, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=10, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=10, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=10, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=10, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=10, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=10, iterations=1)
run_command(languages='spanish', limit=3, model='gpt-4o-mini', subset=10, iterations=1)

# MODEL
for model in supported_models:
    # run_command(languages=' '.join(supported_languages), limit=3, model=model, subset=1, iterations=1)
    # Around $0.50 with subset 100
    run_command(languages='spanish', limit=3, model=model, subset=20, iterations=1)