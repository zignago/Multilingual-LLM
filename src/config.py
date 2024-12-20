LABEL_MAP = {
    0: "Entailment",
    1: "Neutral",
    2: "Contradiction"
}

# Models that have been tested and verified to work with this tool
# Info on GPT models: https://platform.openai.com/docs/models
# Info on LLAMA models: https://docs.llama-api.com/quickstart
SUPPORTED_LLM_MODELS = {
    "gpt": [
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
    ],
    "llama": [
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
        "Nous-Hermes-2-Mixtral-8x7B-DPO",
    ],
    "gemini": [
        "gemini-1.5-pro",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash"
    ]
}

# Used for deep translator reverse translate
LANGUAGE_CODES = {
    'afrikaans': 'af',
    'albanian': 'sq',
    'arabic': 'ar',
    'armenian': 'hy',
    'aymara': 'ay',
    'azerbaijani': 'az',
    'bambara': 'bm',
    'basque': 'eu',
    'belarusian': 'be',
    'bengali': 'bn',
    'bhojpuri': 'bho',
    'bosnian': 'bs',
    'bulgarian': 'bg',
    'catalan': 'ca',
    'cebuano': 'ceb',
    'chichewa': 'ny',
    'chinese': 'zh-CN', # (simplified)
    'chinese (traditional)': 'zh-TW',
    'corsican': 'co',
    'croatian': 'hr',
    'czech': 'cs',
    'danish': 'da',
    'dhivehi': 'dv',
    'dogri': 'doi',
    'dutch': 'nl',
    'english': 'en',
    'esperanto': 'eo',
    'estonian': 'et',
    'ewe': 'ee',
    'filipino': 'tl',
    'finnish': 'fi',
    'french': 'fr',
    'frisian': 'fy',
    'galician': 'gl',
    'georgian': 'ka',
    'german': 'de',
    'greek': 'el',
    'guarani': 'gn',
    'gujarati': 'gu',
    'haitian': 'ht',
    'hausa': 'ha',
    'hawaiian': 'haw',
    'hebrew': 'iw',
    'hindi': 'hi',
    'hmong': 'hmn',
    'hungarian': 'hu',
    'icelandic': 'is',
    'igbo': 'ig',
    'ilocano': 'ilo',
    'indonesian': 'id',
    'irish': 'ga',
    'italian': 'it',
    'japanese': 'ja',
    'javanese': 'jw',
    'kannada': 'kn',
    'kazakh': 'kk',
    'khmer': 'km',
    'kinyarwanda': 'rw',
    'konkani': 'gom',
    # 'korean': 'ko', 
    'krio': 'kri',
    'kurdish (kurmanji)': 'ku',
    'kurdish (sorani)': 'ckb',
    'kyrgyz': 'ky',
    'lao': 'lo',
    'latin': 'la',
    'latvian': 'lv',
    'lingala': 'ln',
    'lithuanian': 'lt',
    'luganda': 'lg',
    'luxembourgish': 'lb',
    'macedonian': 'mk',
    'maithili': 'mai',
    'malagasy': 'mg',
    'malay': 'ms',
    'malayalam': 'ml',
    'maltese': 'mt',
    'maori': 'mi',
    'marathi': 'mr',
    'meiteilon (manipuri)': 'mni-Mtei',
    'mizo': 'lus',
    'mongolian': 'mn',
    'myanmar': 'my',
    'nepali': 'ne',
    'norwegian': 'no',
    'odia (oriya)': 'or',
    'oromo': 'om',
    'pashto': 'ps',
    'persian': 'fa',
    'polish': 'pl',
    'portuguese': 'pt',
    'punjabi': 'pa',
    'quechua': 'qu',
    'romanian': 'ro',
    'russian': 'ru',
    'samoan': 'sm',
    'sanskrit': 'sa',
    'scots gaelic': 'gd',
    'sepedi': 'nso',
    'serbian': 'sr',
    'sesotho': 'st',
    'shona': 'sn',
    'sindhi': 'sd',
    'sinhala': 'si',
    'slovak': 'sk',
    'slovenian': 'sl',
    'somali': 'so',
    'spanish': 'es',
    'sundanese': 'su',
    'swahili': 'sw',
    'swedish': 'sv',
    'tajik': 'tg',
    'tamil': 'ta',
    'tatar': 'tt',
    'telugu': 'te',
    'thai': 'th',
    'tigrinya': 'ti',
    'tsonga': 'ts',
    'turkish': 'tr',
    'turkmen': 'tk',
    'twi': 'ak',
    'ukrainian': 'uk',
    'urdu': 'ur',
    'uyghur': 'ug',
    'uzbek': 'uz',
    'vietnamese': 'vi',
    'xhosa': 'xh',
    'yiddish': 'yi',
    'yoruba': 'yo',
    'zulu': 'zul'
}

TRANSLATION_MODELS = {
    'afrikaans': 'Helsinki-NLP/opus-mt-en-af',
    'albanian': 'Helsinki-NLP/opus-mt-en-sq',
    'arabic': 'Helsinki-NLP/opus-mt-tc-big-en-ar',
    'armenian': 'Helsinki-NLP/opus-mt-en-hy',
    'aymara': None,
    'azerbaijani': 'Helsinki-NLP/opus-mt-en-az',
    'bambara': None,
    'basque': 'Helsinki-NLP/opus-mt-en-eu',
    'belarusian': None,
    'bengali': 'shhossain/opus-mt-en-to-bn',
    'bhojpuri': None,
    'bosnian': None,
    'bulgarian': 'Helsinki-NLP/opus-mt-tc-big-en-bg',
    'catalan': 'Helsinki-NLP/opus-mt-en-ca',
    'cebuano': 'Helsinki-NLP/opus-mt-en-ceb',
    'chichewa': 'Helsinki-NLP/opus-mt-en-ny',
    'chinese': 'Helsinki-NLP/opus-mt-en-zh',  # (simplified)
    'chinese (traditional)': None,
    'corsican': None,
    'croatian': None,
    'czech': 'Helsinki-NLP/opus-mt-en-cs',
    'danish': 'Helsinki-NLP/opus-mt-en-da',
    'dhivehi': None,
    'dogri': None,
    'dutch': 'Helsinki-NLP/opus-mt-en-nl',
    'english': None,
    'esperanto': 'Helsinki-NLP/opus-mt-en-eo',
    'estonian': 'Helsinki-NLP/opus-mt-tc-big-en-et',
    'ewe': 'Helsinki-NLP/opus-mt-en-ee',
    'filipino': 'Helsinki-NLP/opus-mt-en-tl',
    'finnish': 'Helsinki-NLP/opus-mt-tc-big-en-fi',
    'french': 'Helsinki-NLP/opus-mt-tc-big-en-fr',
    'frisian': None,
    'galician': 'Helsinki-NLP/opus-mt-en-gl',
    'georgian': None,
    'german': 'Helsinki-NLP/opus-mt-en-de',
    'greek': 'Helsinki-NLP/opus-mt-tc-big-en-el',
    'guarani': None,
    'gujarati': None,
    'haitian': 'Helsinki-NLP/opus-mt-en-ht', #haitian creole
    'hausa': 'Helsinki-NLP/opus-mt-en-ha',
    'hebrew': 'Helsinki-NLP/opus-mt-en-he',
    'hindi': 'Helsinki-NLP/opus-mt-en-hi',
    'hmong': None,
    'hungarian': 'Helsinki-NLP/opus-mt-tc-big-en-hu',
    'icelandic': 'Helsinki-NLP/opus-mt-en-is',
    'igbo': 'Helsinki-NLP/opus-mt-en-ig',
    'ilocano': 'Helsinki-NLP/opus-mt-en-ilo',
    'indonesian': 'Helsinki-NLP/opus-mt-en-id',
    'irish': 'Helsinki-NLP/opus-mt-en-ga',
    'italian': 'Helsinki-NLP/opus-mt-tc-big-en-it',
    'japanese': 'Helsinki-NLP/opus-mt-en-jap',
    'javanese': None,
    'kannada': None,
    'kazakh': None,
    'khmer': 'Helsinki-NLP/opus-mt-en-mkh',
    'kinyarwanda': 'Helsinki-NLP/opus-mt-en-rw',
    'konkani': None,
    # 'korean': 'Helsinki-NLP/opus-mt-tc-big-en-ko',
    'krio': None,
    'kurdish (kurmanji)': None,
    'kurdish (sorani)': None,
    'kyrgyz': None,
    'lao': None,
    'latin': None,
    'latvian': 'Helsinki-NLP/opus-mt-tc-big-en-lv',
    'lingala': 'Helsinki-NLP/opus-mt-en-ln',
    'lithuanian': 'Helsinki-NLP/opus-mt-tc-big-en-lt',
    'luganda': 'Helsinki-NLP/opus-mt-en-lg',
    'luxembourgish': None,
    'macedonian': 'Helsinki-NLP/opus-mt-en-mk',
    'maithili': None,
    'malagasy': 'Helsinki-NLP/opus-mt-en-mg',
    'malay': None,
    'malayalam': 'Helsinki-NLP/opus-mt-en-ml',
    'maltese': 'Helsinki-NLP/opus-mt-en-mt',
    'maori': None,
    'marathi': 'Helsinki-NLP/opus-mt-en-mr',
    'meiteilon (manipuri)': None,
    'mizo': 'Helsinki-NLP/opus-mt-en-lus',
    'mongolian': None,
    'myanmar': None,
    'nepali': None,
    'norwegian': None,
    'odia (oriya)': None,
    'oromo': 'Helsinki-NLP/opus-mt-en-om',
    'pashto': None,
    'persian': None,
    'polish': None,
    'portuguese': 'Helsinki-NLP/opus-mt-tc-big-en-pt',
    'punjabi': None,
    'quechua': None,
    'romanian': 'Helsinki-NLP/opus-mt-tc-big-en-ro',
    'russian': 'Helsinki-NLP/opus-mt-en-ru',
    'samoan': 'Helsinki-NLP/opus-mt-en-sm',
    'sanskrit': None,
    'scots gaelic': None,
    'sepedi': 'Helsinki-NLP/opus-mt-en-nso',
    'serbian': None,
    'sesotho': 'Helsinki-NLP/opus-mt-en-st',
    'shona': 'Helsinki-NLP/opus-mt-en-sn',
    'sindhi': None,
    'sinhala': None,
    'slovak': 'Helsinki-NLP/opus-mt-en-sk',
    'slovenian': None,
    'somali': None,
    'spanish': 'Helsinki-NLP/opus-mt-tc-big-en-es',
    'sundanese': None,
    'swahili': 'Helsinki-NLP/opus-mt-en-sw',
    'swedish': 'Helsinki-NLP/opus-mt-en-sv',
    'tajik': None,
    'tamil': None,
    'tatar': None,
    'telugu': None,
    'thai': None,
    'tigrinya': 'Helsinki-NLP/opus-mt-en-ti',
    'tsonga': 'Helsinki-NLP/opus-mt-en-ts',
    'turkish': 'Helsinki-NLP/opus-mt-tc-big-en-tr',
    'turkmen': None,
    'twi': None,
    'ukrainian': 'Helsinki-NLP/opus-mt-en-uk',
    'urdu': 'Helsinki-NLP/opus-mt-en-ur',
    'uyghur': None,
    'uzbek': None,
    'vietnamese': 'Helsinki-NLP/opus-mt-en-vi',
    'xhosa': 'Helsinki-NLP/opus-mt-en-xh',
    'yiddish': None,
    'yoruba': None,
    'zulu': None
}