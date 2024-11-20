import os
from llamaapi import LlamaAPI

class LlamaHandler:
    def __init__(self, api_key=None):
        """
        Initialize the LlamaHandler with the LlamaAPI instance.
        """
        self.api_key = api_key or os.getenv("LLAMA_API_KEY")
        if not self.api_key:
            raise ValueError("API key for LlamaAPI is not set.")
        self.llama = LlamaAPI(self.api_key)

    def run_prompt(self, prompt, model_name="llama3.1-70b", **kwargs):
        """
        Send a prompt to the Llama API and return the response.
        :param prompt: The input prompt for the Llama model.
        :param model_name: The model name to use for the request.
        :param kwargs: Additional parameters for the API request.
        :return: The response from the Llama API.
        """
        api_request_json = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "functions": [
                {
                    "name": "extract_keywords",
                    "description": "Identify the top keywords from premise and hypothesis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "premise": {
                                "type": "string",
                                "description": "The premise statement."
                            },
                            "hypothesis": {
                                "type": "string",
                                "description": "The hypothesis statement."
                            },
                            "label": {
                                "type": "string",
                                "description": "The relationship label: entailment, contradiction, or neutral."
                            }
                        },
                        "required": ["premise", "hypothesis", "label"]
                    }
                }
            ],
            "stream": False,
            "function_call": "extract_keywords",
        }
        # Merge any additional parameters into the request
        api_request_json.update(kwargs)

        # Make the API call and return the response
        response = self.llama.run(api_request_json)
        return response.json()["choices"][0]["message"]["content"]



# import json
# import os
# from llamaapi import LlamaAPI

# llama = LlamaAPI(os.getenv("LLAMA_API_KEY"))

# prompt = """Identify the top 4 keywords relevant to understanding the relationship between the premise and hypothesis. Include only words from the premise and hypothesis. Do not include any other words. Do not include punctuation or commas ('.' or ',') in keywords.
# Premise: The new rights are nice enough
# Hypothesis: Everyone really likes the newest benefits
# Label: Neutral

# Return keywords in array format like ["a", "b", "c"]. Include only single words, no phrases."""

# # Build the API request
# api_request_json = {
#     "model": "llama3.1-70b",
#     "messages": [
#         {"role": "user", "content": prompt},
#     ],
#     "functions": [
#         {
#             "name": "extract_keywords",
#             "description": "Identify the top keywords from premise and hypothesis.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "premise": {
#                         "type": "string",
#                         "description": "The premise statement."
#                     },
#                     "hypothesis": {
#                         "type": "string",
#                         "description": "The hypothesis statement."
#                     },
#                     "label": {
#                         "type": "string",
#                         "description": "The relationship label: entailment, contradiction, or neutral."
#                     }
#                 },
#                 "required": ["premise", "hypothesis", "label"]
#             }
#         }
#     ],
#     "stream": False,
#     "function_call": "extract_keywords",
# }

# # Execute the Request
# response = llama.run(api_request_json)
# #print(json.dumps(response.json(), indent=2))
# print(response.json()["choices"][0]["message"]["content"])