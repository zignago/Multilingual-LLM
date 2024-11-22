import os
import re
import google.generativeai as genai

class GeminiHandler:
    """
    Handles interactions with the Gemini API using the google.generativeai library.
    """

    def __init__(self):
        # Fetch API key from environment variable
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment variables.")

        # Configure the generative AI client
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")  # Replace with desired model version

    def run_prompt(self, prompt, max_tokens=50):
        """
        Sends a prompt to the Gemini API and retrieves the response.
        """
        try:
            # Generate content using the Gemini model
            response = self.model.generate_content(prompt)

            # Extract the text from the response
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Error during Gemini API call: {e}")

    def parse_keywords(self, response_text, premise, hypothesis):
        """
        Extracts and validates keywords from Gemini API responses.
        """
        keywords = []
        try:
            # Attempt to extract keywords in JSON array format
            match = re.search(r'\["(.*?)"\]', response_text)
            if match:
                keywords = match.group(1).split('", "')
            else:
                keywords = [word.strip() for word in response_text.strip("[]").split(",")]
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")

        # Normalize and filter keywords
        allowed_words = set(premise.lower().split() + hypothesis.lower().split())
        return [kw.strip().lower() for kw in keywords if kw.lower() in allowed_words]
