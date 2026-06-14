import os
import sys
import time
import random

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VERTEXAI_PROJECT, VERTEXAI_LOCATION, GEMINI_MODEL


class GeminiLLM:
    def __init__(self) -> None:
        self.max_retries = 6 # max retry attempts for API calls

        vertexai.init(project=VERTEXAI_PROJECT, location=VERTEXAI_LOCATION)
        self.model = GenerativeModel(GEMINI_MODEL)

    def __call__(self, prompt: str) -> str:
        """Generate text response from a prompt string."""
        retry_count=0

        # exponential backoff with jitter for API calls
        while True:
            try:
                if retry_count > self.max_retries:
                    print(f"[WARNING] Maximum retry attempts exceeded: {e}")
                    return "ERROR: Unable to generate response after multiple attempts."

                retry_count += 1
                response = self.model.generate_content(prompt)

                retry_count = 0  # reset retry count on success
                return response.text
            except Exception as e:
                wait_time = min(60, (2 ** retry_count))  # max limit 60 seconds
                jitter = random.uniform(0, 1)

                print(f"\n[Warning]: {e}")
                print(f"Waiting for {wait_time + jitter:.2f} seconds...")

                time.sleep(wait_time + jitter)

    def generate_with_image(self, image_path: str, prompt: str) -> str:
        """Send image + custom prompt to Gemini, return text response."""
        retry_count=0
        while True:
            try:
                if retry_count > self.max_retries:
                    print(f"[WARNING] Maximum retry attempts exceeded: {e}")
                    return "ERROR: Unable to generate response after multiple attempts."

                retry_count += 1
                image = Part.from_image(Image.load_from_file(image_path))
                response = self.model.generate_content([image, prompt])

                retry_count = 0  # reset retry count on success
                return response.text
            except Exception as e:
                wait_time = min(60, (2 ** retry_count))  # max limit 60 seconds
                jitter = random.uniform(0, 1)

                print(f"\n[Warning]: {e}")
                print(f"Waiting for {wait_time + jitter:.2f} seconds...")

                time.sleep(wait_time + jitter)

    def describe_image(self, image_path: str) -> str:
        """Send image to Gemini and get a text description for indexing."""
        prompt = (
            "Describe this image concisely for a search index (2-4 sentences).\n"
            "Include:\n"
            "- The type of visual (chart, diagram, table, photo, etc.)\n"
            "- The main topic/subject\n"
            "- Key labels, categories, or axis names visible\n"
            "- Specific numeric values or data points if visible (e.g., percentages, scores, "
            "measurements, model names being compared)\n"
            "- Any important comparisons or relationships shown\n"
            "Write in the same language as the text inside the image, if any."
        )
        return self.generate_with_image(image_path, prompt)