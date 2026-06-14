class ImageAnalystAgent:
    """Analyzes document images using GeminiLLM."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, image_path: str, question: str) -> str:
        """Send image + question to Gemini Vision, return analysis text."""
        print(f"  [ImageAnalystAgent] Analyzing image: {image_path}")
        prompt = (
            "You are analyzing an image extracted from a technical document "
            "(it may be a chart, table, diagram, or photo).\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "- Focus only on the parts of the image relevant to the question.\n"
            "- If the image contains numbers, axis labels, legends, categories, or a table, "
            "read them carefully and report the exact values and labels.\n"
            "- If the image is not relevant to the question, briefly say so.\n"
            "- Be concise and factual; do not speculate beyond what is visible in the image.\n"
            "- Answer in the same language as the question."
        )
        result = self.llm.generate_with_image(image_path, prompt)
        print(f"  [ImageAnalystAgent] Analysis complete ({len(result)} chars)")
        return result
