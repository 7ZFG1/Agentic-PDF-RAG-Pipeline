class ValidatorAgent:
    """Validates that the answer is grounded in the retrieved context."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, question: str, answer: str, context: str) -> str:
        """Check if answer is supported by context. Correct if not."""
        print(f"  [ValidatorAgent] Validating answer...")
        prompt = self._build_prompt(question, answer, context)
        result = self.llm(prompt)
        print(f"  [ValidatorAgent] Validation complete")
        return result

    def _build_prompt(self, question: str, answer: str, context: str) -> str:
        """Build the validation prompt from question, draft answer and context."""
        return (
            "You are a validation agent for a document QA system. Check whether the draft answer "
            "is fully supported by the given context, and correct it if not.\n\n"
            f"Question: {question}\n\n"
            f"Draft answer: {answer}\n\n"
            f"Context:\n{context}\n\n"
            "Instructions:\n"
            "1. Check each factual claim in the draft answer against the context.\n"
            "2. If every claim is supported, return the draft answer as is (only minor wording "
            "cleanup is allowed — do not add or remove information).\n"
            "3. If some claims are not supported by or contradict the context, correct or remove "
            "them using ONLY information found in the context. Do not introduce new facts.\n"
            "4. If the context contains no information relevant to the question, respond exactly: "
            "'The document does not contain enough information to answer this question.'\n"
            "5. Preserve numbers, units, names, and labels exactly as they appear in the context — "
            "do not round, approximate, translate, or rephrase them.\n"
            "6. Respond in the same language as the question.\n\n"
            "Return ONLY the validated or corrected answer text — no explanations, labels, or preambles."
        )
