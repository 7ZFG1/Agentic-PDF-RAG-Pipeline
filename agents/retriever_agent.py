import json

class RetrieverAgent:
    """Wraps the Retriever as a callable tool for the orchestrator."""

    def __init__(self, retriever) -> None:
        self.retriever = retriever

    def __call__(self, query: str) -> str:
        """Search both text and image indexes, return JSON string."""
        print(f"  [RetrieverAgent] Searching: '{query}'")
        text_results, image_results = self.retriever(query)
        print(f"  [RetrieverAgent] Found {len(text_results)} text, {len(image_results)} image chunks")
        return self._format_results(text_results, image_results)

    def _format_results(self, text_results: list[dict], image_results: list[dict]) -> str:
        """Convert retrieval results to JSON for the orchestrator."""
        output = []
        for r in text_results:
            output.append({
                "type": "text",
                "chunk_id": r["chunk_id"],
                "source_pdf": r.get("source_pdf", "unknown"),
                "text": r["text"],
                "pages": r.get("pages", []),
                "score": r["score"]
            })
        for r in image_results:
            output.append({
                "type": "image",
                "chunk_id": r["chunk_id"],
                "source_pdf": r.get("source_pdf", "unknown"),
                "image_path": r.get("image_path", ""),
                "caption": r.get("caption", ""),
                "description": r.get("description", ""),
                "page": r.get("page", 0),
                "score": r["score"]
            })
        return json.dumps(output, ensure_ascii=False)