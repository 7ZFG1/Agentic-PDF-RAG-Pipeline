import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.retriever_agent import RetrieverAgent


class FakeRetriever:
    """Fake retriever that returns predefined text and image results for testing purposes."""

    def __init__(self, text_results, image_results):
        self.text_results = text_results
        self.image_results = image_results
        self.last_query = None

    def __call__(self, query):
        self.last_query = query
        return self.text_results, self.image_results


def test_format_results_with_text_and_image_chunks():
    text_results = [{
        "chunk_id": "t_0",
        "text": "merhaba dünya",
        "source_pdf": "a.pdf",
        "pages": [1],
        "score": 0.9,
    }]
    image_results = [{
        "chunk_id": "img_0",
        "source_pdf": "a.pdf",
        "image_path": "x.png",
        "caption": "Şekil 1",
        "description": "bir grafik",
        "page": 2,
        "score": 0.5,
    }]

    agent = RetrieverAgent(FakeRetriever(text_results, image_results))
    output = json.loads(agent("soru"))

    assert len(output) == 2

    assert output[0]["type"] == "text"
    assert output[0]["text"] == "merhaba dünya"
    assert output[0]["source_pdf"] == "a.pdf"

    assert output[1]["type"] == "image"
    assert output[1]["image_path"] == "x.png"
    assert output[1]["caption"] == "Şekil 1"


def test_format_results_empty_when_nothing_found():
    agent = RetrieverAgent(FakeRetriever([], []))
    output = json.loads(agent("soru"))
    assert output == []


def test_format_results_uses_defaults_for_missing_fields():
    text_results = [{"chunk_id": "t_1", "text": "deneme", "score": 0.8}]
    agent = RetrieverAgent(FakeRetriever(text_results, []))
    output = json.loads(agent("soru"))

    assert output[0]["source_pdf"] == "unknown"
    assert output[0]["pages"] == []


def test_agent_passes_query_to_retriever():
    fake = FakeRetriever([], [])
    agent = RetrieverAgent(fake)
    agent("YOLOv8 mAP nedir?")

    assert fake.last_query == "YOLOv8 mAP nedir?"


if __name__ == "__main__":
    pytest.main([__file__])