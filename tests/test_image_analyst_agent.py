import os 
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.image_analyst_agent import ImageAnalystAgent

class FakeLLM:
    """Instead of calling the real LLM, we just return a fixed response for testing purposes."""

    def __init__(self, response="resimde bir çubuk grafik var"):
        self.response = response
        self.last_image_path = None

    def generate_with_image(self, image_path, prompt):
        self.last_image_path = image_path
        return self.response


def test_analyze_image_returns_llm_response():
    llm = FakeLLM("bu bir performans karşılaştırma grafiği")
    agent = ImageAnalystAgent(llm)

    result = agent("img1.png", "Grafikte ne gösteriliyor?")

    assert result == "bu bir performans karşılaştırma grafiği"


def test_analyze_image_passes_correct_image_path():
    llm = FakeLLM()
    agent = ImageAnalystAgent(llm)

    agent("cache/yolov8_p3_img0.png", "Bu nedir?")

    assert llm.last_image_path == "cache/yolov8_p3_img0.png"


if __name__ == "__main__":
    pytest.main([__file__])