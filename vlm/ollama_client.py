import requests
from PIL import Image

from config import OLLAMA_MODEL, OLLAMA_URL, VLM_TIMEOUT
from vlm.client import VLMClient, encode_image_b64


class OllamaClient(VLMClient):
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def query(self, images: list[Image.Image], prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [encode_image_b64(img) for img in images],
            "stream": False,
            "options": {"temperature": 0},
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=VLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
