import base64
import io
from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class VLMClient(ABC):
    @abstractmethod
    def query(self, images: list[Image.Image], prompt: str) -> str:
        """Send images + prompt, return raw model response string."""


def encode_image_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
