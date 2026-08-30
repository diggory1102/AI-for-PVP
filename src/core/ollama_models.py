import io
import base64
import requests
from PIL import Image
from src.core.base_model import BaseTextModel, BaseVisionModel

class OllamaTextModel(BaseTextModel):
    def __init__(self, model_name="qwen2.5:7b", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def generate_text(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(f"{self.host}/api/generate", json=payload, timeout=180)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error from Ollama ({response.status_code}): {response.text}"
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

class OllamaVisionModel(BaseVisionModel):
    def __init__(self, model_name="qwen2.5vl:7b", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        try:
            img_b64 = self._image_to_base64(image)
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False
            }
            response = requests.post(f"{self.host}/api/generate", json=payload, timeout=180)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error from Ollama Vision ({response.status_code}): {response.text}"
        except Exception as e:
            return f"Error connecting to Ollama Vision: {e}"
