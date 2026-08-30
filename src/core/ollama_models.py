import io
import base64
import requests
from PIL import Image
from src.core.base_model import BaseTextModel, BaseVisionModel

class OllamaTextModel(BaseTextModel):
    def __init__(self, model_name="qwen2.5:3b", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.15
            }
        }
        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=180)
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
            else:
                return f"Error from Ollama ({response.status_code}): {response.text}"
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

class OllamaVisionModel(BaseVisionModel):
    def __init__(self, model_name="qwen2.5vl:3b", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze_image(self, image: Image.Image, prompt: str, system_prompt: str = None) -> str:
        try:
            img_b64 = self._image_to_base64(image)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            user_msg = {
                "role": "user",
                "content": prompt,
                "images": [img_b64]
            }
            messages.append(user_msg)

            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.15
                }
            }
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=180)
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
            else:
                return f"Error from Ollama Vision ({response.status_code}): {response.text}"
        except Exception as e:
            return f"Error connecting to Ollama Vision: {e}"
