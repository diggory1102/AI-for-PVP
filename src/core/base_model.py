from abc import ABC, abstractmethod
from PIL import Image

class BaseTextModel(ABC):
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generates a plain text response for a given prompt."""
        pass

class BaseVisionModel(ABC):
    @abstractmethod
    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        """Analyzes a screen/video frame image using a multimodal prompt."""
        pass
