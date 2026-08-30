from src.core.base_model import BaseTextModel, BaseVisionModel

class AIAgent:
    def __init__(self, text_model: BaseTextModel, vision_model: BaseVisionModel = None):
        self.text_model = text_model
        self.vision_model = vision_model

    def generate_response(self, query, context_docs=None, current_image=None):
        """
        query: user question
        context_docs: list of dicts from retriever
        current_image: PIL Image of current screen selection, if any
        """
        # Build prompt from context documents
        context_str = ""
        if context_docs:
            context_str = "Below are relevant reference materials retrieved from the RAG database:\n\n"
            for i, doc in enumerate(context_docs):
                meta = doc.get("metadata", {})
                source = meta.get("source_type", "unknown")
                fname = meta.get("file_name", "unknown")
                ts = meta.get("timestamp", "")
                ts_str = f" [Time: {ts}]" if ts else ""
                
                context_str += f"--- Document {i+1} (Source: {source}, File: {fname}{ts_str}) ---\n"
                context_str += f"{doc['text']}\n\n"

        # Base instruction with detailed game bot role, semantic labels and few-shot examples
        system_instruction = (
            "You are a Windows PVP Game Bot / AI Assistant. Your main job is to analyze the active screen/window image "
            "and execute precise mouse clicks and keystrokes to achieve the user's target.\n\n"
            "COORDINATE SYSTEM:\n"
            "- The coordinate system starts at (0, 0) at the TOP-LEFT corner of the provided window image.\n"
            "- You must estimate the pixel coordinates (x, y) of target elements relative to this image space.\n\n"
            "ACTION COMMAND FORMAT:\n"
            "- To click a coordinate, you MUST specify the text label of the target button: [CLICK: x, y, LABEL: \"button_name\"]\n"
            "- To type text, output: [TYPE: text]\n"
            "- When you have fully achieved the user's target goal, output: [FINISHED]\n\n"
            "CRITICAL RULES:\n"
            "1. If the user asks you to click, find, or activate something on the screen, you MUST output the command in the exact format [CLICK: x, y, LABEL: \"button_name\"].\n"
            "2. Make sure the LABEL argument corresponds to the text written on the button (e.g. \"Khiêu chiến\", \"Xác nhận\", \"Skill 1\").\n"
            "3. If the goal is completed, output [FINISHED] immediately.\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "Example 1:\n"
            "User: Click on the 'Start Battle' button\n"
            "Assistant: I see the 'Start Battle' button in the bottom-right region at coordinates (720, 550). Clicking now: [CLICK: 720, 550, LABEL: \"Start Battle\"]\n\n"
            "Example 2:\n"
            "User: Enter the code '1234'\n"
            "Assistant: First, I will click the input field at (300, 250): [CLICK: 300, 250, LABEL: \"Input Field\"]. Then, I will type the code: [TYPE: 1234]\n\n"
            "Example 3:\n"
            "User: Go back to home if battle is done\n"
            "Assistant: The battle is over and I am on the reward screen. I will click the Confirm button at (400, 300): [CLICK: 400, 300, LABEL: \"Confirm\"]. Goal achieved: [FINISHED]\n\n"
        )
        
        prompt = system_instruction
        if context_str:
            prompt += context_str + "\n"
        
        prompt += f"User Question: {query}\nAnswer:"

        # OPTIMIZATION: If image is present, query the vision model directly once.
        # This bypasses the double model calling latency.
        if current_image and self.vision_model:
            return self.vision_model.analyze_image(current_image, prompt)
        
        # Otherwise, run the standard text model
        return self.text_model.generate_text(prompt)

if __name__ == "__main__":
    print("AIAgent module loaded with Single Inference Optimization.")
