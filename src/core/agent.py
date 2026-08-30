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
        screen_analysis = ""
        if current_image and self.vision_model:
            vision_prompt = (
                "Identify and extract all relevant text, UI components, code, "
                "or details from this image to help answer: " + query
            )
            screen_analysis = self.vision_model.analyze_image(current_image, vision_prompt)

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

        prompt = (
            "You are a helpful Windows AI Assistant. Use the provided context documents "
            "and the active screen area analysis to answer the user's question.\n\n"
        )
        
        if context_str:
            prompt += context_str + "\n"
            
        if screen_analysis:
            prompt += f"--- Active Screen Area Analysis ---\n{screen_analysis}\n\n"
            
        prompt += f"User Question: {query}\nAnswer:"
        
        return self.text_model.generate_text(prompt)

if __name__ == "__main__":
    print("AIAgent module loaded.")
