import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiAgent:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            print("Warning: GEMINI_API_KEY environment variable is not set.")
            
    def generate_response(self, query, context_docs=None, current_image=None):
        """
        query: user question
        context_docs: list of dicts from retriever
        current_image: PIL Image of current screen selection, if any
        """
        if not self.api_key:
            return "Error: Gemini API Key is missing. Please set it in the settings or .env file."
            
        model = genai.GenerativeModel(self.model_name)
        
        # Build prompt from context documents
        context_str = ""
        if context_docs:
            context_str = "Below are relevant reference materials retrieved from your RAG database:\n\n"
            for i, doc in enumerate(context_docs):
                meta = doc.get("metadata", {})
                source = meta.get("source_type", "unknown")
                fname = meta.get("file_name", "unknown")
                ts = meta.get("timestamp", "")
                ts_str = f" [Time: {ts}]" if ts else ""
                
                context_str += f"--- Document {i+1} (Source: {source}, File: {fname}{ts_str}) ---\n"
                context_str += f"{doc['text']}\n\n"
        
        system_instruction = (
            "You are a helpful Windows AI Assistant. Use the provided context documents "
            "and the current screen image (if provided) to answer the user's question. "
            "If the answer cannot be found in the context or image, answer using your general knowledge "
            "but clearly state that it is not in the provided documents."
        )
        
        contents = [system_instruction]
        
        if context_str:
            contents.append(context_str)
            
        if current_image:
            contents.append("Below is the image/screenshot of the user's selected screen area:")
            contents.append(current_image)
            
        contents.append(f"User Question: {query}\nAnswer:")
        
        try:
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini API: {e}"

if __name__ == "__main__":
    agent = GeminiAgent()
    print("Gemini Agent loaded.")
