import sys
import os
from PyQt6.QtWidgets import QApplication
from dotenv import load_dotenv

# Ensure the root directories exist
os.makedirs("data/raw_documents", exist_ok=True)
os.makedirs("data/raw_videos", exist_ok=True)
os.makedirs("data/vector_store", exist_ok=True)

from src.capture.screen_picker import pick_screen_area
from src.capture.video_stream import ScreenCaptureStream
from src.rag.vector_db import get_chroma_client, get_or_create_collection, add_documents
from src.rag.retriever import retrieve_relevant_context
from src.core.agent import GeminiAgent
from src.ui.app_gui import AssistantGUI

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize ChromaDB
    chroma_client = get_chroma_client()
    collection = get_or_create_collection(chroma_client)
    
    # Initialize Gemini Agent
    agent = GeminiAgent()
    
    # Start QApplication
    app = QApplication(sys.argv)
    
    gui = AssistantGUI(
        picker_func=pick_screen_area,
        stream_class=ScreenCaptureStream,
        db_collection=collection,
        agent_instance=agent,
        add_docs_func=add_documents,
        retrieve_func=retrieve_relevant_context
    )
    
    gui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
