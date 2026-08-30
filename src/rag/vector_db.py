import os
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Set up Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="models/text-embedding-004"):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        # Call Gemini Embedding API
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=input,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            print(f"Error calling Gemini Embedding API: {e}")
            # Fallback mock/zero embedding if API fails to prevent crashes
            return [[0.0] * 768 for _ in input]

def get_chroma_client(persist_dir="data/vector_store"):
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)

def get_or_create_collection(client, name="windows_assistant_knowledge"):
    emb_fn = GeminiEmbeddingFunction()
    return client.get_or_create_collection(name=name, embedding_function=emb_fn)

def add_documents(collection, doc_list):
    """
    doc_list: list of dicts like {"text": "...", "metadata": {...}}
    """
    if not doc_list:
        return
        
    ids = [f"id_{hash(doc['text'])}_{i}" for i, doc in enumerate(doc_list)]
    documents = [doc['text'] for doc in doc_list]
    metadatas = [doc['metadata'] for doc in doc_list]
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

if __name__ == "__main__":
    client = get_chroma_client()
    col = get_or_create_collection(client)
    print("ChromaDB Client initialized. Collection count:", col.count())
