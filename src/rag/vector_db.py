import os
import chromadb
from chromadb.utils import embedding_functions

def get_chroma_client(persist_dir="data/vector_store"):
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)

def get_or_create_collection(client, name="windows_assistant_knowledge"):
    # Initialize a local sentence-transformers embedding model
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
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
    print("ChromaDB Client initialized with local embeddings. Collection count:", col.count())
