import os
import pypdf
import docx

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def parse_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    file_name = os.path.basename(file_path)
    chunks = chunk_text(content)
    
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "text": chunk.strip(),
            "metadata": {
                "source_type": "doc",
                "file_name": file_name,
                "chunk_index": i
            }
        })
    return documents

def parse_pdf(file_path):
    file_name = os.path.basename(file_path)
    documents = []
    
    with open(file_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Chunk page content if it's too long, or treat page as unit
                page_chunks = chunk_text(text, chunk_size=800, overlap=100)
                for chunk_idx, chunk in enumerate(page_chunks):
                    documents.append({
                        "text": chunk.strip(),
                        "metadata": {
                            "source_type": "doc",
                            "file_name": file_name,
                            "page": page_idx + 1,
                            "chunk_index": chunk_idx
                        }
                    })
    return documents

def parse_docx(file_path):
    file_name = os.path.basename(file_path)
    doc = docx.Document(file_path)
    
    # Group paragraphs to form chunks
    current_text = []
    current_length = 0
    documents = []
    chunk_index = 0
    
    for para in doc.paragraphs:
        if para.text.strip():
            current_text.append(para.text)
            current_length += len(para.text)
            
            if current_length >= 1000:
                combined = "\n".join(current_text)
                documents.append({
                    "text": combined,
                    "metadata": {
                        "source_type": "doc",
                        "file_name": file_name,
                        "chunk_index": chunk_index
                    }
                })
                chunk_index += 1
                current_text = []
                current_length = 0
                
    if current_text:
        combined = "\n".join(current_text)
        documents.append({
            "text": combined,
            "metadata": {
                "source_type": "doc",
                "file_name": file_name,
                "chunk_index": chunk_index
            }
        })
        
    return documents

def parse_document(file_path):
    """
    Auto-detects extension and parses the document into list of dicts:
    [{"text": "...", "metadata": {...}}]
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return parse_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        return parse_docx(file_path)
    else:
        # Default text/code parser
        return parse_txt(file_path)

if __name__ == "__main__":
    # Test text parsing
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as f:
        f.write("Hello World!\n" * 100)
        temp_path = f.name
        
    try:
        docs = parse_document(temp_path)
        print(f"Parsed {len(docs)} chunks from test txt file. Sample:\n", docs[0])
    finally:
        os.unlink(temp_path)
