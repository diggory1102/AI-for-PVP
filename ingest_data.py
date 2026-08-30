import os
import glob
from src.rag.vector_db import get_chroma_client, get_or_create_collection, add_documents
from src.parsers.doc_parser import parse_document
from src.parsers.video_parser import parse_video_multimodal
from src.core.ollama_models import OllamaVisionModel

def ingest_all_data():
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    
    # Initialize vision model for video keyframe analysis (using the fast 3b model)
    vision_model = OllamaVisionModel(model_name="qwen2.5vl:3b")
    
    # 1. Quét và nạp tất cả tài liệu trong data/raw_documents/
    doc_folder = "./data/raw_documents"
    doc_files = glob.glob(f"{doc_folder}/*.*")
    print(f"--> Tìm thấy {len(doc_files)} tệp tài liệu trong {doc_folder}...")
    
    for file_path in doc_files:
        filename = os.path.basename(file_path)
        if filename == ".gitkeep":
            continue
        try:
            chunks = parse_document(file_path)  # Trả về list các dict có format RAG chuẩn
            if chunks:
                add_documents(collection, chunks)
                print(f" [ĐÃ NẠP XONG]: {filename} ({len(chunks)} chunks)")
            else:
                print(f" [BỎ QUA - KHÔNG CÓ CHỮ]: {filename}")
        except Exception as e:
            print(f" [LỖI KHI NẠP TÀI LIỆU {file_path}]: {e}")

    # 2. Quét và phân tích âm thanh/video trong data/raw_videos/
    video_folder = "./data/raw_videos"
    video_files = glob.glob(f"{video_folder}/*.mp4") + glob.glob(f"{video_folder}/*.mkv")
    print(f"\n--> Tìm thấy {len(video_files)} video trong {video_folder}...")
    
    for vid_path in video_files:
        filename = os.path.basename(vid_path)
        if filename == ".gitkeep":
            continue
        try:
            print(f"Đang bóc tách âm thanh (Whisper) & hình ảnh (Qwen2.5-VL) từ video: {filename}...")
            chunks = parse_video_multimodal(vid_path, vision_model, whisper_model_size="base", sample_rate_sec=5)
            if chunks:
                add_documents(collection, chunks)
                print(f" [ĐÃ NẠP XONG VIDEO ĐA PHƯƠNG THỨC]: {filename} ({len(chunks)} segments)")
            else:
                print(f" [BỎ QUA - KHÔNG TRÍCH XUẤT ĐƯỢC NỘI DUNG]: {filename}")
        except Exception as e:
            print(f" [LỖI KHI NẠP VIDEO {vid_path}]: {e}")

if __name__ == "__main__":
    ingest_all_data()
