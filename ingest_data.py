import os
import glob
from src.rag.vector_db import get_chroma_client, get_or_create_collection, add_documents
from src.parsers.doc_parser import parse_document
from src.parsers.video_parser import parse_video_audio

def ingest_all_data():
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    
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
            print(f"Đang bóc tách âm thanh và dịch bằng local Whisper: {filename}...")
            # Sử dụng mô hình whisper chạy local (base model)
            chunks = parse_video_audio(vid_path, model_size="base")
            if chunks:
                add_documents(collection, chunks)
                print(f" [ĐÃ NẠP XONG VIDEO]: {filename} ({len(chunks)} segments)")
            else:
                print(f" [BỎ QUA - KHÔNG TRÍCH XUẤT ĐƯỢC GIỌNG NÓI]: {filename}")
        except Exception as e:
            print(f" [LỖI KHI NẠP VIDEO {vid_path}]: {e}")

if __name__ == "__main__":
    ingest_all_data()
