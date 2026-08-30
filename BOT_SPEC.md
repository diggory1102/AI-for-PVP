# AI Windows Assistant - Technical Specification

## 1. Overview & Objectives
Bot AI chạy trên Windows có khả năng:
- Bắt và hiểu vùng màn hình/video được chỉ định.
- Nạp tài liệu ngoài (PDF, Docx, Code) và video file có sẵn.
- Quản lý bộ nhớ tập trung qua Vector Database (RAG) và hỗ trợ hỏi đáp, đối chiếu chéo.

## 2. Tech Stack
- Core: Python 3.10+
- Capture & Vision: `mss`, `opencv-python`, `pillow`, `pygetwindow`
- Transcription & Media: `faster-whisper`, `moviepy`
- LLM & RAG: `google-generativeai` (Gemini 1.5/2.0), `langchain`, `chromadb`, `sentence-transformers`
- UI (Tùy chọn): `PyQt6`

## 3. Metadata Standard
Mọi mẩu dữ liệu đưa vào Vector DB phải có cấu trúc:
- text: string (Nội dung tri thức)
- metadata:
  - source_type: "screen_crop" | "video_stream" | "video_file" | "doc"
  - file_name: string
  - timestamp: string

## 4. Implementation Steps for Agent
1. Khởi tạo cấu trúc dự án và file `requirements.txt`.
2. Viết module chụp vùng màn hình và bóc tách nội dung (`src/capture/`).
3. Viết module nạp tài liệu và file video (`src/parsers/`).
4. Thiết lập Vector Database ChromaDB và luồng RAG (`src/rag/`).
5. Tạo giao diện hoặc CLI để kích hoạt chụp màn hình và hỏi đáp (`main.py`).