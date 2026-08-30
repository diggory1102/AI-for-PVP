Dưới đây là mẫu file `README.md` chuẩn chuyên nghiệp để bạn đưa vào thư mục gốc của dự án:

```markdown
# 🖥️ Windows Vision AI Agent

> Trợ lý AI thông minh tích hợp trên Windows: Tự động chụp và đọc màn hình, phân tích video thời gian thực, bóc tách tài liệu đa nguồn và lưu trữ tri thức tập trung bằng RAG (Retrieval-Augmented Generation).

---

## 📌 Tính năng nổi bật

- **🔍 Đọc & Hiểu màn hình thông minh:** 
  - Kéo chọn vùng màn hình bất kỳ hoặc chỉ định cửa sổ ứng dụng để phân tích.
  - Tự động trích xuất chữ viết (OCR), nhận diện giao diện, sơ đồ, bảng biểu và mã nguồn.
- **🎥 Phân tích Video đa luồng:**
  - **Video trực tiếp (Real-time Stream):** Tự động phát hiện chuyển động khung hình (Scene/Keyframe detection) và bóc băng âm thanh hệ thống để tóm tắt bài giảng/cuộc họp.
  - **Tệp video có sẵn (`.mp4`, `.mkv`, `.avi`):** Bóc tách slide, mã code kèm mốc thời gian (timestamp) và giọng thuyết minh.
- **📚 Cơ sở tri thức hợp nhất (Multi-source RAG):**
  - Nạp đồng thời tài liệu ngoài (`.pdf`, `.docx`, `.txt`, `.py`, `.java`, `.sql`).
  - Gắn nhãn Metadata chi tiết (nguồn, tên file, thời gian) để truy vấn chính xác.
- **💡 Đối chiếu chéo (Cross-referencing):**
  - Hỏi đáp thông minh: Vừa nhìn thao tác trên màn hình vừa đối chiếu với sách/tài liệu lý thuyết để phát hiện lỗi sai và hướng dẫn giải pháp.

---

## 🏗️ Kiến trúc hệ thống


```

[ Màn hình Windows / Video Stream ] ──► [ OpenCV / MSS / Whisper ] ──┐
│
[ Tệp Media (.mp4 / .mkv) ]          ──► [ MoviePy / Gemini File API ] ┼──► [ Chunker & Embedding ]
│            │
[ Tài liệu ngoài (PDF, Docx, Code) ] ──► [ Document Loaders ]         ──┘            ▼
[ Vector DB (ChromaDB) ]
│
▼
[ Trợ lý AI hỏi đáp (RAG) ]

```

---

## 🛠️ Công nghệ sử dụng

- **Ngôn ngữ:** Python 3.10+
- **Bắt màn hình & Xử lý hình ảnh:** `mss`, `opencv-python`, `pillow`, `pygetwindow`
- **Xử lý âm thanh & Video:** `faster-whisper`, `moviepy`
- **Mô hình ngôn ngữ & Thị giác (VLM):** Google Gemini 1.5 Flash / Gemini 2.0 API (`google-generativeai`)
- **Quản lý tri thức & Vector DB:** `langchain`, `chromadb`, `sentence-transformers`
- **Giao diện người dùng:** `PyQt6` / CLI

---

## 📁 Cấu trúc thư mục


```

windows-ai-agent/
├── data/
│   ├── raw_documents/       # Thư mục chứa tài liệu nạp ngoài (PDF, Docx, Code)
│   ├── raw_videos/          # Thư mục chứa file video cần xử lý
│   └── vector_store/        # Cơ sở dữ liệu Vector (ChromaDB)
├── src/
│   ├── capture/             # Module chụp màn hình & bắt luồng video
│   ├── parsers/             # Module bóc tách nội dung từ tài liệu & video
│   ├── rag/                 # Quản lý Vector DB, Embeddings và Retrieval
│   ├── core/                # Bộ não Agent & điều phối LLM/VLM
│   └── ui/                  # Giao diện ứng dụng
├── .env.example
├── requirements.txt
├── main.py
└── README.md

```

---

## 🚀 Hướng dẫn cài đặt & Chạy dự án

### 1. Yêu cầu tiên quyết
- Python 3.10 trở lên
- Đã cài đặt [FFmpeg](https://ffmpeg.org/) (hỗ trợ xử lý media và video)

### 2. Cài đặt môi trường

```bash
# Clone dự án hoặc mở thư mục dự án
cd windows-ai-agent

# Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

```

### 3. Cấu hình khóa API

Tạo file `.env` từ file `.env.example`:

```bash
copy .env.example .env

```

Điền API Key của bạn vào file `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here

```

### 4. Khởi chạy ứng dụng

```bash
python main.py

```

---

## 📖 Hướng dẫn sử dụng nhanh

1. **Chụp và học từ màn hình:** Nhấn tổ hợp phím tắt hoặc nút quét trên giao diện $\rightarrow$ Chọn vùng cần học $\rightarrow$ AI tự động phân tích và lưu vào cơ sở tri thức.
2. **Nạp tài liệu / Video:** Thả file vào thư mục `data/raw_documents/` hoặc `data/raw_videos/` $\rightarrow$ Chạy lệnh cập nhật kiến thức.
3. **Hỏi đáp với AI:** Nhập câu hỏi vào khung chat để bot trả lời dựa trên những gì đang hiển thị trên màn hình và toàn bộ tài liệu đã học.

---

## 📄 Giấy phép

Dự án được phân phối dưới giấy phép [MIT License](https://www.google.com/search?q=LICENSE).

```

```
