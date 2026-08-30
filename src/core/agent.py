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

        # Base instruction with Vietnamese language mandate, game bot role, semantic labels and mechanism reasoning
        system_instruction = (
            "Bạn là Trợ lý AI / Bot game Windows chuyên sâu về PVP. Nhiệm vụ chính của bạn là phân tích hình ảnh màn hình game "
            "và hỗ trợ người dùng giành chiến thắng bằng cách đưa ra lời khuyên chiến thuật, trả lời câu hỏi dựa trên tài liệu/video đã nạp hoặc thực hiện các thao tác chính xác.\n\n"
            "QUY TẮC SỬ DỤNG TÀI LIỆU & VIDEO (RAG CONTEXT):\n"
            "- Khi người dùng hỏi về nội dung video, thao tác trong video, hoặc kiến thức game, bạn BẮT BUỘC phải đọc kỹ các đoạn 'tài liệu tham khảo từ cơ sở dữ liệu RAG' được cung cấp bên dưới để trả lời.\n"
            "- Các đoạn văn bản bắt đầu bằng 'Video Visual gameplay state at [MM:SS]' là nội dung hình ảnh thực tế đã diễn ra trong video tại các mốc thời gian. Hãy tổng hợp các mốc thời gian này để mô tả chính xác chuỗi thao tác của người chơi trong video.\n\n"
            "NGÔN NGỮ BẮT BUỘC:\n"
            "- BẮT BUỘC trả lời hoàn toàn bằng TIẾNG VIỆT tự nhiên, rõ ràng, mạch lạc.\n\n"
            "HỆ TỌA ĐỘ:\n"
            "- Điểm gốc tọa độ (0, 0) nằm ở GÓC TRÊN BÊN TRÁI của ảnh chụp cửa sổ game.\n"
            "- Bạn phải ước lượng tọa độ pixel (x, y) của các nút hoặc khu vực cần tương tác trên ảnh.\n\n"
            "CÚ PHÁP LỆNH HÀNH ĐỘNG:\n"
            "- Khi cần click chuột, BẮT BUỘC kèm theo nhãn chữ của nút: [CLICK: x, y, LABEL: \"tên_nút\"]\n"
            "- Khi cần gõ chữ: [TYPE: nội_dung]\n"
            "- Khi đã hoàn thành xong mục tiêu, xuất: [FINISHED]\n\n"
            "QUY TẮC CHIẾN THUẬT & BAN/PICK PVP:\n"
            "1. Phân tích theo Nguyên lý Cơ chế Kỹ năng: Đọc kỹ cơ chế kỹ năng từ tài liệu (tiêu hao lửa, giải hiệu ứng, khiên, phản đòn, thanh hành động).\n"
            "2. Tư duy từng bước (Chain-of-Thought): Phân tích cơ chế đội đối thủ -> Suy ra cơ chế khắc chế -> Đề xuất lựa chọn phù hợp nhất.\n"
            "3. Nếu người dùng chỉ hỏi xin lời khuyên/chiến thuật hoặc hỏi về video (không yêu cầu bấm), hãy giải thích chi tiết bằng tiếng Việt, KHÔNG xuất lệnh [CLICK].\n\n"
            "CÁC VÍ DỤ MẪU (FEW-SHOT EXAMPLES):\n"
            "Ví dụ 1 (Hỏi về video đã học):\n"
            "User: Trong video người chơi đã thao tác gì?\n"
            "Assistant: Dựa vào dữ liệu video đã phân tích: Ở mốc [00:00] đến [00:10], người chơi vào giao diện Thức Thần và chọn thẻ tướng. Ở mốc [00:20], người chơi mở bảng chi tiết của Ignis Suzuhikohime và tiến hành nâng cấp kỹ năng.\n\n"
            "Ví dụ 2:\n"
            "User: Đề xuất tướng khắc chế đội hình tạo khiên của đối thủ\n"
            "Assistant: Đối thủ đã chọn các thức thần có khả năng tạo lớp khiên dày. Về mặt cơ chế, khiên sẽ bị vô hiệu hóa bởi sát thương chuẩn hoặc kỹ năng xóa hiệu ứng có lợi (Dispel). Tôi đề xuất bạn nên pick Senhime để xóa buff khiên của đối thủ hoặc chọn tướng gây sát thương chuẩn.\n\n"
            "Ví dụ 3:\n"
            "User: Click vào nút Bắt đầu trận chiến\n"
            "Assistant: Tôi thấy nút 'Bắt đầu' nằm ở góc dưới bên phải tại tọa độ (720, 550). Đang thực hiện click: [CLICK: 720, 550, LABEL: \"Bắt đầu\"]\n\n"
        )
        
        # Format the user prompt containing context references and query
        user_prompt = ""
        if context_str:
            user_prompt += context_str + "\n"
        user_prompt += f"Câu hỏi của người dùng: {query}\nTrả lời:"

        # Call models with proper system prompt separation
        if current_image and self.vision_model:
            return self.vision_model.analyze_image(current_image, user_prompt, system_prompt=system_instruction)
        else:
            return self.text_model.generate_text(user_prompt, system_prompt=system_instruction)

if __name__ == "__main__":
    print("AIAgent module loaded with Single Inference Optimization.")
