"""
llm_helper.py - Gemini LLM Integration
Xử lý mô tả nhân vật, phân tích ảnh, tối ưu tham số
"""

import os
import json
import base64
from typing import Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Cấu hình Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiAssistant:
    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Khởi tạo Gemini Assistant
        model_name: gemini-2.5-flash (nhanh, rẻ) hoặc gemini-2.5-pro (mạnh hơn)
        """
        self.model = genai.GenerativeModel(model_name)
        print(f"   🤖 Gemini Model: {model_name}")
    
    def analyze_character_description(self, user_input: str) -> Dict:
        """
        Phân tích mô tả nhân vật của user và trả về thông tin có cấu trúc
        
        Input: "một cậu bé đội mũ rơm, có sẹo dưới mắt"
        Output: {
            "character_name": "luffy",
            "full_name": "Monkey D. Luffy",
            "anime": "One Piece",
            "description": "...",
            "suggested_images": ["luffy", "luffy_gear5"]
        }
        """
        prompt = f"""
Bạn là chuyên gia phân tích nhân vật anime/manga. Người dùng mô tả: "{user_input}"

Nhiệm vụ:
1. Nhận diện nhân vật (nếu có)
2. Đề xuất tên file ảnh (viết thường, không dấu, không khoảng trắng)
3. Nếu không nhận diện được, đề xuất nhân vật tương tự

Trả về JSON CHÍNH XÁC (không markdown, không giải thích):
{{
    "character_name": "luffy",
    "full_name": "Monkey D. Luffy",
    "anime": "One Piece",
    "description": "Thuyền trưởng băng Mũ Rơm, có sẹo dưới mắt trái",
    "suggested_images": ["luffy", "luffy_gear5", "luffy_wano"],
    "confidence": "high"
}}

Nếu không chắc chắn, đặt confidence là "low" và đề xuất nhiều lựa chọn hơn.
"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Xử lý trường hợp Gemini trả về markdown
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(text)
            return result
        
        except json.JSONDecodeError as e:
            print(f"   ⚠️  Lỗi parse JSON từ Gemini: {e}")
            print(f"   Raw response: {response.text}")
            # Fallback: trả về kết quả mặc định
            return {
                "character_name": "unknown",
                "full_name": "Unknown Character",
                "anime": "Unknown",
                "description": user_input,
                "suggested_images": ["luffy", "naruto", "goku"],
                "confidence": "low"
            }
        except Exception as e:
            print(f"   ❌ Lỗi khi gọi Gemini: {e}")
            raise
    
    def analyze_image_for_drawing(self, image_path: str) -> Dict:
        """
        Phân tích ảnh và đề xuất tham số vẽ tối ưu
        
        Output: {
            "max_size": 150,
            "threshold": 127,
            "complexity": "medium",
            "estimated_time_minutes": 5,
            "recommendations": [...]
        }
        """
        # Đọc và encode ảnh
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Upload ảnh lên Gemini
        image_parts = [
            {
                "mime_type": "image/jpeg" if image_path.endswith(".jpg") else "image/png",
                "data": image_data
            }
        ]
        
        prompt = """
Phân tích ảnh này để vẽ tự động bằng Paint. Đánh giá:
1. Độ phức tạp (simple/medium/complex)
2. Số lượng chi tiết
3. Kích thước tối ưu để vẽ (100-300 pixels)
4. Ngưỡng đen trắng tối ưu (0-255)
5. Thời gian ước tính (phút)

Trả về JSON (không markdown):
{
    "max_size": 150,
    "threshold": 127,
    "complexity": "medium",
    "estimated_time_minutes": 5,
    "detail_level": "high",
    "recommendations": [
        "Nên giảm kích thước xuống 120px vì ảnh có nhiều chi tiết",
        "Tăng ngưỡng lên 140 để giảm nhiễu"
    ]
}
"""
        
        try:
            response = self.model.generate_content([prompt, image_parts[0]])
            text = response.text.strip()
            
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(text)
            return result
        
        except Exception as e:
            print(f"   ⚠️  Không thể phân tích ảnh bằng Gemini: {e}")
            # Fallback
            return {
                "max_size": 150,
                "threshold": 127,
                "complexity": "medium",
                "estimated_time_minutes": 5,
                "detail_level": "medium",
                "recommendations": ["Sử dụng tham số mặc định"]
            }
    
    def generate_drawing_instructions(self, image_path: str) -> List[Dict]:
        """
        Tạo hướng dẫn vẽ từng bước (dành cho tương lai)
        
        Output: [
            {"step": 1, "instruction": "Vẽ hình tròn cho đầu", "region": "center"},
            ...
        ]
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        image_parts = [{
            "mime_type": "image/jpeg" if image_path.endswith(".jpg") else "image/png",
            "data": image_data
        }]
        
        prompt = """
Phân tích ảnh này và tạo hướng dẫn vẽ từng bước cho người mới bắt đầu.
Chia thành 5-8 bước, từ dễ đến khó.

Trả về JSON (không markdown):
{
    "steps": [
        {"step": 1, "instruction": "Vẽ hình oval cho đầu ở giữa canvas", "region": "center"},
        {"step": 2, "instruction": "Thêm 2 hình tròn nhỏ cho mắt", "region": "upper_center"},
        ...
    ]
}
"""
        
        try:
            response = self.model.generate_content([prompt, image_parts[0]])
            text = response.text.strip()
            
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(text)
            return result.get("steps", [])
        
        except Exception as e:
            print(f"   ⚠️  Không thể tạo hướng dẫn: {e}")
            return []


# ============ HÀM TIỆN ÍCH ============

def find_image_by_suggestions(suggested_names: List[str], image_folder: str = "images") -> Optional[str]:
    """
    Tìm ảnh trong thư mục dựa trên danh sách gợi ý
    
    Args:
        suggested_names: ["luffy", "luffy_gear5", ...]
        image_folder: Thư mục chứa ảnh
    
    Returns:
        Đường dẫn ảnh nếu tìm thấy, None nếu không
    """
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for name in suggested_names:
        for ext in extensions:
            path = os.path.join(image_folder, name + ext)
            if os.path.exists(path):
                return path
    
    return None


def interactive_character_selection(gemini: GeminiAssistant) -> Optional[str]:
    """
    Hỏi user mô tả nhân vật, dùng Gemini phân tích, và tìm ảnh
    
    Returns:
        Đường dẫn ảnh nếu tìm thấy, None nếu không
    """
    print("\n" + "="*70)
    print("🤖 GEMINI CHARACTER ANALYZER")
    print("="*70)
    
    user_input = input("\n🎭 Mô tả nhân vật bạn muốn vẽ (hoặc Enter để dùng Luffy): ").strip()
    
    if not user_input:
        user_input = "một cậu bé đội mũ rơm, có sẹo dưới mắt, là hải tặc"
        print(f"   📝 Dùng mô tả mặc định: {user_input}")
    
    print("\n   🔍 Gemini đang phân tích...")
    result = gemini.analyze_character_description(user_input)
    
    print(f"\n   ✅ Nhận diện: {result['full_name']} ({result['anime']})")
    print(f"   📖 {result['description']}")
    print(f"   🎯 Độ tin cậy: {result['confidence']}")
    
    # Tìm ảnh
    print(f"\n   🔎 Đang tìm ảnh trong thư mục 'images/'...")
    image_path = find_image_by_suggestions(result['suggested_images'])
    
    if image_path:
        print(f"   ✅ Tìm thấy: {image_path}")
        return image_path
    else:
        print(f"\n   ❌ Không tìm thấy ảnh. Hãy thêm một trong các file sau:")
        for name in result['suggested_images'][:3]:
            print(f"      - images/{name}.jpg")
        return None


# ============ TEST ============

if __name__ == "__main__":
    # Test thử
    gemini = GeminiAssistant()
    
    # Test 1: Phân tích mô tả
    print("\n=== TEST 1: Character Analysis ===")
    result = gemini.analyze_character_description("một ninja tóc vàng, có râu cáo")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Test 2: Tìm ảnh
    print("\n=== TEST 2: Find Image ===")
    image_path = interactive_character_selection(gemini)
    print(f"Result: {image_path}")
