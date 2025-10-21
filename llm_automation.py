import pyautogui
import time
import base64
import os
from datetime import datetime
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Chọn LLM provider (thay đổi theo nhu cầu)
USE_OPENAI = False  # False = dùng Gemini
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


def screenshot_to_base64():
    """Chụp màn hình và chuyển thành base64"""
    screenshot = pyautogui.screenshot()
    buffered = BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def ask_llm_openai(prompt, screenshot_b64):
    """Gửi screenshot + prompt đến OpenAI GPT-4o"""
    from openai import OpenAI
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Bạn là AI điều khiển máy tính. Phân tích screenshot và cho tôi các lệnh PyAutoGUI.

Yêu cầu: {prompt}

Trả về JSON format:
{{
    "analysis": "Mô tả những gì bạn thấy",
    "steps": [
        {{"action": "hotkey", "keys": ["win", "r"]}},
        {{"action": "write", "text": "notepad", "interval": 0.1}},
        {{"action": "press", "key": "enter"}},
        {{"action": "wait", "seconds": 1}},
        {{"action": "click", "x": 100, "y": 200}}
    ]
}}

Chỉ trả về JSON, không thêm text nào khác."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    return response.choices[0].message.content


def ask_llm_gemini(prompt, screenshot_b64):
    """Gửi screenshot + prompt đến Google Gemini"""
    import google.generativeai as genai
    
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Decode base64 để gửi cho Gemini
    import base64
    image_data = base64.b64decode(screenshot_b64)
    
    response = model.generate_content([
        f"""Bạn là AI điều khiển máy tính. Phân tích screenshot và cho tôi các lệnh PyAutoGUI.

Yêu cầu: {prompt}

Trả về JSON format:
{{
    "analysis": "Mô tả những gì bạn thấy",
    "steps": [
        {{"action": "hotkey", "keys": ["win", "r"]}},
        {{"action": "write", "text": "notepad", "interval": 0.1}},
        {{"action": "press", "key": "enter"}},
        {{"action": "wait", "seconds": 1}},
        {{"action": "click", "x": 100, "y": 200}}
    ]
}}

Chỉ trả về JSON, không thêm text nào khác.""",
        {"mime_type": "image/png", "data": image_data}
    ])
    
    return response.text


def execute_step(step):
    """Thực thi một bước lệnh PyAutoGUI"""
    action = step.get('action')
    
    if action == 'hotkey':
        pyautogui.hotkey(*step['keys'])
        print(f"   ✓ Nhấn phím tắt: {' + '.join(step['keys'])}")
    
    elif action == 'write':
        pyautogui.write(step['text'], interval=step.get('interval', 0.05))
        print(f"   ✓ Gõ: {step['text']}")
    
    elif action == 'press':
        pyautogui.press(step['key'])
        print(f"   ✓ Nhấn phím: {step['key']}")
    
    elif action == 'click':
        pyautogui.click(step['x'], step['y'])
        print(f"   ✓ Click tại: ({step['x']}, {step['y']})")
    
    elif action == 'moveTo':
        pyautogui.moveTo(step['x'], step['y'], duration=step.get('duration', 0.5))
        print(f"   ✓ Di chuyển đến: ({step['x']}, {step['y']})")
    
    elif action == 'wait':
        time.sleep(step['seconds'])
        print(f"   ⏳ Chờ {step['seconds']}s")
    
    else:
        print(f"   ⚠ Không hiểu action: {action}")


def run_automation(task_description, max_iterations=5):
    """Chạy automation với LLM"""
    print(f"\n{'='*60}")
    print(f"TASK: {task_description}")
    print(f"{'='*60}\n")
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        
        # 1. Chụp màn hình
        print("📸 Chụp màn hình...")
        screenshot_b64 = screenshot_to_base64()
        
        # 2. Gửi đến LLM
        print("🤖 Hỏi LLM...")
        try:
            if USE_OPENAI:
                response = ask_llm_openai(task_description, screenshot_b64)
            else:
                response = ask_llm_gemini(task_description, screenshot_b64)
            
            print(f"📝 LLM response:\n{response}\n")
            
            # 3. Parse JSON
            import json
            # Loại bỏ markdown code block nếu có
            response = response.strip()
            if response.startswith('```'):
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]
            
            instructions = json.loads(response)
            
            print(f"💭 Analysis: {instructions['analysis']}\n")
            print("⚙️ Executing steps...")
            
            # 4. Thực thi các bước
            for step in instructions['steps']:
                execute_step(step)
                time.sleep(0.2)  # Delay nhỏ giữa các bước
            
            print("\n✅ Hoàn thành iteration!")
            
            # 5. Hỏi người dùng có muốn tiếp tục không
            cont = input("\nTiếp tục? (y/n): ").strip().lower()
            if cont != 'y':
                print("🛑 Dừng automation.")
                break
                
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            break


# ============ MAIN ============

if __name__ == "__main__":
    print("=== AUTOMATION VỚI LLM THẬT ===\n")
    
    # Kiểm tra API key
    if USE_OPENAI:
        if not OPENAI_API_KEY:
            print("❌ Chưa có OPENAI_API_KEY trong .env")
            print("Tạo file .env với nội dung:")
            print("OPENAI_API_KEY=sk-...")
            exit(1)
        print("✓ Dùng OpenAI GPT-4o")
    else:
        if not GOOGLE_API_KEY:
            print("❌ Chưa có GEMINI_API_KEY trong .env")
            print("Tạo file .env với nội dung:")
            print("GOOGLE_API_KEY=...")
            exit(1)
        print("✓ Dùng Google Gemini")
    
    print("\n🚀 Bắt đầu automation...")
    print("⚠️  Đưa chuột vào góc màn hình để dừng khẩn cấp!\n")
    time.sleep(2)
    
    # Ví dụ task
    run_automation("Mở Notepad và viết 'Hello from LLM!'")