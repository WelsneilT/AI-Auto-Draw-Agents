"""
======================================================================================
ANIME STORY STUDIO V5 - PURE COMPUTER VISION (NO HOTKEYS)
100% Mouse Click - Không dùng phím tắt Windows
======================================================================================
"""

import os
import time
import base64
import json
import subprocess
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import pyautogui
import cv2
import numpy as np
from typing import TypedDict, List, Dict, Tuple, Optional
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import traceback
import pyperclip

# Load environment
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Tắt fail-safe
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05

# ============ SHARED STATE ============

class StoryArtState(TypedDict):
    """State chung cho cả 2 agents"""
    # User Input
    character_name: str
    story_theme: str
    story_length: int
    mode: str
    
    # Story Writer State
    character_info: Dict
    story_outline: Dict
    story_content: str
    story_file_path: str
    
    # Art Creator State
    reference_image_b64: str
    paint_screenshot_b64: str
    tool_positions: Dict
    color_positions: Dict
    canvas_area: Dict
    drawing_steps: List[Dict]
    current_step: int
    total_steps: int
    
    # Window Management State
    paint_window_ready: bool
    paint_maximized: bool
    
    # Shared
    execution_log: List[str]
    is_complete: bool
    error: str


# ============ UTILITY FUNCTIONS ============

def screenshot_to_base64(region=None) -> str:
    """Chụp màn hình và convert sang base64"""
    if region:
        screenshot = pyautogui.screenshot(region=region)
    else:
        screenshot = pyautogui.screenshot()
    buffered = BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def image_to_base64(image_path: str) -> str:
    """Convert ảnh file sang base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def call_gemini(prompt: str) -> str:
    """Gọi Gemini text-only"""
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content(prompt)
    return response.text

def call_gemini_vision(prompt: str, images: List[Image.Image]) -> str:
    """Gọi Gemini với vision"""
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    content = images + [prompt]
    response = model.generate_content(content)
    return response.text

def simplify_image_for_drawing(image_path: str, output_path: str = "simplified.png"):
    """Đơn giản hóa ảnh thành dạng cartoon/sketch dễ vẽ"""
    img = cv2.imread(image_path)
    
    # Resize về kích thước vừa phải
    height, width = img.shape[:2]
    max_size = 300
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Chuyển sang cartoon style
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # Detect edges
    edges = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        9, 9
    )
    
    # Smooth colors
    color = cv2.bilateralFilter(img, 9, 250, 250)
    
    # Combine
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    cv2.imwrite(output_path, cartoon)
    print(f"   ✅ Simplified image saved: {output_path}")
    return output_path


# ============ COMPUTER VISION MODULE ============

def find_button_by_template(screenshot_b64: str, button_name: str) -> Optional[Tuple[int, int]]:
    """
    Tìm button bằng template matching
    """
    # Decode screenshot
    img_data = base64.b64decode(screenshot_b64)
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Template path (cần chuẩn bị trước)
    template_path = f"templates/{button_name}.png"
    
    if not os.path.exists(template_path):
        return None
    
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    # Template matching
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val > 0.7:  # Threshold
        # Trả về tâm của button
        h, w = template.shape
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        return (center_x, center_y)
    
    return None


def detect_window_state_with_llm(screenshot_b64: str) -> Dict:
    """
    Dùng LLM để phân tích trạng thái window hiện tại
    """
    img = Image.open(BytesIO(base64.b64decode(screenshot_b64)))
    img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    
    prompt = """
Phân tích màn hình hiện tại và trả về JSON:

{{
  "window_type": "paint" | "dialog" | "desktop" | "other",
  "is_maximized": true/false,
  "has_dialog": true/false,
  "dialog_type": "share" | "save" | "close_confirm" | null,
  "action_needed": "maximize" | "close_dialog" | "click_canvas" | "ready",
  "maximize_button_position": {{"x": 1234, "y": 56}} or null,
  "close_button_position": {{"x": 1234, "y": 56}} or null,
  "canvas_visible": true/false,
  "description": "Mô tả ngắn gọn"
}}

QUY TẮC:
1. Nếu thấy dialog "Share" → action_needed = "close_dialog"
2. Nếu Paint chưa maximize (thấy nút maximize ở góc trên phải) → action_needed = "maximize"
3. Nếu Paint đã maximize và không có dialog → action_needed = "ready"
4. Xác định tọa độ CHÍNH XÁC của nút cần click
"""
    
    response = call_gemini_vision(prompt, [img])
    
    # Extract JSON
    json_str = response
    if '```json' in response:
        json_str = response.split('```json')[1].split('```')[0]
    elif '```' in response:
        json_str = response.split('```')[1].split('```')[0]
    
    return json.loads(json_str.strip())


def find_maximize_button_cv(screenshot_b64: str) -> Optional[Tuple[int, int]]:
    """
    Tìm nút Maximize bằng Computer Vision (heuristic)
    Nút maximize thường ở góc trên phải, trước nút X
    """
    img_data = base64.b64decode(screenshot_b64)
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    
    height, width = img.shape[:2]
    
    # Vùng title bar (phía trên cùng)
    title_bar = img[0:50, width-200:width]
    
    # Convert to grayscale
    gray = cv2.cvtColor(title_bar, cv2.COLOR_BGR2GRAY)
    
    # Tìm các cạnh (buttons thường có viền)
    edges = cv2.Canny(gray, 50, 150)
    
    # Tìm contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc contours có kích thước giống button (20-40 pixels)
    buttons = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 15 < w < 50 and 15 < h < 50:
            # Tọa độ tuyệt đối
            abs_x = (width - 200) + x + w // 2
            abs_y = y + h // 2
            buttons.append((abs_x, abs_y))
    
    # Nút maximize thường là nút thứ 2 từ phải sang (trước nút X)
    if len(buttons) >= 2:
        buttons.sort(key=lambda p: p[0], reverse=True)
        return buttons[1]  # Nút thứ 2
    
    # Fallback: ước lượng vị trí
    return (width - 90, 15)


def find_close_dialog_button_cv(screenshot_b64: str) -> Optional[Tuple[int, int]]:
    """
    Tìm nút X để đóng dialog
    """
    img_data = base64.b64decode(screenshot_b64)
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    
    height, width = img.shape[:2]
    
    # Dialog thường ở giữa màn hình
    dialog_region = img[height//4:3*height//4, width//4:3*width//4]
    
    # Tìm nút X (thường ở góc trên phải của dialog)
    gray = cv2.cvtColor(dialog_region, cv2.COLOR_BGR2GRAY)
    
    # Template matching cho nút X
    # Hoặc tìm vùng tối (nút X thường có màu đỏ/tối)
    
    # Fallback: Ước lượng vị trí nút X của dialog
    # Thường ở góc trên phải của dialog
    dialog_x = width // 2 + 300  # Giả sử dialog rộng 600px
    dialog_y = height // 4 + 10
    
    return (dialog_x, dialog_y)


# ============ WINDOW MANAGEMENT AGENT ============

def window_setup_agent(state: StoryArtState) -> StoryArtState:
    """
    Agent setup Paint window - PURE VISION (NO HOTKEYS)
    """
    
    if state['mode'] == 'story_only' or state['error']:
        return state
    
    print("\n" + "="*70)
    print("🪟 WINDOW SETUP AGENT (Pure Vision)")
    print("="*70)
    
    MAX_ATTEMPTS = 10
    
    try:
        for attempt in range(MAX_ATTEMPTS):
            print(f"\n   🔄 Attempt {attempt + 1}/{MAX_ATTEMPTS}")
            
            # 1. Chụp màn hình hiện tại
            print("      📸 Capturing current screen...")
            screenshot_b64 = screenshot_to_base64()
            
            # 2. LLM phân tích trạng thái
            print("      🧠 LLM analyzing window state...")
            window_state = detect_window_state_with_llm(screenshot_b64)
            
            print(f"      📊 State: {window_state.get('description', 'N/A')}")
            print(f"      🎯 Action needed: {window_state.get('action_needed')}")
            
            action = window_state.get('action_needed')
            
            # 3. Thực hiện action tương ứng
            if action == "ready":
                print("      ✅ Paint is ready!")
                state['paint_window_ready'] = True
                state['paint_maximized'] = True
                state['paint_screenshot_b64'] = screenshot_b64
                break
                
            elif action == "close_dialog":
                print("      🚪 Closing dialog...")
                
                # Lấy vị trí nút close từ LLM
                close_pos = window_state.get('close_button_position')
                
                if not close_pos:
                    # Fallback: Dùng CV
                    print("         🔍 CV finding close button...")
                    close_pos_tuple = find_close_dialog_button_cv(screenshot_b64)
                    if close_pos_tuple:
                        close_pos = {'x': close_pos_tuple[0], 'y': close_pos_tuple[1]}
                
                if close_pos:
                    print(f"         🖱️  Clicking close at ({close_pos['x']}, {close_pos['y']})")
                    pyautogui.click(close_pos['x'], close_pos['y'])
                    time.sleep(1)
                else:
                    # Fallback cuối: ESC key (ngoại lệ duy nhất)
                    print("         ⚠️  Using ESC as fallback...")
                    pyautogui.press('esc')
                    time.sleep(0.5)
                
            elif action == "maximize":
                print("      📐 Maximizing window...")
                
                # Lấy vị trí nút maximize từ LLM
                max_pos = window_state.get('maximize_button_position')
                
                if not max_pos:
                    # Fallback: Dùng CV
                    print("         🔍 CV finding maximize button...")
                    max_pos_tuple = find_maximize_button_cv(screenshot_b64)
                    if max_pos_tuple:
                        max_pos = {'x': max_pos_tuple[0], 'y': max_pos_tuple[1]}
                
                if max_pos:
                    print(f"         🖱️  Clicking maximize at ({max_pos['x']}, {max_pos['y']})")
                    pyautogui.click(max_pos['x'], max_pos['y'])
                    time.sleep(1)
                else:
                    # Fallback: Double-click title bar
                    print("         🔄 Double-clicking title bar...")
                    screen_width, screen_height = pyautogui.size()
                    pyautogui.doubleClick(screen_width // 2, 10)
                    time.sleep(1)
                
            elif action == "click_canvas":
                print("      🖱️  Clicking canvas to focus...")
                screen_width, screen_height = pyautogui.size()
                pyautogui.click(screen_width // 2, screen_height // 2)
                time.sleep(0.5)
                
            else:
                print(f"      ⚠️  Unknown action: {action}")
                time.sleep(1)
        
        if not state.get('paint_window_ready'):
            raise Exception("Failed to setup Paint window after max attempts")
        
        state['execution_log'].append("✓ Window Setup: Paint ready")
        
    except Exception as e:
        print(f"❌ Window setup error: {e}")
        state['error'] = f"Window setup failed: {e}"
        traceback.print_exc()
    
    return state


# ============ AGENT 1: STORY WRITER (UNCHANGED) ============

def story_research_agent(state: StoryArtState) -> StoryArtState:
    """Agent tìm kiếm thông tin về nhân vật"""
    
    if state['mode'] == 'art_only':
        print("\n⏩ Skipping story research (art_only mode)")
        return state
    
    print("\n" + "="*70)
    print("📚 STORY RESEARCH AGENT")
    print("="*70)
    
    try:
        prompt = f"""
Tìm kiếm thông tin về nhân vật hoạt hình "{state['character_name']}".

Trả về JSON với format:
{{
    "name": "Tên nhân vật",
    "origin": "Xuất xứ (anime/cartoon nào)",
    "personality": ["Tính cách 1", "Tính cách 2", ...],
    "appearance": "Mô tả ngoại hình",
    "special_abilities": ["Khả năng đặc biệt 1", ...],
    "famous_quotes": ["Câu nói nổi tiếng 1", ...]
}}
"""
        
        response = call_gemini(prompt)
        
        # Extract JSON
        json_str = response
        if '```json' in response:
            json_str = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            json_str = response.split('```')[1].split('```')[0]
        
        character_info = json.loads(json_str.strip())
        state['character_info'] = character_info
        
        print(f"\n✅ Found info about: {character_info['name']}")
        print(f"   Origin: {character_info['origin']}")
        
        state['execution_log'].append(f"✓ Research: Found info about {character_info['name']}")
        
    except Exception as e:
        print(f"❌ Research error: {e}")
        state['error'] = f"Research failed: {e}"
    
    return state


def story_outline_agent(state: StoryArtState) -> StoryArtState:
    """Agent tạo dàn ý truyện"""
    
    if state['mode'] == 'art_only' or state['error']:
        return state
    
    print("\n" + "="*70)
    print("📝 STORY OUTLINE AGENT")
    print("="*70)
    
    try:
        char_info = state['character_info']
        
        prompt = f"""
Tạo dàn ý cho một câu chuyện ngắn về {char_info['name']}.

THÔNG TIN NHÂN VẬT:
- Xuất xứ: {char_info['origin']}
- Tính cách: {', '.join(char_info['personality'])}

CHỦ ĐỀ: {state['story_theme']}
ĐỘ DÀI: {state['story_length']} từ

JSON FORMAT:
{{
    "title": "Tiêu đề hấp dẫn",
    "setting": "Bối cảnh",
    "act1": "Mở đầu",
    "act2": "Phát triển",
    "act3": "Kết thúc",
    "moral": "Bài học"
}}
"""
        
        response = call_gemini(prompt)
        
        json_str = response
        if '```json' in response:
            json_str = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            json_str = response.split('```')[1].split('```')[0]
        
        outline = json.loads(json_str.strip())
        state['story_outline'] = outline
        
        print(f"\n✅ Outline: {outline['title']}")
        state['execution_log'].append("✓ Outline created")
        
    except Exception as e:
        print(f"❌ Outline error: {e}")
        state['error'] = f"Outline failed: {e}"
    
    return state


def story_writer_agent(state: StoryArtState) -> StoryArtState:
    """Agent viết nội dung truyện"""
    
    if state['mode'] == 'art_only' or state['error']:
        return state
    
    print("\n" + "="*70)
    print("✍️ STORY WRITER AGENT")
    print("="*70)
    
    try:
        char_info = state['character_info']
        outline = state['story_outline']
        
        prompt = f"""
Viết một câu chuyện hoàn chỉnh về {char_info['name']}.

TIÊU ĐỀ: {outline['title']}
DÀN Ý:
- Mở đầu: {outline['act1']}
- Phát triển: {outline['act2']}
- Kết thúc: {outline['act3']}

YÊU CẦU:
- Độ dài: {state['story_length']} từ
- Ngôn ngữ: Tiếng Việt, sinh động
- Có thoại trực tiếp

Chỉ viết nội dung truyện.
"""
        
        print("   🤖 AI đang viết truyện...")
        response = call_gemini(prompt)
        
        story_content = f"""
{'='*70}
{outline['title'].upper()}
{'='*70}

{response.strip()}

{'='*70}
Bài học: {outline['moral']}
{'='*70}
"""
        
        state['story_content'] = story_content
        
        word_count = len(response.split())
        print(f"\n✅ Story written: {word_count} words")
        state['execution_log'].append(f"✓ Story: {word_count} words")
        
    except Exception as e:
        print(f"❌ Writer error: {e}")
        state['error'] = f"Writer failed: {e}"
    
    return state


def story_formatter_agent(state: StoryArtState) -> StoryArtState:
    """Agent mở Notepad và ghi truyện - DÙNG MOUSE CLICK"""
    
    if state['mode'] == 'art_only' or state['error']:
        return state
    
    print("\n" + "="*70)
    print("💾 STORY FORMATTER AGENT")
    print("="*70)
    
    try:
        # 1. Mở Notepad
        print("   📝 Opening Notepad...")
        subprocess.Popen(['notepad'])
        time.sleep(2)
        
        # 2. Maximize bằng double-click title bar (KHÔNG DÙNG HOTKEY)
        print("   📐 Maximizing Notepad...")
        screen_width, screen_height = pyautogui.size()
        pyautogui.doubleClick(screen_width // 2, 10)
        time.sleep(1)
        
        # 3. Paste nội dung
        print("   ⌨️  Pasting story content...")
        pyperclip.copy(state['story_content'])
        pyautogui.hotkey('ctrl', 'v')  # Ctrl+V là OK (không phải Windows hotkey)
        time.sleep(1)
        
        # 4. Save file bằng menu click (KHÔNG DÙNG Ctrl+S)
        print("   💾 Saving file...")
        
        # Click vào File menu
        pyautogui.click(15, 30)  # Vị trí File menu
        time.sleep(0.5)
        
        # Click vào Save
        pyautogui.click(15, 80)  # Vị trí Save
        time.sleep(1)
        
        # Nhập tên file
        char_name = state['character_name'].replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"story_{char_name}_{timestamp}.txt"
        
        pyperclip.copy(filename)
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.2)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.5)
        
        pyautogui.press('enter')
        time.sleep(0.5)
        
        state['story_file_path'] = filename
        
        print(f"\n✅ Story saved: {filename}")
        state['execution_log'].append(f"✓ Saved: {filename}")
        
        # Đóng Notepad bằng click nút X
        print("   🚪 Closing Notepad...")
        pyautogui.click(screen_width - 15, 10)
        time.sleep(1)
        
    except Exception as e:
        print(f"❌ Formatter error: {e}")
        state['error'] = f"Formatter failed: {e}"
    
    return state


# ============ AGENT 2: ART CREATOR ============

def art_preparation_agent(state: StoryArtState) -> StoryArtState:
    """Agent chuẩn bị ảnh tham khảo và mở Paint - NO HOTKEYS"""
    
    if state['mode'] == 'story_only' or state['error']:
        return state
    
    print("\n" + "="*70)
    print("🎨 ART PREPARATION AGENT")
    print("="*70)
    
    IMAGE_FOLDER = "images"
    char_name = state['character_name'].replace(' ', '_').lower()
    REFERENCE_IMAGE_PATH = os.path.join(IMAGE_FOLDER, f"{char_name}.jpg")
    
    try:
        # 1. Kiểm tra ảnh tham khảo
        if not os.path.exists(REFERENCE_IMAGE_PATH):
            REFERENCE_IMAGE_PATH = os.path.join(IMAGE_FOLDER, f"{char_name}.png")
            
            if not os.path.exists(REFERENCE_IMAGE_PATH):
                print(f"\n⚠️  Không tìm thấy ảnh: {REFERENCE_IMAGE_PATH}")
                
                import webbrowser
                search_url = f"https://www.google.com/search?q={state['character_name']}+simple+drawing&tbm=isch"
                webbrowser.open(search_url)
                
                input("\n   ⏸️  Tải ảnh và lưu vào images/, nhấn Enter...")
                
                if not os.path.exists(REFERENCE_IMAGE_PATH):
                    raise FileNotFoundError(f"Không tìm thấy: {REFERENCE_IMAGE_PATH}")
        
        print(f"   ✅ Found: {REFERENCE_IMAGE_PATH}")
        
        # 2. Đơn giản hóa ảnh
        print("   🖼️  Simplifying image...")
        simplified_path = simplify_image_for_drawing(REFERENCE_IMAGE_PATH)
        
        # 3. Load ảnh vào state
        state['reference_image_b64'] = image_to_base64(simplified_path)
        
        # 4. Đóng Paint cũ bằng taskkill
        print("\n   🧹 Closing old Paint...")
        try:
            subprocess.run(['taskkill', '/F', '/IM', 'mspaint.exe'], 
                          capture_output=True, 
                          timeout=3)
            time.sleep(1)
        except:
            pass
        
        # 5. Mở Paint mới
        print("   🎨 Opening Paint...")
        subprocess.Popen(['mspaint'])
        time.sleep(3)
        
        state['execution_log'].append("✓ Preparation: Image ready, Paint opened")
        
    except Exception as e:
        print(f"❌ Preparation error: {e}")
        state['error'] = f"Preparation failed: {e}"
        traceback.print_exc()
    
    return state


def detect_canvas_area(img_cv: np.ndarray) -> Dict:
    """Tìm vùng canvas (vùng trắng lớn nhất) trong Paint"""
    print("   🔍 Detecting canvas area...")
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise Exception("Cannot find canvas area!")
    
    canvas_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(canvas_contour)
    
    margin = 20
    canvas = {
        'x': x + margin,
        'y': y + margin,
        'width': w - 2*margin,
        'height': h - 2*margin
    }
    
    print(f"   ✅ Canvas: ({canvas['x']}, {canvas['y']}) - {canvas['width']}x{canvas['height']}")
    
    return canvas


def art_cv_analyzer_agent(state: StoryArtState) -> StoryArtState:
    """Agent phân tích giao diện Paint bằng Computer Vision"""
    
    if state['mode'] == 'story_only' or state['error']:
        return state
    
    print("\n" + "="*70)
    print("🔬 COMPUTER VISION ANALYSIS")
    print("="*70)
    
    try:
        # Decode screenshot
        img_data = base64.b64decode(state['paint_screenshot_b64'])
        img_pil = Image.open(BytesIO(img_data))
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        screen_height, screen_width = img_cv.shape[:2]
        print(f"   📐 Screen: {screen_width}x{screen_height}")
        
        # 1. Detect Canvas
        canvas_area = detect_canvas_area(img_cv)
        state['canvas_area'] = canvas_area
        
        # 2. Detect Colors & Tools bằng LLM Vision
        print("\n   🧠 LLM detecting tools and colors...")
        
        prompt = """
Phân tích giao diện Paint và trả về JSON:

{{
  "tool_positions": {{
    "pencil": {{"x": 334, "y": 65}},
    "brush": {{"x": 350, "y": 81}},
    "fill": {{"x": 306, "y": 81}},
    "eraser": {{"x": 290, "y": 65}},
    "color_picker": {{"x": 274, "y": 81}},
    "text": {{"x": 386, "y": 65}}, 
    "line": {{"x": 370, "y": 65}},
    "curve": {{"x": 386, "y": 81}},
    "rectangle": {{"x": 418, "y": 65}},
    "polygon": {{"x": 434, "y": 65}},
    "ellipse": {{"x": 402, "y": 81}},
    "rounded_rectangle": {{"x": 450, "y": 65}}
  }},
  "color_positions": {{
    "black": {{"x": 573, "y": 63}},
    "white": {{"x": 619, "y": 99}},
    "gray": {{"x": 589, "y": 63}},
    "red": {{"x": 638, "y": 63}},
    "orange": {{"x": 654, "y": 63}},
    "yellow": {{"x": 692, "y": 63}},
    "green": {{"x": 710, "y": 63}},
    "blue": {{"x": 728, "y": 63}},
    "purple": {{"x": 746, "y": 63}},
    "brown": {{"x": 670, "y": 63}}
  }},
  "other_buttons": {{
    "size": {{"x": 48, "y": 133}},
    "outline": {{"x": 244, "y": 81}}
  }}
}}

YÊU CẦU:
- Xác định tọa độ CHÍNH XÁC của TỪNG tool trong toolbar
- Xác định tọa độ CHÍNH XÁC của TỪNG màu trong color palette
- Tọa độ phải là pixel tuyệt đối trên màn hình
"""
        
        img_pil_small = img_pil.copy()
        img_pil_small.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        
        response = call_gemini_vision(prompt, [img_pil_small])
        
        # Extract JSON
        json_str = response
        if '```json' in response:
            json_str = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            json_str = response.split('```')[1].split('```')[0]
        
        detection = json.loads(json_str.strip())
        
        state['tool_positions'] = detection.get('tool_positions', {})
        state['color_positions'] = detection.get('color_positions', {})
        
        print(f"\n   ✅ Detected {len(state['tool_positions'])} tools")
        print(f"   ✅ Detected {len(state['color_positions'])} colors")
        
        state['execution_log'].append("✓ CV Analysis: Interface mapped")
        
    except Exception as e:
        print(f"❌ CV Analysis error: {e}")
        state['error'] = f"CV Analysis failed: {e}"
        traceback.print_exc()
    
    return state


def art_planner_agent(state: StoryArtState) -> StoryArtState:
    """Agent tạo kế hoạch vẽ ĐƠN GIẢN"""
    
    if state['mode'] == 'story_only' or state['error']:
        return state
    
    print("\n" + "="*70)
    print("🧠 ART PLANNER AGENT")
    print("="*70)
    
    try:
        # Load reference image
        ref_img = Image.open(BytesIO(base64.b64decode(state['reference_image_b64'])))
        ref_img.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        canvas = state['canvas_area']
        
        prompt = f"""
Tạo kế hoạch vẽ SIÊU ĐƠN GIẢN cho nhân vật "{state['character_name']}" trong Paint.

ẢNH THAM KHẢO: Ảnh đã được đơn giản hóa

THÔNG TIN CANVAS:
- Vị trí: ({canvas['x']}, {canvas['y']})
- Kích thước: {canvas['width']}x{canvas['height']}

TOOLS CÓ SẴN:
{json.dumps(list(state['tool_positions'].keys()), indent=2)}

COLORS CÓ SẴN:
{json.dumps(list(state['color_positions'].keys()), indent=2)}

TẠO KẾ HOẠCH VẼ:
- CHỈ 6-8 BƯỚC ĐƠN GIẢN
- Dùng ellipse và rectangle để tạo hình cơ bản
- Tô màu 2-3 màu chính
- KHÔNG VẼ CHI TIẾT NHỎ

JSON FORMAT:
{{
  "character_analysis": {{
    "main_shapes": ["head (ellipse)", "body (rectangle)", ...],
    "main_colors": ["yellow", "red", "black"],
    "complexity": "simple"
  }},
  "steps": [
    {{
      "step": 1,
      "description": "Vẽ đầu (ellipse lớn)",
      "tool": "ellipse",
      "color": "black",
      "action": "drag",
      "start_x": 600,
      "start_y": 300,
      "end_x": 800,
      "end_y": 450,
      "note": "Outline của đầu"
    }},
    {{
      "step": 2,
      "description": "Tô màu đầu",
      "tool": "fill",
      "color": "yellow",
      "action": "click",
      "click_x": 700,
      "click_y": 375,
      "note": "Fill màu da/lông"
    }},
    {{
      "step": 3,
      "description": "Vẽ thân",
      "tool": "rectangle",
      "color": "black",
      "action": "drag",
      "start_x": 650,
      "start_y": 430,
      "end_x": 750,
      "end_y": 600,
      "note": "Outline của thân"
    }}
  ]
}}

LƯU Ý:
- Tọa độ phải nằm TRONG canvas: x ∈ [{canvas['x']}, {canvas['x'] + canvas['width']}], y ∈ [{canvas['y']}, {canvas['y'] + canvas['height']}]
- action = "drag" cho ellipse/rectangle/line (cần start_x, start_y, end_x, end_y)
- action = "click" cho fill (chỉ cần click_x, click_y)
- CHỈ 6-8 BƯỚC
"""
        
        print("   🤖 AI creating simple drawing plan...")
        response = call_gemini_vision(prompt, [ref_img])
        
        # Extract JSON
        json_str = response
        if '```json' in response:
            json_str = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            json_str = response.split('```')[1].split('```')[0]
        
        plan = json.loads(json_str.strip())
        
        state['drawing_steps'] = plan.get('steps', [])
        state['current_step'] = 0
        state['total_steps'] = len(state['drawing_steps'])
        
        print(f"\n✅ Plan created:")
        print(f"   Character: {plan.get('character_analysis', {}).get('main_shapes', [])}")
        print(f"   Colors: {plan.get('character_analysis', {}).get('main_colors', [])}")
        print(f"   Total steps: {state['total_steps']}")
        
        # In ra các bước
        for step in state['drawing_steps']:
            print(f"   Step {step['step']}: {step['description']}")
        
        state['execution_log'].append(f"✓ Plan: {state['total_steps']} steps")
        
    except Exception as e:
        print(f"❌ Planner error: {e}")
        state['error'] = f"Planner failed: {e}"
        traceback.print_exc()
    
    return state


def safe_click_tool(tool_name: str, tool_positions: Dict, retry: int = 2) -> bool:
    """Click vào tool với retry"""
    if tool_name not in tool_positions:
        print(f"      ⚠️  Tool '{tool_name}' not found in positions")
        return False
    
    pos = tool_positions[tool_name]
    
    for attempt in range(retry):
        try:
            print(f"      🔧 Clicking {tool_name} at ({pos['x']}, {pos['y']})...")
            pyautogui.click(pos['x'], pos['y'])
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"         ⚠️  Attempt {attempt+1} failed: {e}")
            time.sleep(0.2)
    
    return False


def safe_click_color(color_name: str, color_positions: Dict, retry: int = 2) -> bool:
    """Click vào màu với retry"""
    if color_name not in color_positions:
        print(f"      ⚠️  Color '{color_name}' not found in positions")
        return False
    
    pos = color_positions[color_name]
    
    for attempt in range(retry):
        try:
            print(f"      🎨 Clicking {color_name} at ({pos['x']}, {pos['y']})...")
            pyautogui.click(pos['x'], pos['y'])
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"         ⚠️  Attempt {attempt+1} failed: {e}")
            time.sleep(0.2)
    
    return False


def safe_drag(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
    """Thực hiện drag an toàn"""
    try:
        print(f"      ✏️  Dragging from ({start_x},{start_y}) to ({end_x},{end_y})...")
        
        # Di chuyển đến vị trí bắt đầu
        pyautogui.moveTo(start_x, start_y, duration=0.2)
        time.sleep(0.1)
        
        # Giữ chuột trái và kéo
        pyautogui.mouseDown(button='left')
        time.sleep(0.1)
        
        pyautogui.moveTo(end_x, end_y, duration=duration)
        time.sleep(0.1)
        
        pyautogui.mouseUp(button='left')
        time.sleep(0.2)
        
        return True
        
    except Exception as e:
        print(f"         ❌ Drag failed: {e}")
        # Đảm bảo thả chuột nếu có lỗi
        try:
            pyautogui.mouseUp(button='left')
        except:
            pass
        return False


def art_executor_agent(state: StoryArtState) -> StoryArtState:
    """Agent thực thi vẽ TOÀN BỘ các bước - PURE MOUSE"""
    
    if state['mode'] == 'story_only' or state['error']:
        return state
    
    print("\n" + "="*70)
    print("⚙️ ART EXECUTOR AGENT - FULL AUTO DRAWING")
    print("="*70)
    
    tool_positions = state['tool_positions']
    color_positions = state['color_positions']
    steps = state['drawing_steps']
    
    try:
        # Click vào canvas trước để focus
        canvas = state['canvas_area']
        canvas_center_x = canvas['x'] + canvas['width'] // 2
        canvas_center_y = canvas['y'] + canvas['height'] // 2
        
        print(f"\n   🖱️  Focusing canvas at ({canvas_center_x}, {canvas_center_y})...")
        pyautogui.click(canvas_center_x, canvas_center_y)
        time.sleep(0.5)
        
        # VẼ TOÀN BỘ KHÔNG DỪNG
        for i, step_data in enumerate(steps):
            step_num = i + 1
            print(f"\n   [{step_num}/{len(steps)}] {step_data['description']}")
            
            tool = step_data.get('tool', 'ellipse')
            color = step_data.get('color', 'black')
            action = step_data.get('action', 'drag')
            
            # 1. Chọn tool
            if not safe_click_tool(tool, tool_positions):
                print(f"      ⚠️  Skipping step {step_num} - tool selection failed")
                continue
            
            # 2. Chọn màu
            if not safe_click_color(color, color_positions):
                print(f"      ⚠️  Warning: color selection failed, continuing anyway...")
            
            # 3. Thực hiện vẽ
            if action == 'drag':
                sx = step_data.get('start_x', 600)
                sy = step_data.get('start_y', 300)
                ex = step_data.get('end_x', 800)
                ey = step_data.get('end_y', 450)
                
                if not safe_drag(sx, sy, ex, ey):
                    print(f"      ⚠️  Step {step_num} drag failed")
                
            elif action == 'click':
                cx = step_data.get('click_x', 700)
                cy = step_data.get('click_y', 400)
                
                print(f"      🖱️  Clicking at ({cx},{cy})...")
                pyautogui.click(cx, cy)
                time.sleep(0.3)
            
            time.sleep(0.5)
            
            print(f"      ✅ Step {step_num} done!")
            state['current_step'] = step_num
        
        # Đánh dấu hoàn thành
        state['is_complete'] = True
        
        print(f"\n   🎉 ALL {len(steps)} STEPS COMPLETED!")
        state['execution_log'].append(f"✓ Drawing: {len(steps)} steps completed")
        
        # Lưu ảnh cuối cùng
        print("\n   💾 Saving final drawing...")
        time.sleep(1)
        
        char_name = state['character_name'].replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drawing_{char_name}_{timestamp}.png"
        
        # Chụp màn hình
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        
        print(f"   ✅ Saved: {filename}")
        state['execution_log'].append(f"✓ Image: {filename}")
        
    except Exception as e:
        print(f"❌ Executor error: {e}")
        state['error'] = f"Executor failed: {e}"
        traceback.print_exc()
    
    return state


# ============ BUILD GRAPH ============

def build_workflow(mode: str):
    """Build workflow dựa trên mode"""
    workflow = StateGraph(StoryArtState)
    
    if mode == "story_only":
        # Chỉ viết truyện
        workflow.add_node("story_research", story_research_agent)
        workflow.add_node("story_outline", story_outline_agent)
        workflow.add_node("story_writer", story_writer_agent)
        workflow.add_node("story_formatter", story_formatter_agent)
        
        workflow.set_entry_point("story_research")
        workflow.add_edge("story_research", "story_outline")
        workflow.add_edge("story_outline", "story_writer")
        workflow.add_edge("story_writer", "story_formatter")
        workflow.add_edge("story_formatter", END)
        
    elif mode == "art_only":
        # Chỉ vẽ tranh - PURE VISION WORKFLOW
        workflow.add_node("art_preparation", art_preparation_agent)
        workflow.add_node("window_setup", window_setup_agent)
        workflow.add_node("cv_analyzer", art_cv_analyzer_agent)
        workflow.add_node("art_planner", art_planner_agent)
        workflow.add_node("art_executor", art_executor_agent)
        
        workflow.set_entry_point("art_preparation")
        workflow.add_edge("art_preparation", "window_setup")
        workflow.add_edge("window_setup", "cv_analyzer")
        workflow.add_edge("cv_analyzer", "art_planner")
        workflow.add_edge("art_planner", "art_executor")
        workflow.add_edge("art_executor", END)
        
    else:  # full
        # Cả 2
        workflow.add_node("story_research", story_research_agent)
        workflow.add_node("story_outline", story_outline_agent)
        workflow.add_node("story_writer", story_writer_agent)
        workflow.add_node("story_formatter", story_formatter_agent)
        workflow.add_node("art_preparation", art_preparation_agent)
        workflow.add_node("window_setup", window_setup_agent)
        workflow.add_node("cv_analyzer", art_cv_analyzer_agent)
        workflow.add_node("art_planner", art_planner_agent)
        workflow.add_node("art_executor", art_executor_agent)
        
        workflow.set_entry_point("story_research")
        workflow.add_edge("story_research", "story_outline")
        workflow.add_edge("story_outline", "story_writer")
        workflow.add_edge("story_writer", "story_formatter")
        workflow.add_edge("story_formatter", "art_preparation")
        workflow.add_edge("art_preparation", "window_setup")
        workflow.add_edge("window_setup", "cv_analyzer")
        workflow.add_edge("cv_analyzer", "art_planner")
        workflow.add_edge("art_planner", "art_executor")
        workflow.add_edge("art_executor", END)
    
    return workflow.compile()


# ============ MAIN ============

def main():
    """Main function"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║          🎨📝 ANIME STORY STUDIO V5                       ║
║          Pure Computer Vision - No Windows Hotkeys        ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("\n📋 MENU:")
    print("  1. Viết truyện + Vẽ nhân vật (Full Auto)")
    print("  2. Chỉ viết truyện (Story Only)")
    print("  3. Chỉ vẽ tranh (Art Only) - PURE VISION")
    print("  0. Thoát")
    
    choice = input("\n👉 Chọn chức năng (1-3): ").strip()
    
    if choice == '0':
        print("👋 Goodbye!")
        return
    
    # Xác định mode
    mode_map = {
        '1': 'full',
        '2': 'story_only',
        '3': 'art_only'
    }
    mode = mode_map.get(choice, 'full')
    
    # Input thông tin
    print("\n" + "="*60)
    character_name = input("🎭 Nhập tên nhân vật (VD: Doraemon, Pikachu, Luffy): ").strip()
    
    if not character_name:
        character_name = "Doraemon"
        print(f"   ⚠️  Sử dụng mặc định: {character_name}")
    
    story_theme = "adventure"
    story_length = 200
    
    if mode in ['full', 'story_only']:
        story_theme = input("📖 Chủ đề truyện (adventure/friendship/funny): ").strip() or "adventure"
        story_length_input = input("📏 Độ dài truyện (số từ, VD: 500): ").strip()
        try:
            story_length = int(story_length_input)
        except:
            story_length = 200
    
    # Khởi tạo state
    initial_state = StoryArtState(
        character_name=character_name,
        story_theme=story_theme,
        story_length=story_length,
        mode=mode,
        character_info={},
        story_outline={},
        story_content="",
        story_file_path="",
        reference_image_b64="",
        paint_screenshot_b64="",
        tool_positions={},
        color_positions={},
        canvas_area={},
        drawing_steps=[],
        current_step=0,
        total_steps=0,
        paint_window_ready=False,
        paint_maximized=False,
        execution_log=[],
        is_complete=False,
        error=""
    )
    
    # Build và chạy workflow
    print("\n🚀 Starting PURE VISION automation...")
    print("⚠️  Script sẽ điều khiển chuột (KHÔNG DÙNG PHÍM TẮT WINDOWS)!")
    print("⚠️  KHÔNG DI CHUYỂN CHUỘT trong quá trình vẽ!")
    print("\n⏳ Bắt đầu sau 3 giây...")
    time.sleep(3)
    
    graph = build_workflow(mode)
    
    try:
        final_state = graph.invoke(initial_state)
        
        # Summary
        print("\n" + "="*70)
        print("📊 EXECUTION SUMMARY")
        print("="*70)
        
        if final_state.get('error'):
            print(f"❌ Error: {final_state['error']}")
        else:
            print("✅ Status: Completed!")
            
            if final_state.get('story_file_path'):
                print(f"📝 Story: {final_state['story_file_path']}")
            
            if final_state.get('is_complete'):
                print(f"🎨 Drawing: {final_state['total_steps']} steps completed")
        
        print("\n📜 Execution Log:")
        for log in final_state.get('execution_log', []):
            print(f"  {log}")
        
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🤖 Automation finished!")
    print("="*70)


if __name__ == "__main__":
    main()
