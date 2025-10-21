"""
======================================================================================
ANIME STORY STUDIO V4 - HYBRID CV + LLM
Computer Vision để detect interface + LLM để tạo lệnh vẽ thông minh
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
from PIL import Image
import pyautogui
import cv2
import numpy as np
from typing import TypedDict, List, Dict, Tuple
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import traceback
import pyperclip
import pygetwindow as gw
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

def detect_canvas_area(img_cv: np.ndarray) -> Dict:
    """
    Tìm vùng canvas (vùng trắng lớn nhất) trong Paint
    """
    print("   🔍 Detecting canvas area...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Threshold để tìm vùng trắng (canvas)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise Exception("Cannot find canvas area!")
    
    # Tìm contour lớn nhất (canvas)
    canvas_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(canvas_contour)
    
    # Thêm margin nhỏ để tránh vẽ sát mép
    margin = 20
    canvas = {
        'x': x + margin,
        'y': y + margin,
        'width': w - 2*margin,
        'height': h - 2*margin
    }
    
    print(f"   ✅ Canvas: ({canvas['x']}, {canvas['y']}) - {canvas['width']}x{canvas['height']}")
    
    return canvas


def detect_color_palette(img_cv: np.ndarray) -> Dict:
    """
    Tìm vị trí các màu trong bảng màu Paint bằng color matching
    """
    print("   🎨 Detecting color palette...")
    
    screen_height, screen_width = img_cv.shape[:2]
    
    # Vùng chứa colors thường ở phía trên, giữa màn hình
    # Điều chỉnh theo layout Paint
    color_region_y1 = 50
    color_region_y2 = 110
    color_region_x1 = int(screen_width * 0.3)
    color_region_x2 = int(screen_width * 0.7)
    
    color_region = img_cv[color_region_y1:color_region_y2, color_region_x1:color_region_x2]
    
    # Định nghĩa màu cần tìm (BGR format + tolerance)
    color_targets = {
        'black': ([0, 0, 0], 30),
        'white': ([255, 255, 255], 30),
        'gray': ([128, 128, 128], 40),
        'red': ([0, 0, 200], 60),
        'orange': ([0, 165, 255], 60),
        'yellow': ([0, 255, 255], 60),
        'green': ([0, 200, 0], 60),
        'blue': ([200, 0, 0], 60),
        'purple': ([200, 0, 200], 60),
        'brown': ([42, 82, 165], 60)
    }
    
    color_positions = {}
    
    for color_name, (target_bgr, tolerance) in color_targets.items():
        # Tạo mask cho màu
        lower = np.array([max(0, c - tolerance) for c in target_bgr])
        upper = np.array([min(255, c + tolerance) for c in target_bgr])
        
        mask = cv2.inRange(color_region, lower, upper)
        
        # Tìm vị trí
        coords = cv2.findNonZero(mask)
        if coords is not None and len(coords) > 10:  # Đủ lớn
            # Lấy tâm của vùng màu
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + color_region_x1
                cy = int(M["m01"] / M["m00"]) + color_region_y1
                color_positions[color_name] = {'x': cx, 'y': cy}
                print(f"      • {color_name}: ({cx}, {cy})")
    
    if len(color_positions) < 3:
        print("      ⚠️  Warning: Only found", len(color_positions), "colors")
    
    return color_positions


def detect_tool_positions(img_cv: np.ndarray) -> Dict:
    """
    Tìm vị trí các tools trong Paint
    Dùng heuristic + pattern matching
    """
    print("   🔧 Detecting tool positions...")
    
    screen_height, screen_width = img_cv.shape[:2]
    
    # Paint tools thường ở bên trái hoặc phía trên
    # Ta dùng heuristic dựa trên layout chuẩn của Paint
    
    # Vùng tools (phía trên, bên trái)
    tool_region_y = 60
    tool_region_x_start = 150
    tool_spacing = 30
    
    tool_positions = {
        'pencil': {'x': tool_region_x_start, 'y': tool_region_y},
        'fill': {'x': tool_region_x_start + tool_spacing, 'y': tool_region_y + 20},
        'eraser': {'x': tool_region_x_start + tool_spacing*2, 'y': tool_region_y},
        'color_picker': {'x': tool_region_x_start + tool_spacing*3, 'y': tool_region_y},
        'text': {'x': tool_region_x_start + tool_spacing*4, 'y': tool_region_y},
        'line': {'x': tool_region_x_start + tool_spacing*5, 'y': tool_region_y},
        'curve': {'x': tool_region_x_start + tool_spacing*6, 'y': tool_region_y},
        'rectangle': {'x': tool_region_x_start + tool_spacing*7, 'y': tool_region_y},
        'oval': {'x': tool_region_x_start + tool_spacing*8, 'y': tool_region_y},
        'brush': {'x': tool_region_x_start + tool_spacing*1, 'y': tool_region_y}
    }
    
    print(f"      • Estimated {len(tool_positions)} tool positions")
    
    return tool_positions


def analyze_paint_interface(screenshot_b64: str) -> Dict:
    """
    MAIN FUNCTION: Phân tích toàn bộ giao diện Paint bằng Computer Vision
    """
    print("\n" + "="*70)
    print("🔬 COMPUTER VISION ANALYSIS")
    print("="*70)
    
    # Decode screenshot
    img_data = base64.b64decode(screenshot_b64)
    img_pil = Image.open(BytesIO(img_data))
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    screen_height, screen_width = img_cv.shape[:2]
    print(f"   📐 Screen resolution: {screen_width}x{screen_height}")
    
    # 1. Detect Canvas
    canvas_area = detect_canvas_area(img_cv)
    
    # 2. Detect Colors
    color_positions = detect_color_palette(img_cv)
    
    # 3. Detect Tools
    tool_positions = detect_tool_positions(img_cv)
    
    print("\n✅ Computer Vision Analysis Complete!")
    
    return {
        'canvas_area': canvas_area,
        'color_positions': color_positions,
        'tool_positions': tool_positions,
        'screen_width': screen_width,
        'screen_height': screen_height
    }


# ============ AGENT 1: STORY WRITER ============

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


def maximize_paint_safely():
    """
    Maximize Paint một cách AN TOÀN - chỉ maximize khi chưa full
    """
    try:
        # Tìm window Paint
        paint_windows = gw.getWindowsWithTitle('Paint')
        
        if not paint_windows:
            print("   ⚠️  Không tìm thấy Paint window!")
            return False
        
        paint_window = paint_windows[0]
        
        # Kiểm tra trạng thái
        if paint_window.isMaximized:
            print("   ✅ Paint đã maximized rồi, không cần làm gì")
        else:
            print("   📐 Maximizing Paint...")
            paint_window.maximize()
            time.sleep(0.5)
        
        # Đảm bảo Paint được focus
        paint_window.activate()
        time.sleep(0.3)
        
        return True
        
    except Exception as e:
        print(f"   ⚠️  Không thể maximize Paint: {e}")
        print("   🔄 Thử phương pháp dự phòng...")
        
        # Fallback: Click vào thanh title bar rồi maximize bằng double-click
        pyautogui.click(500, 10)  # Click vào title bar
        time.sleep(0.2)
        pyautogui.doubleClick(500, 10)  # Double-click để maximize
        time.sleep(0.5)
        
        return True
    

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
    """Agent mở Notepad và ghi truyện"""
    
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
        
        # 2. Maximize
        pyautogui.hotkey('win', 'up')
        time.sleep(0.5)
        
        # 3. Paste nội dung
        print("   ⌨️  Pasting story content...")
        pyperclip.copy(state['story_content'])
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(1)
        
        # 4. Save file
        print("   💾 Saving file...")
        pyautogui.hotkey('ctrl', 's')
        time.sleep(1)
        
        # Tạo tên file
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
        
        # Đóng Notepad
        print("   🚪 Closing Notepad...")
        time.sleep(1)
        pyautogui.hotkey('alt', 'f4')
        time.sleep(1)
        
    except Exception as e:
        print(f"❌ Formatter error: {e}")
        state['error'] = f"Formatter failed: {e}"
    
    return state


# ============ AGENT 2: ART CREATOR ============
def art_preparation_agent(state: StoryArtState) -> StoryArtState:
    """Agent chuẩn bị ảnh tham khảo và mở Paint - FIXED VERSION"""
    
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
        
        # 4. Setup Paint - PHƯƠNG PHÁP AN TOÀN (không dùng activate())
        print("\n   🔍 Setting up Paint...")
        paint_windows = gw.getWindowsWithTitle('Paint')
        
        if paint_windows:
            print("   ✅ Paint đã mở")
            paint_window = paint_windows[0]
            
            # Chỉ dùng maximize và click, KHÔNG dùng activate()
            try:
                if not paint_window.isMaximized:
                    print("   📐 Maximizing Paint...")
                    paint_window.maximize()
                    time.sleep(1.0)
                else:
                    print("   ✅ Paint đã maximized")
            except Exception as e:
                print(f"   ⚠️  Maximize error: {e}, using keyboard shortcut...")
                pyautogui.hotkey('win', 'up')
                time.sleep(1.0)
            
            # Focus bằng cách click vào window (AN TOÀN hơn activate())
            print("   🖱️  Focusing Paint window...")
            try:
                # Click vào giữa window
                x, y, w, h = paint_window.left, paint_window.top, paint_window.width, paint_window.height
                pyautogui.click(x + w // 2, y + h // 2)
                time.sleep(0.5)
            except:
                # Fallback: click vào giữa màn hình
                screen_w, screen_h = pyautogui.size()
                pyautogui.click(screen_w // 2, screen_h // 2)
                time.sleep(0.5)
            
        else:
            # 5. Mở Paint mới
            print("   🎨 Opening new Paint...")
            subprocess.Popen(['mspaint'])
            time.sleep(3.5)
            
            # Maximize bằng keyboard
            print("   📐 Maximizing Paint...")
            pyautogui.hotkey('win', 'up')
            time.sleep(1.0)
            
            # Click để focus
            screen_w, screen_h = pyautogui.size()
            pyautogui.click(screen_w // 2, screen_h // 2)
            time.sleep(0.5)
        
        # 6. Click vào canvas area để đảm bảo ready
        print("   🖱️  Preparing canvas...")
        pyautogui.click(700, 400)
        time.sleep(0.5)
        
        # 7. Chụp screenshot Paint
        print("   📸 Capturing Paint interface...")
        time.sleep(1.0)
        state['paint_screenshot_b64'] = screenshot_to_base64()
        
        print("   ✅ Paint setup complete!")
        state['execution_log'].append("✓ Preparation: Paint ready")
        
    except Exception as e:
        print(f"❌ Preparation error: {e}")
        state['error'] = f"Preparation failed: {e}"
        traceback.print_exc()
    
    return state

def art_cv_analyzer_agent(state: StoryArtState) -> StoryArtState:
    """
    Agent phân tích giao diện Paint bằng Computer Vision
    """
    
    if state['mode'] == 'story_only' or state['error']:
        return state
    
    try:
        # Phân tích bằng CV
        cv_result = analyze_paint_interface(state['paint_screenshot_b64'])
        
        # Lưu vào state
        state['canvas_area'] = cv_result['canvas_area']
        state['color_positions'] = cv_result['color_positions']
        state['tool_positions'] = cv_result['tool_positions']
        
        state['execution_log'].append("✓ CV Analysis: Interface detected")
        
    except Exception as e:
        print(f"❌ CV Analyzer error: {e}")
        state['error'] = f"CV Analysis failed: {e}"
        traceback.print_exc()
    
    return state


def art_llm_planner_agent(state: StoryArtState) -> StoryArtState:
    """
    Agent dùng LLM để tạo kế hoạch vẽ DựA TRÊN TỌA ĐỘ THẬT từ CV
    """
    
    if state['mode'] == 'story_only' or state['error']:
        return state
    
    print("\n" + "="*70)
    print("🧠 LLM PLANNER AGENT (AI Strategy)")
    print("="*70)
    
    try:
        canvas = state['canvas_area']
        colors = list(state['color_positions'].keys())
        tools = list(state['tool_positions'].keys())
        
        # Load reference image
        ref_img = Image.open(BytesIO(base64.b64decode(state['reference_image_b64'])))
        ref_img.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        prompt = f"""
Bạn là AI Drawing Planner. Tạo kế hoạch vẽ "{state['character_name']}" trong Paint.

THÔNG TIN CANVAS (đã được Computer Vision phát hiện):
- Vị trí: ({canvas['x']}, {canvas['y']})
- Kích thước: {canvas['width']}x{canvas['height']}
- Vùng vẽ hợp lệ:
  * X: từ {canvas['x']} đến {canvas['x'] + canvas['width']}
  * Y: từ {canvas['y']} đến {canvas['y'] + canvas['height']}

TOOLS CÓ SẴN: {', '.join(tools)}
COLORS CÓ SẴN: {', '.join(colors)}

YÊU CẦU:
1. Phân tích ảnh tham khảo
2. Tạo 5-7 bước vẽ ĐƠN GIẢN
3. Dùng oval/rectangle cho hình cơ bản
4. Dùng fill để tô màu
5. TỌA ĐỘ PHẢI NẰM TRONG CANVAS!

JSON FORMAT:
{{
  "analysis": "Mô tả ngắn về nhân vật và cách vẽ",
  "steps": [
    {{
      "step": 1,
      "description": "Vẽ đầu (oval lớn)",
      "tool": "oval",
      "color": "black",
      "action": "drag",
      "start_x": {canvas['x'] + 200},
      "start_y": {canvas['y'] + 100},
      "end_x": {canvas['x'] + 400},
      "end_y": {canvas['y'] + 250}
    }},
    {{
      "step": 2,
      "description": "Tô màu đầu",
      "tool": "fill",
      "color": "yellow",
      "action": "click",
      "click_x": {canvas['x'] + 300},
      "click_y": {canvas['y'] + 175}
    }},
    {{
      "step": 3,
      "description": "Vẽ thân (rectangle)",
      "tool": "rectangle",
      "color": "red",
      "action": "drag",
      "start_x": {canvas['x'] + 250},
      "start_y": {canvas['y'] + 250},
      "end_x": {canvas['x'] + 350},
      "end_y": {canvas['y'] + 400}
    }}
  ]
}}

⚠️ QUAN TRỌNG:
- action="drag" cần: start_x, start_y, end_x, end_y
- action="click" cần: click_x, click_y
- TẤT CẢ tọa độ PHẢI trong khoảng canvas đã cho
- CHỈ dùng tools và colors có sẵn
"""
        
        print("   🤖 LLM analyzing reference image and creating plan...")
        response = call_gemini_vision(prompt, [ref_img])
        
        # Extract JSON
        json_str = response
        if '```json' in response:
            json_str = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            json_str = response.split('```')[1].split('```')[0]
        
        plan = json.loads(json_str.strip())
        
        print(f"\n   📋 AI Analysis: {plan.get('analysis', 'N/A')}")
        
        # VALIDATE và CLAMP tọa độ
        validated_steps = []
        for step in plan.get('steps', []):
            # Kiểm tra tool và color có tồn tại không
            if step.get('tool') not in tools:
                print(f"      ⚠️  Step {step['step']}: Tool '{step.get('tool')}' not available, using 'pencil'")
                step['tool'] = 'pencil' if 'pencil' in tools else tools[0]
            
            if step.get('color') not in colors:
                print(f"      ⚠️  Step {step['step']}: Color '{step.get('color')}' not available, using 'black'")
                step['color'] = 'black' if 'black' in colors else colors[0]
            
            # Clamp tọa độ vào canvas
            if step.get('action') == 'drag':
                step['start_x'] = max(canvas['x'], min(step['start_x'], canvas['x'] + canvas['width']))
                step['start_y'] = max(canvas['y'], min(step['start_y'], canvas['y'] + canvas['height']))
                step['end_x'] = max(canvas['x'], min(step['end_x'], canvas['x'] + canvas['width']))
                step['end_y'] = max(canvas['y'], min(step['end_y'], canvas['y'] + canvas['height']))
            
            elif step.get('action') == 'click':
                step['click_x'] = max(canvas['x'], min(step['click_x'], canvas['x'] + canvas['width']))
                step['click_y'] = max(canvas['y'], min(step['click_y'], canvas['y'] + canvas['height']))
            
            validated_steps.append(step)
        
        state['drawing_steps'] = validated_steps
        state['total_steps'] = len(validated_steps)
        state['current_step'] = 0
        
        print(f"\n✅ Drawing plan created: {state['total_steps']} steps")
        for step in validated_steps:
            print(f"   Step {step['step']}: {step['description']}")
            if step.get('action') == 'drag':
                print(f"      → Drag from ({step['start_x']},{step['start_y']}) to ({step['end_x']},{step['end_y']})")
            elif step.get('action') == 'click':
                print(f"      → Click at ({step['click_x']},{step['click_y']})")
        
        state['execution_log'].append(f"✓ LLM Plan: {state['total_steps']} steps")
        
    except Exception as e:
        print(f"❌ LLM Planner error: {e}")
        state['error'] = f"LLM Planner failed: {e}"
        traceback.print_exc()
    
    return state


def art_executor_agent(state: StoryArtState) -> StoryArtState:
    """Agent thực thi vẽ TOÀN BỘ các bước"""
    
    if state['mode'] == 'story_only' or state['error']:
        return state
    
    print("\n" + "="*70)
    print("⚙️ ART EXECUTOR AGENT - FULL AUTO DRAWING")
    print("="*70)
    
    tool_positions = state['tool_positions']
    color_positions = state['color_positions']
    steps = state['drawing_steps']
    
    try:
        for i, step_data in enumerate(steps):
            step_num = i + 1
            print(f"\n   [{step_num}/{len(steps)}] {step_data['description']}")
            
            tool = step_data.get('tool', 'pencil')
            color = step_data.get('color', 'black')
            action = step_data.get('action', 'drag')
            
            # 1. Chọn tool
            if tool in tool_positions:
                tool_pos = tool_positions[tool]
                print(f"      🔧 Tool: {tool} at ({tool_pos['x']}, {tool_pos['y']})")
                pyautogui.click(tool_pos['x'], tool_pos['y'])
                time.sleep(0.3)
            else:
                print(f"      ⚠️  Tool '{tool}' not found!")
            
            # 2. Chọn màu
            if color in color_positions:
                color_pos = color_positions[color]
                print(f"      🎨 Color: {color} at ({color_pos['x']}, {color_pos['y']})")
                pyautogui.click(color_pos['x'], color_pos['y'])
                time.sleep(0.3)
            else:
                print(f"      ⚠️  Color '{color}' not found!")
            
            # 3. Thực hiện vẽ
            if action == 'drag':
                sx = step_data.get('start_x')
                sy = step_data.get('start_y')
                ex = step_data.get('end_x')
                ey = step_data.get('end_y')
                
                print(f"      ✏️  Dragging from ({sx},{sy}) to ({ex},{ey})...")
                
                pyautogui.moveTo(sx, sy, duration=0.2)
                time.sleep(0.1)
                pyautogui.drag(ex - sx, ey - sy, duration=0.5, button='left')
                
            elif action == 'click':
                cx = step_data.get('click_x')
                cy = step_data.get('click_y')
                
                print(f"      🖱️  Clicking at ({cx},{cy})...")
                pyautogui.click(cx, cy)
            
            time.sleep(0.5)
            print(f"      ✅ Step {step_num} done!")
        
        # Hoàn thành
        state['current_step'] = len(steps)
        state['is_complete'] = True
        
        print(f"\n   🎉 ALL {len(steps)} STEPS COMPLETED!")
        state['execution_log'].append(f"✓ Drawing: {len(steps)} steps completed")
        
        # Lưu ảnh
        print("\n   💾 Saving final drawing...")
        time.sleep(1)
        
        char_name = state['character_name'].replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drawing_{char_name}_{timestamp}.png"
        
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        
        print(f"   ✅ Saved: {filename}")
        state['execution_log'].append(f"✓ Image: {filename}")
        
    except Exception as e:
        print(f"❌ Executor error: {e}")
        state['error'] = f"Executor failed: {e}"
        traceback.print_exc()
    
    return state


# ============ BUILD WORKFLOW ============

def build_workflow(mode: str):
    """Build workflow dựa trên mode"""
    workflow = StateGraph(StoryArtState)
    
    if mode == "story_only":
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
        workflow.add_node("art_preparation", art_preparation_agent)
        workflow.add_node("art_cv_analyzer", art_cv_analyzer_agent)
        workflow.add_node("art_llm_planner", art_llm_planner_agent)
        workflow.add_node("art_executor", art_executor_agent)
        
        workflow.set_entry_point("art_preparation")
        workflow.add_edge("art_preparation", "art_cv_analyzer")
        workflow.add_edge("art_cv_analyzer", "art_llm_planner")
        workflow.add_edge("art_llm_planner", "art_executor")
        workflow.add_edge("art_executor", END)
        
    else:  # full
        workflow.add_node("story_research", story_research_agent)
        workflow.add_node("story_outline", story_outline_agent)
        workflow.add_node("story_writer", story_writer_agent)
        workflow.add_node("story_formatter", story_formatter_agent)
        workflow.add_node("art_preparation", art_preparation_agent)
        workflow.add_node("art_cv_analyzer", art_cv_analyzer_agent)
        workflow.add_node("art_llm_planner", art_llm_planner_agent)
        workflow.add_node("art_executor", art_executor_agent)
        
        workflow.set_entry_point("story_research")
        workflow.add_edge("story_research", "story_outline")
        workflow.add_edge("story_outline", "story_writer")
        workflow.add_edge("story_writer", "story_formatter")
        workflow.add_edge("story_formatter", "art_preparation")
        workflow.add_edge("art_preparation", "art_cv_analyzer")
        workflow.add_edge("art_cv_analyzer", "art_llm_planner")
        workflow.add_edge("art_llm_planner", "art_executor")
        workflow.add_edge("art_executor", END)
    
    return workflow.compile()


# ============ MAIN ============

def main():
    """Main function"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║          🎨📝 ANIME STORY STUDIO V4                       ║
║          Hybrid CV + LLM - Smart Drawing                  ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("\n📋 MENU:")
    print("  1. Viết truyện + Vẽ nhân vật (Full Auto)")
    print("  2. Chỉ viết truyện (Story Only)")
    print("  3. Chỉ vẽ tranh (Art Only)")
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
    character_name = input("🎭 Nhập tên nhân vật (VD: Doraemon, Pikachu, Naruto): ").strip()
    
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
        execution_log=[],
        is_complete=False,
        error=""
    )
    
    # Build và chạy workflow
    print("\n🚀 Starting automation...")
    print("⚠️  Script sẽ điều khiển chuột và bàn phím!")
    print("⚠️  KHÔNG DI CHUYỂN CHUỘT trong quá trình vẽ!")
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


if __name__ == "__main__":
    main()