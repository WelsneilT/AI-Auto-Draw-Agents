"""
======================================================================================
ANIME STORY STUDIO V7.3 - RELIABLE WINDOW MAXIMIZE
Zero Hotkeys + Guaranteed Maximize + CV-Powered
======================================================================================
"""

import os
import time
import json
import subprocess
import ctypes
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image, ImageGrab
import pyautogui
import cv2
import numpy as np
from typing import Dict, Tuple
import traceback
import pygetwindow as gw # <<< THÊM THƯ VIỆN ĐỂ ĐIỀU KHIỂN CỬA SỔ
from paint_cv_detector import detect_paint_interface_cv
from llm_helper import GeminiAssistant, interactive_character_selection
# Load environment
load_dotenv()

# Config pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.1

# Global scale factor
SCALE_FACTOR = 1.0

# ============ SCREEN SCALING ============

def get_screen_scale_factor() -> float:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        pass
    logical_width, logical_height = pyautogui.size()
    screenshot = ImageGrab.grab()
    physical_width, physical_height = screenshot.size
    scale_factor = physical_width / logical_width
    
    print(f"\n   📐 Screen Detection:")
    print(f"      Logical size:  {logical_width}x{logical_height}")
    print(f"      Physical size: {physical_width}x{physical_height}")
    print(f"      Scale factor:  {scale_factor:.2f}x ({scale_factor*100:.0f}%)")
    
    return scale_factor

def scale_coords(x: int, y: int) -> Tuple[int, int]:
    global SCALE_FACTOR
    return (int(x / SCALE_FACTOR), int(y / SCALE_FACTOR))

# ============ UTILITY FUNCTIONS ============

def screenshot_full() -> Image.Image:
    return ImageGrab.grab()

def slow_click(x: int, y: int, clicks: int = 1):
    scaled_x, scaled_y = scale_coords(x, y)
    pyautogui.moveTo(scaled_x, scaled_y, duration=0.3)
    pyautogui.click(clicks=clicks)
    time.sleep(0.3)

# ============ VISION FUNCTIONS ============

def detect_paint_interface(screenshot: Image.Image) -> Dict:
    """Phân tích giao diện Paint bằng Computer Vision."""
    print("   🔬 Analyzing Paint interface via CV...")
    return detect_paint_interface_cv(screenshot, templates_dir="templates")

# ============ IMAGE PROCESSING ============

def preprocess_image(image_path: str, max_size: int = 200) -> np.ndarray:
    print(f"\n   📐 Processing image: {image_path}")
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    height, width = original.shape[:2]
    scale = min(max_size / width, max_size / height)
    new_width, new_height = int(width * scale), int(height * scale)
    
    resized = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    print(f"   ✅ Processed: {new_width}x{new_height} pixels")
    
    preview_path = "preview_bw.png"
    cv2.imwrite(preview_path, bw)
    print(f"   💾 Preview saved: {preview_path}")
    
    return bw

# ============ MAIN WORKFLOW ============

def setup_paint_window():
    # <<< HÀM NÀY ĐƯỢC VIẾT LẠI HOÀN TOÀN ĐỂ ĐẢM BẢO ĐỘ TIN CẬY
    """
    Mở, tìm, và chắc chắn phóng to cửa sổ Paint.
    """
    print("\n" + "="*70)
    print("🎨 SETUP PAINT WINDOW (RELIABLE)")
    print("="*70)
    
    # 1. Đóng tất cả các tiến trình Paint cũ
    print("\n   🧹 Closing old Paint processes...")
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'mspaint.exe'], capture_output=True, timeout=3)
        time.sleep(1)
    except Exception as e:
        print(f"      (Info) No old Paint process found or could not kill: {e}")

    # 2. Mở một tiến trình Paint mới
    print("   🚀 Opening a new Paint instance...")
    subprocess.Popen(['mspaint'])
    time.sleep(4) # Chờ cho Paint có thời gian khởi động

    # 3. Tìm cửa sổ Paint vừa mở
    print("   🔍 Searching for the Paint window...")
    paint_window = None
    # Thử tìm trong vài giây, vì cửa sổ có thể xuất hiện chậm
    for _ in range(5):
        # Tìm kiếm các tiêu đề phổ biến của Paint
        windows = gw.getWindowsWithTitle('Untitled - Paint') + gw.getWindowsWithTitle('Paint')
        if windows:
            paint_window = windows[0]
            print(f"   ✅ Found window: '{paint_window.title}'")
            break
        time.sleep(1)

    if not paint_window:
        raise Exception("Fatal Error: Could not find the Paint window after opening it.")

    # 4. Kích hoạt và Phóng to cửa sổ một cách chắc chắn
    print("   📐 Activating and Maximizing window...")
    paint_window.activate()
    time.sleep(0.5)

    if not paint_window.isMaximized:
        paint_window.maximize()
        print("      Window was not maximized. Sent maximize command.")
        time.sleep(1) # Chờ cho hiệu ứng phóng to hoàn tất
    else:
        print("      Window is already maximized.")

    # 5. Chụp lại màn hình sau khi đã chắc chắn phóng to
    print("   📸 Taking screenshot of the maximized and ready window...")
    return screenshot_full()


def execute_drawing(reference_image_path: str):
    """Quy trình thực thi vẽ chính."""
    global SCALE_FACTOR
    
    print("\n" + "="*70)
    print("🎨 DRAWING EXECUTION (CV-POWERED)")
    print("="*70)
    
    SCALE_FACTOR = get_screen_scale_factor()
    bw_image = preprocess_image(reference_image_path, max_size=150)
    paint_screenshot = setup_paint_window()
    
    interface = detect_paint_interface(paint_screenshot)
    
    canvas = interface.get('canvas_area', {})
    pencil = interface.get('tools', {}).get('pencil', {})
    black = interface.get('colors', {}).get('black', {})
    
    print(f"   ✅ Canvas found: {canvas}")
    print(f"   ✅ Pencil found: {pencil}")
    print(f"   ✅ Black color found: {black}")
    print(f"   🖼️ Debug image saved at: {interface.get('debug', {}).get('path')}")
    
    if not all([canvas, pencil, black]):
        print("\n❌ LỖI: Không thể xác định đầy đủ giao diện Paint bằng CV.")
        print("   Vui lòng kiểm tra file debug/ui_detect_debug.png.")
        print("   Hãy chắc chắn ảnh mẫu trong 'templates/' cắt chuẩn và Paint đang ở đúng trạng thái.")
        raise Exception("Cannot detect Paint interface properly by CV!")
    
    draw_pixel_by_pixel_runs(
        bw_image,
        canvas['x'],
        canvas['y'],
        pencil,
        black
    )
    
    print("\n   💾 Saving final result...")
    time.sleep(1)
    
    final_screenshot = screenshot_full()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"drawing_final_{timestamp}.png"
    final_screenshot.save(output_path)
    
    print(f"   ✅ Saved: {output_path}")
    
    print("\n" + "="*70)
    print("🎉 MISSION COMPLETE! DRAWING FINISHED!")
    print("="*70)

def draw_pixel_by_pixel_runs(bw_image: np.ndarray, canvas_x: int, canvas_y: int, 
                             pencil_pos: Dict, black_pos: Dict):
    """Vẽ theo các đoạn pixel liên tiếp (nhanh hơn)."""
    print("\n" + "="*70)
    print("✏️  RUN-LENGTH DRAWING (OPTIMIZED)")
    print("="*70)
    
    print(f"\n   🖊️  Selecting pencil...")
    slow_click(pencil_pos['x'], pencil_pos['y'])
    
    print(f"   🎨 Selecting black color...")
    slow_click(black_pos['x'], black_pos['y'])
    
    print(f"\n   ✍️  Drawing {bw_image.shape[1]}x{bw_image.shape[0]} image...")
    
    h, w = bw_image.shape
    total_runs = 0
    original_pause = pyautogui.PAUSE
    pyautogui.PAUSE = 0.001
    
    for y in range(h):
        row = bw_image[y]
        x = 0
        while x < w:
            if row[x] == 0:
                start_x = x
                while x < w and row[x] == 0: x += 1
                end_x = x - 1
                
                canvas_start_x, canvas_y_pos, canvas_end_x = canvas_x + start_x, canvas_y + y, canvas_x + end_x
                
                scaled_start_x, scaled_y = scale_coords(canvas_start_x, canvas_y_pos)
                scaled_end_x, _ = scale_coords(canvas_end_x, canvas_y_pos)
                
                pyautogui.moveTo(scaled_start_x, scaled_y)
                pyautogui.mouseDown()
                if scaled_start_x != scaled_end_x:
                    pyautogui.moveTo(scaled_end_x, scaled_y)
                pyautogui.mouseUp()
                
                total_runs += 1
            else:
                x += 1
    
    pyautogui.PAUSE = original_pause
    print(f"\n   ✅ Drawing completed with {total_runs} runs.")

# ============ MAIN ============

def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║          🎨 ANIME STORY STUDIO V8.0                       ║
║          Gemini-Powered Character Recognition             ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Khởi tạo Gemini
    gemini = GeminiAssistant(model_name="gemini-2.5-flash")  # Hoặc gemini-1.5-pro
    
    # Cho phép user mô tả nhân vật bằng ngôn ngữ tự nhiên
    image_path = interactive_character_selection(gemini)
    
    if not image_path:
        print("\n❌ Không thể tiếp tục vì không tìm thấy ảnh.")
        return
    
    # (TÙY CHỌN) Phân tích ảnh để tối ưu tham số
    optimize = input("\n🤖 Cho phép Gemini tối ưu tham số vẽ? (y/n): ").strip().lower()
    
    if optimize == 'y':
        print("\n   🔬 Gemini đang phân tích ảnh...")
        params = gemini.analyze_image_for_drawing(image_path)
        
        print(f"\n   📊 Đề xuất từ Gemini:")
        print(f"      Kích thước: {params['max_size']}px")
        print(f"      Ngưỡng: {params['threshold']}")
        print(f"      Độ phức tạp: {params['complexity']}")
        print(f"      Thời gian ước tính: {params['estimated_time_minutes']} phút")
        
        for rec in params['recommendations']:
            print(f"      💡 {rec}")
        
        apply = input("\n   ✅ Áp dụng? (y/n): ").strip().lower()
        if apply == 'y':
            # Sẽ dùng params này trong preprocess_image
            # (Cần sửa hàm preprocess_image để nhận tham số)
            pass
    
    print("\n⚠️  CHUỘT SẼ DI CHUYỂN TỰ ĐỘNG!")
    print("⏳ Bắt đầu sau 5 giây...")
    
    for i in range(5, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    try:
        execute_drawing(image_path)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()