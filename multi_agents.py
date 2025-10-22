"""
HYBRID DRAWING SYSTEM V3.3
FIX:
1. ✅ Giữ nét bút liên tục bằng dragTo() thay vì moveTo()
2. ✅ Canvas detection bằng template matching
3. ✅ Tối ưu tốc độ vẽ
4. ✅ Loại bỏ mouseDown/mouseUp thủ công
"""

import os
import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageGrab
import google.generativeai as genai
from dotenv import load_dotenv
import time

load_dotenv()

# ============ PAINT SETUP (TEMPLATE MATCHING) ============

class PaintSetup:
    """Setup Paint với template matching"""
    
    @staticmethod
    def setup_paint():
        """Mở Paint và detect canvas bằng template matching"""
        print("\n🎨 [PAINT SETUP] Opening Paint...")
        os.system("start mspaint")
        time.sleep(3)
        
        # Maximize window
        try:
            import pygetwindow as gw
            paint_windows = gw.getWindowsWithTitle("Paint")
            if not paint_windows:
                paint_windows = gw.getWindowsWithTitle("Untitled")
            
            if paint_windows:
                paint_windows[0].maximize()
                time.sleep(1)
                print("   ✅ Paint maximized")
        except Exception as e:
            print(f"   ⚠️ Cannot maximize: {e}")
        
        # Detect canvas bằng template matching
        print("   Detecting canvas (template matching)...")
        canvas_region = PaintSetup._detect_canvas_template()
        
        print(f"   ✅ Canvas detected: {canvas_region}")
        return canvas_region
    
    @staticmethod
    def _detect_canvas_template():
        """Detect canvas bằng template matching"""
        try:
            # Chụp màn hình
            screenshot = ImageGrab.grab()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Load template (canvas trắng)
            template_path = "templates/canvas.png"
            if not os.path.exists(template_path):
                print(f"   ⚠️ Template not found: {template_path}")
                raise Exception("Template not found")
            
            template = cv2.imread(template_path)
            
            # Template matching
            result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            print(f"   Template match confidence: {max_val:.3f}")
            
            if max_val < 0.7:  # Threshold
                raise Exception(f"Template match too low: {max_val:.3f}")
            
            # Lấy vị trí canvas
            top_left = max_loc
            h, w = template.shape[:2]
            
            # Thêm margin (canvas thực tế lớn hơn template)
            margin = 50
            canvas_region = {
                "x": top_left[0] + margin,
                "y": top_left[1] + margin,
                "width": w - 2 * margin,
                "height": h - 2 * margin
            }
            
            return canvas_region
        
        except Exception as e:
            print(f"   ⚠️ Template matching failed: {e}")
            print("   Using fallback method...")
            return PaintSetup._detect_canvas_fallback()
    
    @staticmethod
    def _detect_canvas_fallback():
        """Fallback: Detect canvas bằng cách tìm vùng trắng lớn nhất"""
        try:
            screenshot = ImageGrab.grab()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)
            
            # Threshold để tìm vùng trắng
            _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Lấy contour lớn nhất
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Lọc contour quá nhỏ
                if w < 500 or h < 300:
                    raise Exception("Canvas too small")
                
                return {
                    "x": x + 10,
                    "y": y + 10,
                    "width": w - 20,
                    "height": h - 20
                }
        
        except Exception as e:
            print(f"   ⚠️ Fallback failed: {e}")
        
        # Last resort: Hardcoded
        screen_width, screen_height = pyautogui.size()
        return {
            "x": int(screen_width * 0.05),
            "y": int(screen_height * 0.15),
            "width": int(screen_width * 0.85),
            "height": int(screen_height * 0.75)
        }

# ============ PAINT DRAWER (✅ FIXED: CONTINUOUS DRAWING) ============

class PaintDrawer:
    """Vẽ contours lên Paint (✅ FIX: Dùng dragTo() thay vì moveTo())"""
    
    def __init__(self, canvas_region: dict):
        self.canvas_region = canvas_region
        print("✏️ [PAINT DRAWER] Initialized")
        print(f"   Canvas: {canvas_region}")
    
    def draw_contours(self, contours: list, image_shape: tuple):
        """Vẽ contours lên Paint (✅ CONTINUOUS DRAWING WITH DRAGTO)"""
        print("\n✏️ [PAINT DRAWER] Drawing contours...")
        
        img_height, img_width = image_shape[:2]
        canvas_width = self.canvas_region['width']
        canvas_height = self.canvas_region['height']
        
        # Tính scale factor
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y) * 0.9
        
        # Tính offset để center
        scaled_width = img_width * scale
        scaled_height = img_height * scale
        offset_x = self.canvas_region['x'] + (canvas_width - scaled_width) / 2
        offset_y = self.canvas_region['y'] + (canvas_height - scaled_height) / 2
        
        print(f"   Scale: {scale:.3f}")
        print(f"   Offset: ({offset_x:.1f}, {offset_y:.1f})")
        print(f"   Scaled size: {scaled_width:.1f}x{scaled_height:.1f}")
        
        # ✅ SETUP PENCIL TOOL
        self._setup_pencil_tool()
        
        total_contours = len(contours)
        
        for i, contour in enumerate(contours):
            print(f"   Drawing contour {i+1}/{total_contours} ({len(contour)} points)")
            
            # ============ ✅ FIX: SỬ DỤNG DRAGTO() ============
            for j, point in enumerate(contour):
                x = int(offset_x + point[0][0] * scale)
                y = int(offset_y + point[0][1] * scale)
                
                if j == 0:
                    # Di chuyển đến điểm đầu (KHÔNG VẼ)
                    pyautogui.moveTo(x, y, duration=0.1)
                    time.sleep(0.05)  # Đợi cursor ổn định
                else:
                    # ✅ VẼ ĐẾN ĐIỂM TIẾP THEO BẰNG DRAGTO (TỰ ĐỘNG GIỮ CHUỘT)
                    pyautogui.dragTo(
                        x, y, 
                        duration=0.02,  # Tốc độ vẽ (càng nhỏ càng nhanh)
                        button='left'
                    )
            
            # ✅ Đóng contour (nối điểm cuối với điểm đầu)
            if len(contour) > 0:
                x = int(offset_x + contour[0][0][0] * scale)
                y = int(offset_y + contour[0][0][1] * scale)
                pyautogui.dragTo(x, y, duration=0.02, button='left')
            
            # Nghỉ giữa các contour
            time.sleep(0.05)
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{total_contours} ({(i+1)/total_contours*100:.1f}%)")
        
        print("   ✅ Drawing completed!")
    
    def _setup_pencil_tool(self):
        """Setup pencil tool bằng template matching"""
        print("   Setting up pencil tool...")
        
        try:
            # Template matching để tìm pencil icon
            template_path = "templates/pencil.png"
            if not os.path.exists(template_path):
                print(f"   ⚠️ Pencil template not found, using keyboard shortcut")
                # Fallback: Dùng keyboard shortcut
                pyautogui.hotkey('ctrl', 'p')  # Pencil tool shortcut (nếu có)
                time.sleep(0.5)
                return
            
            # Chụp màn hình
            screenshot = ImageGrab.grab()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Load template
            template = cv2.imread(template_path)
            
            # Template matching
            result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.7:  # Tìm thấy pencil icon
                # Click vào pencil
                pencil_x = max_loc[0] + template.shape[1] // 2
                pencil_y = max_loc[1] + template.shape[0] // 2
                
                pyautogui.click(pencil_x, pencil_y)
                time.sleep(0.3)
                print(f"   ✅ Pencil tool selected (confidence: {max_val:.3f})")
            else:
                print(f"   ⚠️ Pencil icon not found (confidence: {max_val:.3f})")
                print("   Assuming pencil is already selected")
        
        except Exception as e:
            print(f"   ⚠️ Pencil setup failed: {e}")
            print("   Assuming pencil is already selected")

# ============ OPENCV PROCESSOR (UNCHANGED) ============

class OpenCVProcessor:
    """Xử lý ảnh và tạo contours"""
    
    def __init__(self, debug=True):
        self.debug = debug
        print("👁️ [OPENCV PROCESSOR] Initialized")
    
    def process_image(self, image_path: str, metadata: dict) -> tuple:
        """Xử lý ảnh và trả về contours"""
        print("\n👁️ [OPENCV PROCESSOR] Processing image...")
        
        # 1. Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Cannot load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Preprocessing
        blur_kernel = metadata.get("preprocessing", {}).get("blur_kernel", 5)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # 3. Edge detection
        canny_low = metadata.get("preprocessing", {}).get("canny_low", 50)
        canny_high = metadata.get("preprocessing", {}).get("canny_high", 150)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # 4. Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Filter small contours
        min_area = 100  # Tăng threshold để lọc noise
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # 6. Sort contours
        contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
        
        print(f"   ✅ Found {len(contours)} contours (after filtering)")
        
        # 7. Simplify contours
        simplified_contours = []
        for contour in contours:
            epsilon = 0.003 * cv2.arcLength(contour, True)  # Giảm epsilon để giữ nhiều chi tiết
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 3:
                simplified_contours.append(approx)
        
        print(f"   ✅ Simplified to {len(simplified_contours)} contours")
        
        # 8. Debug: Visualize contours
        if self.debug:
            self._visualize_contours(img, simplified_contours)
        
        return simplified_contours, img.shape
    
    def _visualize_contours(self, img: np.ndarray, contours: list):
        """Visualize contours để debug"""
        debug_img = img.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        
        # Save debug image
        cv2.imwrite("debug_contours.jpg", debug_img)
        print("   🐛 Debug image saved: debug_contours.jpg")

# ============ LLM PLANNER (UNCHANGED) ============

class LLMPlanner:
    """Phân tích ảnh và đề xuất chiến lược vẽ"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    def analyze_image(self, image_path: str) -> dict:
        """Phân tích ảnh"""
        print("\n🧠 [LLM PLANNER] Analyzing image...")
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        prompt = """
Phân tích ảnh này và trả về JSON (KHÔNG MARKDOWN):

{
    "subject": "tên nhân vật/vật thể",
    "complexity": "simple/medium/complex",
    "main_shapes": ["circle", "oval", "line", "curve"],
    "drawing_order": ["head", "eyes", "mouth", "body", "arms", "legs"],
    "estimated_contours": 50,
    "preprocessing": {
        "blur_kernel": 5,
        "canny_low": 50,
        "canny_high": 150
    }
}
"""
        
        image_part = {
            "mime_type": "image/jpeg" if image_path.endswith(('.jpg', '.jpeg')) else "image/png",
            "data": image_data
        }
        
        response = self.model.generate_content([prompt, image_part])
        text = response.text.strip()
        
        # Clean markdown
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        elif text.startswith("```"):
            text = text.replace("```", "").strip()
        
        import json
        metadata = json.loads(text)
        
        print(f"   ✅ Subject: {metadata['subject']}")
        print(f"   ✅ Complexity: {metadata['complexity']}")
        
        return metadata

# ============ ORCHESTRATOR ============

class HybridDrawingSystem:
    """Điều phối LLM + OpenCV + PyAutoGUI"""
    
    def __init__(self, gemini_api_key: str, debug=True):
        self.planner = LLMPlanner(gemini_api_key)
        self.processor = OpenCVProcessor(debug=debug)
        self.drawer = None
    
    def draw(self, image_path: str):
        """Quy trình vẽ hybrid"""
        print("\n" + "="*70)
        print("🎨 HYBRID DRAWING SYSTEM V3.3 (FIXED)")
        print("="*70)
        
        # 0. Setup Paint (AUTO)
        canvas_region = PaintSetup.setup_paint()
        
        # 1. LLM phân tích
        metadata = self.planner.analyze_image(image_path)
        
        # 2. OpenCV xử lý
        contours, image_shape = self.processor.process_image(image_path, metadata)
        
        # 3. PyAutoGUI vẽ
        self.drawer = PaintDrawer(canvas_region)
        
        print("\n⚠️ Vẽ sẽ bắt đầu sau 3 giây...")
        print("   Đảm bảo Paint window đang active!")
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        self.drawer.draw_contours(contours, image_shape)
        
        print("\n" + "="*70)
        print("🎉 DRAWING COMPLETED!")
        print("="*70)

# ============ MAIN ============

if __name__ == "__main__":
    api_key = os.getenv("GOOGLE_API_KEY")
    
    system = HybridDrawingSystem(api_key, debug=True)
    system.draw("images/luffy.jpg")
