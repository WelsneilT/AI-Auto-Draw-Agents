# paint_cv_detector.py (VERSION 2 - COLOR MATCHING)
import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List

def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    """Chuyển đổi ảnh từ định dạng PIL Image sang OpenCV (NumPy array)."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def nms_boxes(boxes: List[Tuple[int,int,int,int,float]], iou_thr=0.3) -> List[Tuple[int,int,int,int,float]]:
    """Loại bỏ các hộp bị trùng lặp (Non-Maximum Suppression)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        b = boxes.pop(0)
        keep.append(b)
        def iou(a,b):
            ax, ay, aw, ah, _ = a; bx, by, bw, bh, _ = b
            x1, y1 = max(ax, bx), max(ay, by)
            x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
            inter = max(0, x2-x1) * max(0, y2-y1)
            union = aw*ah + bw*bh - inter
            return inter/union if union > 0 else 0.0
        boxes = [bb for bb in boxes if iou(bb, b) < iou_thr]
    return keep

def multi_scale_match(src_bgr: np.ndarray, tmpl_bgr: np.ndarray, scales=(0.85,1.0,1.25), thr=0.85) -> List[Tuple[int,int,int,int,float]]:
    """Tìm kiếm ảnh mẫu MÀU trên ảnh nguồn MÀU ở nhiều tỷ lệ."""
    h_t, w_t = tmpl_bgr.shape[:2]
    found_boxes = []
    for s in scales:
        tw, th = int(w_t * s), int(h_t * s)
        if tw == 0 or th == 0: continue
        
        tmpl_s = cv2.resize(tmpl_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(src_bgr, tmpl_s, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= thr)
        
        for y, x in zip(loc[0], loc[1]):
            score = float(res[y, x])
            found_boxes.append((x, y, tw, th, score))
            
    return nms_boxes(found_boxes, iou_thr=0.3)

def draw_debug_info(img: np.ndarray, box: Tuple[int,int,int,int], color=(0,255,0), text=""):
    """Vẽ hộp và nhãn lên ảnh debug."""
    x,y,w,h = box
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    if text:
        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def detect_tool_or_color(src_bgr: np.ndarray, tmpl_bgr: np.ndarray, debug_img: Optional[np.ndarray]=None, color=(0,255,0), label="item", thr=0.85) -> Optional[Dict]:
    """Hàm chung để phát hiện bút chì và màu sắc bằng MÀU."""
    h, w = src_bgr.shape[:2]
    roi = src_bgr[min(40,h):min(200,h), 0:w] # Chỉ tìm trong vùng ribbon
    
    boxes = multi_scale_match(roi, tmpl_bgr, thr=thr)
    if not boxes: return None
    
    b = boxes[0]
    bx, by, bw, bh, sc = b
    # Tọa độ trả về là của toàn màn hình
    cx, cy = bx + bw//2, min(40,h) + by + bh//2 
    
    if debug_img is not None:
        draw_debug_info(debug_img, (bx, min(40,h)+by, bw, bh), color=color, text=f"{label} {sc:.2f}")
    
    return {"x": int(cx), "y": int(cy), "score": sc}

def detect_canvas(src_bgr: np.ndarray, debug_img: Optional[np.ndarray]=None) -> Optional[Dict]:
    """Phát hiện vùng canvas trắng (hàm này vẫn dùng grayscale vì hiệu quả hơn)."""
    h, w = src_bgr.shape[:2]
    roi_top = min(200, h // 5)
    roi = src_bgr[roi_top:h, 0:w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
        
    c = max(contours, key=cv2.contourArea)
    x,y,ww,hh = cv2.boundingRect(c)
    
    if ww*hh < (w*h*0.1): return None
        
    if debug_img is not None:
        draw_debug_info(debug_img, (x, roi_top+y, ww, hh), color=(0,255,255), text=f"Canvas {ww}x{hh}")
        
    return {"x": x, "y": roi_top + y, "width": ww, "height": hh}

def load_template_color(path: str) -> np.ndarray:
    """Tải ảnh mẫu và giữ nguyên MÀU SẮC."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ảnh mẫu không tồn tại: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR) # THAY ĐỔI QUAN TRỌNG
    if img is None:
        raise IOError(f"Không thể đọc file ảnh mẫu: {path}")
    return img

def detect_paint_interface_cv(screenshot_pil: Image.Image, templates_dir="templates") -> Dict:
    """
    Hàm chính: Chụp ảnh màn hình, tìm các thành phần và trả về kết quả.
    """
    src_bgr = pil_to_cv(screenshot_pil)
    debug_img = src_bgr.copy()

    try:
        # Tải template ở dạng MÀU
        pencil_tmpl = load_template_color(os.path.join(templates_dir, "pencil.png"))
        black_tmpl = load_template_color(os.path.join(templates_dir, "colors_black.png"))
    except (FileNotFoundError, IOError) as e:
        print(f"LỖI: {e}. Hãy chắc chắn bạn đã tạo ảnh mẫu đúng trong thư mục '{templates_dir}/'.")
        return {"error": str(e)}

    # Thực hiện nhận diện bằng MÀU
    pencil = detect_tool_or_color(src_bgr, pencil_tmpl, debug_img, color=(0,255,0), label="Pencil", thr=0.80)
    black = detect_tool_or_color(src_bgr, black_tmpl, debug_img, color=(0,0,255), label="Black", thr=0.88)
    # Canvas vẫn dùng grayscale
    canvas = detect_canvas(src_bgr, debug_img)

    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/ui_detect_debug.png", debug_img)

    return {
        "canvas_area": canvas or {},
        "tools": {"pencil": pencil or {}},
        "colors": {"black": black or {}},
        "debug": {"path": "debug/ui_detect_debug.png"}
    }