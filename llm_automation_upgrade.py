import pyautogui
import time
import math

# --- CÁC THIẾT LẬP AN TOÀN VÀ CHUẨN BỊ ---
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05 # Thêm một khoảng nghỉ nhỏ sau mỗi hành động

# --- CÁC HÀM TIỆN ÍCH ---

def open_and_prepare_paint():
    """Mở MS Paint, phóng to và trả về vùng vẽ."""
    print(">>> Mở Microsoft Paint và chuẩn bị...")
    pyautogui.hotkey('win', 'r')
    time.sleep(1)
    pyautogui.write('mspaint', interval=0.1)
    pyautogui.press('enter')
    time.sleep(3) # Đợi cửa sổ Paint xuất hiện

    # Phóng to cửa sổ Paint để chiếm toàn màn hình
    pyautogui.hotkey('win', 'up')
    time.sleep(1)
    
    # Lấy cửa sổ Paint đang hoạt động
    try:
        paint_window = pyautogui.getWindowsWithTitle('Paint')[0]
        if not paint_window:
            print("!!! Lỗi: Không tìm thấy cửa sổ Paint.")
            return None
        print(f"   ✓ Đã tìm thấy cửa sổ Paint tại: ({paint_window.left}, {paint_window.top})")
        return paint_window
    except IndexError:
        print("!!! Lỗi: Không thể lấy thông tin cửa sổ Paint.")
        return None

def select_color(paint_window, color_name):
    """
    Click vào một màu trên bảng màu của Paint.
    Lưu ý: Tọa độ này là tương đối so với góc trên bên trái của cửa sổ Paint
    và có thể cần điều chỉnh cho phiên bản Paint hoặc độ phân giải khác.
    """
    colors = {
        'black': (910, 55),
        'red': (980, 55)
    }
    if color_name in colors:
        print(f"   - Chọn màu: {color_name.upper()}...")
        x, y = colors[color_name]
        pyautogui.click(paint_window.left + x, paint_window.top + y)
        time.sleep(0.5)

def draw_circle_robust(center_x, center_y, radius, duration_per_segment=0.01):
    """Vẽ hình tròn bằng cách kéo chuột liên tục."""
    start_x = center_x + radius
    start_y = center_y
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    
    # Kéo qua các điểm trên vòng tròn
    for i in range(10, 370, 10): # Bước nhảy 10 độ
        angle = math.radians(i)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        pyautogui.dragTo(x, y, duration=duration_per_segment, button='left')
        
    pyautogui.mouseUp()

# --- HÀM VẼ CHÍNH ---

def draw_doraemon_smart(paint_window):
    """Hàm chính điều phối việc vẽ, sử dụng tọa độ tương đối của cửa sổ."""
    if not paint_window:
        return

    # Xác định trung tâm vùng vẽ (ước lượng)
    # Dựa trên cửa sổ đã phóng to
    canvas_center_x = paint_window.left + (paint_window.width // 2)
    canvas_center_y = paint_window.top + (paint_window.height // 2) + 50 # Dịch xuống 1 chút

    print(">>> Bắt đầu vẽ khuôn mặt...")
    
    # 1. Vẽ đầu (màu đen)
    select_color(paint_window, 'black')
    print("   - Vẽ đầu...")
    head_radius = 200
    draw_circle_robust(canvas_center_x, canvas_center_y, head_radius)
    time.sleep(0.5)

    # 2. Vẽ khuôn mặt bên trong
    print("   - Vẽ khuôn mặt...")
    face_radius = 160
    draw_circle_robust(canvas_center_x, canvas_center_y + 20, face_radius)
    time.sleep(0.5)
    
    # 3. Vẽ mắt
    print("   - Vẽ mắt...")
    eye_radius = 35
    eye_offset_x = 45
    eye_y = canvas_center_y - 70
    draw_circle_robust(canvas_center_x - eye_offset_x, eye_y, eye_radius) # Mắt trái
    draw_circle_robust(canvas_center_x + eye_offset_x, eye_y, eye_radius) # Mắt phải
    
    # Vẽ tròng đen
    pyautogui.click(canvas_center_x - eye_offset_x + 15, eye_y)
    pyautogui.click(canvas_center_x + eye_offset_x - 15, eye_y)
    time.sleep(0.5)

    # 4. Vẽ mũi (màu đỏ)
    select_color(paint_window, 'red')
    print("   - Vẽ mũi...")
    nose_radius = 20
    nose_y = canvas_center_y - 15
    draw_circle_robust(canvas_center_x, nose_y, nose_radius)
    time.sleep(0.5)
    
    # 5. Vẽ miệng và đường giữa (quay lại màu đen)
    select_color(paint_window, 'black')
    print("   - Vẽ miệng...")
    pyautogui.moveTo(canvas_center_x, nose_y + nose_radius)
    pyautogui.dragTo(canvas_center_x, canvas_center_y + 80, duration=0.4)
    pyautogui.moveTo(canvas_center_x - 120, canvas_center_y + 30)
    pyautogui.dragTo(canvas_center_x + 120, canvas_center_y + 30, duration=0.7)
    time.sleep(0.5)

    # 6. Vẽ râu
    print("   - Vẽ râu...")
    whisker_start_x_left = canvas_center_x - 80
    whisker_start_x_right = canvas_center_x + 80
    whisker_y_base = canvas_center_y + 20
    whisker_length = 100
    
    # Râu trái
    pyautogui.moveTo(whisker_start_x_left, whisker_y_base - 20)
    pyautogui.dragRel(-whisker_length, -15, duration=0.3)
    pyautogui.moveTo(whisker_start_x_left, whisker_y_base)
    pyautogui.dragRel(-whisker_length, 0, duration=0.3)
    pyautogui.moveTo(whisker_start_x_left, whisker_y_base + 20)
    pyautogui.dragRel(-whisker_length, 15, duration=0.3)

    # Râu phải
    pyautogui.moveTo(whisker_start_x_right, whisker_y_base - 20)
    pyautogui.dragRel(whisker_length, -15, duration=0.3)
    pyautogui.moveTo(whisker_start_x_right, whisker_y_base)
    pyautogui.dragRel(whisker_length, 0, duration=0.3)
    pyautogui.moveTo(whisker_start_x_right, whisker_y_base + 20)
    pyautogui.dragRel(whisker_length, 15, duration=0.3)
    
    print("\n   ✓ Hoàn thành bản vẽ Doraemon!")

# --- THỰC THI SCRIPT ---

if __name__ == "__main__":
    print("=== DEMO NÂNG CẤP: LLM VẼ DORAEMON ===")
    print("!!! Sẽ bắt đầu sau 5 giây. Vui lòng không chạm vào máy tính.")
    
    for i in range(5, 0, -1):
        print(f"Bắt đầu trong {i}...")
        time.sleep(1)
        
    paint_window_info = open_and_prepare_paint()
    if paint_window_info:
        draw_doraemon_smart(paint_window_info)
    
    print("\n=== KẾT THÚC DEMO ===")