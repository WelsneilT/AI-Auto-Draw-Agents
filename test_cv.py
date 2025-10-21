# test_cv.py
import time
import json
from PIL import ImageGrab
import os

# Quan trọng: Import hàm detect từ file bạn đã tạo
from paint_cv_detector import detect_paint_interface_cv

def run_test():
    """
    Chạy thử nghiệm nhận diện giao diện Paint bằng Computer Vision.
    """
    print("============================================")
    print(" BẮT ĐẦU TEST MODULE COMPUTER VISION")
    print("============================================")
    print("\n⚠️  BẠN CÓ 3 GIÂY ĐỂ CHUYỂN SANG CỬA SỔ PAINT ĐANG MAXIMIZE...")

    # Đếm ngược để bạn có thời gian chuẩn bị
    for i in range(3, 0, -1):
        print(f"   ...{i}")
        time.sleep(1)

    print("\n📸  Đang chụp ảnh màn hình...")
    try:
        screenshot = ImageGrab.grab()
        print("   ✅ Chụp ảnh thành công!")
    except Exception as e:
        print(f"   ❌ Lỗi khi chụp ảnh màn hình: {e}")
        return

    print("\n🔬  Đang phân tích giao diện Paint...")
    # Gọi hàm detect chính
    results = detect_paint_interface_cv(screenshot, templates_dir="templates")

    print("\n📊  KẾT QUẢ NHẬN DIỆN:")
    # In kết quả ra một cách dễ đọc
    print(json.dumps(results, indent=2))

    # Hướng dẫn kiểm tra file debug
    if "debug" in results and "path" in results["debug"]:
        debug_path = os.path.abspath(results["debug"]["path"])
        print(f"\n\n============================================")
        print("  KIỂM TRA VISUAL")
        print(f"============================================")
        print(f"\n👉  Hãy mở file sau để xem kết quả nhận diện:")
        print(f"    {debug_path}")
        print("\n- Khung XANH LÁ phải bao quanh bút chì.")
        print("- Khung ĐỎ phải bao quanh ô màu đen.")
        print("- Khung VÀNG phải bao quanh vùng vẽ canvas.")
    else:
        print("\n❌ Không thể tạo file debug. Hãy kiểm tra lại code trong 'paint_cv_detector.py'.")

if __name__ == "__main__":
    run_test()