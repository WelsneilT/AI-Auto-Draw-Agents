# ======================================================================================
# ======================================================================================
"""
LangGraph Multi-Agent System for Paint Automation
Optimized for drawing from a local file.
Architecture: Preparation → Planner → Analyzer → Executor → Validator
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
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import traceback
import webbrowser 

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# ============ STATE DEFINITION ============

class DrawingState(TypedDict):
    """State shared across all agents"""
    subject: str
    search_query: str
    reference_image_b64: str
    screenshot_b64: str
    canvas_bounds: Dict[str, int]
    overall_plan: Dict
    current_phase: str
    current_step: int
    total_steps: int
    step_actions: List[Dict]
    execution_log: List[str]
    validation_result: Dict
    is_complete: bool
    error: str


# ============ UTILITY FUNCTIONS ============

def screenshot_to_base64() -> str:
    """Chụp màn hình và convert sang base64"""
    screenshot = pyautogui.screenshot()
    buffered = BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def save_screenshot(prefix: str = "verify") -> str:
    """Lưu screenshot với timestamp"""
    screenshot = pyautogui.screenshot()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    screenshot.save(filename)
    print(f"   💾 Saved: {filename}")
    return filename


def image_to_base64(image_path: str) -> str:
    """Convert ảnh file sang base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()



def call_gemini_vision(prompt: str, images: List[Image.Image]) -> str:
    """Gọi Gemini với vision"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    content = images + [prompt]
    # Gọi hàm một cách bình thường, không có request_options
    response = model.generate_content(content)
    return response.text

# ============ AGENT 0: PREPARATION AGENT (SIMPLIFIED) ============

def preparation_agent(state: DrawingState) -> DrawingState:
    """
    Agent này chuẩn bị môi trường: mở trình duyệt cho người dùng tìm ảnh,
    chờ người dùng tải ảnh thủ công, sau đó mở Paint.
    """
    print("\n" + "="*70)
    print("🚀 PREPARATION AGENT")
    print("="*70)
    
    # --- Cấu hình đường dẫn ảnh ---
    IMAGE_FOLDER = "images"
    REFERENCE_IMAGE_FILENAME = "doraemon.jpg"
    REFERENCE_IMAGE_PATH = os.path.join(IMAGE_FOLDER, REFERENCE_IMAGE_FILENAME)
    
    try:
        # 1. Mở Chrome để người dùng tìm ảnh (hỗ trợ)
        query = state['search_query']
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch"
        print(f"\n🌐 Mở Chrome để bạn tìm ảnh: {search_url}")
        webbrowser.open(search_url)

        # 2. Hướng dẫn và chờ người dùng tải ảnh
        print("\n" + "!"*70)
        print("ACTION REQUIRED: Vui lòng tìm và tải ảnh bạn muốn vẽ.")
        print(f"👉 Hãy lưu ảnh vào đường dẫn sau: {REFERENCE_IMAGE_PATH}")
        print("!"*70)
        
        # Vòng lặp chờ cho đến khi file tồn tại
        while not os.path.exists(REFERENCE_IMAGE_PATH):
            input("Nhấn Enter khi bạn đã lưu ảnh xong...")
            if not os.path.exists(REFERENCE_IMAGE_PATH):
                print(f"❌ Không tìm thấy file tại '{REFERENCE_IMAGE_PATH}'. Vui lòng kiểm tra lại.")

        print(f"\n✅ Đã tìm thấy ảnh tham khảo: {REFERENCE_IMAGE_PATH}")
        state['reference_image_b64'] = image_to_base64(REFERENCE_IMAGE_PATH)
        
        # 3. Mở Paint
        print("\n🎨 Mở Paint...")
        subprocess.Popen(['mspaint'])
        print("⏳ Chờ Paint mở (3 giây)...")
        time.sleep(3)
        
        # 4. Maximize Paint
        pyautogui.hotkey('win', 'up')
        time.sleep(0.5)
        
        # 5. Chụp screenshot Paint ban đầu
        print("\n📸 Chụp màn hình canvas Paint...")
        state['screenshot_b64'] = screenshot_to_base64()
        state['execution_log'].append("✓ Preparation: User provided reference image.")
        
    except Exception as e:
        print(f"❌ Preparation error: {e}")
        state['error'] = f"Preparation failed: {e}"
        traceback.print_exc()

    return state


# ============ AGENT 1: PLANNER ============

def planner_agent(state: DrawingState) -> DrawingState:
    """
    Agent lập kế hoạch tổng thể
    """
    print("\n" + "="*70)
    print("🧠 PLANNER AGENT")
    print("="*70)
    
    if state['error']: return state

    try:
        # Prepare images
        ref_img = Image.open(BytesIO(base64.b64decode(state['reference_image_b64'])))
        screen_img = Image.open(BytesIO(base64.b64decode(state['screenshot_b64'])))

        # Tối ưu hóa kích thước ảnh để tránh timeout
        MAX_SIZE = (1024, 1024)
        ref_img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
        screen_img.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
        print(f"   🖼️  Reference image resized to: {ref_img.size}")
        print(f"   🖼️  Screenshot resized to: {screen_img.size}")

        prompt = f"""Bạn là Planner Agent chuyên nghiệp.

    NHIỆM VỤ: Tạo kế hoạch chi tiết để vẽ "{state['subject']}" trong Paint, dựa trên ảnh tham khảo.

    ẢNH 1: Ảnh tham khảo (ảnh mẫu do người dùng cung cấp).
    ẢNH 2: Screenshot màn hình Paint trống.

    PHÂN TÍCH VÀ LẬP KẾ HOẠCH:
    1.  Phân tích kỹ lưỡng Ảnh 1 để xác định các thành phần chính (đầu, mắt, mũi, miệng, thân, tay, chân, phụ kiện...).
    2.  Xác định vùng canvas Paint từ Ảnh 2.
    3.  Tạo một kế hoạch vẽ chi tiết, chia thành 3 GIAI ĐOẠN: SKETCHING (phác thảo), DETAILING (vẽ chi tiết), và COLORING (tô màu).
    4.  Mỗi giai đoạn phải có từ 4-6 BƯỚC nhỏ, cụ thể. Mỗi bước phải chỉ rõ: công cụ (`tool`), vị trí (`position`), kích thước (`size`), và màu sắc (`color`).
    5.  Các tọa độ phải được tính toán để nằm trong vùng canvas.

    YÊU CẦU ĐỊNH DẠNG: Chỉ trả về một đối tượng JSON hợp lệ, không có bất kỳ văn bản giải thích nào khác.
    
    Ví dụ JSON:
    {{
      "canvas_bounds": {{"x": 290, "y": 140, "width": 1600, "height": 800}},
      "reference_analysis": "Ảnh tham khảo là Doraemon đang đứng thẳng. Các thành phần chính bao gồm đầu tròn màu xanh, mặt trắng, mắt, mũi, miệng cười, vòng cổ đỏ với chuông vàng, thân xanh, túi bán nguyệt và tay chân tròn.",
      "color_palette": ["blue", "white", "red", "yellow", "black"],
      "phases": [
        {{
          "phase_name": "SKETCHING",
          "description": "Vẽ các hình dạng cơ bản cho đầu và thân.",
          "steps": [
            {{
              "step_id": 1,
              "description": "Vẽ một hình tròn lớn cho cái đầu",
              "tool": "oval",
              "position": {{"center_x": 700, "center_y": 450}},
              "size": {{"width": 300, "height": 300}},
              "color": "black"
            }}
          ]
        }},
        {{
            "phase_name": "DETAILING",
            "description": "Thêm các chi tiết cho khuôn mặt và cơ thể.",
            "steps": []
        }},
        {{
            "phase_name": "COLORING",
            "description": "Tô màu cho bức tranh.",
            "steps": []
        }}
      ]
    }}
    """
        response = call_gemini_vision(prompt, [ref_img, screen_img])
        
        # Logic trích xuất JSON mạnh mẽ
        json_string = response
        if '```json' in response:
            json_string = response.split('```json', 1)[1].rsplit('```', 1)
        elif '```' in response:
             json_string = response.split('```', 1).rsplit('```', 1)[0]
        
        plan = json.loads(json_string.strip())
        
        # Cập nhật state
        state['overall_plan'] = plan
        state['canvas_bounds'] = plan.get('canvas_bounds', {'x': 290, 'y': 140, 'width': 900, 'height': 600})
        state['current_phase'] = plan['phases'][0]['phase_name']
        state['current_step'] = 0
        
        total = sum(len(p.get('steps', [])) for p in plan.get('phases', []))
        state['total_steps'] = total
        
        print("\n📊 Plan created successfully:")
        print(f"   Canvas: {state['canvas_bounds']}")
        print(f"   Colors: {plan.get('color_palette', [])}")
        print(f"   Total phases: {len(plan.get('phases', []))}")
        print(f"   Total steps: {total}")
        
        state['execution_log'].append(f"✓ Planner: Created plan with {total} steps.")
        
    except Exception as e:
        print(f"❌ Planner error: {e}")
        state['error'] = f"Planner failed: {e}"
        traceback.print_exc()
    
    return state


# ============ AGENT 2: ANALYZER ============

def analyzer_agent(state: DrawingState) -> DrawingState:
    """
    Agent phân tích step hiện tại và sinh actions cụ thể
    """
    print("\n" + "="*70)
    print("🔍 ANALYZER AGENT")
    print("="*70)
    
    if state['error']: return state
    
    plan = state['overall_plan']
    current_step_idx = state['current_step']
    
    # Tìm step hiện tại
    step_count = 0
    current_step_data = None
    current_phase_name = None
    
    for phase in plan.get('phases', []):
        for step in phase.get('steps', []):
            if step_count == current_step_idx:
                current_step_data = step
                current_phase_name = phase.get('phase_name', 'Unknown')
                break
            step_count += 1
        if current_step_data:
            break
    
    if not current_step_data:
        print("⚠️ No more steps found in the plan!")
        state['is_complete'] = True
        return state
    
    print(f"\n📍 Phase: {current_phase_name}")
    print(f"📍 Step {current_step_idx + 1}/{state['total_steps']}")
    print(f"📝 Description: {current_step_data.get('description', 'N/A')}")
    
    # Sinh actions
    actions = []
    
    if current_phase_name in ["SKETCHING", "DETAILING"]:
        tool = current_step_data.get('tool', 'oval')
        pos = current_step_data.get('position', {})
        size = current_step_data.get('size', {})
        color = current_step_data.get('color', 'black').lower()
        
        tool_map = {
            'oval': (402, 81), 'rectangle': (418, 65), 'line': (350, 65)
        }
        tool_pos = tool_map.get(tool, (402, 81))
        actions.append({'action': 'click', 'x': tool_pos[0], 'y': tool_pos[1], 'reason': f"Chọn tool: {tool}"})
        
        color_map = {
            'black': (573, 63), 'white': (619, 99), 'red': (638, 63), 
            'yellow': (692, 63), 'blue': (728, 63)
        }
        color_pos = color_map.get(color, (573, 63))
        actions.append({'action': 'click', 'x': color_pos[0], 'y': color_pos[1], 'reason': f"Chọn màu: {color}"})
        
        if tool in ['oval', 'rectangle']:
            cx, cy = pos.get('center_x', 600), pos.get('center_y', 400)
            w, h = size.get('width', 100), size.get('height', 100)
            sx, sy, ex, ey = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
            actions.append({'action': 'drag', 'start_x': sx, 'start_y': sy, 'end_x': ex, 'end_y': ey, 'duration': 0.5})
    
    elif current_phase_name == "COLORING":
        color = current_step_data.get('color', 'blue').lower()
        fill_pos = current_step_data.get('fill_position', {})
        
        actions.append({'action': 'click', 'x': 306, 'y': 81, 'reason': "Chọn Fill tool"}) # Fill bucket
        
        color_map = {
            'black': (573, 63), 'white': (619, 99), 'red': (638, 63), 
            'yellow': (692, 63), 'blue': (728, 63)
        }
        color_pos = color_map.get(color, (728, 63))
        actions.append({'action': 'click', 'x': color_pos[0], 'y': color_pos[1], 'reason': f"Chọn màu tô: {color}"})
        
        fx, fy = fill_pos.get('x', 600), fill_pos.get('y', 400)
        actions.append({'action': 'click', 'x': fx, 'y': fy, 'reason': f"Tô màu tại ({fx},{fy})"})
    
    actions.append({'action': 'screenshot_verify'})
    
    state['step_actions'] = actions
    state['execution_log'].append(f"✓ Analyzer: Generated {len(actions)} actions for step {current_step_idx + 1}")
    
    return state


# ============ AGENT 3: EXECUTOR ============

def executor_agent(state: DrawingState) -> DrawingState:
    """
    Agent thực thi các actions
    """
    print("\n" + "="*70)
    print("⚙️ EXECUTOR AGENT")
    print("="*70)
    
    if state['error']: return state
    
    for action in state.get('step_actions', []):
        act_type = action.get('action')
        print(f"   -> Executing: {act_type} - {action.get('reason', '')}")
        try:
            time.sleep(0.5) # Thêm độ trễ nhỏ giữa các action
            if act_type == 'click':
                pyautogui.click(action['x'], action['y'])
            elif act_type == 'drag':
                pyautogui.moveTo(action['start_x'], action['start_y'], duration=0.2)
                pyautogui.dragTo(action['end_x'], action['end_y'], duration=action.get('duration', 0.5), button='left')
            elif act_type == 'screenshot_verify':
                save_screenshot(f"step_{state['current_step'] + 1:02d}")
                state['screenshot_b64'] = screenshot_to_base64()
        except Exception as e:
            error_msg = f"Executor failed on action '{act_type}': {e}"
            print(f"   ❌ Execution Error: {error_msg}")
            state['error'] = error_msg
            return state
    
    state['execution_log'].append(f"✓ Executor: Completed actions for step {state['current_step'] + 1}")
    print("\n✅ Execution for this step is complete!")
    return state


# ============ AGENT 4: VALIDATOR ============

def validator_agent(state: DrawingState) -> DrawingState:
    """
    Agent kiểm tra kết quả
    """
    print("\n" + "="*70)
    print("✅ VALIDATOR AGENT")
    print("="*70)

    if state['error']: return state

    screen_img = Image.open(BytesIO(base64.b64decode(state['screenshot_b64'])))
    ref_img = Image.open(BytesIO(base64.b64decode(state['reference_image_b64'])))
    
    prompt = f"""Bạn là Validator Agent, một chuyên gia đánh giá nghệ thuật.

    ẢNH 1: Ảnh tham khảo gốc.
    ẢNH 2: Tác phẩm đang vẽ trong Paint.

    NHIỆM VỤ: So sánh tác phẩm hiện tại (Ảnh 2) với ảnh tham khảo (Ảnh 1), đánh giá chất lượng và quyết định hành động tiếp theo.

    ĐỊNH DẠNG JSON:
    {{
      "quality_score": 8.0,
      "analysis": "Đã vẽ xong phần đầu. Tỷ lệ khá chính xác. Cần tiếp tục với phần thân.",
      "decision": "continue"
    }}
    (Quyết định có thể là "continue", "retry", hoặc "complete")
    """
    try:
        response = call_gemini_vision(prompt, [ref_img, screen_img])
        
        json_string = response
        if '```json' in response:
            json_string = response.split('```json', 1)[1].rsplit('```', 1)
        
        result = json.loads(json_string.strip())
        state['validation_result'] = result
        
        score = result.get('quality_score', 5)
        decision = result.get('decision', 'continue').lower()
        
        print(f"\n📊 Quality Score: {score}/10")
        print(f"📝 Analysis: {result.get('analysis', 'N/A')}")
        print(f"🎯 AI Decision: {decision.upper()}")
        
        if decision == 'complete':
            state['is_complete'] = True
        elif decision == 'continue':
            state['current_step'] += 1
        
        state['execution_log'].append(f"✓ Validator: Score={score}, Decision={decision}")
        
    except Exception as e:
        print(f"❌ Validator error: {e}. Defaulting to 'continue'.")
        state['current_step'] += 1
        state['validation_result'] = {'decision': 'continue', 'error': str(e)}
    
    return state


# ============ ROUTING ============

def should_continue(state: DrawingState) -> str:
    """Routing logic"""
    if state.get('error'): return "end"
    if state.get('is_complete'): return "end"
    if state['current_step'] >= state['total_steps']:
        print("\n🏁 All planned steps have been executed.")
        return "end"
    
    user_input = input(f"\n⏸️ Press Enter to continue to step {state['current_step'] + 1}, or 'n' to stop: ").strip().lower()
    return "end" if user_input == 'n' else "continue"


# ============ BUILD GRAPH ============

def build_graph():
    """Build LangGraph workflow"""
    workflow = StateGraph(DrawingState)
    
    workflow.add_node("preparation", preparation_agent)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("executor", executor_agent)
    workflow.add_node("validator", validator_agent)
    
    workflow.set_entry_point("preparation")
    workflow.add_edge("preparation", "planner")
    workflow.add_edge("planner", "analyzer")
    workflow.add_edge("analyzer", "executor")
    workflow.add_edge("executor", "validator")
    
    workflow.add_conditional_edges(
        "validator",
        should_continue,
        {"continue": "analyzer", "end": END}
    )
    
    return workflow.compile()


# ============ MAIN ============

def run_automation():
    print("="*70)
    print("🎨 LANGGRAPH PAINT AUTOMATION (FILE-BASED)")
    print("="*70)
    
    initial_state = DrawingState(
        subject="Doraemon",
        search_query="simple doraemon drawing for kids easy",
        reference_image_b64="",
        screenshot_b64="",
        canvas_bounds={},
        overall_plan={},
        current_phase="",
        current_step=0,
        total_steps=0,
        step_actions=[],
        execution_log=[],
        validation_result={},
        is_complete=False,
        error=""
    )
    
    graph = build_graph()
    
    try:
        final_state = graph.invoke(initial_state)
        print("\n" + "="*70)
        print("📋 EXECUTION SUMMARY")
        print("="*70)
        print(f"Status: {'✅ Complete' if final_state.get('is_complete') or not final_state.get('error') else '❌ Halted'}")
        if final_state.get('error'):
            print(f"🚨 Error: {final_state['error']}")
        print("\n📜 Log:")
        for log in final_state.get('execution_log', []):
            print(f"  {log}")
    except Exception as e:
        print(f"\n💥 A critical error occurred during the workflow: {e}")
        traceback.print_exc()

    print("\n" + "="*70)
    print("🤖 Automation finished.")


if __name__ == "__main__":
    print("\n🚨 IMPORTANT: This script will take control of your mouse and keyboard.")
    print("    To stop it manually, move your mouse to any corner of the screen.")
    print("    You have 5 seconds to cancel (Ctrl+C).")
    time.sleep(5)
    
    run_automation()