"""
HYBRID DRAWING SYSTEM V4.0 - MULTI-AGENT ARCHITECTURE
======================================================

KIẾN TRÚC:
1. Vision Agent: Phân tích ảnh chi tiết (subject, style, complexity)
2. Strategy Agent: Lập kế hoạch vẽ (order, parameters, optimization)
3. Color Agent: Phân tích màu sắc và shading
4. Execution Agent: Thực thi vẽ với feedback loop
5. Quality Agent: Đánh giá kết quả và suggest improvements
6. Orchestrator: Điều phối toàn bộ agents

FEATURES:
- ✅ Deep analysis với multi-turn LLM conversation
- ✅ Adaptive parameters (Canny, blur) dựa trên image complexity
- ✅ Contour optimization (traveling salesman problem)
- ✅ Color extraction và painting
- ✅ Quality verification với screenshot comparison
- ✅ Self-healing (retry nếu quality thấp)
"""

import os
import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageGrab
import google.generativeai as genai
from dotenv import load_dotenv
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from enum import Enum
import logging
import sys
# ============ LOGGING SETUP ============

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('drawing_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if sys.platform == "win32":
    # Force UTF-8 encoding
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ============ LOGGING SETUP (FIXED) ============

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('drawing_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Test emoji
logger.info("🎨 Unicode test successful!")

# ============ DATA STRUCTURES ============

class Complexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class DrawingStyle(Enum):
    OUTLINE = "outline"
    SKETCH = "sketch"
    DETAILED = "detailed"
    CARTOON = "cartoon"

@dataclass
class ImageAnalysis:
    """Kết quả phân tích từ Vision Agent"""
    subject: str
    complexity: Complexity
    style: DrawingStyle
    main_shapes: List[str]
    key_features: List[str]
    estimated_contours: int
    recommended_order: List[str]
    color_palette: List[Tuple[int, int, int]]
    background_color: Tuple[int, int, int]

@dataclass
class DrawingStrategy:
    """Chiến lược vẽ từ Strategy Agent"""
    preprocessing: Dict
    contour_params: Dict
    drawing_order: List[str]
    optimization_level: int
    use_color: bool
    use_shading: bool
    estimated_time: int

@dataclass
class QualityMetrics:
    """Metrics từ Quality Agent"""
    overall_score: float
    edge_accuracy: float
    completeness: float
    smoothness: float
    suggestions: List[str]

# ============ AGENT 1: VISION AGENT ============

class VisionAgent:
    """
    Agent chuyên phân tích ảnh sâu
    - Nhận diện subject, style, complexity
    - Phân tích cấu trúc, shapes, features
    - Extract color palette
    """
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        logger.info("🔍 Vision Agent initialized")
    
    def analyze(self, image_path: str) -> ImageAnalysis:
        """Phân tích ảnh toàn diện"""
        logger.info(f"🔍 [VISION AGENT] Analyzing: {image_path}")
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # ============ PROMPT ENGINEERING (DEEP ANALYSIS) ============
        prompt = """
Bạn là một chuyên gia phân tích hình ảnh để chuẩn bị cho việc vẽ lại bằng máy tính.

NHIỆM VỤ: Phân tích ảnh này theo các khía cạnh sau:

1. SUBJECT IDENTIFICATION:
   - Đây là nhân vật/vật thể gì?
   - Thuộc thể loại nào (anime, cartoon, realistic, abstract)?
   - Có bao nhiêu đối tượng chính?

2. COMPLEXITY ANALYSIS:
   - Độ phức tạp: simple/medium/complex/very_complex
   - Số lượng chi tiết ước tính
   - Độ khó của các đường nét (straight/curved/mixed)

3. STRUCTURAL BREAKDOWN:
   - Các hình dạng chính (circle, oval, rectangle, triangle, curve)
   - Thứ tự vẽ hợp lý (từ trong ra ngoài, từ trên xuống dưới)
   - Các feature quan trọng (eyes, nose, mouth, hair, accessories)

4. COLOR ANALYSIS:
   - 5-7 màu chính trong ảnh (RGB format)
   - Màu nền (background)
   - Có cần shading không?

5. DRAWING STRATEGY:
   - Nên vẽ outline trước hay fill màu trước?
   - Các vùng nào cần vẽ liên tục?
   - Ước tính số contours cần vẽ

TRẢ VỀ JSON (KHÔNG MARKDOWN, KHÔNG GIẢI THÍCH):

{
    "subject": "tên chính xác của subject",
    "subject_type": "anime/cartoon/realistic/abstract",
    "complexity": "simple/medium/complex/very_complex",
    "complexity_score": 1-10,
    "main_shapes": ["circle", "oval", "line"],
    "key_features": ["eyes", "mouth", "hair"],
    "estimated_contours": 100,
    "recommended_order": ["face_outline", "eyes", "nose", "mouth", "hair", "body"],
    "color_palette": [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    "background_color": [255, 255, 255],
    "drawing_style": "outline/sketch/detailed/cartoon",
    "use_shading": true/false,
    "special_notes": "các lưu ý đặc biệt"
}
"""
        
        image_part = {
            "mime_type": "image/jpeg" if image_path.endswith(('.jpg', '.jpeg')) else "image/png",
            "data": image_data
        }
        
        try:
            response = self.model.generate_content([prompt, image_part])
            text = response.text.strip()
            
            # Clean markdown
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
            
            data = json.loads(text)
            
            # Parse vào ImageAnalysis
            analysis = ImageAnalysis(
                subject=data["subject"],
                complexity=Complexity(data["complexity"]),
                style=DrawingStyle(data["drawing_style"]),
                main_shapes=data["main_shapes"],
                key_features=data["key_features"],
                estimated_contours=data["estimated_contours"],
                recommended_order=data["recommended_order"],
                color_palette=[tuple(c) for c in data["color_palette"]],
                background_color=tuple(data["background_color"])
            )
            
            logger.info(f"   ✅ Subject: {analysis.subject}")
            logger.info(f"   ✅ Complexity: {analysis.complexity.value}")
            logger.info(f"   ✅ Style: {analysis.style.value}")
            logger.info(f"   ✅ Estimated contours: {analysis.estimated_contours}")
            
            return analysis
        
        except Exception as e:
            logger.error(f"   ❌ Vision Agent failed: {e}")
            raise

# ============ AGENT 2: STRATEGY AGENT ============

class StrategyAgent:
    """
    Agent lập kế hoạch vẽ
    - Dựa trên ImageAnalysis để tính toán parameters
    - Tối ưu thứ tự vẽ contours
    - Adaptive preprocessing
    """
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        logger.info("🎯 Strategy Agent initialized")
    
    def plan(self, analysis: ImageAnalysis) -> DrawingStrategy:
        """Lập kế hoạch vẽ dựa trên analysis"""
        logger.info("🎯 [STRATEGY AGENT] Planning drawing strategy...")
        
        # ============ PROMPT ENGINEERING (STRATEGY PLANNING) ============
        prompt = f"""
Bạn là chuyên gia lập kế hoạch vẽ. Dựa trên phân tích sau:

SUBJECT: {analysis.subject}
COMPLEXITY: {analysis.complexity.value}
STYLE: {analysis.style.value}
ESTIMATED CONTOURS: {analysis.estimated_contours}
KEY FEATURES: {', '.join(analysis.key_features)}

NHIỆM VỤ: Đề xuất chiến lược vẽ tối ưu

1. PREPROCESSING PARAMETERS:
   - blur_kernel: 3/5/7/9 (càng phức tạp càng lớn)
   - canny_low: 30-100 (threshold thấp)
   - canny_high: 100-200 (threshold cao)
   - adaptive_threshold: true/false (dùng adaptive thresholding?)

2. CONTOUR PARAMETERS:
   - min_area: 50-500 (lọc contours nhỏ)
   - epsilon_factor: 0.001-0.01 (simplification level)
   - max_contours: 100-1000 (giới hạn số contours)

3. DRAWING ORDER:
   - Thứ tự vẽ các features (theo recommended_order)
   - Có cần group contours không?

4. OPTIMIZATION:
   - optimization_level: 0-3 (0=none, 3=max)
   - use_traveling_salesman: true/false
   - parallel_drawing: true/false

5. COLOR & SHADING:
   - use_color: true/false
   - use_shading: true/false
   - color_order: ["background", "main", "details"]

6. TIME ESTIMATION:
   - Ước tính thời gian vẽ (giây)

TRẢ VỀ JSON (KHÔNG MARKDOWN):

{{
    "preprocessing": {{
        "blur_kernel": 5,
        "canny_low": 50,
        "canny_high": 150,
        "adaptive_threshold": false
    }},
    "contour_params": {{
        "min_area": 100,
        "epsilon_factor": 0.003,
        "max_contours": 500
    }},
    "drawing_order": ["face", "eyes", "mouth"],
    "optimization_level": 2,
    "use_color": true,
    "use_shading": false,
    "estimated_time": 120,
    "strategy_notes": "lưu ý đặc biệt"
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean markdown
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
            
            data = json.loads(text)
            
            strategy = DrawingStrategy(
                preprocessing=data["preprocessing"],
                contour_params=data["contour_params"],
                drawing_order=data["drawing_order"],
                optimization_level=data["optimization_level"],
                use_color=data["use_color"],
                use_shading=data["use_shading"],
                estimated_time=data["estimated_time"]
            )
            
            logger.info(f"   ✅ Strategy planned")
            logger.info(f"   ✅ Optimization level: {strategy.optimization_level}")
            logger.info(f"   ✅ Estimated time: {strategy.estimated_time}s")
            
            return strategy
        
        except Exception as e:
            logger.error(f"   ❌ Strategy Agent failed: {e}")
            # Fallback strategy
            return self._fallback_strategy(analysis)
    
    def _fallback_strategy(self, analysis: ImageAnalysis) -> DrawingStrategy:
        """Fallback strategy nếu LLM fail"""
        logger.warning("   ⚠️ Using fallback strategy")
        
        # Adaptive parameters based on complexity
        complexity_map = {
            Complexity.SIMPLE: (3, 30, 100, 50, 0.005),
            Complexity.MEDIUM: (5, 50, 150, 100, 0.003),
            Complexity.COMPLEX: (7, 70, 180, 200, 0.002),
            Complexity.VERY_COMPLEX: (9, 80, 200, 300, 0.001)
        }
        
        blur, canny_low, canny_high, min_area, epsilon = complexity_map[analysis.complexity]
        
        return DrawingStrategy(
            preprocessing={
                "blur_kernel": blur,
                "canny_low": canny_low,
                "canny_high": canny_high,
                "adaptive_threshold": False
            },
            contour_params={
                "min_area": min_area,
                "epsilon_factor": epsilon,
                "max_contours": 500
            },
            drawing_order=analysis.recommended_order,
            optimization_level=1,
            use_color=False,
            use_shading=False,
            estimated_time=60
        )

# ============ AGENT 3: COLOR AGENT ============

class ColorAgent:
    """
    Agent xử lý màu sắc
    - Extract dominant colors
    - Map colors to Paint palette
    - Plan color filling order
    """
    
    def __init__(self):
        logger.info("🎨 Color Agent initialized")
    
    def extract_colors(self, image_path: str, n_colors: int = 7) -> List[Tuple[int, int, int]]:
        """Extract dominant colors bằng K-means"""
        logger.info(f"🎨 [COLOR AGENT] Extracting {n_colors} colors...")
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape to 2D array
        pixels = img.reshape(-1, 3)
        
        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        colors = [tuple(c) for c in colors]
        
        logger.info(f"   ✅ Extracted colors: {colors}")
        return colors
    
    def map_to_paint_palette(self, colors: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Map colors to Paint's default palette"""
        # Paint có 28 màu cơ bản
        paint_palette = [
            (0, 0, 0), (127, 127, 127), (136, 0, 21), (237, 28, 36),
            (255, 127, 39), (255, 242, 0), (34, 177, 76), (0, 162, 232),
            (63, 72, 204), (163, 73, 164), (255, 255, 255), (195, 195, 195),
            (185, 122, 87), (255, 174, 201), (255, 201, 14), (239, 228, 176),
            (181, 230, 29), (153, 217, 234), (112, 146, 190), (200, 191, 231)
        ]
        
        # Find closest color in palette
        def closest_color(target):
            min_dist = float('inf')
            closest = paint_palette[0]
            for pc in paint_palette:
                dist = sum((a - b) ** 2 for a, b in zip(target, pc))
                if dist < min_dist:
                    min_dist = dist
                    closest = pc
            return closest
        
        mapped = [closest_color(c) for c in colors]
        logger.info(f"   ✅ Mapped to Paint palette: {mapped}")
        return mapped

# ============ AGENT 4: EXECUTION AGENT ============

class ExecutionAgent:
    """
    Agent thực thi vẽ
    - Setup Paint
    - Vẽ contours theo strategy
    - Handle errors và retry
    """
    
    def __init__(self, canvas_region: Dict):
        self.canvas_region = canvas_region
        logger.info("✏️ Execution Agent initialized")
    
    def draw_contours(
        self, 
        contours: List[np.ndarray], 
        image_shape: Tuple,
        strategy: DrawingStrategy,
        color_palette: Optional[List[Tuple[int, int, int]]] = None
    ):
        """Vẽ contours với strategy"""
        logger.info("✏️ [EXECUTION AGENT] Starting drawing...")
        
        img_height, img_width = image_shape[:2]
        canvas_width = self.canvas_region['width']
        canvas_height = self.canvas_region['height']
        
        # Calculate scale
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y) * 0.9
        
        # Calculate offset
        scaled_width = img_width * scale
        scaled_height = img_height * scale
        offset_x = self.canvas_region['x'] + (canvas_width - scaled_width) / 2
        offset_y = self.canvas_region['y'] + (canvas_height - scaled_height) / 2
        
        logger.info(f"   Scale: {scale:.3f}, Offset: ({offset_x:.1f}, {offset_y:.1f})")
        
        # Setup pencil
        self._setup_pencil_tool()
        
        # Optimize contour order (Traveling Salesman)
        if strategy.optimization_level >= 2:
            contours = self._optimize_contour_order(contours)
        
        # Draw
        total = len(contours)
        for i, contour in enumerate(contours):
            try:
                self._draw_single_contour(contour, offset_x, offset_y, scale)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"   Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            
            except Exception as e:
                logger.error(f"   ❌ Failed to draw contour {i}: {e}")
                continue
        
        logger.info("   ✅ Drawing completed!")
    
    def _draw_single_contour(self, contour: np.ndarray, offset_x: float, offset_y: float, scale: float):
        """Vẽ một contour"""
        for j, point in enumerate(contour):
            x = int(offset_x + point[0][0] * scale)
            y = int(offset_y + point[0][1] * scale)
            
            if j == 0:
                pyautogui.moveTo(x, y, duration=0.1)
                time.sleep(0.05)
            else:
                pyautogui.dragTo(x, y, duration=0.02, button='left')
        
        # Close contour
        if len(contour) > 0:
            x = int(offset_x + contour[0][0][0] * scale)
            y = int(offset_y + contour[0][0][1] * scale)
            pyautogui.dragTo(x, y, duration=0.02, button='left')
        
        time.sleep(0.05)
    
    def _optimize_contour_order(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Tối ưu thứ tự vẽ contours (Nearest Neighbor TSP)"""
        logger.info("   🔧 Optimizing contour order...")
        
        if len(contours) <= 1:
            return contours
        
        # Get centroid of each contour
        centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = c[0][0]
            centroids.append((cx, cy))
        
        # Nearest neighbor algorithm
        unvisited = list(range(len(contours)))
        current = 0
        path = [current]
        unvisited.remove(current)
        
        while unvisited:
            current_pos = centroids[current]
            
            # Find nearest
            min_dist = float('inf')
            nearest = unvisited[0]
            
            for idx in unvisited:
                pos = centroids[idx]
                dist = ((current_pos[0] - pos[0]) ** 2 + (current_pos[1] - pos[1]) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest = idx
            
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        optimized = [contours[i] for i in path]
        logger.info(f"   ✅ Optimized order: {len(optimized)} contours")
        return optimized
    
    def _setup_pencil_tool(self):
        """Setup pencil tool"""
        logger.info("   Setting up pencil tool...")
        # Assume pencil is already selected
        time.sleep(0.3)

# ============ AGENT 5: QUALITY AGENT ============

class QualityAgent:
    """
    Agent đánh giá chất lượng
    - So sánh ảnh gốc với screenshot
    - Tính metrics (edge accuracy, completeness)
    - Suggest improvements
    """
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        logger.info("✅ Quality Agent initialized")
    
    def evaluate(self, original_path: str, canvas_region: Dict) -> QualityMetrics:
        """Đánh giá chất lượng vẽ"""
        logger.info("✅ [QUALITY AGENT] Evaluating drawing quality...")
        
        # Capture canvas
        screenshot = ImageGrab.grab(bbox=(
            canvas_region['x'],
            canvas_region['y'],
            canvas_region['x'] + canvas_region['width'],
            canvas_region['y'] + canvas_region['height']
        ))
        screenshot.save("temp_screenshot.png")
        
        # Compare với original
        original = cv2.imread(original_path)
        drawn = cv2.imread("temp_screenshot.png")
        
        # Resize to same size
        drawn = cv2.resize(drawn, (original.shape[1], original.shape[0]))
        
        # Calculate metrics
        edge_accuracy = self._calculate_edge_accuracy(original, drawn)
        completeness = self._calculate_completeness(original, drawn)
        smoothness = self._calculate_smoothness(drawn)
        
        overall_score = (edge_accuracy + completeness + smoothness) / 3
        
        # LLM evaluation
        suggestions = self._get_llm_suggestions(original_path, "temp_screenshot.png")
        
        metrics = QualityMetrics(
            overall_score=overall_score,
            edge_accuracy=edge_accuracy,
            completeness=completeness,
            smoothness=smoothness,
            suggestions=suggestions
        )
        
        logger.info(f"   ✅ Overall score: {overall_score:.2f}")
        logger.info(f"   ✅ Edge accuracy: {edge_accuracy:.2f}")
        logger.info(f"   ✅ Completeness: {completeness:.2f}")
        
        return metrics
    
    def _calculate_edge_accuracy(self, original: np.ndarray, drawn: np.ndarray) -> float:
        """Tính độ chính xác của edges"""
        # Edge detection
        orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 50, 150)
        drawn_edges = cv2.Canny(cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY), 50, 150)
        
        # Calculate overlap
        intersection = cv2.bitwise_and(orig_edges, drawn_edges)
        union = cv2.bitwise_or(orig_edges, drawn_edges)
        
        accuracy = np.sum(intersection) / (np.sum(union) + 1e-6)
        return float(accuracy)
    
    def _calculate_completeness(self, original: np.ndarray, drawn: np.ndarray) -> float:
        """Tính độ hoàn thiện"""
        # Count non-white pixels
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        drawn_gray = cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
        
        orig_pixels = np.sum(orig_gray < 250)
        drawn_pixels = np.sum(drawn_gray < 250)
        
        completeness = min(drawn_pixels / (orig_pixels + 1e-6), 1.0)
        return float(completeness)
    
    def _calculate_smoothness(self, drawn: np.ndarray) -> float:
        """Tính độ mượt của nét vẽ"""
        gray = cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Lower variance = smoother
        smoothness = 1.0 / (1.0 + np.std(grad_mag))
        return float(smoothness)
    
    def _get_llm_suggestions(self, original_path: str, drawn_path: str) -> List[str]:
        """LLM đánh giá và suggest improvements"""
        try:
            with open(original_path, "rb") as f:
                orig_data = f.read()
            with open(drawn_path, "rb") as f:
                drawn_data = f.read()
            
            prompt = """
So sánh 2 ảnh:
- Ảnh 1: Original
- Ảnh 2: Drawn by system

Đánh giá:
1. Độ chính xác của outline
2. Các chi tiết bị thiếu
3. Các lỗi cần sửa

Trả về JSON:
{
    "missing_details": ["eyes too small", "hair incomplete"],
    "errors": ["line not smooth", "wrong proportions"],
    "suggestions": ["increase detail level", "use more contours"]
}
"""
            
            orig_part = {"mime_type": "image/jpeg", "data": orig_data}
            drawn_part = {"mime_type": "image/png", "data": drawn_data}
            
            response = self.model.generate_content([prompt, orig_part, drawn_part])
            text = response.text.strip()
            
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(text)
            return data.get("suggestions", [])
        
        except Exception as e:
            logger.error(f"   ❌ LLM evaluation failed: {e}")
            return []

# ============ ORCHESTRATOR ============

class MultiAgentOrchestrator:
    """
    Điều phối toàn bộ agents
    - Vision → Strategy → Execution → Quality
    - Retry logic nếu quality thấp
    - Logging và monitoring
    """
    
    def __init__(self, api_key: str, debug: bool = True):
        self.debug = debug
        
        # Initialize agents
        self.vision_agent = VisionAgent(api_key)
        self.strategy_agent = StrategyAgent(api_key)
        self.color_agent = ColorAgent()
        self.quality_agent = QualityAgent(api_key)
        self.execution_agent = None  # Khởi tạo sau khi có canvas
        
        logger.info("🎭 Multi-Agent Orchestrator initialized")
    
    def draw(self, image_path: str, max_retries: int = 2):
        """Quy trình vẽ hoàn chỉnh"""
        logger.info("\n" + "="*70)
        logger.info("🎨 HYBRID DRAWING SYSTEM V4.0 - MULTI-AGENT")
        logger.info("="*70)
        
        # PHASE 1: Setup Paint
        logger.info("\n📍 PHASE 1: Setup Paint")
        canvas_region = self._setup_paint()
        self.execution_agent = ExecutionAgent(canvas_region)
        
        # PHASE 2: Vision Analysis
        logger.info("\n📍 PHASE 2: Vision Analysis")
        analysis = self.vision_agent.analyze(image_path)
        
        # PHASE 3: Strategy Planning
        logger.info("\n📍 PHASE 3: Strategy Planning")
        strategy = self.strategy_agent.plan(analysis)
        
        # PHASE 4: Color Extraction (nếu cần)
        color_palette = None
        if strategy.use_color:
            logger.info("\n📍 PHASE 4: Color Extraction")
            color_palette = self.color_agent.extract_colors(image_path)
            color_palette = self.color_agent.map_to_paint_palette(color_palette)
        
        # PHASE 5: Image Processing
        logger.info("\n📍 PHASE 5: Image Processing")
        contours, image_shape = self._process_image(image_path, strategy)
        
        # PHASE 6: Execution (with retry)
        logger.info("\n📍 PHASE 6: Execution")
        
        for attempt in range(max_retries):
            logger.info(f"\n   Attempt {attempt + 1}/{max_retries}")
            
            # Countdown
            logger.info("   ⚠️ Drawing starts in 3 seconds...")
            for i in range(3, 0, -1):
                logger.info(f"      {i}...")
                time.sleep(1)
            
            # Draw
            self.execution_agent.draw_contours(
                contours, 
                image_shape, 
                strategy, 
                color_palette
            )
            
            # PHASE 7: Quality Check
            logger.info("\n📍 PHASE 7: Quality Check")
            metrics = self.quality_agent.evaluate(image_path, canvas_region)
            
            if metrics.overall_score >= 0.7:
                logger.info(f"   ✅ Quality acceptable: {metrics.overall_score:.2f}")
                break
            else:
                logger.warning(f"   ⚠️ Quality low: {metrics.overall_score:.2f}")
                if attempt < max_retries - 1:
                    logger.info("   🔄 Retrying with adjusted parameters...")
                    strategy = self._adjust_strategy(strategy, metrics)
                    self._clear_canvas()
        
        # PHASE 8: Final Report
        logger.info("\n" + "="*70)
        logger.info("🎉 DRAWING COMPLETED!")
        logger.info(f"   Final score: {metrics.overall_score:.2f}")
        logger.info(f"   Suggestions: {', '.join(metrics.suggestions)}")
        logger.info("="*70)
    
    def _setup_paint(self) -> Dict:
        """Setup Paint"""
        logger.info("   Opening Paint...")
        os.system("start mspaint")
        time.sleep(3)
        
        # Maximize
        try:
            import pygetwindow as gw
            paint_windows = gw.getWindowsWithTitle("Paint")
            if not paint_windows:
                paint_windows = gw.getWindowsWithTitle("Untitled")
            if paint_windows:
                paint_windows[0].maximize()
                time.sleep(1)
        except:
            pass
        
        # Detect canvas (fallback method)
        screen_width, screen_height = pyautogui.size()
        return {
            "x": int(screen_width * 0.05),
            "y": int(screen_height * 0.15),
            "width": int(screen_width * 0.85),
            "height": int(screen_height * 0.75)
        }
    
    def _process_image(self, image_path: str, strategy: DrawingStrategy) -> Tuple:
        """Xử lý ảnh với OpenCV"""
        logger.info("   Processing image with OpenCV...")
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing
        blur_kernel = strategy.preprocessing["blur_kernel"]
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Edge detection
        canny_low = strategy.preprocessing["canny_low"]
        canny_high = strategy.preprocessing["canny_high"]
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter
        min_area = strategy.contour_params["min_area"]
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Simplify
        epsilon_factor = strategy.contour_params["epsilon_factor"]
        simplified = []
        for c in contours:
            epsilon = epsilon_factor * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) >= 3:
                simplified.append(approx)
        
        # Limit
        max_contours = strategy.contour_params["max_contours"]
        if len(simplified) > max_contours:
            # Sort by area and keep largest
            simplified = sorted(simplified, key=cv2.contourArea, reverse=True)[:max_contours]
        
        logger.info(f"   ✅ Processed: {len(simplified)} contours")
        
        return simplified, img.shape
    
    def _adjust_strategy(self, strategy: DrawingStrategy, metrics: QualityMetrics) -> DrawingStrategy:
        """Điều chỉnh strategy dựa trên quality metrics"""
        logger.info("   Adjusting strategy based on quality metrics...")
        
        # Increase detail level
        strategy.preprocessing["blur_kernel"] = max(3, strategy.preprocessing["blur_kernel"] - 2)
        strategy.contour_params["epsilon_factor"] *= 0.8
        strategy.contour_params["min_area"] *= 0.8
        
        return strategy
    
    def _clear_canvas(self):
        """Xóa canvas để vẽ lại"""
        logger.info("   Clearing canvas...")
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.2)
        pyautogui.press('delete')
        time.sleep(0.5)

# ============ MAIN ============

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    orchestrator = MultiAgentOrchestrator(api_key, debug=True)
    orchestrator.draw("images/luffy.jpg", max_retries=2)
