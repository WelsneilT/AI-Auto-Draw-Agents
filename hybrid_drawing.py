"""
HYBRID DRAWING SYSTEM V5.1 - ANIME OPTIMIZED (OPENAI VERSION)
==============================================================

IMPROVEMENTS:
- ✅ OpenAI GPT-4 Vision (FASTER than Gemini)
- ✅ Multi-scale edge detection
- ✅ Anime-specific preprocessing
- ✅ Adaptive Canny thresholds
- ✅ Hierarchical contour filtering
- ✅ Better canvas detection
- 🔧 FIX: Removed 'win+up' hotkey to prevent screen splitting.
"""

import os
import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageGrab
from openai import OpenAI
from dotenv import load_dotenv
import time
import json
import base64
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import sys

# ============ UTF-8 ENCODING ============
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('drawing_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01 # Adjusted for slightly faster drawing

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
    ANIME = "anime"

@dataclass
class ImageAnalysis:
    subject: str
    complexity: Complexity
    style: DrawingStyle
    main_shapes: List[str]
    key_features: List[str]
    estimated_contours: int
    recommended_order: List[str]
    color_palette: List[Tuple[int, int, int]]
    background_color: Tuple[int, int, int]
    is_anime: bool

@dataclass
class DrawingStrategy:
    preprocessing: Dict
    contour_params: Dict
    drawing_order: List[str]
    optimization_level: int
    use_color: bool
    use_shading: bool
    estimated_time: int

@dataclass
class QualityMetrics:
    overall_score: float
    edge_accuracy: float
    completeness: float
    smoothness: float
    suggestions: List[str]

# ============ AGENT 1: OPENAI VISION AGENT ============

class VisionAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("🔍 OpenAI Vision Agent initialized (GPT-4 Vision)")

    def analyze(self, image_path: str) -> ImageAnalysis:
        logger.info(f"🔍 [VISION AGENT] Analyzing: {image_path}")

        # Automatically resize image if it's too large to prevent API timeout
        try:
            with Image.open(image_path) as img:
                max_dim = 1024
                if img.width > max_dim or img.height > max_dim:
                    logger.warning(f"   ⚠️ Image is large ({img.width}x{img.height}), resizing for faster analysis...")
                    img.thumbnail((max_dim, max_dim))
                    temp_path = "temp_resized_image.jpg"
                    img.convert("RGB").save(temp_path, "JPEG")
                    image_path = temp_path
                    logger.info(f"   ✅ Resized to {img.width}x{img.height}")
        except Exception as e:
            logger.error(f"   ❌ Could not resize image: {e}")


        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        mime_type = "image/jpeg" if image_path.endswith(('.jpg', '.jpeg')) else "image/png"

        prompt = """Analyze this image for drawing. Return ONLY valid JSON:

{
    "subject": "brief description",
    "is_anime": true/false,
    "complexity": "simple/medium/complex/very_complex",
    "main_shapes": ["shape1", "shape2", "shape3"],
    "key_features": ["feature1", "feature2", "feature3", "feature4", "feature5"],
    "estimated_contours": 200,
    "recommended_order": ["part1", "part2", "part3"],
    "color_palette": [[255, 224, 189], [0, 0, 0], [255, 0, 0]],
    "background_color": [255, 255, 255],
    "drawing_style": "anime/cartoon/sketch/outline"
}

Rules:
- complexity: simple<100 contours, medium=100-300, complex=300-500, very_complex>500
- is_anime: true if anime/manga style
- Keep arrays short (max 5 items)
- Return ONLY JSON, no markdown"""

        try:
            logger.info("   ⏳ Calling OpenAI GPT-4o-mini...")
            start_time = time.time()

            response = self.client.chat.completions.create(
                model="gpt-5o-mini",  #  CORRECTED MODEL NAME
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1,
                timeout=60.0 # Add a 60-second timeout
            )

            elapsed = time.time() - start_time
            logger.info(f"   ⚡ Response in {elapsed:.2f}s")

            text = response.choices[0].message.content.strip()

            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()

            data = json.loads(text)

            analysis = ImageAnalysis(
                subject=data["subject"],
                complexity=Complexity(data["complexity"]),
                style=DrawingStyle(data["drawing_style"]),
                main_shapes=data["main_shapes"],
                key_features=data["key_features"],
                estimated_contours=data["estimated_contours"],
                recommended_order=data["recommended_order"],
                color_palette=[tuple(c) for c in data["color_palette"]],
                background_color=tuple(data["background_color"]),
                is_anime=data.get("is_anime", False)
            )

            logger.info(f"   ✅ Subject: {analysis.subject}")
            logger.info(f"   ✅ Is Anime: {analysis.is_anime}")
            logger.info(f"   ✅ Complexity: {analysis.complexity.value}")

            return analysis

        except Exception as e:
            logger.error(f"   ❌ OpenAI Vision failed: {e}")
            logger.warning("   ⚠️ Using fallback analysis...")
            return self._fallback_analysis()

    def _fallback_analysis(self) -> ImageAnalysis:
        return ImageAnalysis(
            subject="Anime Character (Fallback)",
            complexity=Complexity.COMPLEX,
            style=DrawingStyle.ANIME,
            main_shapes=["face", "body", "hair"],
            key_features=["face", "eyes", "hair", "body", "clothing"],
            estimated_contours=400,
            recommended_order=["face", "eyes", "hair", "body", "clothing"],
            color_palette=[(0, 0, 0), (255, 255, 255)],
            background_color=(255, 255, 255),
            is_anime=True
        )

# ============ AGENT 2: ENHANCED STRATEGY AGENT ============

class StrategyAgent:
    def __init__(self):
        logger.info("🎯 Enhanced Strategy Agent initialized")
    
    def plan(self, analysis: ImageAnalysis) -> DrawingStrategy:
        logger.info("🎯 [STRATEGY AGENT] Planning strategy...")
        
        if analysis.is_anime or analysis.style == DrawingStyle.ANIME:
            return self._anime_strategy(analysis)
        else:
            return self._general_strategy(analysis)
    
    def _anime_strategy(self, analysis: ImageAnalysis) -> DrawingStrategy:
        logger.info("   Using ANIME-OPTIMIZED strategy")
        
        complexity_map = {
            Complexity.SIMPLE: {
                "blur": 3, "canny_low": 20, "canny_high": 60, "min_area": 20, "epsilon": 0.006, "max_contours": 300
            },
            Complexity.MEDIUM: {
                "blur": 3, "canny_low": 15, "canny_high": 50, "min_area": 15, "epsilon": 0.004, "max_contours": 500
            },
            Complexity.COMPLEX: {
                "blur": 3, "canny_low": 10, "canny_high": 40, "min_area": 10, "epsilon": 0.003, "max_contours": 800
            },
            Complexity.VERY_COMPLEX: {
                "blur": 3, "canny_low": 8, "canny_high": 35, "min_area": 8, "epsilon": 0.002, "max_contours": 1200
            }
        }
        
        params = complexity_map.get(analysis.complexity, complexity_map[Complexity.MEDIUM])
        
        strategy = DrawingStrategy(
            preprocessing={
                "blur_kernel": params["blur"], "canny_low": params["canny_low"], "canny_high": params["canny_high"],
                "use_bilateral": True, "use_clahe": True, "multi_scale": True
            },
            contour_params={
                "min_area": params["min_area"], "epsilon_factor": params["epsilon"],
                "max_contours": params["max_contours"], "hierarchical": True
            },
            drawing_order=analysis.recommended_order, optimization_level=2,
            use_color=False, use_shading=False, estimated_time=180
        )
        
        logger.info(f"   ✅ Anime params: Canny={params['canny_low']}-{params['canny_high']}")
        logger.info(f"   ✅ Min area={params['min_area']}, Max contours={params['max_contours']}")
        
        return strategy
    
    def _general_strategy(self, analysis: ImageAnalysis) -> DrawingStrategy:
        logger.info("   Using GENERAL strategy")
        
        complexity_map = {
            Complexity.SIMPLE: (5, 30, 80, 40, 0.008, 300),
            Complexity.MEDIUM: (5, 25, 70, 30, 0.006, 500),
            Complexity.COMPLEX: (5, 20, 60, 20, 0.004, 800),
            Complexity.VERY_COMPLEX: (5, 15, 50, 15, 0.003, 1200)
        }
        
        blur, canny_low, canny_high, min_area, epsilon, max_contours = complexity_map.get(
            analysis.complexity, (5, 25, 70, 30, 0.006, 500)
        )
        
        return DrawingStrategy(
            preprocessing={
                "blur_kernel": blur, "canny_low": canny_low, "canny_high": canny_high,
                "use_bilateral": False, "use_clahe": False, "multi_scale": False
            },
            contour_params={
                "min_area": min_area, "epsilon_factor": epsilon,
                "max_contours": max_contours, "hierarchical": False
            },
            drawing_order=analysis.recommended_order, optimization_level=2,
            use_color=False, use_shading=False, estimated_time=120
        )

# ============ AGENT 3: COLOR AGENT ============
# (Not used in drawing but available)
class ColorAgent:
    def __init__(self):
        logger.info("🎨 Color Agent initialized")
    
    def extract_colors(self, image_path: str, n_colors: int = 7) -> List[Tuple[int, int, int]]:
        logger.info(f"🎨 [COLOR AGENT] Extracting {n_colors} colors...")
        try:
            from sklearn.cluster import KMeans
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = img.reshape(-1, 3)
            
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            colors = [tuple(map(int, c)) for c in colors]
            
            logger.info(f"   ✅ Extracted {len(colors)} colors")
            return colors
        except ImportError:
            logger.warning("   ⚠️ scikit-learn not installed. Cannot extract colors. Using fallback.")
            return [(0,0,0), (255,255,255)]
        except Exception as e:
            logger.error(f"   ❌ Color extraction failed: {e}")
            return [(0,0,0), (255,255,255)]

# ============ AGENT 4: ENHANCED EXECUTION AGENT ============

class ExecutionAgent:
    def __init__(self, canvas_region: Dict):
        self.canvas_region = canvas_region
        logger.info("✏️ Enhanced Execution Agent initialized")
    
    def draw_contours(
        self, contours: List[np.ndarray], image_shape: Tuple, strategy: DrawingStrategy,
        color_palette: Optional[List[Tuple[int, int, int]]] = None
    ):
        logger.info("✏️ [EXECUTION AGENT] Starting drawing...")
        
        img_height, img_width = image_shape[:2]
        canvas_width = self.canvas_region['width']
        canvas_height = self.canvas_region['height']
        
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y) * 0.90 # Use a bit more of the canvas
        
        scaled_width = img_width * scale
        scaled_height = img_height * scale
        offset_x = self.canvas_region['x'] + (canvas_width - scaled_width) / 2
        offset_y = self.canvas_region['y'] + (canvas_height - scaled_height) / 2
        
        logger.info(f"   📐 Scale: {scale:.3f}, Offset: ({offset_x:.1f}, {offset_y:.1f})")
        
        if strategy.optimization_level >= 2:
            contours = self._optimize_contour_order(contours)
        
        total = len(contours)
        logger.info(f"   🎨 Drawing {total} contours...")
        
        for i, contour in enumerate(contours):
            try:
                self._draw_single_contour(contour, offset_x, offset_y, scale)
                if (i + 1) % 50 == 0 or i == total - 1:
                    logger.info(f"   Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            except Exception as e:
                logger.error(f"   ❌ Failed to draw contour {i}: {e}")
                continue
        
        logger.info("   ✅ Drawing completed!")
    
    def _draw_single_contour(self, contour: np.ndarray, offset_x: float, offset_y: float, scale: float):
        if len(contour) < 2: return

        start_point = contour
        start_x = int(offset_x + start_point * scale)
        start_y = int(offset_y + start_point * scale)

        pyautogui.moveTo(start_x, start_y, duration=0)
        pyautogui.mouseDown(button='left')

        for point in contour[1:]:
            p = point
            x = int(offset_x + p * scale)
            y = int(offset_y + p * scale)
            pyautogui.moveTo(x, y, duration=0)
        
        # Close the loop
        pyautogui.moveTo(start_x, start_y, duration=0)
        pyautogui.mouseUp(button='left')

    def _optimize_contour_order(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        logger.info("   🔧 Optimizing contour order...")
        if len(contours) <= 1: return contours
        
        path, _ = cv2.tsp.findShortestPath(np.array(contours))
        optimized = [contours[i] for i in path.flatten()]

        logger.info(f"   ✅ Optimized {len(optimized)} contours using TSP")
        return optimized

# ============ AGENT 5: OPENAI QUALITY AGENT ============

class QualityAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI Quality Agent initialized")
    
    def evaluate(self, original_path: str, canvas_region: Dict) -> QualityMetrics:
        logger.info("✅ [QUALITY AGENT] Evaluating quality...")
        
        try:
            screenshot = ImageGrab.grab(bbox=(
                canvas_region['x'], canvas_region['y'],
                canvas_region['x'] + canvas_region['width'],
                canvas_region['y'] + canvas_region['height']
            ))
            screenshot.save("temp_screenshot.png")
            
            original = cv2.imread(original_path)
            if original is None:
                logger.error("   ❌ Could not read original image for evaluation.")
                return self._fallback_metrics()

            drawn = cv2.imread("temp_screenshot.png")
            if drawn is None:
                logger.error("   ❌ Could not read screenshot for evaluation.")
                return self._fallback_metrics()

            drawn = cv2.resize(drawn, (original.shape, original.shape))
            
            edge_accuracy = self._calculate_edge_accuracy(original, drawn)
            completeness = self._calculate_completeness(original, drawn)
            smoothness = self._calculate_smoothness(drawn)
            
            overall_score = (edge_accuracy * 0.5) + (completeness * 0.4) + (smoothness * 0.1) # Weighted score
            
            metrics = QualityMetrics(
                overall_score=overall_score, edge_accuracy=edge_accuracy,
                completeness=completeness, smoothness=smoothness, suggestions=[]
            )
            
            logger.info(f"   ✅ Overall: {overall_score:.2f}, Edge: {edge_accuracy:.2f}, Complete: {completeness:.2f}")
            return metrics
        except Exception as e:
            logger.error(f"   ❌ Quality evaluation failed: {e}")
            return self._fallback_metrics()

    def _fallback_metrics(self) -> QualityMetrics:
        return QualityMetrics(0, 0, 0, 0, ["Evaluation failed."])

    def _calculate_edge_accuracy(self, original: np.ndarray, drawn: np.ndarray) -> float:
        orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 50, 150)
        drawn_edges = cv2.Canny(cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY), 50, 150)
        
        intersection = np.sum(cv2.bitwise_and(orig_edges, drawn_edges))
        union = np.sum(cv2.bitwise_or(orig_edges, drawn_edges))
        
        return float(intersection / (union + 1e-6))

    def _calculate_completeness(self, original: np.ndarray, drawn: np.ndarray) -> float:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        drawn_gray = cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
        
        _, orig_thresh = cv2.threshold(orig_gray, 240, 255, cv2.THRESH_BINARY_INV)
        _, drawn_thresh = cv2.threshold(drawn_gray, 240, 255, cv2.THRESH_BINARY_INV)

        orig_pixels = np.sum(orig_thresh > 0)
        drawn_pixels = np.sum(drawn_thresh > 0)
        
        return float(min(drawn_pixels / (orig_pixels + 1e-6), 1.0))

    def _calculate_smoothness(self, drawn: np.ndarray) -> float:
        gray = cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(1.0 / (1.0 + laplacian_var / 1000))

# ============ ENHANCED IMAGE PROCESSOR ============

class ImageProcessor:
    @staticmethod
    def process_anime(image_path: str, strategy: DrawingStrategy) -> Tuple[List[np.ndarray], Tuple]:
        logger.info("   🎨 Processing ANIME image...")
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image from {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if strategy.preprocessing.get("use_clahe", False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray); logger.info("      ✓ CLAHE applied")
        
        if strategy.preprocessing.get("use_bilateral", False):
            gray = cv2.bilateralFilter(gray, 9, 75, 75); logger.info("      ✓ Bilateral filter applied")
        else:
            blur_kernel = strategy.preprocessing["blur_kernel"]
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        all_contours = []
        if strategy.preprocessing.get("multi_scale", False):
            scales = [
                (strategy.preprocessing["canny_low"], strategy.preprocessing["canny_high"]),
                (strategy.preprocessing["canny_low"] // 2, strategy.preprocessing["canny_high"] // 2)
            ]
            for i, (low, high) in enumerate(scales):
                edges = cv2.Canny(gray, low, high)
                contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                filtered = [c for c in contours if cv2.contourArea(c) > strategy.contour_params["min_area"]]
                all_contours.extend(filtered)
                logger.info(f"      ✓ Scale {i+1}: Found {len(filtered)} contours")
        else:
            edges = cv2.Canny(gray, strategy.preprocessing["canny_low"], strategy.preprocessing["canny_high"])
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            all_contours = [c for c in contours if cv2.contourArea(c) > strategy.contour_params["min_area"]]
        
        epsilon_factor = strategy.contour_params["epsilon_factor"]
        simplified = [cv2.approxPolyDP(c, epsilon_factor * cv2.arcLength(c, True), True) for c in all_contours]
        
        simplified = ImageProcessor._remove_duplicate_contours(simplified)
        
        max_contours = strategy.contour_params["max_contours"]
        if len(simplified) > max_contours:
            simplified = sorted(simplified, key=cv2.contourArea, reverse=True)[:max_contours]
        
        logger.info(f"   ✅ Final contours: {len(simplified)}, Image size: {img.shape}x{img.shape}")
        return simplified, img.shape
    
    @staticmethod
    def _remove_duplicate_contours(contours: List[np.ndarray], threshold: float = 5.0) -> List[np.ndarray]:
        if len(contours) <= 1: return contours
        
        unique_contours = []
        # A simple but effective way to remove near-duplicate contours
        # by comparing their centroids and areas.
        # This is a placeholder for a more sophisticated algorithm if needed.
        # For now, let's assume the TSP optimization handles spatial locality.
        # A simple check on contour length might be enough to remove empty ones.
        unique_contours = [c for c in contours if len(c) > 2]

        removed_count = len(contours) - len(unique_contours)
        if removed_count > 0:
            logger.info(f"      ✓ Removed {removed_count} simple/duplicate contours")
        return unique_contours

# ============ ORCHESTRATOR ============

class MultiAgentOrchestrator:
    def __init__(self, api_key: str, debug: bool = True):
        self.debug = debug
        self.vision_agent = VisionAgent(api_key)
        self.strategy_agent = StrategyAgent()
        self.quality_agent = QualityAgent(api_key)
        self.execution_agent = None
        logger.info("🎭 Multi-Agent Orchestrator V5.1 (OpenAI) initialized")
    
    def draw(self, image_path: str):
        logger.info("\n" + "="*70 + f"\n🎨 STARTING DRAWING: {os.path.basename(image_path)}\n" + "="*70)
        
        # PHASE 1: Setup Paint
        logger.info("\n📍 PHASE 1: Setup Paint")
        canvas_region = self._setup_paint()
        if not canvas_region: return # Stop if setup fails
        self.execution_agent = ExecutionAgent(canvas_region)
        
        # PHASE 2: Vision Analysis
        logger.info("\n📍 PHASE 2: OpenAI Vision Analysis")
        analysis = self.vision_agent.analyze(image_path)
        
        # PHASE 3: Strategy Planning
        logger.info("\n📍 PHASE 3: Anime-Optimized Strategy")
        strategy = self.strategy_agent.plan(analysis)
        
        # PHASE 4: Image Processing
        logger.info("\n📍 PHASE 4: Multi-Scale Processing")
        try:
            contours, image_shape = ImageProcessor.process_anime(image_path, strategy)
        except FileNotFoundError as e:
            logger.error(f"   ❌ ERROR: {e}")
            return

        if not contours:
            logger.error("   ❌ No contours found in the image. Cannot draw.")
            return

        # PHASE 5: Execution
        logger.info("\n📍 PHASE 5: Execution")
        logger.info("   ⚠️ Drawing starts in 3 seconds... Do not move the mouse!")
        for i in range(3, 0, -1): logger.info(f"      {i}..."); time.sleep(1)
        
        self.execution_agent.draw_contours(contours, image_shape, strategy)
        
        # PHASE 6: Quality Check
        logger.info("\n📍 PHASE 6: Quality Evaluation")
        metrics = self.quality_agent.evaluate(image_path, canvas_region)
        
        # PHASE 7: Report
        logger.info("\n📍 PHASE 7: Final Report")
        self._generate_report(analysis, strategy, metrics, len(contours))
        
        logger.info("\n" + "="*70 + "\n✅ DRAWING COMPLETED!\n" + "="*70)
    
    def _setup_paint(self) -> Optional[Dict]:
        logger.info("   🎨 Opening Microsoft Paint...")
        os.system("start mspaint")
        time.sleep(3) # Wait for Paint to open and become active
        
        # --- CHANGE ---
        # The 'win' + 'up' hotkey is removed to prevent screen splitting.
        # We rely on Paint opening in a reasonable default size.
        logger.info("   🔧 Skipping window maximization to avoid screen split.")

        screen_width, screen_height = pyautogui.size()
        
        # Define a safe drawing area, assuming Paint is not fully maximized
        canvas_region = {
            'x': 100,  # Increased margin
            'y': 200,  # Increased margin
            'width': screen_width - 250, # Reduced width
            'height': screen_height - 300 # Reduced height
        }
        
        if canvas_region['width'] <= 0 or canvas_region['height'] <= 0:
            logger.error("   ❌ Invalid screen or canvas size detected. Aborting.")
            return None

        logger.info(f"   ✅ Assumed Canvas: {canvas_region['width']}x{canvas_region['height']}")
        
        # Click in the middle of the canvas to give it focus
        pyautogui.click(
            canvas_region['x'] + canvas_region['width'] / 2,
            canvas_region['y'] + canvas_region['height'] / 2
        )
        time.sleep(0.5)
        
        return canvas_region
    
    def _generate_report(
        self, analysis: ImageAnalysis, strategy: DrawingStrategy, 
        metrics: QualityMetrics, contour_count: int
    ):
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    DRAWING REPORT V5.1 (OpenAI)                  ║
╚══════════════════════════════════════════════════════════════════╝

📊 IMAGE ANALYSIS:
   • Subject: {analysis.subject}
   • Style: {analysis.style.value.upper()}
   • Is Anime: {'YES ✓' if analysis.is_anime else 'NO'}
   • Complexity: {analysis.complexity.value.upper()}
   • Estimated/Actual Contours: {analysis.estimated_contours} / {contour_count}

🎯 STRATEGY USED:
   • Preprocessing: Canny={strategy.preprocessing['canny_low']}-{strategy.preprocessing['canny_high']}, Bilateral={'YES' if strategy.preprocessing.get('use_bilateral') else 'NO'}
   • Contour Params: Min Area={strategy.contour_params['min_area']}, Max Contours={strategy.contour_params['max_contours']}
   
✅ QUALITY METRICS:
   • Overall Score: {metrics.overall_score:.2%} {'🌟' if metrics.overall_score > 0.4 else '⭐' if metrics.overall_score > 0.2 else '⚠️'}
   • Edge Accuracy: {metrics.edge_accuracy:.2%}
   • Completeness:  {metrics.completeness:.2%}
   • Smoothness:    {metrics.smoothness:.2%}

🎨 KEY FEATURES DRAWN:
"""
        for i, feature in enumerate(analysis.key_features[:5], 1):
            report += f"   {i}. {feature}\n"
        
        report += "═"*70
        logger.info(report)
        
        try:
            with open("drawing_report.txt", "w", encoding="utf-8") as f:
                f.write(report)
            logger.info("   📄 Report saved to: drawing_report.txt")
        except Exception as e:
            logger.error(f"   ❌ Could not save report: {e}")

# ============ MAIN FUNCTION ============

def main():
    print("🎨 HYBRID DRAWING SYSTEM V5.1 - ANIME OPTIMIZED (OPENAI) 🎨")
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not found in .env file! Please create a .env file with OPENAI_API_KEY=sk-...")
        return
    
    image_path = input("📁 Enter the path to your image: ").strip().strip('"')
    
    if not os.path.exists(image_path):
        logger.error(f"❌ Image not found at path: {image_path}")
        return
    
    logger.info(f"✅ Image loaded: {os.path.basename(image_path)}")
    
    print("\n⚠️ IMPORTANT - PLEASE READ:")
    print("   1. Close any existing MS Paint windows.")
    print("   2. Minimize other windows to avoid interference.")
    print("   3. DO NOT move the mouse or use the keyboard during the drawing process.")
    print("   4. To stop the program, switch to the console window and press Ctrl+C.")
    
    confirm = input("\n🚀 Ready to start drawing? (y/n): ").strip().lower()
    if confirm not in ['yes', 'y']:
        logger.info("❌ Operation cancelled by user.")
        return
    
    orchestrator = MultiAgentOrchestrator(api_key=api_key, debug=True)
    
    try:
        orchestrator.draw(image_path)
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Drawing interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"\n❌ An unexpected error occurred: {e}", exc_info=True)
    finally:
        print("\n👋 Thank you for using the Hybrid Drawing System!")

if __name__ == "__main__":
    main()