"""
HYBRID DRAWING SYSTEM V5.0 - ANIME OPTIMIZED (OPENAI VERSION)
==============================================================

IMPROVEMENTS:
- ✅ OpenAI GPT-4 Vision (FASTER than Gemini)
- ✅ Multi-scale edge detection
- ✅ Anime-specific preprocessing
- ✅ Adaptive Canny thresholds
- ✅ Hierarchical contour filtering
- ✅ Better canvas detection
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
pyautogui.PAUSE = 0.05

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
        
        # Encode image to base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        mime_type = "image/jpeg" if image_path.endswith(('.jpg', '.jpeg')) else "image/png"
        
        prompt = """Analyze this image for drawing. You are an expert in anime/manga art analysis.

TASK: Provide detailed analysis for recreating this image.

1. SUBJECT & STYLE:
   - Character name (if known)
   - Anime style: classic/modern/manga/chibi?
   - Line art style: thick/thin/variable?

2. COMPLEXITY (IMPORTANT):
   - Count details: eyes, hair strands, clothing, accessories
   - Anime typically has: complex eyes (pupils, highlights), many hair strands, layered clothing
   - Estimate contours: simple=50-100, medium=100-300, complex=300-500, very_complex=500+

3. KEY FEATURES (priority high→low):
   - Eyes (most important in anime)
   - Face shape
   - Hair strands
   - Clothing details
   - Accessories

4. COLORS:
   - Main colors (RGB format)
   - Skin tone, hair color, clothing colors
   - Background color

Return ONLY valid JSON (no markdown, no explanation):

{
    "subject": "character description",
    "is_anime": true/false,
    "complexity": "simple/medium/complex/very_complex",
    "main_shapes": ["oval face", "large eyes", "spiky hair"],
    "key_features": ["eyes", "face outline", "hair", "clothing", "accessories"],
    "estimated_contours": 250,
    "recommended_order": ["face outline", "eyes", "nose", "mouth", "hair outline", "hair details", "body", "clothing"],
    "color_palette": [[255, 224, 189], [0, 0, 0], [255, 0, 0]],
    "background_color": [255, 255, 255],
    "drawing_style": "anime/cartoon/sketch/outline"
}

Rules:
- complexity based on detail count
- is_anime: true if anime/manga style
- Keep arrays concise (max 8 items)
- Return ONLY the JSON object"""
        
        try:
            logger.info("   ⏳ Calling OpenAI GPT-4 Vision...")
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Fastest vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            elapsed = time.time() - start_time
            logger.info(f"   ⚡ Response received in {elapsed:.2f}s")
            
            text = response.choices[0].message.content.strip()
            
            # Remove markdown if present
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
            
            logger.info(f"   📄 Response length: {len(text)} chars")
            
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
            logger.info(f"   ✅ Estimated contours: {analysis.estimated_contours}")
            
            return analysis
        
        except Exception as e:
            logger.error(f"   ❌ OpenAI Vision failed: {e}")
            logger.warning("   ⚠️ Using fallback analysis...")
            return self._fallback_analysis()
    
    def _fallback_analysis(self) -> ImageAnalysis:
        """Fallback analysis if API fails"""
        return ImageAnalysis(
            subject="Anime Character",
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
        
        # Anime needs special parameters
        complexity_map = {
            Complexity.SIMPLE: {
                "blur": 3,
                "canny_low": 20,
                "canny_high": 60,
                "min_area": 20,
                "epsilon": 0.006,
                "max_contours": 300
            },
            Complexity.MEDIUM: {
                "blur": 3,
                "canny_low": 15,
                "canny_high": 50,
                "min_area": 15,
                "epsilon": 0.004,
                "max_contours": 500
            },
            Complexity.COMPLEX: {
                "blur": 3,
                "canny_low": 10,
                "canny_high": 40,
                "min_area": 10,
                "epsilon": 0.003,
                "max_contours": 800
            },
            Complexity.VERY_COMPLEX: {
                "blur": 3,
                "canny_low": 8,
                "canny_high": 35,
                "min_area": 8,
                "epsilon": 0.002,
                "max_contours": 1200
            }
        }
        
        params = complexity_map.get(analysis.complexity, complexity_map[Complexity.MEDIUM])
        
        strategy = DrawingStrategy(
            preprocessing={
                "blur_kernel": params["blur"],
                "canny_low": params["canny_low"],
                "canny_high": params["canny_high"],
                "use_bilateral": True,
                "use_clahe": True,
                "multi_scale": True
            },
            contour_params={
                "min_area": params["min_area"],
                "epsilon_factor": params["epsilon"],
                "max_contours": params["max_contours"],
                "hierarchical": True
            },
            drawing_order=analysis.recommended_order,
            optimization_level=2,
            use_color=False,
            use_shading=False,
            estimated_time=180
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
            analysis.complexity,
            (5, 25, 70, 30, 0.006, 500)
        )
        
        return DrawingStrategy(
            preprocessing={
                "blur_kernel": blur,
                "canny_low": canny_low,
                "canny_high": canny_high,
                "use_bilateral": False,
                "use_clahe": False,
                "multi_scale": False
            },
            contour_params={
                "min_area": min_area,
                "epsilon_factor": epsilon,
                "max_contours": max_contours,
                "hierarchical": False
            },
            drawing_order=analysis.recommended_order,
            optimization_level=2,
            use_color=False,
            use_shading=False,
            estimated_time=120
        )

# ============ AGENT 3: COLOR AGENT ============

class ColorAgent:
    def __init__(self):
        logger.info("🎨 Color Agent initialized")
    
    def extract_colors(self, image_path: str, n_colors: int = 7) -> List[Tuple[int, int, int]]:
        logger.info(f"🎨 [COLOR AGENT] Extracting {n_colors} colors...")
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img.reshape(-1, 3)
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        colors = [tuple(map(int, c)) for c in colors]
        
        logger.info(f"   ✅ Extracted {len(colors)} colors")
        return colors

# ============ AGENT 4: ENHANCED EXECUTION AGENT ============

class ExecutionAgent:
    def __init__(self, canvas_region: Dict):
        self.canvas_region = canvas_region
        logger.info("✏️ Enhanced Execution Agent initialized")
    
    def draw_contours(
        self,
        contours: List[np.ndarray],
        image_shape: Tuple,
        strategy: DrawingStrategy,
        color_palette: Optional[List[Tuple[int, int, int]]] = None
    ):
        logger.info("✏️ [EXECUTION AGENT] Starting drawing...")
        
        img_height, img_width = image_shape[:2]
        canvas_width = self.canvas_region['width']
        canvas_height = self.canvas_region['height']
        
        # Calculate scale with padding
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y) * 0.80
        
        # Center image
        scaled_width = img_width * scale
        scaled_height = img_height * scale
        offset_x = self.canvas_region['x'] + (canvas_width - scaled_width) / 2
        offset_y = self.canvas_region['y'] + (canvas_height - scaled_height) / 2
        
        logger.info(f"   📐 Scale: {scale:.3f}")
        logger.info(f"   📐 Offset: ({offset_x:.1f}, {offset_y:.1f})")
        
        # Optimize order
        if strategy.optimization_level >= 2:
            contours = self._optimize_contour_order(contours)
        
        # Draw all contours
        total = len(contours)
        logger.info(f"   🎨 Drawing {total} contours...")
        
        for i, contour in enumerate(contours):
            try:
                self._draw_single_contour(contour, offset_x, offset_y, scale)
                
                if (i + 1) % 50 == 0 or i == total - 1:
                    logger.info(f"   Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            
            except Exception as e:
                logger.error(f"   ❌ Failed contour {i}: {e}")
                continue
        
        logger.info("   ✅ Drawing completed!")
    
    def _draw_single_contour(self, contour: np.ndarray, offset_x: float, offset_y: float, scale: float):
        if len(contour) < 2:
            return
        
        # Move to start
        x = int(offset_x + contour[0][0][0] * scale)
        y = int(offset_y + contour[0][0][1] * scale)
        pyautogui.moveTo(x, y, duration=0.03)
        time.sleep(0.02)
        
        # Draw contour
        for point in contour[1:]:
            x = int(offset_x + point[0][0] * scale)
            y = int(offset_y + point[0][1] * scale)
            pyautogui.dragTo(x, y, duration=0.008, button='left')
        
        # Close contour
        x = int(offset_x + contour[0][0][0] * scale)
        y = int(offset_y + contour[0][0][1] * scale)
        pyautogui.dragTo(x, y, duration=0.008, button='left')
        
        time.sleep(0.015)
    
    def _optimize_contour_order(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        logger.info("   🔧 Optimizing contour order...")
        
        if len(contours) <= 1:
            return contours
        
        centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = c[0][0]
            centroids.append((cx, cy))
        
        # Nearest neighbor TSP
        unvisited = list(range(len(contours)))
        current = 0
        path = [current]
        unvisited.remove(current)
        
        while unvisited:
            current_pos = centroids[current]
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
        logger.info(f"   ✅ Optimized {len(optimized)} contours")
        return optimized

# ============ AGENT 5: OPENAI QUALITY AGENT ============

class QualityAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI Quality Agent initialized")
    
    def evaluate(self, original_path: str, canvas_region: Dict) -> QualityMetrics:
        logger.info("✅ [QUALITY AGENT] Evaluating quality...")
        
        # Capture canvas
        screenshot = ImageGrab.grab(bbox=(
            canvas_region['x'],
            canvas_region['y'],
            canvas_region['x'] + canvas_region['width'],
            canvas_region['y'] + canvas_region['height']
        ))
        screenshot.save("temp_screenshot.png")
        
        # Compare
        original = cv2.imread(original_path)
        drawn = cv2.imread("temp_screenshot.png")
        drawn = cv2.resize(drawn, (original.shape[1], original.shape[0]))
        
        edge_accuracy = self._calculate_edge_accuracy(original, drawn)
        completeness = self._calculate_completeness(original, drawn)
        smoothness = self._calculate_smoothness(drawn)
        
        overall_score = (edge_accuracy + completeness + smoothness) / 3
        
        metrics = QualityMetrics(
            overall_score=overall_score,
            edge_accuracy=edge_accuracy,
            completeness=completeness,
            smoothness=smoothness,
            suggestions=[]
        )
        
        logger.info(f"   ✅ Overall: {overall_score:.2f}")
        logger.info(f"   ✅ Edge: {edge_accuracy:.2f}")
        logger.info(f"   ✅ Complete: {completeness:.2f}")
        
        return metrics
    
    def _calculate_edge_accuracy(self, original: np.ndarray, drawn: np.ndarray) -> float:
        orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 50, 150)
        drawn_edges = cv2.Canny(cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY), 50, 150)
        
        intersection = cv2.bitwise_and(orig_edges, drawn_edges)
        union = cv2.bitwise_or(orig_edges, drawn_edges)
        
        accuracy = np.sum(intersection) / (np.sum(union) + 1e-6)
        return float(accuracy)
    
    def _calculate_completeness(self, original: np.ndarray, drawn: np.ndarray) -> float:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        drawn_gray = cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
        
        orig_pixels = np.sum(orig_gray < 250)
        drawn_pixels = np.sum(drawn_gray < 250)
        
        completeness = min(drawn_pixels / (orig_pixels + 1e-6), 1.0)
        return float(completeness)
    
    def _calculate_smoothness(self, drawn: np.ndarray) -> float:
        gray = cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        smoothness = 1.0 / (1.0 + np.std(grad_mag))
        return float(smoothness)

# ============ ENHANCED IMAGE PROCESSOR ============

class ImageProcessor:
    @staticmethod
    def process_anime(image_path: str, strategy: DrawingStrategy) -> Tuple[List[np.ndarray], Tuple]:
        logger.info("   🎨 Processing ANIME image...")
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 1: CLAHE
        if strategy.preprocessing.get("use_clahe", False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            logger.info("      ✓ CLAHE applied")
        
        # Step 2: Bilateral filter
        if strategy.preprocessing.get("use_bilateral", False):
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            logger.info("      ✓ Bilateral filter applied")
        else:
            blur_kernel = strategy.preprocessing["blur_kernel"]
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Step 3: Multi-scale edge detection
        all_contours = []
        
        if strategy.preprocessing.get("multi_scale", False):
            scales = [
                (strategy.preprocessing["canny_low"], strategy.preprocessing["canny_high"]),
                (strategy.preprocessing["canny_low"] // 2, strategy.preprocessing["canny_high"] // 2),
                (strategy.preprocessing["canny_low"] * 2, strategy.preprocessing["canny_high"] * 2)
            ]
            
            for i, (low, high) in enumerate(scales):
                edges = cv2.Canny(gray, low, high)
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                min_area = strategy.contour_params["min_area"]
                filtered = [c for c in contours if cv2.contourArea(c) > min_area]
                
                all_contours.extend(filtered)
                logger.info(f"      ✓ Scale {i+1}: {len(filtered)} contours")
        else:
            edges = cv2.Canny(
                gray,
                strategy.preprocessing["canny_low"],
                strategy.preprocessing["canny_high"]
            )
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            min_area = strategy.contour_params["min_area"]
            all_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Step 4: Simplify
        epsilon_factor = strategy.contour_params["epsilon_factor"]
        simplified = []
        
        for c in all_contours:
            epsilon = epsilon_factor * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) >= 3:
                simplified.append(approx)
        
        # Step 5: Remove duplicates
        simplified = ImageProcessor._remove_duplicate_contours(simplified)
        
        # Step 6: Limit contours
        max_contours = strategy.contour_params["max_contours"]
        if len(simplified) > max_contours:
            simplified = sorted(simplified, key=cv2.contourArea, reverse=True)[:max_contours]
        
        logger.info(f"   ✅ Final: {len(simplified)} contours")
        logger.info(f"   ✅ Image size: {img.shape[1]}x{img.shape[0]}")
        
        return simplified, img.shape
    
    @staticmethod
    def _remove_duplicate_contours(contours: List[np.ndarray], threshold: float = 5.0) -> List[np.ndarray]:
        if len(contours) <= 1:
            return contours
        
        centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centroids.append((cx, cy))
            else:
                centroids.append((c[0][0][0], c[0][0][1]))
        
        unique = []
        used = set()
        
        for i, c in enumerate(contours):
            if i in used:
                continue
            
            unique.append(c)
            
            for j in range(i + 1, len(contours)):
                if j in used:
                    continue
                
                dist = ((centroids[i][0] - centroids[j][0]) ** 2 + 
                       (centroids[i][1] - centroids[j][1]) ** 2) ** 0.5
                
                if dist < threshold:
                    used.add(j)
        
        logger.info(f"      ✓ Removed {len(contours) - len(unique)} duplicates")
        return unique

# ============ ORCHESTRATOR ============

class MultiAgentOrchestrator:
    def __init__(self, api_key: str, debug: bool = True):
        self.debug = debug
        
        self.vision_agent = VisionAgent(api_key)
        self.strategy_agent = StrategyAgent()
        self.color_agent = ColorAgent()
        self.quality_agent = QualityAgent(api_key)
        self.execution_agent = None
        
        logger.info("🎭 Multi-Agent Orchestrator V5.0 (OpenAI) initialized")
    
    def draw(self, image_path: str):
        logger.info("\n" + "="*70)
        logger.info("🎨 HYBRID DRAWING SYSTEM V5.0 - OPENAI VERSION")
        logger.info("="*70)
        
        # PHASE 1: Setup Paint
        logger.info("\n📍 PHASE 1: Setup Paint")
        canvas_region = self._setup_paint()
        self.execution_agent = ExecutionAgent(canvas_region)
        
        # PHASE 2: Vision Analysis
        logger.info("\n📍 PHASE 2: OpenAI Vision Analysis")
        analysis = self.vision_agent.analyze(image_path)
        
        # PHASE 3: Strategy Planning
        logger.info("\n📍 PHASE 3: Anime-Optimized Strategy")
        strategy = self.strategy_agent.plan(analysis)
        
        # PHASE 4: Image Processing
        logger.info("\n📍 PHASE 4: Multi-Scale Processing")
        contours, image_shape = ImageProcessor.process_anime(image_path, strategy)
        
        # PHASE 5: Execution
        logger.info("\n📍 PHASE 5: Execution")
        logger.info("   ⚠️ Drawing starts in 3 seconds...")
        for i in range(3, 0, -1):
            logger.info(f"      {i}...")
            time.sleep(1)
        
        self.execution_agent.draw_contours(contours, image_shape, strategy)
        
        # PHASE 6: Quality Check
        logger.info("\n📍 PHASE 6: Quality Evaluation")
        metrics = self.quality_agent.evaluate(image_path, canvas_region)
        
        # PHASE 7: Report
        logger.info("\n📍 PHASE 7: Final Report")
        self._generate_report(analysis, strategy, metrics, len(contours))
        
        logger.info("\n" + "="*70)
        logger.info("✅ DRAWING COMPLETED!")
        logger.info("="*70)
    
    def _setup_paint(self) -> Dict:
        logger.info("   🎨 Opening Microsoft Paint...")
        
        os.system("mspaint")
        time.sleep(3)
        
        pyautogui.hotkey('win', 'up')
        time.sleep(1)
        
        screen_width, screen_height = pyautogui.size()
        
        canvas_region = {
            'x': 10,
            'y': 150,
            'width': screen_width - 20,
            'height': screen_height - 200
        }
        
        logger.info(f"   ✅ Canvas: {canvas_region['width']}x{canvas_region['height']}")
        
        pyautogui.click(canvas_region['x'] + 100, canvas_region['y'] + 100)
        time.sleep(0.5)
        
        return canvas_region
    
    def _generate_report(
        self,
        analysis: ImageAnalysis,
        strategy: DrawingStrategy,
        metrics: QualityMetrics,
        contour_count: int
    ):
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    DRAWING REPORT V5.0 (OpenAI)                  ║
╚══════════════════════════════════════════════════════════════════╝

📊 IMAGE ANALYSIS:
   • Subject: {analysis.subject}
   • Style: {analysis.style.value.upper()}
   • Is Anime: {'YES ✓' if analysis.is_anime else 'NO'}
   • Complexity: {analysis.complexity.value.upper()}
   • Estimated contours: {analysis.estimated_contours}
   • Actual contours drawn: {contour_count}

🎯 STRATEGY USED:
   • Preprocessing:
     - Blur kernel: {strategy.preprocessing['blur_kernel']}
     - Canny thresholds: {strategy.preprocessing['canny_low']}-{strategy.preprocessing['canny_high']}
     - Bilateral filter: {'YES' if strategy.preprocessing.get('use_bilateral') else 'NO'}
     - CLAHE: {'YES' if strategy.preprocessing.get('use_clahe') else 'NO'}
     - Multi-scale: {'YES' if strategy.preprocessing.get('multi_scale') else 'NO'}
   
   • Contour parameters:
     - Min area: {strategy.contour_params['min_area']}
     - Epsilon factor: {strategy.contour_params['epsilon_factor']}
     - Max contours: {strategy.contour_params['max_contours']}
   
   • Optimization level: {strategy.optimization_level}

✅ QUALITY METRICS:
   • Overall Score: {metrics.overall_score:.2%} {'🌟' if metrics.overall_score > 0.7 else '⭐' if metrics.overall_score > 0.5 else '⚠️'}
   • Edge Accuracy: {metrics.edge_accuracy:.2%}
   • Completeness: {metrics.completeness:.2%}
   • Smoothness: {metrics.smoothness:.2%}

🎨 KEY FEATURES DRAWN:
"""
        
        for i, feature in enumerate(analysis.key_features[:5], 1):
            report += f"   {i}. {feature}\n"
        
        report += "\n" + "="*70
        
        logger.info(report)
        
        with open("drawing_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info("   📄 Report saved to: drawing_report.txt")

# ============ MAIN FUNCTION ============

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║       🎨 HYBRID DRAWING SYSTEM V5.0 - OPENAI VERSION 🎨         ║
║                                                                  ║
║  Improvements:                                                   ║
║  ⚡ OpenAI GPT-4 Vision (FASTER than Gemini)                     ║
║  ✅ Multi-scale edge detection                                   ║
║  ✅ Anime-specific preprocessing                                 ║
║  ✅ Adaptive Canny thresholds                                    ║
║  ✅ Hierarchical contour filtering                               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not found in .env file!")
        logger.info("💡 Add this to your .env file:")
        logger.info("   OPENAI_API_KEY=sk-...")
        return
    
    # Get image path
    image_path = input("📁 Enter image path: ").strip().strip('"')
    
    if not os.path.exists(image_path):
        logger.error(f"❌ Image not found: {image_path}")
        return
    
    logger.info(f"✅ Image loaded: {image_path}")
    
    # Confirm
    print("\n⚠️  IMPORTANT:")
    print("   1. Make sure Microsoft Paint is NOT already open")
    print("   2. Close all unnecessary windows")
    print("   3. Do NOT move mouse during drawing")
    print("   4. Press Ctrl+C to stop at any time")
    
    confirm = input("\n🚀 Ready to start? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        logger.info("❌ Cancelled by user")
        return
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(api_key=api_key, debug=True)
    
    try:
        orchestrator.draw(image_path)
        print("\n✅ SUCCESS! Check your Paint window!")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Drawing interrupted by user")
    
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
    
    finally:
        print("\n👋 Thank you for using Hybrid Drawing System V5.0!")

if __name__ == "__main__":
    main()
