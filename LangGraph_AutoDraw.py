# HYBRID DRAWING SYSTEM V6.0 - LANGGRAPH VERSION
# ==============================================================
# IMPROVEMENTS:
# - ✅ All 5 agents now use LLM intelligently
# - ✅ Converted the orchestrator to a LangGraph graph for clear workflow visualization and state management.
# - ✅ Added explicit conditional logic for handling cases where no contours are found.
# - ✅ Encapsulated all logic into a single, runnable script.

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
from typing import Dict, List, Tuple, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum
import logging
import sys
import subprocess

from langgraph.graph import StateGraph, END

# ============ UTF-8 ENCODING & LOGGING ============
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('drawing_system_langgraph.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

# ===============================================================
# SECTION 1: DATA STRUCTURES AND AGENT DEFINITIONS
# (These classes are kept from your original code as they are well-defined)
# ===============================================================

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
    priority_regions: List[str]
    detail_levels: Dict[str, str]

@dataclass
class DrawingStrategy:
    preprocessing: Dict
    contour_params: Dict
    drawing_order: List[str]
    optimization_level: int
    use_color: bool
    use_shading: bool
    estimated_time: int
    region_strategies: Dict[str, Dict]
    adaptive_speed: Dict[str, float]

@dataclass
class QualityMetrics:
    overall_score: float
    edge_accuracy: float
    completeness: float
    smoothness: float
    suggestions: List[str]
    region_scores: Dict[str, float]

class VisionAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("🔍 Enhanced OpenAI Vision Agent initialized")

    def analyze(self, image_path: str) -> ImageAnalysis:
        logger.info(f"🔍 [VISION AGENT] Analyzing: {image_path}")
        try:
            with Image.open(image_path) as img:
                max_dim = 1024
                if img.width > max_dim or img.height > max_dim:
                    logger.warning(f"   ⚠️ Resizing large image ({img.width}x{img.height})...")
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
        prompt = """Analyze this image for drawing with MAXIMUM DETAIL. Return ONLY valid JSON:
{
    "subject": "detailed description", "is_anime": true/false, "complexity": "simple/medium/complex/very_complex",
    "main_shapes": ["shape1", "shape2", "shape3"], "key_features": ["feature1", "feature2", "feature3", "feature4", "feature5"],
    "estimated_contours": 500, "recommended_order": ["part1", "part2", "part3"],
    "priority_regions": ["most_important_part", "second_important", "third"],
    "detail_levels": {"face": "very_high", "hair": "high", "body": "medium", "background": "low"},
    "color_palette": [[255, 224, 189], [0, 0, 0], [255, 0, 0]], "background_color": [255, 255, 255],
    "drawing_style": "anime/cartoon/sketch/outline"
}
Rules:
- complexity: simple<100, medium=100-300, complex=300-800, very_complex>800
- is_anime: true if anime/manga style
- priority_regions: parts to draw first (most important)
- detail_levels: very_high/high/medium/low for each region
- Return ONLY JSON, no markdown"""
        try:
            logger.info("   ⏳ Calling OpenAI GPT-4o-mini with enhanced prompt...")
            start_time = time.time()
            response = self.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}", "detail": "high"}}]}], max_tokens=800, temperature=0.1, timeout=60.0)
            elapsed = time.time() - start_time
            logger.info(f"   ⚡ Response in {elapsed:.2f}s")
            text = response.choices[0].message.content.strip()
            if text.startswith("```json"): text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"): text = text.replace("```", "").strip()
            data = json.loads(text)
            analysis = ImageAnalysis(subject=data["subject"], complexity=Complexity(data["complexity"]), style=DrawingStyle(data["drawing_style"]), main_shapes=data["main_shapes"], key_features=data["key_features"], estimated_contours=data["estimated_contours"], recommended_order=data["recommended_order"], color_palette=[tuple(c) for c in data["color_palette"]], background_color=tuple(data["background_color"]), is_anime=data.get("is_anime", False), priority_regions=data.get("priority_regions", []), detail_levels=data.get("detail_levels", {}))
            logger.info(f"   ✅ Subject: {analysis.subject}")
            logger.info(f"   ✅ Is Anime: {analysis.is_anime}")
            logger.info(f"   ✅ Complexity: {analysis.complexity.value}")
            logger.info(f"   ✅ Priority Regions: {analysis.priority_regions}")
            return analysis
        except Exception as e:
            logger.error(f"   ❌ OpenAI Vision failed: {e}")
            logger.warning("   ⚠️ Using fallback analysis...")
            return self._fallback_analysis()
    def _fallback_analysis(self) -> ImageAnalysis:
        return ImageAnalysis(subject="Anime Character (Fallback)", complexity=Complexity.COMPLEX, style=DrawingStyle.ANIME, main_shapes=["face", "body", "hair"], key_features=["face", "eyes", "hair", "body", "clothing"], estimated_contours=400, recommended_order=["face", "eyes", "hair", "body", "clothing"], color_palette=[(0, 0, 0), (255, 255, 255)], background_color=(255, 255, 255), is_anime=True, priority_regions=["face", "eyes", "hair"], detail_levels={"face": "very_high", "hair": "high", "body": "medium"})

class StrategyAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("🎯 LLM-Enhanced Strategy Agent initialized")
    def plan(self, analysis: ImageAnalysis) -> DrawingStrategy:
        logger.info("🎯 [STRATEGY AGENT] Planning with LLM assistance...")
        strategy_params = self._get_llm_strategy(analysis)
        if strategy_params: return self._build_strategy_from_llm(strategy_params, analysis)
        else:
            logger.warning("   ⚠️ LLM strategy failed, using enhanced fallback")
            return self._enhanced_fallback_strategy(analysis)
    def _get_llm_strategy(self, analysis: ImageAnalysis) -> Optional[Dict]:
        prompt = f"""You are an expert in computer vision and image processing. Given this image analysis:
Subject: {analysis.subject}
Complexity: {analysis.complexity.value}
Style: {analysis.style.value}
Is Anime: {analysis.is_anime}
Key Features: {', '.join(analysis.key_features)}
Priority Regions: {', '.join(analysis.priority_regions)}
Detail Levels: {json.dumps(analysis.detail_levels)}
Recommend optimal OpenCV parameters for contour detection. Return ONLY valid JSON:
{{
    "global_params": {{"blur_kernel": 3, "canny_low": 10, "canny_high": 40, "min_area": 2, "epsilon_factor": 0.001, "max_contours": 2000}},
    "region_strategies": {{"face": {{"canny_low": 5, "canny_high": 25, "min_area": 1, "priority": 1}}, "hair": {{"canny_low": 8, "canny_high": 35, "min_area": 2, "priority": 2}}, "body": {{"canny_low": 12, "canny_high": 45, "min_area": 3, "priority": 3}}}},
    "adaptive_speed": {{"face": 0.02, "hair": 0.01, "body": 0.005}},
    "use_clahe": true, "use_bilateral": true, "use_morphology": true, "multi_scale": true
}}
Guidelines:
- Lower Canny thresholds = more details (5-15 for very detailed areas)
- Higher thresholds = cleaner lines (20-60 for simple areas)
- min_area: 1-5 (lower = more small details)
- epsilon_factor: 0.0005-0.002 (lower = more accurate contours)
- For anime: prioritize face/eyes with lowest thresholds
- adaptive_speed: slower (0.02-0.05) for detailed areas, faster (0.005-0.01) for simple areas
Return ONLY JSON."""
        try:
            logger.info("   ⏳ Asking LLM for optimal parameters...")
            response = self.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are an expert in computer vision and OpenCV parameter optimization."}, {"role": "user", "content": prompt}], max_tokens=800, temperature=0.2, timeout=30.0)
            text = response.choices.message.content.strip()
            if text.startswith("```json"): text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"): text = text.replace("```", "").strip()
            params = json.loads(text)
            logger.info(f"   ✅ LLM recommended: Canny={params['global_params']['canny_low']}-{params['global_params']['canny_high']}")
            logger.info(f"   ✅ Max contours: {params['global_params']['max_contours']}")
            return params
        except Exception as e:
            logger.error(f"   ❌ LLM strategy request failed: {e}")
            return None
    def _build_strategy_from_llm(self, params: Dict, analysis: ImageAnalysis) -> DrawingStrategy:
        global_params = params["global_params"]
        strategy = DrawingStrategy(preprocessing={"blur_kernel": global_params["blur_kernel"], "canny_low": global_params["canny_low"], "canny_high": global_params["canny_high"], "use_bilateral": params.get("use_bilateral", True), "use_clahe": params.get("use_clahe", True), "multi_scale": params.get("multi_scale", True), "use_morphology": params.get("use_morphology", True)}, contour_params={"min_area": global_params["min_area"], "epsilon_factor": global_params["epsilon_factor"], "max_contours": global_params["max_contours"], "hierarchical": True, "filter_nested": True}, drawing_order=analysis.recommended_order, optimization_level=3, use_color=False, use_shading=False, estimated_time=400, region_strategies=params.get("region_strategies", {}), adaptive_speed=params.get("adaptive_speed", {}))
        logger.info("   ✅ Strategy built from LLM recommendations")
        return strategy
    def _enhanced_fallback_strategy(self, analysis: ImageAnalysis) -> DrawingStrategy:
        complexity_map = {Complexity.SIMPLE: (3, 20, 60, 5, 0.005, 800), Complexity.MEDIUM: (3, 12, 45, 3, 0.003, 1500), Complexity.COMPLEX: (3, 8, 30, 2, 0.002, 2500), Complexity.VERY_COMPLEX: (3, 5, 25, 1, 0.001, 4000)}
        blur, canny_low, canny_high, min_area, epsilon, max_contours = complexity_map.get(analysis.complexity, (3, 10, 40, 2, 0.003, 2000))
        return DrawingStrategy(preprocessing={"blur_kernel": blur, "canny_low": canny_low, "canny_high": canny_high, "use_bilateral": True, "use_clahe": True, "use_morphology": True, "multi_scale": True}, contour_params={"min_area": min_area, "epsilon_factor": epsilon, "max_contours": max_contours, "hierarchical": True, "filter_nested": True}, drawing_order=analysis.recommended_order, optimization_level=2, use_color=False, use_shading=False, estimated_time=300, region_strategies={}, adaptive_speed={})

class ImageProcessor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("🖼️ LLM-Enhanced Image Processor initialized")
    def process(self, image_path: str, strategy: DrawingStrategy, analysis: ImageAnalysis) -> Tuple[List[np.ndarray], Tuple]:
        logger.info("   🎨 Processing with LLM-enhanced pipeline...")
        img = cv2.imread(image_path)
        if img is None: raise FileNotFoundError(f"Could not read image from {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if strategy.preprocessing.get("use_clahe", False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); gray = clahe.apply(gray); logger.info("      ✓ CLAHE applied")
        if strategy.preprocessing.get("use_bilateral", False):
            gray = cv2.bilateralFilter(gray, 9, 75, 75); logger.info("      ✓ Bilateral filter applied")
        else: blur_kernel = strategy.preprocessing["blur_kernel"]; gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        if strategy.preprocessing.get("use_morphology", False):
            kernel = np.ones((2, 2), np.uint8); gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel); logger.info("      ✓ Morphological gradient applied")
        all_contours = []
        if strategy.preprocessing.get("multi_scale", False):
            scales = [(strategy.preprocessing["canny_low"], strategy.preprocessing["canny_high"]), (max(5, strategy.preprocessing["canny_low"] // 2), strategy.preprocessing["canny_high"] // 2), (strategy.preprocessing["canny_low"] * 2, min(255, strategy.preprocessing["canny_high"] * 2))]
            for i, (low, high) in enumerate(scales):
                edges = cv2.Canny(gray, low, high); kernel = np.ones((2, 2), np.uint8); edges = cv2.dilate(edges, kernel, iterations=1); contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE); filtered = [c for c in contours if cv2.contourArea(c) > strategy.contour_params["min_area"]]; all_contours.extend(filtered); logger.info(f"      ✓ Scale {i+1} ({low}-{high}): Found {len(filtered)} contours")
        else:
            edges = cv2.Canny(gray, strategy.preprocessing["canny_low"], strategy.preprocessing["canny_high"]); contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE); all_contours = [c for c in contours if cv2.contourArea(c) > strategy.contour_params["min_area"]]
        epsilon_factor = strategy.contour_params["epsilon_factor"]; simplified = []
        for c in all_contours:
            approx = cv2.approxPolyDP(c, epsilon_factor * cv2.arcLength(c, True), True)
            if len(approx) >= 3: simplified.append(approx)
        simplified = self._remove_duplicate_contours(simplified); simplified = sorted(simplified, key=cv2.contourArea, reverse=True)
        max_contours = strategy.contour_params["max_contours"]
        if len(simplified) > max_contours: simplified = simplified[:max_contours]
        logger.info(f"   ✅ Final contours: {len(simplified)}, Image size: {img.shape[1]}x{img.shape[0]}")
        if len(simplified) < 50 and analysis.complexity != Complexity.SIMPLE:
            logger.warning(f"   ⚠️ Only {len(simplified)} contours found, asking LLM for adjustment...")
            adjustment = self._ask_llm_for_adjustment(len(simplified), analysis, strategy)
            if adjustment: logger.info("   🔄 Reprocessing with adjusted parameters..."); return self._reprocess_with_adjustment(image_path, strategy, adjustment)
        return simplified, img.shape
    def _remove_duplicate_contours(self, contours: List[np.ndarray], threshold: float = 0.95) -> List[np.ndarray]:
        if len(contours) <= 1: return contours
        unique = []
        for c in contours:
            is_duplicate = False
            for u in unique:
                match = cv2.matchShapes(c, u, cv2.CONTOURS_MATCH_I2, 0)
                if match < threshold: is_duplicate = True; break
            if not is_duplicate: unique.append(c)
        return unique
    def _ask_llm_for_adjustment(self, current_count: int, analysis: ImageAnalysis, strategy: DrawingStrategy) -> Optional[Dict]:
        prompt = f"""The contour detection found only {current_count} contours, which seems too low.
Image info: - Subject: {analysis.subject} - Complexity: {analysis.complexity.value} - Expected contours: {analysis.estimated_contours}
Current parameters: - Canny: {strategy.preprocessing['canny_low']}-{strategy.preprocessing['canny_high']} - Min area: {strategy.contour_params['min_area']} - Epsilon: {strategy.contour_params['epsilon_factor']}
Should we adjust? Return JSON: {{ "should_adjust": true/false, "new_canny_low": 5, "new_canny_high": 20, "new_min_area": 1, "reason": "explanation" }}
Return ONLY JSON."""
        try:
            response = self.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are an expert in computer vision parameter tuning."}, {"role": "user", "content": prompt}], max_tokens=200, temperature=0.3, timeout=20.0)
            text = response.choices[0].message.content.strip()
            if text.startswith("```json"): text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"): text = text.replace("```", "").strip()
            adjustment = json.loads(text)
            if adjustment.get("should_adjust", False):
                logger.info(f"   ✅ LLM suggests adjustment: {adjustment['reason']}"); return adjustment
            return None
        except Exception as e:
            logger.error(f"   ❌ LLM adjustment request failed: {e}"); return None
    def _reprocess_with_adjustment(self, image_path: str, strategy: DrawingStrategy, adjustment: Dict) -> Tuple[List[np.ndarray], Tuple]:
        strategy.preprocessing["canny_low"] = adjustment["new_canny_low"]; strategy.preprocessing["canny_high"] = adjustment["new_canny_high"]; strategy.contour_params["min_area"] = adjustment["new_min_area"]
        img = cv2.imread(image_path); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if strategy.preprocessing.get("use_clahe", False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); gray = clahe.apply(gray)
        if strategy.preprocessing.get("use_bilateral", False): gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, strategy.preprocessing["canny_low"], strategy.preprocessing["canny_high"]); contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [c for c in contours if cv2.contourArea(c) > strategy.contour_params["min_area"]]
        epsilon_factor = strategy.contour_params["epsilon_factor"]; simplified = []
        for c in filtered:
            approx = cv2.approxPolyDP(c, epsilon_factor * cv2.arcLength(c, True), True)
            if len(approx) >= 3: simplified.append(approx)
        simplified = sorted(simplified, key=cv2.contourArea, reverse=True)
        max_contours = strategy.contour_params["max_contours"]
        if len(simplified) > max_contours: simplified = simplified[:max_contours]
        logger.info(f"   ✅ After adjustment: {len(simplified)} contours")
        return simplified, img.shape

class ExecutionAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("✏️ LLM-Enhanced Execution Agent initialized")
    def draw_contours(self, contours: List[np.ndarray], image_shape: Tuple, strategy: DrawingStrategy, analysis: ImageAnalysis, canvas_region: Dict):
        logger.info("✏️ [EXECUTION AGENT] Starting LLM-guided drawing...")
        img_height, img_width = image_shape[:2]; canvas_width = canvas_region['width']; canvas_height = canvas_region['height']
        scale_x = canvas_width / img_width; scale_y = canvas_height / img_height; scale = min(scale_x, scale_y) * 0.85
        scaled_width = img_width * scale; scaled_height = img_height * scale
        offset_x = canvas_region['x'] + (canvas_width - scaled_width) / 2; offset_y = canvas_region['y'] + (canvas_height - scaled_height) / 2
        logger.info(f"   📐 Scale: {scale:.3f}, Offset: ({offset_x:.1f}, {offset_y:.1f})")
        optimized_contours = self._llm_optimize_order(contours, analysis, strategy)
        total = len(optimized_contours)
        logger.info(f"   🎨 Drawing {total} contours with adaptive speed...")
        for i, contour in enumerate(optimized_contours):
            try:
                speed = self._get_adaptive_speed(contour, strategy, analysis)
                self._draw_single_contour(contour, offset_x, offset_y, scale, speed)
                if (i + 1) % 50 == 0 or i == total - 1: logger.info(f"   Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            except Exception as e:
                logger.error(f"   ❌ Failed to draw contour {i}: {e}"); continue
        logger.info("   ✅ Drawing completed!")
    def _llm_optimize_order(self, contours: List[np.ndarray], analysis: ImageAnalysis, strategy: DrawingStrategy) -> List[np.ndarray]:
        if len(contours) <= 1: return contours
        baseline_order = self._greedy_tsp(contours)
        if analysis.complexity in [Complexity.COMPLEX, Complexity.VERY_COMPLEX] and analysis.priority_regions:
            logger.info("   🤖 Asking LLM for region-based ordering...")
            try:
                region_groups = self._group_contours_by_region(baseline_order, analysis)
                prompt = f"""Given these drawing regions for a {analysis.subject}:
Priority regions: {', '.join(analysis.priority_regions)}
Current groups: {list(region_groups.keys())}
What order should we draw them? Return JSON: {{ "order": ["region1", "region2", "region3"], "reason": "explanation" }}
Return ONLY JSON."""
                response = self.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are an expert in drawing order optimization."}, {"role": "user", "content": prompt}], max_tokens=200, temperature=0.3, timeout=15.0)
                text = response.choices.message.content.strip()
                if text.startswith("```json"): text = text.replace("```json", "").replace("```", "").strip()
                elif text.startswith("```"): text = text.replace("```", "").strip()
                order_data = json.loads(text); recommended_order = order_data.get("order", [])
                ordered_contours = []
                for region in recommended_order:
                    if region in region_groups: ordered_contours.extend(region_groups[region])
                for region, contours_list in region_groups.items():
                    if region not in recommended_order: ordered_contours.extend(contours_list)
                logger.info(f"   ✅ LLM recommended order: {' → '.join(recommended_order)}"); return ordered_contours
            except Exception as e:
                logger.error(f"   ❌ LLM ordering failed: {e}, using TSP"); return baseline_order
        return baseline_order
    def _group_contours_by_region(self, contours: List[np.ndarray], analysis: ImageAnalysis) -> Dict[str, List[np.ndarray]]:
        groups = {region: [] for region in analysis.priority_regions}; groups["other"] = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cy = M["m01"] / M["m00"]
                if cy < 0.33: region = analysis.priority_regions[0] if len(analysis.priority_regions) > 0 else "other"
                elif cy < 0.66: region = analysis.priority_regions[1] if len(analysis.priority_regions) > 1 else "other"
                else: region = analysis.priority_regions[2] if len(analysis.priority_regions) > 2 else "other"
                if region in groups: groups[region].append(contour)
                else: groups["other"].append(contour)
            else: groups["other"].append(contour)
        return groups
    def _greedy_tsp(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        if len(contours) <= 1: return contours
        centroids = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0: cx = M["m10"] / M["m00"]; cy = M["m01"] / M["m00"]
            else: cx, cy = c[0][0]
            centroids.append((cx, cy))
        unvisited = list(range(len(contours))); current = 0; path = [current]; unvisited.remove(current)
        while unvisited:
            current_pos = centroids[current]; min_dist = float('inf'); nearest = unvisited[0]
            for idx in unvisited:
                pos = centroids[idx]
                dist = ((current_pos[0] - pos[0]) ** 2 + (current_pos[1] - pos[1]) ** 2) ** 0.5
                if dist < min_dist: min_dist = dist; nearest = idx
            path.append(nearest); unvisited.remove(nearest); current = nearest
        return [contours[i] for i in path]
    def _get_adaptive_speed(self, contour: np.ndarray, strategy: DrawingStrategy, analysis: ImageAnalysis) -> float:
        if not strategy.adaptive_speed: return 0.01
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cy = M["m01"] / M["m00"]
            if cy < 0.33 and len(analysis.priority_regions) > 0: region = analysis.priority_regions[0]
            elif cy < 0.66 and len(analysis.priority_regions) > 1: region = analysis.priority_regions[1]
            elif len(analysis.priority_regions) > 2: region = analysis.priority_regions[2]
            else: return 0.01
            return strategy.adaptive_speed.get(region, 0.01)
        return 0.01
    def _draw_single_contour(self, contour: np.ndarray, offset_x: float, offset_y: float, scale: float, speed: float = 0.01):
        if len(contour) < 2: return
        start_x = int(offset_x + contour[0][0][0] * scale); start_y = int(offset_y + contour[0][0][1] * scale)
        pyautogui.moveTo(start_x, start_y, duration=speed)
        pyautogui.mouseDown(button='left')
        for point in contour[1:]:
            x = int(offset_x + point[0][0] * scale); y = int(offset_y + point[0][1] * scale)
            pyautogui.dragTo(x, y, duration=0)
        pyautogui.dragTo(start_x, start_y, duration=0)
        pyautogui.mouseUp(button='left')
        time.sleep(0.01)

class QualityAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("✅ Enhanced OpenAI Quality Agent initialized")
    def evaluate(self, original_path: str, canvas_region: Dict, analysis: ImageAnalysis) -> QualityMetrics:
        logger.info("✅ [QUALITY AGENT] Evaluating with LLM assistance...")
        try:
            screenshot = ImageGrab.grab(bbox=(canvas_region['x'], canvas_region['y'], canvas_region['x'] + canvas_region['width'], canvas_region['y'] + canvas_region['height']))
            screenshot.save("temp_screenshot.png")
            original = cv2.imread(original_path); drawn = cv2.imread("temp_screenshot.png")
            if original is None or drawn is None: return self._fallback_metrics()
            drawn = cv2.resize(drawn, (original.shape[1], original.shape[0]))
            edge_accuracy = self._calculate_edge_accuracy(original, drawn); completeness = self._calculate_completeness(original, drawn); smoothness = self._calculate_smoothness(drawn)
            overall_score = (edge_accuracy * 0.5) + (completeness * 0.4) + (smoothness * 0.1)
            llm_suggestions = self._get_llm_assessment(original_path, "temp_screenshot.png", analysis, overall_score)
            region_scores = self._calculate_region_scores(original, drawn, analysis)
            metrics = QualityMetrics(overall_score=overall_score, edge_accuracy=edge_accuracy, completeness=completeness, smoothness=smoothness, suggestions=llm_suggestions, region_scores=region_scores)
            logger.info(f"   ✅ Overall: {overall_score:.2f}, Edge: {edge_accuracy:.2f}, Complete: {completeness:.2f}")
            logger.info(f"   ✅ Region scores: {region_scores}")
            return metrics
        except Exception as e:
            logger.error(f"   ❌ Quality evaluation failed: {e}"); return self._fallback_metrics()
    def _get_llm_assessment(self, original_path: str, drawn_path: str, analysis: ImageAnalysis, score: float) -> List[str]:
        try:
            with open(original_path, "rb") as f: original_data = base64.b64encode(f.read()).decode('utf-8')
            with open(drawn_path, "rb") as f: drawn_data = base64.b64encode(f.read()).decode('utf-8')
            prompt = f"""Compare these two images:
1. Original image (left)
2. Drawn version (right)
Subject: {analysis.subject}
Quantitative score: {score:.2f}/1.0
Provide 3-5 specific suggestions for improvement. Return JSON: {{ "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"], "strengths": ["strength 1", "strength 2"], "overall_assessment": "brief assessment" }}
Return ONLY JSON."""
            response = self.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_data}", "detail": "low"}}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{drawn_data}", "detail": "low"}}]}], max_tokens=500, temperature=0.3, timeout=30.0)
            text = response.choices[0].message.content.strip()
            if text.startswith("```json"): text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"): text = text.replace("```", "").strip()
            assessment = json.loads(text)
            logger.info(f"   ✅ LLM assessment: {assessment['overall_assessment']}")
            logger.info(f"   ✅ Strengths: {', '.join(assessment['strengths'])}")
            return assessment["suggestions"]
        except Exception as e:
            logger.error(f"   ❌ LLM assessment failed: {e}")
            return ["Could not get LLM assessment"]
    def _calculate_region_scores(self, original: np.ndarray, drawn: np.ndarray, analysis: ImageAnalysis) -> Dict[str, float]:
        height = original.shape; region_scores = {}
        regions = {"top": (0, height // 3), "middle": (height // 3, 2 * height // 3), "bottom": (2 * height // 3, height)}
        for region_name, (start, end) in regions.items():
            orig_region = original[start:end, :]; drawn_region = drawn[start:end, :]
            edge_acc = self._calculate_edge_accuracy(orig_region, drawn_region); completeness = self._calculate_completeness(orig_region, drawn_region)
            region_scores[region_name] = (edge_acc + completeness) / 2
        return region_scores
    def _fallback_metrics(self) -> QualityMetrics:
        return QualityMetrics(0, 0, 0, 0, ["Evaluation failed."], {})
    def _calculate_edge_accuracy(self, original: np.ndarray, drawn: np.ndarray) -> float:
        orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 50, 150)
        drawn_edges = cv2.Canny(cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY), 50, 150)
        intersection = np.sum(cv2.bitwise_and(orig_edges, drawn_edges))
        union = np.sum(cv2.bitwise_or(orig_edges, drawn_edges))
        return float(intersection / (union + 1e-6))
    def _calculate_completeness(self, original: np.ndarray, drawn: np.ndarray) -> float:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY); drawn_gray = cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
        _, orig_thresh = cv2.threshold(orig_gray, 240, 255, cv2.THRESH_BINARY_INV)
        _, drawn_thresh = cv2.threshold(drawn_gray, 240, 255, cv2.THRESH_BINARY_INV)
        orig_pixels = np.sum(orig_thresh > 0); drawn_pixels = np.sum(drawn_thresh > 0)
        return float(min(drawn_pixels / (orig_pixels + 1e-6), 1.0))
    def _calculate_smoothness(self, drawn: np.ndarray) -> float:
        gray = cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(1.0 / (1.0 + laplacian_var / 1000))


# ===============================================================
# SECTION 2: LANGGRAPH IMPLEMENTATION
# This section replaces the MultiAgentOrchestrator class.
# ===============================================================

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    This state is passed between all nodes in the graph.
    """
    image_path: str
    api_key: str
    canvas_region: Optional[Dict]
    analysis: Optional[ImageAnalysis]
    strategy: Optional[DrawingStrategy]
    contours: Optional[List[np.ndarray]]
    image_shape: Optional[Tuple]
    metrics: Optional[QualityMetrics]
    error_message: Optional[str]

class DrawingGraphOrchestrator:
    """
    This class orchestrates the drawing process using a LangGraph StateGraph.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize all agents once
        self.vision_agent = VisionAgent(api_key)
        self.strategy_agent = StrategyAgent(api_key)
        self.image_processor = ImageProcessor(api_key)
        self.execution_agent = ExecutionAgent(api_key)
        self.quality_agent = QualityAgent(api_key)

    # =================== NODE DEFINITIONS ===================
    
    def setup_paint(self, state: GraphState) -> Dict:
        """Node to open MS Paint and define the canvas region."""
        logger.info("\n📍 NODE: Setup Paint")
        try:
            subprocess.Popen("mspaint", shell=True)
            time.sleep(5)
            screen_width, screen_height = pyautogui.size()
            canvas_region = {
                'x': 100, 'y': 200,
                'width': screen_width - 250, 'height': screen_height - 300
            }
            if canvas_region['width'] <= 0 or canvas_region['height'] <= 0:
                return {"error_message": "Invalid canvas size detected."}

            logger.info(f"   ✅ Canvas: {canvas_region['width']}x{canvas_region['height']}")
            pyautogui.click(
                canvas_region['x'] + canvas_region['width'] / 2,
                canvas_region['y'] + canvas_region['height'] / 2
            )
            time.sleep(0.5)
            return {"canvas_region": canvas_region}
        except Exception as e:
            return {"error_message": f"Failed to set up Paint: {e}"}

    def analyze_image(self, state: GraphState) -> Dict:
        """Node for the VisionAgent to analyze the image."""
        logger.info("\n📍 NODE: OpenAI Vision Analysis")
        analysis = self.vision_agent.analyze(state["image_path"])
        return {"analysis": analysis}

    def plan_strategy(self, state: GraphState) -> Dict:
        """Node for the StrategyAgent to create a drawing plan."""
        logger.info("\n📍 NODE: LLM-Enhanced Strategy Planning")
        strategy = self.strategy_agent.plan(state["analysis"])
        return {"strategy": strategy}

    def process_image(self, state: GraphState) -> Dict:
        """Node for the ImageProcessor to find contours."""
        logger.info("\n📍 NODE: LLM-Enhanced Image Processing")
        try:
            contours, image_shape = self.image_processor.process(
                state["image_path"], state["strategy"], state["analysis"]
            )
            return {"contours": contours, "image_shape": image_shape}
        except Exception as e:
            return {"error_message": f"Image processing failed: {e}", "contours": []}

    def execute_drawing(self, state: GraphState) -> Dict:
        """Node for the ExecutionAgent to draw the contours."""
        logger.info("\n📍 NODE: LLM-Guided Drawing Execution")
        logger.info("   ⚠️ Starting in 3 seconds...")
        for i in range(3, 0, -1):
            logger.info(f"      {i}...")
            time.sleep(1)
        
        self.execution_agent.draw_contours(
            state["contours"], state["image_shape"], 
            state["strategy"], state["analysis"], state["canvas_region"]
        )
        return {} # No state change needed

    def evaluate_quality(self, state: GraphState) -> Dict:
        """Node for the QualityAgent to evaluate the final drawing."""
        logger.info("\n📍 NODE: LLM-Enhanced Quality Check")
        metrics = self.quality_agent.evaluate(
            state["image_path"], state["canvas_region"], state["analysis"]
        )
        return {"metrics": metrics}

    def generate_report(self, state: GraphState) -> Dict:
        """Node to generate the final comprehensive report."""
        logger.info("\n" + "="*70)
        logger.info("📊 COMPREHENSIVE DRAWING REPORT")
        logger.info("="*70)

        if state.get("error_message"):
            logger.error(f"   ❌ PROCESS FAILED: {state['error_message']}")
            logger.info("\n" + "="*70)
            return {}

        analysis = state["analysis"]
        strategy = state["strategy"]
        metrics = state["metrics"]
        contour_count = len(state["contours"])
        
        logger.info("\n🔍 IMAGE ANALYSIS:")
        logger.info(f"   Subject: {analysis.subject}")
        logger.info(f"   Style: {analysis.style.value}, Complexity: {analysis.complexity.value}")
        logger.info(f"   Priority Regions: {', '.join(analysis.priority_regions)}")
        
        logger.info("\n🎯 STRATEGY:")
        logger.info(f"   Preprocessing Params: Canny Low={strategy.preprocessing['canny_low']}, Canny High={strategy.preprocessing['canny_high']}")
        logger.info(f"   Contour Params: Min Area={strategy.contour_params['min_area']}, Max Contours={strategy.contour_params['max_contours']}")
        
        logger.info("\n📈 EXECUTION:")
        logger.info(f"   Total Contours Drawn: {contour_count}")
        
        logger.info("\n✅ QUALITY METRICS:")
        logger.info(f"   Overall Score: {metrics.overall_score:.2%}")
        logger.info(f"   Edge Accuracy: {metrics.edge_accuracy:.2%}, Completeness: {metrics.completeness:.2%}")
        
        if metrics.suggestions:
            logger.info("\n💡 LLM SUGGESTIONS FOR IMPROVEMENT:")
            for i, suggestion in enumerate(metrics.suggestions, 1):
                logger.info(f"   {i}. {suggestion}")
        
        logger.info("\n" + "="*70)
        return {}
        
    # =================== CONDITIONAL EDGE ===================
    
    def should_continue_drawing(self, state: GraphState) -> str:
        """
        Determines the next step after image processing.
        If contours are found, proceed to drawing. Otherwise, end the process.
        """
        logger.info("\n📍 DECISION: Check for Contours")
        if state.get("error_message"):
             logger.error(f"   ❌ Error detected: {state['error_message']}. Halting process.")
             return "end_process"
        
        contours = state["contours"]
        if contours and len(contours) > 0:
            logger.info(f"   ✅ Found {len(contours)} contours. Proceeding to drawing.")
            return "continue_to_draw"
        else:
            logger.warning("   ⚠️ No contours found. Skipping drawing and ending process.")
            state["error_message"] = "No contours were found during image processing."
            return "end_process"

    def run(self, image_path: str):
        """
        Builds and runs the LangGraph workflow.
        """
        workflow = StateGraph(GraphState)

        # Add nodes to the graph
        workflow.add_node("setup_paint", self.setup_paint)
        workflow.add_node("analyze_image", self.analyze_image)
        workflow.add_node("plan_strategy", self.plan_strategy)
        workflow.add_node("process_image", self.process_image)
        workflow.add_node("execute_drawing", self.execute_drawing)
        workflow.add_node("evaluate_quality", self.evaluate_quality)
        workflow.add_node("generate_report", self.generate_report)

        # Define the workflow edges
        workflow.set_entry_point("setup_paint")
        workflow.add_edge("setup_paint", "analyze_image")
        workflow.add_edge("analyze_image", "plan_strategy")
        workflow.add_edge("plan_strategy", "process_image")
        
        # Add the conditional edge
        workflow.add_conditional_edges(
            "process_image",
            self.should_continue_drawing,
            {
                "continue_to_draw": "execute_drawing",
                "end_process": "generate_report" # Go directly to report on failure
            }
        )
        
        workflow.add_edge("execute_drawing", "evaluate_quality")
        workflow.add_edge("evaluate_quality", "generate_report")
        workflow.add_edge("generate_report", END)

        # Compile the graph
        app = workflow.compile()

        # Optional: Visualize the graph
        try:
            with open("drawing_graph.png", "wb") as f:
                f.write(app.get_graph().draw_png())
            logger.info("✅ Workflow graph saved to drawing_graph.png")
        except Exception as e:
            logger.warning(f"Could not generate graph visualization: {e}")


        # Run the workflow
        logger.info("\n" + "="*70)
        logger.info(f"🎨 STARTING LANGGRAPH DRAWING: {os.path.basename(image_path)}")
        logger.info("="*70)
        
        initial_state = {"image_path": image_path, "api_key": self.api_key}
        app.invoke(initial_state)

        logger.info("\n" + "="*70)
        logger.info("✅ LANGGRAPH DRAWING COMPLETED!")
        logger.info("="*70)

# ===============================================================
# SECTION 3: MAIN EXECUTION
# ===============================================================

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not found in environment!")
        return
    
    orchestrator = DrawingGraphOrchestrator(api_key)
    
    # Create a dummy image file for the demo if it doesn't exist
    image_dir = "images"
    image_path = os.path.join(image_dir, "doraemon_clean.jpg")
    
    if not os.path.exists(image_path):
        logger.warning(f"Image not found at {image_path}. Please place your image there.")
        # As a placeholder, you could create a simple black image, but it's better for the user to provide one.
        logger.error("Please provide an image named 'doraemon_clean.jpg' inside an 'images' folder.")
        return
    
    orchestrator.run(image_path)

if __name__ == "__main__":
    main()