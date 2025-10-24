"""
HYBRID DRAWING SYSTEM V6.0 - FULL LLM-ENHANCED VERSION
==============================================================
IMPROVEMENTS:
- ✅ All 5 agents now use LLM intelligently
- ✅ StrategyAgent uses LLM for dynamic parameter optimization
- ✅ ExecutionAgent uses LLM for intelligent drawing order
- ✅ ImageProcessor uses LLM for preprocessing guidance
- ✅ Enhanced error handling and fallback mechanisms
- ✅ Real-time LLM-guided adjustments
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
import subprocess

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
pyautogui.PAUSE = 0.01

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
    priority_regions: List[str]  # ✅ NEW
    detail_levels: Dict[str, str]  # ✅ NEW

@dataclass
class DrawingStrategy:
    preprocessing: Dict
    contour_params: Dict
    drawing_order: List[str]
    optimization_level: int
    use_color: bool
    use_shading: bool
    estimated_time: int
    region_strategies: Dict[str, Dict]  # ✅ NEW: Different params per region
    adaptive_speed: Dict[str, float]  # ✅ NEW: Speed per region

@dataclass
class QualityMetrics:
    overall_score: float
    edge_accuracy: float
    completeness: float
    smoothness: float
    suggestions: List[str]
    region_scores: Dict[str, float]  # ✅ NEW

# ============ AGENT 1: ENHANCED VISION AGENT ============

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
    "subject": "detailed description",
    "is_anime": true/false,
    "complexity": "simple/medium/complex/very_complex",
    "main_shapes": ["shape1", "shape2", "shape3"],
    "key_features": ["feature1", "feature2", "feature3", "feature4", "feature5"],
    "estimated_contours": 500,
    "recommended_order": ["part1", "part2", "part3"],
    "priority_regions": ["most_important_part", "second_important", "third"],
    "detail_levels": {
        "face": "very_high",
        "hair": "high",
        "body": "medium",
        "background": "low"
    },
    "color_palette": [[255, 224, 189], [0, 0, 0], [255, 0, 0]],
    "background_color": [255, 255, 255],
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

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}",
                                    "detail": "high"  # ✅ Changed to "high" for better analysis
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,  # ✅ Increased for more detailed response
                temperature=0.1,
                timeout=60.0
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
                is_anime=data.get("is_anime", False),
                priority_regions=data.get("priority_regions", []),
                detail_levels=data.get("detail_levels", {})
            )

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
            is_anime=True,
            priority_regions=["face", "eyes", "hair"],
            detail_levels={"face": "very_high", "hair": "high", "body": "medium"}
        )

# ============ AGENT 2: LLM-ENHANCED STRATEGY AGENT ============

class StrategyAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("🎯 LLM-Enhanced Strategy Agent initialized")
    
    def plan(self, analysis: ImageAnalysis) -> DrawingStrategy:
        logger.info("🎯 [STRATEGY AGENT] Planning with LLM assistance...")
        
        # ✅ Ask LLM for optimal parameters
        strategy_params = self._get_llm_strategy(analysis)
        
        if strategy_params:
            return self._build_strategy_from_llm(strategy_params, analysis)
        else:
            logger.warning("   ⚠️ LLM strategy failed, using enhanced fallback")
            return self._enhanced_fallback_strategy(analysis)
    
    def _get_llm_strategy(self, analysis: ImageAnalysis) -> Optional[Dict]:
        """✅ NEW: Ask LLM for optimal OpenCV parameters"""
        
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
    "global_params": {{
        "blur_kernel": 3,
        "canny_low": 10,
        "canny_high": 40,
        "min_area": 2,
        "epsilon_factor": 0.003,
        "max_contours": 2000
    }},
    "region_strategies": {{
        "face": {{
            "canny_low": 5,
            "canny_high": 25,
            "min_area": 1,
            "priority": 1
        }},
        "hair": {{
            "canny_low": 8,
            "canny_high": 35,
            "min_area": 2,
            "priority": 2
        }},
        "body": {{
            "canny_low": 12,
            "canny_high": 45,
            "min_area": 3,
            "priority": 3
        }}
    }},
    "adaptive_speed": {{
        "face": 0.02,
        "hair": 0.01,
        "body": 0.005
    }},
    "use_clahe": true,
    "use_bilateral": true,
    "use_morphology": true,
    "multi_scale": true
}}

Guidelines:
- Lower Canny thresholds = more details (5-15 for very detailed areas)
- Higher thresholds = cleaner lines (20-60 for simple areas)
- min_area: 1-5 (lower = more small details)
- epsilon_factor: 0.001-0.01 (lower = more accurate contours)
- For anime: prioritize face/eyes with lowest thresholds
- adaptive_speed: slower (0.02-0.05) for detailed areas, faster (0.005-0.01) for simple areas

Return ONLY JSON."""

        try:
            logger.info("   ⏳ Asking LLM for optimal parameters...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in computer vision and OpenCV parameter optimization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.2,
                timeout=30.0
            )
            
            text = response.choices[0].message.content.strip()
            
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
            
            params = json.loads(text)
            logger.info(f"   ✅ LLM recommended: Canny={params['global_params']['canny_low']}-{params['global_params']['canny_high']}")
            logger.info(f"   ✅ Max contours: {params['global_params']['max_contours']}")
            
            return params
            
        except Exception as e:
            logger.error(f"   ❌ LLM strategy request failed: {e}")
            return None
    
    def _build_strategy_from_llm(self, params: Dict, analysis: ImageAnalysis) -> DrawingStrategy:
        """✅ Build strategy from LLM recommendations"""
        
        global_params = params["global_params"]
        
        strategy = DrawingStrategy(
            preprocessing={
                "blur_kernel": global_params["blur_kernel"],
                "canny_low": global_params["canny_low"],
                "canny_high": global_params["canny_high"],
                "use_bilateral": params.get("use_bilateral", True),
                "use_clahe": params.get("use_clahe", True),
                "multi_scale": params.get("multi_scale", True),
                "use_morphology": params.get("use_morphology", True)
            },
            contour_params={
                "min_area": global_params["min_area"],
                "epsilon_factor": global_params["epsilon_factor"],
                "max_contours": global_params["max_contours"],
                "hierarchical": True,
                "filter_nested": True
            },
            drawing_order=analysis.recommended_order,
            optimization_level=3,  # ✅ Highest level with LLM
            use_color=False,
            use_shading=False,
            estimated_time=400,
            region_strategies=params.get("region_strategies", {}),
            adaptive_speed=params.get("adaptive_speed", {})
        )
        
        logger.info("   ✅ Strategy built from LLM recommendations")
        return strategy
    
    def _enhanced_fallback_strategy(self, analysis: ImageAnalysis) -> DrawingStrategy:
        """✅ Enhanced fallback with better defaults"""
        
        complexity_map = {
            Complexity.SIMPLE: (3, 20, 60, 5, 0.005, 800),
            Complexity.MEDIUM: (3, 12, 45, 3, 0.003, 1500),
            Complexity.COMPLEX: (3, 8, 30, 2, 0.002, 2500),
            Complexity.VERY_COMPLEX: (3, 5, 25, 1, 0.001, 4000)
        }
        
        blur, canny_low, canny_high, min_area, epsilon, max_contours = complexity_map.get(
            analysis.complexity, (3, 10, 40, 2, 0.003, 2000)
        )
        
        return DrawingStrategy(
            preprocessing={
                "blur_kernel": blur,
                "canny_low": canny_low,
                "canny_high": canny_high,
                "use_bilateral": True,
                "use_clahe": True,
                "multi_scale": True,
                "use_morphology": True
            },
            contour_params={
                "min_area": min_area,
                "epsilon_factor": epsilon,
                "max_contours": max_contours,
                "hierarchical": True,
                "filter_nested": True
            },
            drawing_order=analysis.recommended_order,
            optimization_level=2,
            use_color=False,
            use_shading=False,
            estimated_time=300,
            region_strategies={},
            adaptive_speed={}
        )

# ============ AGENT 3: LLM-ENHANCED IMAGE PROCESSOR ============

class ImageProcessor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("🖼️ LLM-Enhanced Image Processor initialized")
    
    def process(self, image_path: str, strategy: DrawingStrategy, analysis: ImageAnalysis) -> Tuple[List[np.ndarray], Tuple]:
        """✅ Process image with LLM-guided adjustments"""
        logger.info("   🎨 Processing with LLM-enhanced pipeline...")
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image from {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ✅ CLAHE for better contrast
        if strategy.preprocessing.get("use_clahe", False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            logger.info("      ✓ CLAHE applied")
        
        # ✅ Bilateral filter for edge preservation
        if strategy.preprocessing.get("use_bilateral", False):
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            logger.info("      ✓ Bilateral filter applied")
        else:
            blur_kernel = strategy.preprocessing["blur_kernel"]
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # ✅ Morphological operations
        if strategy.preprocessing.get("use_morphology", False):
            kernel = np.ones((2, 2), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            logger.info("      ✓ Morphological gradient applied")
        
        all_contours = []
        
        # ✅ Multi-scale edge detection
        if strategy.preprocessing.get("multi_scale", False):
            scales = [
                (strategy.preprocessing["canny_low"], strategy.preprocessing["canny_high"]),
                (max(5, strategy.preprocessing["canny_low"] // 2), strategy.preprocessing["canny_high"] // 2),
                (strategy.preprocessing["canny_low"] * 2, min(255, strategy.preprocessing["canny_high"] * 2))
            ]
            
            for i, (low, high) in enumerate(scales):
                edges = cv2.Canny(gray, low, high)
                
                # Dilate edges slightly
                kernel = np.ones((2, 2), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
                
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                filtered = [c for c in contours if cv2.contourArea(c) > strategy.contour_params["min_area"]]
                all_contours.extend(filtered)
                logger.info(f"      ✓ Scale {i+1} ({low}-{high}): Found {len(filtered)} contours")
        else:
            edges = cv2.Canny(gray, strategy.preprocessing["canny_low"], strategy.preprocessing["canny_high"])
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            all_contours = [c for c in contours if cv2.contourArea(c) > strategy.contour_params["min_area"]]
        
        # ✅ Simplify contours
        epsilon_factor = strategy.contour_params["epsilon_factor"]
        simplified = []
        for c in all_contours:
            approx = cv2.approxPolyDP(c, epsilon_factor * cv2.arcLength(c, True), True)
            if len(approx) >= 3:
                simplified.append(approx)
        
        # ✅ Remove duplicates and sort
        simplified = self._remove_duplicate_contours(simplified)
        simplified = sorted(simplified, key=cv2.contourArea, reverse=True)
        
        # ✅ Limit contours
        max_contours = strategy.contour_params["max_contours"]
        if len(simplified) > max_contours:
            simplified = simplified[:max_contours]
        
        logger.info(f"   ✅ Final contours: {len(simplified)}, Image size: {img.shape[1]}x{img.shape[0]}")
        
        # ✅ Ask LLM if we should adjust parameters
        if len(simplified) < 50:
            logger.warning(f"   ⚠️ Only {len(simplified)} contours found, asking LLM for adjustment...")
            adjustment = self._ask_llm_for_adjustment(len(simplified), analysis, strategy)
            if adjustment:
                logger.info("   🔄 Reprocessing with adjusted parameters...")
                return self._reprocess_with_adjustment(image_path, strategy, adjustment)
        
        return simplified, img.shape
    
    def _remove_duplicate_contours(self, contours: List[np.ndarray], threshold: float = 0.95) -> List[np.ndarray]:
        """Remove nearly identical contours"""
        if len(contours) <= 1:
            return contours
        
        unique = []
        for c in contours:
            is_duplicate = False
            for u in unique:
                match = cv2.matchShapes(c, u, cv2.CONTOURS_MATCH_I2, 0)
                if match < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(c)
        
        return unique
    
    def _ask_llm_for_adjustment(self, current_count: int, analysis: ImageAnalysis, strategy: DrawingStrategy) -> Optional[Dict]:
        """✅ Ask LLM if parameters need adjustment"""
        
        prompt = f"""The contour detection found only {current_count} contours, which seems too low.

Image info:
- Subject: {analysis.subject}
- Complexity: {analysis.complexity.value}
- Expected contours: {analysis.estimated_contours}

Current parameters:
- Canny: {strategy.preprocessing['canny_low']}-{strategy.preprocessing['canny_high']}
- Min area: {strategy.contour_params['min_area']}
- Epsilon: {strategy.contour_params['epsilon_factor']}

Should we adjust? Return JSON:
{{
    "should_adjust": true/false,
    "new_canny_low": 5,
    "new_canny_high": 20,
    "new_min_area": 1,
    "reason": "explanation"
}}

Return ONLY JSON."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in computer vision parameter tuning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3,
                timeout=20.0
            )
            
            text = response.choices[0].message.content.strip()
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
            
            adjustment = json.loads(text)
            
            if adjustment.get("should_adjust", False):
                logger.info(f"   ✅ LLM suggests adjustment: {adjustment['reason']}")
                return adjustment
            
            return None
            
        except Exception as e:
            logger.error(f"   ❌ LLM adjustment request failed: {e}")
            return None
    
    def _reprocess_with_adjustment(self, image_path: str, strategy: DrawingStrategy, adjustment: Dict) -> Tuple[List[np.ndarray], Tuple]:
        """Reprocess with adjusted parameters"""
        
        # Update strategy with new parameters
        strategy.preprocessing["canny_low"] = adjustment["new_canny_low"]
        strategy.preprocessing["canny_high"] = adjustment["new_canny_high"]
        strategy.contour_params["min_area"] = adjustment["new_min_area"]
        
        # Reprocess (without LLM check to avoid infinite loop)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if strategy.preprocessing.get("use_clahe", False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        if strategy.preprocessing.get("use_bilateral", False):
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        edges = cv2.Canny(gray, strategy.preprocessing["canny_low"], strategy.preprocessing["canny_high"])
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered = [c for c in contours if cv2.contourArea(c) > strategy.contour_params["min_area"]]
        
        epsilon_factor = strategy.contour_params["epsilon_factor"]
        simplified = []
        for c in filtered:
            approx = cv2.approxPolyDP(c, epsilon_factor * cv2.arcLength(c, True), True)
            if len(approx) >= 3:
                simplified.append(approx)
        
        simplified = sorted(simplified, key=cv2.contourArea, reverse=True)
        max_contours = strategy.contour_params["max_contours"]
        if len(simplified) > max_contours:
            simplified = simplified[:max_contours]
        
        logger.info(f"   ✅ After adjustment: {len(simplified)} contours")
        
        return simplified, img.shape

# ============ AGENT 4: LLM-ENHANCED EXECUTION AGENT ============

class ExecutionAgent:
    def __init__(self, canvas_region: Dict, api_key: str):
        self.canvas_region = canvas_region
        self.client = OpenAI(api_key=api_key)
        logger.info("✏️ LLM-Enhanced Execution Agent initialized")
    
    def draw_contours(
        self, contours: List[np.ndarray], image_shape: Tuple, 
        strategy: DrawingStrategy, analysis: ImageAnalysis
    ):
        logger.info("✏️ [EXECUTION AGENT] Starting LLM-guided drawing...")
        
        img_height, img_width = image_shape[:2]
        canvas_width = self.canvas_region['width']
        canvas_height = self.canvas_region['height']
        
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y) * 0.85
        
        scaled_width = img_width * scale
        scaled_height = img_height * scale
        offset_x = self.canvas_region['x'] + (canvas_width - scaled_width) / 2
        offset_y = self.canvas_region['y'] + (canvas_height - scaled_height) / 2
        
        logger.info(f"   📐 Scale: {scale:.3f}, Offset: ({offset_x:.1f}, {offset_y:.1f})")
        
        # ✅ Ask LLM for intelligent drawing order
        optimized_contours = self._llm_optimize_order(contours, analysis, strategy)
        
        total = len(optimized_contours)
        logger.info(f"   🎨 Drawing {total} contours with adaptive speed...")
        
        for i, contour in enumerate(optimized_contours):
            try:
                # ✅ Use adaptive speed if available
                speed = self._get_adaptive_speed(contour, strategy, analysis)
                self._draw_single_contour(contour, offset_x, offset_y, scale, speed)
                
                if (i + 1) % 50 == 0 or i == total - 1:
                    logger.info(f"   Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            except Exception as e:
                logger.error(f"   ❌ Failed to draw contour {i}: {e}")
                continue
        
        logger.info("   ✅ Drawing completed!")
    
    def _llm_optimize_order(self, contours: List[np.ndarray], analysis: ImageAnalysis, strategy: DrawingStrategy) -> List[np.ndarray]:
        """✅ Ask LLM for intelligent contour ordering"""
        
        if len(contours) <= 1:
            return contours
        
        # Use greedy TSP as baseline
        baseline_order = self._greedy_tsp(contours)
        
        # For complex images, ask LLM for region-based ordering
        if analysis.complexity in [Complexity.COMPLEX, Complexity.VERY_COMPLEX] and analysis.priority_regions:
            logger.info("   🤖 Asking LLM for region-based ordering...")
            
            try:
                # Group contours by region (simplified - in production, use actual region detection)
                region_groups = self._group_contours_by_region(baseline_order, analysis)
                
                prompt = f"""Given these drawing regions for a {analysis.subject}:
Priority regions: {', '.join(analysis.priority_regions)}
Current groups: {list(region_groups.keys())}

What order should we draw them? Return JSON:
{{
    "order": ["region1", "region2", "region3"],
    "reason": "explanation"
}}

Return ONLY JSON."""

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert in drawing order optimization."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3,
                    timeout=15.0
                )
                
                text = response.choices[0].message.content.strip()
                if text.startswith("```json"):
                    text = text.replace("```json", "").replace("```", "").strip()
                elif text.startswith("```"):
                    text = text.replace("```", "").strip()
                
                order_data = json.loads(text)
                recommended_order = order_data.get("order", [])
                
                # Reorder contours based on LLM recommendation
                ordered_contours = []
                for region in recommended_order:
                    if region in region_groups:
                        ordered_contours.extend(region_groups[region])
                
                # Add any remaining contours
                for region, contours_list in region_groups.items():
                    if region not in recommended_order:
                        ordered_contours.extend(contours_list)
                
                logger.info(f"   ✅ LLM recommended order: {' → '.join(recommended_order)}")
                return ordered_contours
                
            except Exception as e:
                logger.error(f"   ❌ LLM ordering failed: {e}, using TSP")
                return baseline_order
        
        return baseline_order
    
    def _group_contours_by_region(self, contours: List[np.ndarray], analysis: ImageAnalysis) -> Dict[str, List[np.ndarray]]:
        """Simple region grouping based on vertical position"""
        
        groups = {region: [] for region in analysis.priority_regions}
        groups["other"] = []
        
        # Simple heuristic: divide image into vertical thirds
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cy = M["m01"] / M["m00"]
                
                # Normalize to 0-1
                # (In production, use actual region detection)
                if cy < 0.33:
                    region = analysis.priority_regions[0] if len(analysis.priority_regions) > 0 else "other"
                elif cy < 0.66:
                    region = analysis.priority_regions[1] if len(analysis.priority_regions) > 1 else "other"
                else:
                    region = analysis.priority_regions[2] if len(analysis.priority_regions) > 2 else "other"
                
                if region in groups:
                    groups[region].append(contour)
                else:
                    groups["other"].append(contour)
            else:
                groups["other"].append(contour)
        
        return groups
    
    def _greedy_tsp(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Greedy nearest neighbor TSP"""
        
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
        
        return [contours[i] for i in path]
    
    def _get_adaptive_speed(self, contour: np.ndarray, strategy: DrawingStrategy, analysis: ImageAnalysis) -> float:
        """Get adaptive drawing speed based on region"""
        
        if not strategy.adaptive_speed:
            return 0.01  # Default speed
        
        # Determine which region this contour belongs to
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cy = M["m01"] / M["m00"]
            
            # Simple heuristic
            if cy < 0.33 and len(analysis.priority_regions) > 0:
                region = analysis.priority_regions[0]
            elif cy < 0.66 and len(analysis.priority_regions) > 1:
                region = analysis.priority_regions[1]
            elif len(analysis.priority_regions) > 2:
                region = analysis.priority_regions[2]
            else:
                return 0.01
            
            return strategy.adaptive_speed.get(region, 0.01)
        
        return 0.01
    
    def _draw_single_contour(self, contour: np.ndarray, offset_x: float, offset_y: float, scale: float, speed: float = 0.01):
        """Draw contour with adaptive speed using dragTo for reliability"""
        
        if len(contour) < 2:
            return

        # Lấy điểm bắt đầu
        start_x = int(offset_x + contour[0][0][0] * scale)
        start_y = int(offset_y + contour[0][0][1] * scale)

        # Di chuyển đến điểm bắt đầu mà không vẽ
        pyautogui.moveTo(start_x, start_y, duration=speed)
        
        # Nhấn giữ chuột trái để bắt đầu vẽ
        pyautogui.mouseDown(button='left')

        # Dùng dragTo để vẽ các điểm tiếp theo trong đường viền
        for point in contour[1:]:
            x = int(offset_x + point[0][0] * scale)
            y = int(offset_y + point[0][1] * scale)
            # THAY ĐỔI QUAN TRỌNG: Dùng dragTo thay vì moveTo
            pyautogui.dragTo(x, y, duration=0)

        # Kéo về điểm bắt đầu để khép kín đường viền (nếu cần)
        pyautogui.dragTo(start_x, start_y, duration=0)
        
        # Thả chuột trái để hoàn thành
        pyautogui.mouseUp(button='left')
        time.sleep(0.01)
# ============ AGENT 5: ENHANCED QUALITY AGENT ============

class QualityAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logger.info("✅ Enhanced OpenAI Quality Agent initialized")
    
    def evaluate(self, original_path: str, canvas_region: Dict, analysis: ImageAnalysis) -> QualityMetrics:
        logger.info("✅ [QUALITY AGENT] Evaluating with LLM assistance...")
        
        try:
            screenshot = ImageGrab.grab(bbox=(
                canvas_region['x'], canvas_region['y'],
                canvas_region['x'] + canvas_region['width'],
                canvas_region['y'] + canvas_region['height']
            ))
            screenshot.save("temp_screenshot.png")
            
            original = cv2.imread(original_path)
            drawn = cv2.imread("temp_screenshot.png")
            
            if original is None or drawn is None:
                return self._fallback_metrics()
            
            drawn = cv2.resize(drawn, (original.shape[1], original.shape[0]))
            
            # Calculate metrics
            edge_accuracy = self._calculate_edge_accuracy(original, drawn)
            completeness = self._calculate_completeness(original, drawn)
            smoothness = self._calculate_smoothness(drawn)
            
            overall_score = (edge_accuracy * 0.5) + (completeness * 0.4) + (smoothness * 0.1)
            
            # ✅ Ask LLM for qualitative assessment
            llm_suggestions = self._get_llm_assessment(original_path, "temp_screenshot.png", analysis, overall_score)
            
            # ✅ Calculate region scores
            region_scores = self._calculate_region_scores(original, drawn, analysis)
            
            metrics = QualityMetrics(
                overall_score=overall_score,
                edge_accuracy=edge_accuracy,
                completeness=completeness,
                smoothness=smoothness,
                suggestions=llm_suggestions,
                region_scores=region_scores
            )
            
            logger.info(f"   ✅ Overall: {overall_score:.2f}, Edge: {edge_accuracy:.2f}, Complete: {completeness:.2f}")
            logger.info(f"   ✅ Region scores: {region_scores}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"   ❌ Quality evaluation failed: {e}")
            return self._fallback_metrics()

    def _get_llm_assessment(self, original_path: str, drawn_path: str, analysis: ImageAnalysis, score: float) -> List[str]:
        """✅ Get LLM's qualitative assessment"""
        
        try:
            # Encode both images
            with open(original_path, "rb") as f:
                original_data = base64.b64encode(f.read()).decode('utf-8')
            with open(drawn_path, "rb") as f:
                drawn_data = base64.b64encode(f.read()).decode('utf-8')
            
            prompt = f"""Compare these two images:
1. Original image (left)
2. Drawn version (right)

Subject: {analysis.subject}
Quantitative score: {score:.2f}/1.0

Provide 3-5 specific suggestions for improvement. Return JSON:
{{
    "suggestions": [
        "suggestion 1",
        "suggestion 2",
        "suggestion 3"
    ],
    "strengths": ["strength 1", "strength 2"],
    "overall_assessment": "brief assessment"
}}

Return ONLY JSON."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{original_data}", "detail": "low"}
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{drawn_data}", "detail": "low"}
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.3,
                timeout=30.0
            )
            
            text = response.choices[0].message.content.strip()
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
            
            assessment = json.loads(text)
            
            logger.info(f"   ✅ LLM assessment: {assessment['overall_assessment']}")
            logger.info(f"   ✅ Strengths: {', '.join(assessment['strengths'])}")
            
            return assessment["suggestions"]
            
        except Exception as e:
            logger.error(f"   ❌ LLM assessment failed: {e}")
            return ["Could not get LLM assessment"]

    def _calculate_region_scores(self, original: np.ndarray, drawn: np.ndarray, analysis: ImageAnalysis) -> Dict[str, float]:
        """Calculate scores for different regions"""
        
        height = original.shape[0]
        region_scores = {}
        
        # Simple vertical division
        regions = {
            "top": (0, height // 3),
            "middle": (height // 3, 2 * height // 3),
            "bottom": (2 * height // 3, height)
        }
        
        for region_name, (start, end) in regions.items():
            orig_region = original[start:end, :]
            drawn_region = drawn[start:end, :]
            
            edge_acc = self._calculate_edge_accuracy(orig_region, drawn_region)
            completeness = self._calculate_completeness(orig_region, drawn_region)
            
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

# ============ ORCHESTRATOR ============

class MultiAgentOrchestrator:
    def __init__(self, api_key: str):
        # FIX 1: Store api_key as an instance attribute so it can be used in other methods.
        self.api_key = api_key
        
        # Now, consistently use self.api_key to initialize all agents.
        self.vision_agent = VisionAgent(self.api_key)
        self.strategy_agent = StrategyAgent(self.api_key)
        self.image_processor = ImageProcessor(self.api_key)
        self.quality_agent = QualityAgent(self.api_key)
        
        # FIX 2: Set execution_agent to None initially.
        # It cannot be fully initialized until the canvas region is known.
        self.execution_agent = None
        logger.info("🎭 Multi-Agent Orchestrator V6.0 (Full LLM) initialized")
    
    def draw(self, image_path: str):
        logger.info("\n" + "="*70)
        logger.info(f"🎨 STARTING LLM-ENHANCED DRAWING: {os.path.basename(image_path)}")
        logger.info("="*70)
        
        logger.info("\n📍 PHASE 1: Setup Paint")
        canvas_region = self._setup_paint()
        if not canvas_region:
            return
            
        # FIX 3: Correctly initialize ExecutionAgent here using the stored self.api_key.
        self.execution_agent = ExecutionAgent(canvas_region, self.api_key)
        
        logger.info("\n📍 PHASE 2: OpenAI Vision Analysis")
        analysis = self.vision_agent.analyze(image_path)
        
        logger.info("\n📍 PHASE 3: LLM-Enhanced Strategy Planning")
        strategy = self.strategy_agent.plan(analysis)
        
        logger.info("\n📍 PHASE 4: LLM-Enhanced Image Processing")
        try:
            contours, image_shape = self.image_processor.process(image_path, strategy, analysis)
        except Exception as e:
            logger.error(f"   ❌ Processing failed: {e}")
            return

        if not contours:
            logger.error("   ❌ No contours found!")
            return

        logger.info("\n📍 PHASE 5: LLM-Guided Drawing Execution")
        logger.info("   ⚠️ Starting in 3 seconds...")
        for i in range(3, 0, -1):
            logger.info(f"      {i}...")
            time.sleep(1)
        
        self.execution_agent.draw_contours(contours, image_shape, strategy, analysis)
        
        logger.info("\n📍 PHASE 6: LLM-Enhanced Quality Check")
        metrics = self.quality_agent.evaluate(image_path, canvas_region, analysis)
        
        logger.info("\n📍 PHASE 7: Comprehensive Report")
        self._generate_report(analysis, strategy, metrics, len(contours))
        
        logger.info("\n" + "="*70)
        logger.info("✅ LLM-ENHANCED DRAWING COMPLETED!")
        logger.info("="*70)
    
    def _setup_paint(self) -> Optional[Dict]:
        logger.info("   🎨 Opening Microsoft Paint...")
        subprocess.Popen("mspaint", shell=True)
        time.sleep(5)
        
        screen_width, screen_height = pyautogui.size()
        
        canvas_region = {
            'x': 100,
            'y': 200,
            'width': screen_width - 250,
            'height': screen_height - 300
        }
        
        if canvas_region['width'] <= 0 or canvas_region['height'] <= 0:
            logger.error("   ❌ Invalid canvas size!")
            return None

        logger.info(f"   ✅ Canvas: {canvas_region['width']}x{canvas_region['height']}")
        
        pyautogui.click(
            canvas_region['x'] + canvas_region['width'] / 2,
            canvas_region['y'] + canvas_region['height'] / 2
        )
        time.sleep(0.5)
        
        return canvas_region
    
    def _generate_report(self, analysis: ImageAnalysis, strategy: DrawingStrategy, metrics: QualityMetrics, contour_count: int):
        """Generate comprehensive report"""
        
        logger.info("\n" + "="*70)
        logger.info("📊 COMPREHENSIVE DRAWING REPORT")
        logger.info("="*70)
        
        logger.info("\n🔍 IMAGE ANALYSIS:")
        logger.info(f"   Subject: {analysis.subject}")
        logger.info(f"   Style: {analysis.style.value}")
        logger.info(f"   Complexity: {analysis.complexity.value}")
        logger.info(f"   Is Anime: {analysis.is_anime}")
        logger.info(f"   Priority Regions: {', '.join(analysis.priority_regions)}")
        logger.info(f"   Detail Levels: {json.dumps(analysis.detail_levels, indent=6)}")
        
        logger.info("\n🎯 STRATEGY:")
        logger.info(f"   Preprocessing: {json.dumps(strategy.preprocessing, indent=6)}")
        logger.info(f"   Contour Params: {json.dumps(strategy.contour_params, indent=6)}")
        logger.info(f"   Optimization Level: {strategy.optimization_level}")
        logger.info(f"   Drawing Order: {' → '.join(strategy.drawing_order)}")
        if strategy.region_strategies:
            logger.info(f"   Region Strategies: {json.dumps(strategy.region_strategies, indent=6)}")
        if strategy.adaptive_speed:
            logger.info(f"   Adaptive Speed: {json.dumps(strategy.adaptive_speed, indent=6)}")
        
        logger.info("\n📈 EXECUTION:")
        logger.info(f"   Total Contours Drawn: {contour_count}")
        logger.info(f"   Estimated Time: {strategy.estimated_time}s")
        
        logger.info("\n✅ QUALITY METRICS:")
        logger.info(f"   Overall Score: {metrics.overall_score:.2%}")
        logger.info(f"   Edge Accuracy: {metrics.edge_accuracy:.2%}")
        logger.info(f"   Completeness: {metrics.completeness:.2%}")
        logger.info(f"   Smoothness: {metrics.smoothness:.2%}")
        
        if metrics.region_scores:
            logger.info("\n   Region Scores:")
            for region, score in metrics.region_scores.items():
                logger.info(f"      {region}: {score:.2%}")
        
        if metrics.suggestions:
            logger.info("\n💡 LLM SUGGESTIONS FOR IMPROVEMENT:")
            for i, suggestion in enumerate(metrics.suggestions, 1):
                logger.info(f"   {i}. {suggestion}")
        
        logger.info("\n" + "="*70)
# ============ MAIN ============

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not found in environment!")
        return
    
    orchestrator = MultiAgentOrchestrator(api_key)
    
    # Example usage
    image_path = "images/luffy.jpg"  # Replace with your image
    
    if not os.path.exists(image_path):
        logger.error(f"❌ Image not found: {image_path}")
        return
    
    orchestrator.draw(image_path)

if __name__ == "__main__":
    main()
