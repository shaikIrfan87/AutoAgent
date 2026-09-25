import os
import subprocess
import tempfile
import time
import ctypes
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image, ImageGrab
import numpy as np


class NativeWinRTOCREngine:
    """
    Zero-LLM, zero-external-binary in-process OCR engine using Windows native Windows.Media.Ocr.
    Recognizes raster text inside opaque canvas/Electron UI elements without downloading Tesseract or neural weights.
    """

    @staticmethod
    def recognize_text(image: Any) -> str:
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            return ""

        temp_path = os.path.join(tempfile.gettempdir(), f"ocr_{time.time_ns()}.png")
        try:
            pil_img.save(temp_path)
            ps_script = f"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }})[0]
Function Await($WinRtTask, $ResultType) {{
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(4000) | Out-Null
    $netTask.Result
}}
[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync('{temp_path}')) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
$ocrResult = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
Write-Output $ocrResult.Text
"""
            res = subprocess.run(
                ["C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=6,
            )
            return res.stdout.strip()
        except Exception:
            return ""
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


class VisualGeometryParser:
    """
    Zero-LLM geometric UI parser. Identifies buttons, input bars,
    and clickable rectangular regions via morphological edge contours.
    """


    @staticmethod
    def extract_bounding_boxes(frame_rgb: np.ndarray, min_area: int = 500, max_area: int = 150000) -> List[Dict[str, Any]]:
        try:
            import cv2
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            abs_grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)

            _, thresh = cv2.threshold(abs_grad, 40, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                aspect_ratio = w / float(max(h, 1))

                if min_area <= area <= max_area and 1.2 <= aspect_ratio <= 12.0:
                    candidates.append({
                        "id": f"contour_{x}_{y}",
                        "name": f"VisualRegion_{x}_{y}",
                        "control_type": "GeometricElement",
                        "bbox": (x, y, x + w, y + h),
                        "center": (x + (w // 2), y + (h // 2)),
                        "clickable": True
                    })
            return candidates
        except ImportError:
            # ponytail: stdlib/numpy fallback for gradient bounding boxes when cv2 is omitted
            if frame_rgb.ndim != 3 or frame_rgb.size == 0:
                return []
            gray = np.mean(frame_rgb[::4, ::4], axis=-1)
            grad = np.abs(np.diff(gray, axis=0)[:, :-1]) + np.abs(np.diff(gray, axis=1)[:-1, :])
            edges = grad > 30
            y_indices, x_indices = np.where(edges)
            if len(x_indices) < 10:
                return []
            x1, x2 = int(np.min(x_indices) * 4), int(np.max(x_indices) * 4)
            y1, y2 = int(np.min(y_indices) * 4), int(np.max(y_indices) * 4)
            return [{
                "id": f"numpy_region_{x1}_{y1}",
                "name": f"VisualRegion_{x1}_{y1}",
                "control_type": "GeometricElement",
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "clickable": True
            }]


from enum import Enum


class AffordanceType(str, Enum):
    ACTION_TRIGGER = "action_trigger"         # Clickable button, action icon
    TEXT_INPUT = "text_input"                 # Editable text field, search bar
    NAVIGATION_CONTROL = "navigation_control" # Close, back, tab, header nav
    SCROLL_CONTAINER = "scroll_container"     # Scrollable view pane
    CONTENT_CONTAINER = "content_container"   # Display / background pane


class SemanticAffordanceClassifier:
    """
    Classifies visual regions and bounding boxes into semantic affordances
    based on spatial geometry, aspect ratios, relative screen placement,
    and control types without requiring heavy external VLM dependencies.
    """

    @staticmethod
    def classify_region(
        bbox: Tuple[int, int, int, int],
        screen_size: Tuple[int, int] = (1920, 1080),
        control_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> AffordanceType:
        if control_type in ("ButtonControl", "MenuItemControl", "Hyperlink"):
            return AffordanceType.ACTION_TRIGGER
        if control_type in ("EditControl", "DocumentControl"):
            return AffordanceType.TEXT_INPUT
        if control_type in ("ScrollBarControl", "PaneControl") and ("scroll" in (name or "").lower()):
            return AffordanceType.SCROLL_CONTAINER

        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        ar = w / float(h)
        scr_w, scr_h = screen_size

        # Navigation controls: small icons near edges
        if area < 15000 and 0.7 <= ar <= 1.4:
            if y1 < 0.15 * scr_h or x1 < 0.10 * scr_w or x2 > 0.90 * scr_w:
                return AffordanceType.NAVIGATION_CONTROL

        # Text inputs: elongated horizontal bars
        if ar >= 3.5 and 20 <= h <= 80 and area < 80000:
            return AffordanceType.TEXT_INPUT

        # Action triggers (buttons): compact rectangles
        if 1.2 <= ar <= 5.5 and 25 <= h <= 95 and area < 65000:
            return AffordanceType.ACTION_TRIGGER

        # Scroll containers: large surface or tall column
        if area > 100000 or (h > 0.4 * scr_h and ar < 0.4):
            return AffordanceType.SCROLL_CONTAINER

        return AffordanceType.CONTENT_CONTAINER


class ScreenPerceptionEngine:
    """
    Deterministic visual sensor extracting screen layout trees, bounding boxes,
    semantic affordances, and visual change deltas without external VLM calls.
    """
    def __init__(self, sample_step: int = 8):
        self.sample_step = sample_step
        self.last_frame: Optional[np.ndarray] = None
        self.geom_parser = VisualGeometryParser()
        self.affordance_classifier = SemanticAffordanceClassifier()
        self.ocr = NativeWinRTOCREngine()
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


    def capture_frame(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Captures active screen buffer into an RGB numpy matrix with headless/service fallback."""
        try:
            screenshot = ImageGrab.grab()
            frame = np.array(screenshot)
            return frame, screenshot.size
        except Exception:
            # Fallback for headless, locked screen, or disconnected desktop DC
            width = ctypes.windll.user32.GetSystemMetrics(0) or 1920
            height = ctypes.windll.user32.GetSystemMetrics(1) or 1080
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            return frame, (width, height)

    def compute_screen_delta(self, current_frame: np.ndarray) -> float:
        """
        Calculates pixel divergence against the previous frame.
        Guarantees sub-millisecond check to prevent compute allocation when static.
        """
        if self.last_frame is None or self.last_frame.shape != current_frame.shape:
            self.last_frame = current_frame
            return 1.0

        step = self.sample_step
        diff = np.abs(
            current_frame[::step, ::step, :3].astype(np.int16) - 
            self.last_frame[::step, ::step, :3].astype(np.int16)
        )
        delta_ratio = float(np.count_nonzero(diff > 25)) / max(diff.size, 1)
        self.last_frame = current_frame
        return delta_ratio

    def wait_for_settle(self, max_wait_sec: float = 0.45, interval_sec: float = 0.05, threshold: float = 0.005) -> float:
        """
        Adaptive settling loop: samples frames at interval_sec until delta stabilizes,
        avoiding hardcoded sleeps and absorbing UI fade/flyout animations.
        """
        t0 = time.time()
        prev_delta = 1.0
        total_shift = 0.0
        while time.time() - t0 < max_wait_sec:
            time.sleep(interval_sec)
            frame, _ = self.capture_frame()
            d = self.compute_screen_delta(frame)
            total_shift = max(total_shift, d)
            if abs(d - prev_delta) < threshold and d < threshold:
                break
            prev_delta = d
        return total_shift

    def inspect_ui_tree(self) -> List[Dict[str, Any]]:
        """
        Inspects live OS accessibility tree via Windows UIAutomation.
        Extracts interactable controls, bounding rectangles, and automation IDs.
        Falls back to geometric contour parsing if UIAutomation tree is sparse.
        """
        elements: List[Dict[str, Any]] = []
        try:
            import uiautomation as uia
            root = uia.GetRootControl()
            for ctrl, depth in uia.WalkControl(root, maxDepth=3):
                rect = ctrl.BoundingRectangle
                if rect.width() > 0 and rect.height() > 0 and ctrl.IsEnabled:
                    elements.append({
                        "id": ctrl.AutomationId or ctrl.Name or f"elem_{len(elements)}",
                        "name": ctrl.Name or "",
                        "control_type": ctrl.ControlTypeName,
                        "bbox": (rect.left, rect.top, rect.right, rect.bottom),
                        "center": ((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2),
                        "clickable": ctrl.IsInvokePatternAvailable() or ctrl.ControlTypeName in ["ButtonControl", "MenuItemControl"]
                    })
        except Exception:
            pass

        if not elements:
            # Fallback 1: Geometric contour detection
            frame, _ = self.capture_frame()
            elements = self.geom_parser.extract_bounding_boxes(frame)

        if not elements:
            # Fallback 2: Root display canvas
            width = ctypes.windll.user32.GetSystemMetrics(0) or 1920
            height = ctypes.windll.user32.GetSystemMetrics(1) or 1080
            elements.append({
                "id": "root_display",
                "name": "DisplayCanvas",
                "control_type": "Window",
                "bbox": (0, 0, width, height),
                "center": (width // 2, height // 2),
                "clickable": True
            })

        # Enrich all discovered elements with semantic affordance classifications
        scr_w = ctypes.windll.user32.GetSystemMetrics(0) or 1920
        scr_h = ctypes.windll.user32.GetSystemMetrics(1) or 1080
        for elem in elements:
            aff = self.affordance_classifier.classify_region(
                elem["bbox"],
                screen_size=(scr_w, scr_h),
                control_type=elem.get("control_type"),
                name=elem.get("name"),
            )
            elem["affordance"] = aff.value

        return elements

    def find_by_affordance(self, affordance: Any) -> List[Dict[str, Any]]:
        """Finds visible UI elements matching the specified semantic affordance."""
        target_val = affordance.value if hasattr(affordance, "value") else str(affordance).lower()
        elements = self.inspect_ui_tree()
        return [e for e in elements if e.get("affordance") == target_val]

    def read_text_at(self, bbox: Tuple[int, int, int, int], frame_rgb: Optional[np.ndarray] = None) -> str:
        """Reads text within bbox using Windows native zero-binary WinRT OCR."""
        x1, y1, x2, y2 = bbox
        if frame_rgb is None:
            frame_rgb, _ = self.capture_frame()
        h, w = frame_rgb.shape[:2]
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return ""
        cropped = frame_rgb[y1:y2, x1:x2]
        return self.ocr.recognize_text(cropped)

    def find_by_text(self, text: str, exact: bool = False, frame_rgb: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Finds elements in the UI tree or canvas whose accessible name or OCR text matches query."""
        if frame_rgb is None:
            frame_rgb, _ = self.capture_frame()
        elements = self.inspect_ui_tree()
        matching = []
        target = text.strip() if exact else text.strip().lower()
        for elem in elements:
            name = elem.get("name", "")
            elem_text = name if exact else name.lower()
            if (exact and target == elem_text) or (not exact and target in elem_text):
                matching.append(elem)
                continue
            if elem.get("bbox"):
                ocr_txt = self.read_text_at(elem["bbox"], frame_rgb=frame_rgb)
                check_txt = ocr_txt if exact else ocr_txt.lower()
                if (exact and target == check_txt) or (not exact and target in check_txt):
                    elem["ocr_text"] = ocr_txt
                    matching.append(elem)
        return matching

