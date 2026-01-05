import yaml
import cv2
import easyocr
import numpy as np
import gradio as gr
from ultralytics import YOLO
from typing import Tuple, Optional, Any, Dict
from utils import fuzzy_check

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ File {path} not found! Using defaults.")
        return {}

CFG = load_config()

class BarrierSystem:
    def __init__(self, config: Dict[str, Any]) -> None:
        print("🛠️ System Initialization...")
        
        model_cfg = config.get("model", {})
        model_path = model_cfg.get("path", "best.pt")
        
        task = model_cfg.get("task", "detect")
        
        ocr_cfg = config.get("ocr", {})
        
        self.conf_threshold: float = model_cfg.get("conf_threshold", 0.25)
        self.prep_cfg = config.get("preprocessing", {})
        self.ocr_allowlist: str = ocr_cfg.get("allowlist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        self.detector = YOLO(model_path, task=task)
        
        self.reader = easyocr.Reader(
            ocr_cfg.get("languages", ['en']),
            gpu=ocr_cfg.get("use_gpu", False),
            quantize=ocr_cfg.get("quantize", True)
        )
        print("\n✅ System Ready!")

    def preprocess_plate(self, img_crop: np.ndarray) -> Optional[np.ndarray]:
        if img_crop is None or img_crop.size == 0:
            return None

        scale = self.prep_cfg.get("scale", 3)
        pad = self.prep_cfg.get("padding", 30)
        bil_d = self.prep_cfg.get("bilateral", {}).get("d", 11)
        bil_sc = self.prep_cfg.get("bilateral", {}).get("sigma_color", 17)
        bil_ss = self.prep_cfg.get("bilateral", {}).get("sigma_space", 17)

        h, w = img_crop.shape[:2]
        img_resized = cv2.resize(img_crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        gray_filtered = cv2.bilateralFilter(gray, bil_d, bil_sc, bil_ss)
        
        return cv2.copyMakeBorder(
            gray_filtered, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
        )

    def detect_and_read(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        results = self.detector.predict(image, conf=self.conf_threshold, verbose=False)[0]
        
        if not results.boxes:
            return None, None, "Plate not detected"
        
        box = results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        crop = image[y1:y2, x1:x2]
        processed = self.preprocess_plate(crop)
        
        if processed is None:
            return crop, None, "Processing error"

        text_res = self.reader.readtext(
            processed, 
            detail=0, 
            allowlist=self.ocr_allowlist
        )
        full_text = "".join(text_res) if text_res else "Unreadable"
        
        return crop, processed, full_text

system = BarrierSystem(CFG)

def _led_html(color: str, text: str, shadow: bool = False) -> str:
    s = f"0 0 20px {color}" if shadow else "none"
    return f"""
    <div style="text-align:center; padding:10px; background:#333; border-radius:10px;">
        <div style="width:50px; height:50px; background:{color}; border-radius:50%; margin:0 auto; box-shadow:{s}; border:2px solid #fff;"></div>
        <h3 style="color:{color}; margin-top:5px;">{text}</h3>
    </div>
    """

def pipeline(
    image: np.ndarray, 
    db_text: str, 
    country_mode: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, str]:
    
    led_colors = CFG.get("app", {}).get("led_colors", {})
    col_wait = led_colors.get("wait", "gray")
    col_allow = led_colors.get("allowed", "#00FF00")
    col_deny = led_colors.get("denied", "#FF0000")

    if image is None:
        return None, None, _led_html(col_wait, "Waiting"), ""

    crop_orig, crop_proc, raw_text = system.detect_and_read(image)
    
    if crop_orig is None:
        return None, None, _led_html(col_deny, "No vehicle"), "Detector failed"

    allowed, clean_num, info = fuzzy_check(raw_text, db_text, country_mode)
    
    log_text = f"Raw OCR: {raw_text}\nResult:  {clean_num}\nStatus:  {info}"
    
    if allowed:
        return crop_orig, crop_proc, _led_html(col_allow, "ACCESS GRANTED", True), log_text
    else:
        return crop_orig, crop_proc, _led_html(col_deny, "ACCESS DENIED"), log_text

with gr.Blocks(theme=CFG.get("app", {}).get("theme", "soft")) as demo:
    gr.Markdown(f"## {CFG.get('app', {}).get('title', 'AI Barrier')}")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_img = gr.Image(label="Camera Feed", type="numpy", height=400)
            country_selector = gr.Radio(
                choices=["Ukraine 🇺🇦", "Europe 🇪🇺", "Auto 🤖"], 
                value="Ukraine 🇺🇦", 
                label="Region Mode",
                info="Select plate standard for better accuracy"
            )

        with gr.Column(scale=1):
            led_init_col = CFG.get("app", {}).get("led_colors", {}).get("wait", "gray")
            led = gr.HTML(_led_html(led_init_col, "Waiting"))
            
            with gr.Row():
                out_crop = gr.Image(label="Crop", height=100)
                out_proc = gr.Image(label="Processed", height=100)
            logs = gr.Textbox(label="System Logs", lines=4)
            
    db = gr.Textbox(label="Allowed Database", value="AA0055BP, KA0132CO, BO0001OO")
    btn = gr.Button("Check Access", variant="primary")
    
    btn.click(
        pipeline, 
        inputs=[input_img, db, country_selector], 
        outputs=[out_crop, out_proc, led, logs]
    )

if __name__ == "__main__":
    demo.launch()