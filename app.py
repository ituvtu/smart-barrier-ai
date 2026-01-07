import yaml
import cv2
import easyocr
import numpy as np
import gradio as gr
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from typing import Tuple, Optional, Any, Dict
from utils import fuzzy_check
from database import DataManager

MODEL_REPO = "ituvtu/yolo-v10-car-plate-detector"
MODEL_FILENAME = "yolo-v10-m.onnx"

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

CFG = load_config()
data_manager = DataManager()

data_manager.allowed_plates = {"AA0055BP", "KA0132CO", "BO0001OO"}
data_manager.source_info = "Default Test Data"

class BarrierSystem:
    def __init__(self, config: Dict[str, Any]) -> None:
        print("🚀 Initializing Smart Barrier System...")

        self.detector = None
        self.initialization_error = None

        try:
            print(f"📥 Downloading model from {MODEL_REPO}...")
            model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)

            self.detector = YOLO(model_path, task='detect')
            print(f"✅ Model loaded successfully: {model_path}")

        except Exception as error:
            print(f"❌ Failed to initialize model: {error}")
            self.initialization_error = str(error)


        ocr_cfg = config.get("ocr", {})
        self.prep_cfg = config.get("preprocessing", {})
        self.conf_threshold: float = config.get("model", {}).get("conf_threshold", 0.25)
        self.ocr_allowlist: str = ocr_cfg.get("allowlist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self.reader = easyocr.Reader(
            ocr_cfg.get("languages", ['en']),
            gpu=False,
            quantize=False
        )
        print("✅ System Ready!")

    def preprocess_plate(self, img_crop: np.ndarray) -> Optional[np.ndarray]:
        if img_crop is None or img_crop.size == 0:
            return None

        scale = self.prep_cfg.get("scale", 3)
        h, w = img_crop.shape[:2]
        img_resized = cv2.resize(img_crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

        bil_d = self.prep_cfg.get("bilateral", {}).get("d", 11)
        gray_filtered = cv2.bilateralFilter(gray, bil_d, 17, 17)

        pad = self.prep_cfg.get("padding", 30)
        return cv2.copyMakeBorder(gray_filtered, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    def detect_and_read(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        if self.detector is None:
            return None, None, f"Model Error: {self.initialization_error}"

        if image is None:
            return None, None, "No Image"

        try:
            results = self.detector.predict(image, conf=self.conf_threshold, verbose=False)[0]
        except Exception as e:
            return None, None, f"Inference Error: {str(e)}"

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
            allowlist=self.ocr_allowlist,
            min_size=10
        )
        full_text = "".join(text_res) if text_res else "Unreadable"

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        return crop_rgb, processed_rgb, full_text

system = BarrierSystem(CFG)

def _led_html(color: str, text: str, status_icon: str) -> str:
    border_color = color if color != "gray" else "#4b5563"
    glow = f"box-shadow: 0 0 20px {color};" if color != "gray" else ""
    return f"""<div style="display: flex; align-items: center; justify-content: space-between; background: #111827; border: 1px solid {border_color}; border-radius: 8px; padding: 0 20px; height: 60px;"><div style="display: flex; align-items: center;"><div style="width: 14px; height: 14px; background: {color}; border-radius: 50%; {glow} margin-right: 12px; border: 2px solid white;"></div><span style="color: #e5e7eb; font-weight: 600; font-size: 1.1rem; letter-spacing: 0.05em;">{text}</span></div><div style="font-size: 1.5rem;">{status_icon}</div></div>"""

def update_database(file, url, query):
    if file:
        success, msg = data_manager.load_source(file_obj=file)
    elif url:
        success, msg = data_manager.load_source(db_url=url, sql_query=query)
    else:
        return "⚠️ Select a source first"
    return f"✅ {msg}" if success else f"❌ {msg}"

def import_datetime():
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

def pipeline_wrapper(image, region_code, allowed_list_str=""):
    col_wait, col_allow, col_deny = "gray", "#10b981", "#ef4444"

    if image is None:
        return None, None, _led_html(col_wait, "SYSTEM IDLE", "💤"), ""

    crop_orig, crop_proc, raw_text = system.detect_and_read(image)

    if crop_orig is None:
        return None, None, _led_html(col_deny, "NO VEHICLE", "🚫"), f"Status: {raw_text}"

    if allowed_list_str.strip():
        db_string = allowed_list_str
        data_manager.source_info = "Manual Input"
    else:
        db_string = ",".join(data_manager.allowed_plates)
    allowed, clean_num, info = fuzzy_check(raw_text, db_string, region_code)

    log_entry = (
        f"🕒 TIME:   {import_datetime()}\n"
        f"📂 SOURCE: {data_manager.source_info}\n"
        f"📷 RAW:    {raw_text}\n"
        f"🎯 RESULT: {clean_num}\n"
        f"ℹ️ STATUS: {info}"
    )

    if allowed:
        return crop_orig, crop_proc, _led_html(col_allow, "ACCESS GRANTED", "🔓"), log_entry
    else:
        return crop_orig, crop_proc, _led_html(col_deny, "ACCESS DENIED", "🔒"), log_entry

dashboard_css = """
.gradio-container { min-width: 100% !important; }
footer { display: none !important; }
"""

theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(title="Smart Barrier AI", theme=theme, css=dashboard_css) as demo:
    with gr.Row():
        gr.Markdown("## 🛡️ Smart Barrier AI (ONNX Engine)")
        led_status = gr.HTML(_led_html("gray", "SYSTEM READY", "✅"))

    with gr.Row():
        with gr.Column(scale=2):
            input_img = gr.Image(label="Camera", type="numpy", height=500)
        with gr.Column(scale=1):
            out_crop = gr.Image(label="Plate", interactive=False, height=200)
            logs = gr.Textbox(label="Logs", lines=10)

    allow_list_input = gr.Textbox(
        label="✅ Allowed Plates (Whitelist)",
        placeholder="Example: AA0055BP, KA1234BC, BO0001OO",
        value="AA0055BP, KA0132CO, BO0001OO",
        lines=2
    )

    scan_btn = gr.Button("🔍 SCAN", variant="primary")

    country_selector = gr.Dropdown(choices=["UA", "EU"], value="UA", visible=False)

    scan_btn.click(
        pipeline_wrapper,
        inputs=[input_img, country_selector, allow_list_input],
        outputs=[out_crop, gr.Image(visible=False), led_status, logs]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
