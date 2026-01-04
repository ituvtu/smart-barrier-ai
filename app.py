import gradio as gr
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from utils import fuzzy_check

class BarrierSystem:
    def __init__(self, model_path="best.pt"):
        print("🛠️ Ініціалізація...")
        self.detector = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=False, quantize=True) 
        print("✅ Готово!")

    def preprocess_plate(self, img_crop):
        if img_crop is None or img_crop.size == 0: return None
        scale = 3
        h, w = img_crop.shape[:2]
        img = cv2.resize(img_crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        # Трохи більший паддінг для надійності
        return cv2.copyMakeBorder(gray, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)

    def detect_and_read(self, image):
        results = self.detector.predict(image, conf=0.25, verbose=False)[0]
        if not results.boxes: return None, None, "Номер не виявлено"
        
        box = results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = image[y1:y2, x1:x2]
        processed = self.preprocess_plate(crop)
        
        text_res = self.reader.readtext(processed, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        full_text = "".join(text_res) if text_res else "Нечитабельно"
        
        return crop, processed, full_text

system = BarrierSystem("best.pt")

# --- ГОЛОВНИЙ ПАЙПЛАЙН ---
def pipeline(image, db_text, country_mode):
    if image is None: return None, None, _led_html("gray", "Очікування"), ""

    crop_orig, crop_proc, raw_text = system.detect_and_read(image)
    
    if crop_orig is None:
        return None, None, _led_html("red", "Немає авто"), "Детектор не бачить номер"

    # Передаємо обраний режим країни
    allowed, clean_num, info = fuzzy_check(raw_text, db_text, country_mode)
    
    log_text = f"Raw OCR: {raw_text}\nResult:  {clean_num}\nStatus:  {info}"
    
    if allowed:
        return crop_orig, crop_proc, _led_html("#00FF00", "ВІДКРИТО", True), log_text
    else:
        return crop_orig, crop_proc, _led_html("red", "ЗАБОРОНЕНО"), log_text

def _led_html(color, text, shadow=False):
    s = "0 0 20px #00FF00" if shadow else "none"
    return f"""<div style="text-align:center; padding:10px; background:#333; border-radius:10px;">
    <div style="width:50px; height:50px; background:{color}; border-radius:50%; margin:0 auto; box-shadow:{s}; border:2px solid #fff;"></div>
    <h3 style="color:{color}; margin-top:5px;">{text}</h3></div>"""

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🛡️ Розумний Шлагбаум v3 (Multi-Region)")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_img = gr.Image(label="Камера", type="numpy", height=400)
            
            # 🔥 НОВИЙ ЕЛЕМЕНТ: ВИБІР РЕГІОНУ
            country_selector = gr.Radio(
                choices=["Україна 🇺🇦", "Європа 🇪🇺", "Авто 🤖"], 
                value="Україна 🇺🇦", 
                label="Режим обробки",
                info="Оберіть стандарт номерів для кращої точності"
            )

        with gr.Column(scale=1):
            led = gr.HTML(_led_html("gray", "Очікування"))
            with gr.Row():
                out_crop = gr.Image(label="Кроп", height=100)
                out_proc = gr.Image(label="Для OCR", height=100)
            logs = gr.Textbox(label="Логи", lines=4)
            
    db = gr.Textbox(label="База номерів", value="AA0055BP, KA0132CO, BO0001OO")
    btn = gr.Button("Перевірити", variant="primary")
    
    btn.click(pipeline, inputs=[input_img, db, country_selector], outputs=[out_crop, out_proc, led, logs])

if __name__ == "__main__":
    demo.launch()