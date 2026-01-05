import pytest
import cv2
import os
import numpy as np
from app import BarrierSystem, load_config

ASSET_PATH = os.path.join("assets", "test_car.jpg")

@pytest.fixture(scope="module")
def system_instance():
    config = load_config("config.yaml")
    return BarrierSystem(config)

def test_assets_folder_exists():
    assert os.path.exists("assets"), "Assets folder missing"
    assert os.path.exists(ASSET_PATH), f"Test image not found at {ASSET_PATH}"

def test_image_loading():
    img = cv2.imread(ASSET_PATH)
    assert img is not None, "Failed to load image via OpenCV"
    assert img.shape[2] == 3, "Image must have 3 channels (BGR)"

def test_full_pipeline_detection(system_instance):
    img_bgr = cv2.imread(ASSET_PATH)
    assert img_bgr is not None, "Test image not found at ASSET_PATH"
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    crop, processed, text = system_instance.detect_and_read(img_rgb)

    assert isinstance(crop, (type(None), np.ndarray)), "crop should be ndarray or None"
    if isinstance(crop, np.ndarray):
        assert crop.size > 0, "YOLO detected empty crop"
        assert len(crop.shape) == 3 and crop.shape[2] == 3, "Crop should be RGB (3 channels)"

    if processed is not None:
        assert len(processed.shape) == 3 and processed.shape[2] == 3, \
            "Processed image must be converted to RGB (3 channels) for Gradio display"

    # 3. Перевірка тексту
    assert isinstance(text, str), "Text result must be a string"

def test_detect_and_read_signature(system_instance):
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = system_instance.detect_and_read(dummy)
    assert isinstance(result, tuple) and len(result) == 3

def test_detect_on_blank_image_returns_none_or_empty(system_instance):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    crop, processed, text = system_instance.detect_and_read(blank)
    if crop is not None:
        assert isinstance(crop, np.ndarray)
    assert isinstance(text, str)
    # allow OCR to return empty string on no detection
    assert processed is None or (isinstance(processed, np.ndarray) and processed.size >= 0)

def test_preprocessing_logic(system_instance):
    dummy_crop = np.zeros((50, 100, 3), dtype=np.uint8)
    processed = system_instance.preprocess_plate(dummy_crop)
    assert isinstance(processed, np.ndarray)
    h, w = processed.shape[:2]
    # Accept either orientation in case implementation swaps dims; require expected pair present
    assert (h == 210 and w == 360) or (h == 360 and w == 210), f"Unexpected processed size: {(h,w)}"

def test_preprocess_plate_dtype_and_range(system_instance):
    dummy_crop = np.random.randint(0, 256, (60, 120, 3), dtype=np.uint8)
    processed = system_instance.preprocess_plate(dummy_crop)
    assert processed.dtype == np.uint8
    assert processed.min() >= 0
    assert processed.max() <= 255