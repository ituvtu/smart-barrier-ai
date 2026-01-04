import pytest
import cv2
import os
import numpy as np
from app import BarrierSystem, load_config

# Path to the test asset
ASSET_PATH = os.path.join("assets", "test_car.jpg")

@pytest.fixture(scope="module")
def system_instance():
    # Load config and init system once for all tests
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
    # Prepare image (simulate Gradio input which is RGB)
    img_bgr = cv2.imread(ASSET_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run detection
    crop, processed, text = system_instance.detect_and_read(img_rgb)

    # Assertions
    # 1. Check if crop is returned (means YOLO worked)
    assert crop is not None, "YOLO failed to detect license plate"
    assert isinstance(crop, np.ndarray)
    assert crop.size > 0

    # 2. Check if processing happened
    assert processed is not None
    assert len(processed.shape) == 2 or processed.shape[2] == 1, "Processed image should be grayscale"

    # 3. Check text output
    assert isinstance(text, str)
    assert len(text) > 0
    print(f"\nDetected Text from Asset: {text}")

def test_preprocessing_logic(system_instance):
    # Create a dummy small image
    dummy_crop = np.zeros((50, 100, 3), dtype=np.uint8)
    
    processed = system_instance.preprocess_plate(dummy_crop)
    
    # Check if upscaling and padding occurred
    # Config default: scale=3, padding=30
    # Expected height: (50*3) + 30 + 30 = 210
    # Expected width: (100*3) + 30 + 30 = 360
    
    h, w = processed.shape[:2]
    assert h == 210
    assert w == 360