# 🛡️ Smart Barrier AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange?style=for-the-badge)
![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ituvtu/smart-barrier-ai)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

**Smart Barrier AI** is an intelligent vehicle access control system designed to automate gate operations. It utilizes a fine-tuned **YOLOv10m** model for license plate detection and **EasyOCR** for text recognition, enhanced by robust image preprocessing and region-specific post-processing logic.

## 🚀 Key Features

* **High-Precision Detection:** Uses a custom-trained YOLOv10m model to accurately locate license plates in various lighting conditions.
* **Advanced OCR Pipeline:** Implements EasyOCR with a custom preprocessing stage (upscaling, denoising, adaptive thresholding) to handle low-quality images.
* **Region Enforcer Logic:**
    * 🇺🇦 **Ukraine:** Enforces specific standard structure (`LL DDDD LL`) with sliding window analysis and character correction.
    * 🇪🇺 **Europe:** Adaptive filtering for EU plates (stripping country codes and artifacts).
* **Fuzzy Matching:** Smart database verification that accounts for common OCR typos (e.g., `0` vs `O`, `8` vs `B`).
* **Interactive UI:** A user-friendly web interface built with **Gradio** featuring simulated LED status indicators and real-time logs.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ituvtu/smart-barrier-ai.git
    cd smart-barrier-ai
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Verify Model Weights:**
    Ensure that the fine-tuned model weights file `best.pt` is located in the root directory.

## 💻 Usage

Run the application locally:

```bash
python app.py

```

Open your browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

### Workflow:

1. **Upload** an image of a vehicle.
2. **Select** the region mode (e.g., Ukraine 🇺🇦).
3. **Populate** the "Database" field with allowed license plates (comma-separated).
4. Click **"Check Access"** to see the detection result and access decision.

## 📂 Project Structure

* `app.py` - Main entry point containing the Gradio interface and pipeline orchestration.
* `utils.py` - Core logic for text cleaning, regex pattern matching, and fuzzy search algorithms.
* `requirements.txt` - Python dependencies.
* `best.pt` - Fine-tuned YOLOv10m weights (ensure this file is present).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Author:** [ituvtu](https://github.com/ituvtu)
