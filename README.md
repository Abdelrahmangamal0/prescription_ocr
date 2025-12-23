# Prescription OCR System

A web-based **Prescription OCR** application that detects and extracts handwritten or printed prescription text and predicts the drug names. Built using **FastAPI**, **PyTorch**, **Transformers (TrOCR)**, and **HTML/CSS** for the GUI.

---

## Features

* **OCR (Text Detection):** Recognizes handwritten or printed prescription text using Microsoft's TrOCR model.
* **Drug Prediction:** Identifies the most probable drug mentioned in the prescription.
* **Web GUI:** Upload prescription images and see the extracted text and predicted drug in a user-friendly interface.
* **FastAPI Backend:** Provides API endpoints for prediction and file handling.
* **Dynamic Prediction:** Works on any uploaded prescription image without requiring manual input.

---

## Project Structure

```
backend/
│
├── main.py              # FastAPI server & routes
├── pipeline.py          # OCR & drug prediction pipeline
├── models/              # Trained ResNet model & label encoder
│   ├── resnet.pth
│   └── label_encoder.pkl
├── templates/           # HTML templates
│   └── index.html
└── static/              # CSS, JS, images
```

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/prescription_ocr.git
cd prescription_ocr/backend
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

**Note:** For PyTorch installation:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers pillow fastapi uvicorn jinja2
```

---

## Usage

1. **Run the backend server:**

```bash
uvicorn main:app --reload
```

2. **Open your browser** and navigate to:

```
http://127.0.0.1:8000
```

3. **Upload a prescription image** in the GUI to see:

* Detected text from the prescription.
* Predicted drug name.

---

## Example

| Prescription Image | Detected Text        | Predicted Drug |
| ------------------ | -------------------- | -------------- |
| image1.jpeg        | Defense scalp        | Napa Extend    |
| image2.jpeg        | # a conversation ... | Azithrocin     |

---

## Notes

* The OCR model may sometimes misread handwritten text; image preprocessing may improve results.
* Predicted drugs are based on a predefined list (`label_encoder.pkl`). Consider using fuzzy matching for better prediction.

---

## License

This project is licensed under the MIT License.


