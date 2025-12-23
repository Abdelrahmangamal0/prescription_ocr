import os
import torch
import pickle
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Base directory
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))

# Paths
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
RESNET_PATH = MODELS_DIR / "resnet.pth"

# Load TrOCR model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=False)
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model.eval()

# Load label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)

# Load ResNet classifier
resnet = models.resnet18(weights=None)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
resnet.load_state_dict(torch.load(RESNET_PATH, map_location="cpu"))
resnet.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# OCR function
def run_ocr(image: Image.Image) -> str:
    pixel_values = processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Classifier function
def classify_word(image: Image.Image) -> str:
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = resnet(x)
    idx = output.argmax(dim=1).item()
    return label_encoder.inverse_transform([idx])[0]

# Pipeline
def run_pipeline(image: Image.Image):
    text = run_ocr(image)
    drug = classify_word(image)
    return {
        "detected_text": text,
        "predicted_drug": drug
    }
