from pathlib import Path
import torch
import pickle

base_dir = Path(r"C:\Inferno-ai\prescription_ocr\backend\models")

resnet_path = base_dir / "resnet.pth"
label_encoder_path = base_dir / "label_encoder.pkl"

if not resnet_path.exists():
    print("❌ ResNet model file not found!")
else:
    print("✅ ResNet model found!")

if not label_encoder_path.exists():
    print("❌ Label encoder file not found!")
else:
    print("✅ Label encoder found!")

try:
    resnet = torch.load(resnet_path, map_location="cpu")
    print("ResNet loaded successfully!")
except Exception as e:
    print("Error loading ResNet:", e)

try:
    label_encoder = pickle.load(open(label_encoder_path, "rb"))
    print("Label encoder loaded successfully!")
except Exception as e:
    print("Error loading label encoder:", e)
