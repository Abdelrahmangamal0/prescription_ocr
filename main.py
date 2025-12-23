from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
import io

from pipeline import run_pipeline

# ======================
# App initialization
# ======================
app = FastAPI(title="Prescription OCR API")

# ======================
# Static & Templates
# ======================
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ======================
# Home Page (GUI)
# ======================
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# ======================
# Prediction Endpoint
# ======================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = run_pipeline(image)

        return {
            "status": "success",
            "detected_text": result.get("detected_text", ""),
            "predicted_drug": result.get("predicted_drug", "")
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
