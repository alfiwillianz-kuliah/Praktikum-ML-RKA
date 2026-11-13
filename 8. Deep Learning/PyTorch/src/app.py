"""REST API klasifikasi Fashion-MNIST (FastAPI + ONNX Runtime, CPU).

Jalankan dari folder ini:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from inference import FashionMNISTClassifier

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

app = FastAPI(
    title="Fashion-MNIST Classifier",
    description="Klasifikasi 10 kelas pakaian dengan CNN (ONNX Runtime, CPU).",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


classifier = FashionMNISTClassifier(
    ASSETS_DIR / "model.onnx", ASSETS_DIR / "class_names.json"
)


class Prediction(BaseModel):
    label: str
    class_id: int
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "CNN + BatchNorm + Dropout",
        "runtime": "onnxruntime",
        "providers": classifier.providers,
        "classes": len(classifier.class_names),
    }


@app.get("/classes")
def classes() -> dict:
    return {
        "classes": [
            {"id": i, "name": name} for i, name in enumerate(classifier.class_names)
        ]
    }


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)) -> Prediction:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File kosong.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")

    probs = classifier.predict(image)
    top = int(np.argmax(probs))

    return Prediction(
        label=classifier.class_names[top],
        class_id=top,
        confidence=round(float(probs[top]), 6),
        probabilities={
            name: round(float(p), 6) for name, p in zip(classifier.class_names, probs)
        },
    )
