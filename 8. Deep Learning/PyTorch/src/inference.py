"""Inference Fashion-MNIST dengan ONNX Runtime (CPU, tanpa PyTorch)."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_SIZE = 28


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp = np.exp(z)
    return exp / exp.sum()


class FashionMNISTClassifier:
    def __init__(self, model_path: Path, class_names_path: Path):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        mapping = json.loads(Path(class_names_path).read_text(encoding="utf-8"))
        self.class_names = [mapping[str(i)] for i in range(len(mapping))]

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def preprocess(self, image: Image.Image) -> np.ndarray:
        # Model dilatih pada grayscale 28x28, piksel 0-255 dinormalisasi ke 0-1,
        # dengan bentuk input (batch, channel, tinggi, lebar).
        img = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)

    def predict(self, image: Image.Image) -> np.ndarray:
        x = self.preprocess(image)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        return softmax(logits[0])
