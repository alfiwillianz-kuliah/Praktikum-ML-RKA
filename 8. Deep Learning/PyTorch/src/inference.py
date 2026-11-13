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
        # Model dilatih pada grayscale 28x28 (objek terang, latar hitam), piksel 0-1,
        # bentuk input (batch, channel, tinggi, lebar).
        # Foto biasa berlatar terang, jadi: transparansi -> putih, latar terang di-invert,
        # lalu objek di-crop dan di-pad ke persegi agar mirip gaya Fashion-MNIST.
        if image.mode in ("RGBA", "LA", "P"):
            rgba = image.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(bg, rgba)
        gray = np.asarray(image.convert("L"), dtype=np.float32)

        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        if border.mean() > 127:
            gray = 255.0 - gray
        gray = gray - np.median(border if border.mean() <= 127 else 255.0 - border)
        gray = np.clip(gray, 0, 255)

        mask = gray > 40
        if mask.any():
            ys, xs = np.where(mask)
            gray = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        h, w = gray.shape
        side = max(h, w)
        canvas = np.zeros((side, side), dtype=np.float32)
        canvas[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = gray

        img = Image.fromarray(canvas.astype(np.uint8)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)

    def predict(self, image: Image.Image) -> np.ndarray:
        x = self.preprocess(image)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        return softmax(logits[0])
