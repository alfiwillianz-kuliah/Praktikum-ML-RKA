# Deep Learning (PyTorch) - Customer Churn & Fashion-MNIST

| Nama | NRP |
|------|-----|
| Malfino Muhammad Willianz | 5054251028 |

## Isi folder

```text
PyTorch/
├── 1_DL_5054251028.ipynb     # Tugas 1: Customer Churn (LR, ANN, Dropout, Class Weight)
├── 2_DL_5054251028.ipynb     # Tugas 2: Fashion-MNIST (ANN, CNN, CNN + BatchNorm + Dropout)
├── assets/                   # artefak model hasil Tugas 2
│   ├── model.onnx            # model terbaik untuk inference (CPU, tanpa PyTorch)
│   ├── model.pth             # state_dict PyTorch
│   ├── model_architecture.py # definisi class CNNBatchNorm
│   └── class_names.json      # pemetaan 10 kelas
├── src/                      # REST API untuk deployment
│   ├── app.py                # FastAPI
│   ├── inference.py          # preprocessing + ONNX Runtime
│   └── requirements.txt
└── data/                     # dataset (di-gitignore, diunduh otomatis oleh notebook)
```

Kedua notebook sudah dijalankan penuh dan menyimpan output. Notebook mengunduh dataset sendiri
jika `data/*.csv` belum ada, dan notebook 2 menulis ulang seluruh isi `assets/`.

Demo: https://fashion.alfiwillianz.my.id

## Menjalankan API

Model yang dipakai adalah **CNN + BatchNorm + Dropout** (accuracy 0,9185 pada test set) dalam
format ONNX, sehingga cukup berjalan di CPU tanpa PyTorch dan tanpa GPU.

```bash
cd "8. Deep Learning/PyTorch"
python -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt

uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000
```

Dokumentasi interaktif tersedia di `http://localhost:8000/docs`.

### Endpoint

| Method | Path | Keterangan |
|---|---|---|
| GET | `/health` | status server, runtime, dan provider ONNX |
| GET | `/classes` | daftar 10 kelas beserta id |
| POST | `/predict` | upload gambar (`multipart/form-data`, field `file`) |

Contoh:

```bash
curl -X POST -F "file=@gambar.png" http://localhost:8000/predict
```

```json
{
  "label": "Ankle boot",
  "class_id": 9,
  "confidence": 0.999971,
  "probabilities": {
    "T-shirt/top": 0.000001,
    "Trouser": 0.000001,
    "Pullover": 0.000001,
    "Dress": 0.000001,
    "Coat": 0.000001,
    "Sandal": 0.000022,
    "Shirt": 0.000001,
    "Sneaker": 0.000006,
    "Bag": 0.000001,
    "Ankle boot": 0.999971
  }
}
```

### Preprocessing yang diterapkan API

1. Gambar dikonversi ke grayscale; latar terang otomatis di-invert (Fashion-MNIST berlatar hitam), objek di-crop dan di-pad persegi, lalu di-resize ke 28×28.
2. Nilai piksel 0-255 dibagi 255 menjadi rentang 0-1, tipe `float32`.
3. Bentuk input menjadi `(1, 1, 28, 28)`.
4. Output model berupa 10 logit, lalu diubah dengan softmax menjadi probabilitas.

Langkah ini identik dengan preprocessing di notebook, sehingga hasil API sama dengan hasil
evaluasi model. Gambar berukuran lain, RGB, atau JPEG tetap diterima karena langkah 1 dan 2
menyeragamkan bentuk dan skala.
