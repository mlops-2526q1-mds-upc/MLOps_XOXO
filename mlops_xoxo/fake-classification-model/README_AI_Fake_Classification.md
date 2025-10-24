# 🧠 AI Fake Face Classification Pipeline

This pipeline trains a lightweight **MobileNetV3-Small** model to classify **real vs fake faces** using a dataset from Hugging Face.

## 📦 Requirements
Install dependencies in your virtual environment:
```bash
pip install torch torchvision scikit-learn opencv-python pillow pandas numpy onnx
```

---

## 1️⃣ Ingest Dataset
Downloads and extracts the dataset ZIP file from Hugging Face.
```bash
python data_ingest.py --url https://huggingface.co/datasets/pujanpaudel/deepfake_face_classification/resolve/main/val.zip --out-dir data/raw/deepfake_val
```
- Output: `data/raw/deepfake_val/`
- Manifest: `ingest_manifest.json`

---

## 2️⃣ Preprocess Data
Performs **face detection, cropping, resizing (224×224), and normalization**.
```bash
python data_preprocess.py --in-dir data/raw/deepfake_val/val --out-dir data/processed/deepfake_val_224 --size 224 --margin 0.25
```
- Output: `data/processed/deepfake_val_224/`
- Files: `normalization.json`, `index.csv`, `skipped.jsonl`

---

## 3️⃣ Split Dataset
Splits the processed dataset into **train**, **val**, and **test** sets.
```bash
python data_split.py --in-dir data/processed/deepfake_val_224 --out-dir data/splits/deepfake_val_224_split --train 0.7 --val 0.15 --test 0.15 --seed 42
```
- Output: `data/splits/deepfake_val_224_split/train|val|test/`
- Manifest: `split_manifest.json`

---

## 4️⃣ Train Model
Fine-tunes a pre-trained **MobileNetV3-Small** for fake/real face classification.
```bash
python train_model.py --splits-dir data/splits/deepfake_val_224_split --norm-json data/processed/deepfake_val_224/normalization.json --epochs 5 --batch-size 64 --lr 1e-3 --export-onnx 1
```
- Output: `artifacts/ai_face/`
- Files:
  - `model_best.pth` (best checkpoint)
  - `model_last.pth`
  - `model_scripted.pt`
  - `model.onnx`
  - `metrics_history.csv`
  - `metrics_val.json`
  - `config.json`

---

## 5️⃣ Evaluate Model
Evaluates the trained model on the **test** split.
```bash
python eval_model.py --splits-dir data/splits/deepfake_val_224_split --split test --norm-json data/processed/deepfake_val_224/normalization.json --checkpoint artifacts/ai_face/model_best.pth
```
- Output: `artifacts/ai_face_eval/`
- Files:
  - `metrics_test.json` (accuracy, precision, recall, F1)
  - `confusion_matrix_test.csv`

---

## 📈 Typical Results
| Split | Accuracy | F1-Score |
|:------|:----------|:---------|
| Validation | ~0.945 | ~0.945 |
| Test | ~0.957 | ~0.957 |

---

## 🧾 Deliverables
- **Dataset Card** – description, size, preprocessing, splits, license.
- **Model Card** – backbone info, hyperparameters, metrics, portability.
- **Trained Artifacts** – `.pth`, `.onnx`, `.pt`, config, metrics.
- **README.md** – usage instructions (this file).

---

## 🧱 Notes
- Model backbone: `MobileNetV3-Small (pre-trained on ImageNet)`
- Goal: small, portable, and fast model (~5–10 MB).
- The `.onnx` and `.pt` files make it deployable to other frameworks.
- Logging: metrics are printed in the console and saved to CSV.
