# Defective Product Detection Using Deep Learning

Deep Learning Lab capstone, Ramdeobaba University, Nagpur (Dept. of Electronics Engineering, 2026-27, Semester V).
Group ID: [your group ID]

Team: Aaditee Bhope (1), Aditi Patel (4), Arya Paunikar (7), Janhavi Gulhane (18)

## What this project does
Classifies an image of a manufactured part as **Normal** or **Defective**. It compares a simple CNN baseline with transfer learning using EfficientNetB0 (ImageNet weights, Global Average Pooling, Dropout, sigmoid output). A Streamlit app lets a user take a photo or upload an image and see the prediction.

## Dataset
Kolektor Surface-Defect Dataset (KolektorSDD), ViCoS Laboratory, University of Ljubljana.
- Page: https://www.vicos.si/resources/kolektorsdd/
- Direct download: https://data.vicos.si/datasets/KSDD/KolektorSDD.zip
- 399 images: 347 normal, 52 defective
- Labels come from the masks: an image whose `_label.bmp` mask has any white pixel is Defective.
- Split 70/15/15 by item (kos folder), so near-identical photos of one item never appear in both train and test.

## Repository contents
| File | Purpose |
|---|---|
| `notebooks/kolektorsdd_training.ipynb` | Training and evaluation (Google Colab, T4 GPU) |
| `models/defect_detector.keras` | Trained EfficientNetB0 model (KolektorSDD) |
| `app/app.py` | Streamlit demo app |
| `defect_detector_cans.keras`, `app_live.py` | Extra experiment: model trained on our own crushed-can photos |
| `screenshots/` | Result and app screenshots |
| `requirements.txt` | Python libraries |

## How to run

### 1. Training (Google Colab)
1. Open `notebooks/kolektorsdd_training.ipynb` in Google Colab.
2. Runtime > Change runtime type > T4 GPU.
3. Run all cells in order. The first cell downloads the dataset automatically.
4. The last cell saves `defect_detector.keras`.

### 2. Demo app (local)
1. Install Python 3.11 (TensorFlow does not support the newest Python versions).
2. In a terminal:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```
3. Put `defect_detector.keras` in the same folder as `app.py`, or change the path in the code.
4. Open http://localhost:8501, then take a photo or upload an image.

## Results (test set)
| Model | Accuracy | Defective precision | Defective recall | Defective F1 |
|---|---|---|---|---|
| CNN baseline | [XX] | [XX] | [XX] | [XX] |
| EfficientNetB0 | [XX] | [XX] | [XX] | [XX] |

## Sample outputs
![Confusion matrix](screenshots/confusion_matrix.png)
![App - defective](screenshots/app_defective.png)
![App - normal](screenshots/app_normal.png)

## Limitations
- Small dataset (52 defective images), so test scores vary a lot.
- The model only works on the kind of images it was trained on: grayscale close-ups of metal parts under fixed lighting. A photo of a crushed can gave about 50% confidence (a guess). This is domain shift.
- Our extra can model was trained on only about 60 images, so it is a demonstration, not a reliable system.
