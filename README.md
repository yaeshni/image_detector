
Report: https://docs.google.com/spreadsheets/d/1aic4hSo2U0DNIsvolFmGBWwMdyzoZYI4/edit?usp=drive_link&ouid=111220777700853621757&rtpof=true&sd=true



# 📸 Spot the Fake Photo
### Real vs. Screen Recapture Classifier

<p align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Logistic%20Regression-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

</p>

A lightweight **binary image classifier** that determines whether an image is:

- 📷 **A real photograph**
- 🖥️ **A photograph of a screen (screen recapture)**

The model outputs a probability score between **0** and **1**.

| Score | Prediction |
|------:|------------|
| **0.0** | Real Photo |
| **1.0** | Screen Recapture |

---

## ✨ Example

```bash
python predict.py some_image.jpg
```

Output

```text
0.93
```

> A score of **0.93** indicates a **high probability** that the image is a **screen recapture**.

---

# 🚀 Features

- ✅ MobileNetV2 transfer learning
- ✅ Frozen feature extractor (pretrained on ImageNet)
- ✅ Logistic Regression classifier
- ✅ Fast CPU inference
- ✅ Lightweight deployment
- ✅ Simple command-line interface

---

# 📂 Project Structure

```
Spot-the-Fake-Photo/
│
├── predict.py          # Prediction script
├── train.py            # Training pipeline
├── model.pkl           # Saved classifier
│
├── data/
│   ├── real/
│   ├── screen/
│   └── val/
│       ├── real/
│       └── screen/
│
└── README.md
```

---

# ⚙️ Installation

### Clone the repository

```bash
git clone <your-repository-url>
cd Spot-the-Fake-Photo
```

### Install dependencies

```bash
pip install tensorflow opencv-python scikit-learn joblib scipy scikit-image
```

**Requirements**

- Python **3.8+**
- TensorFlow **2.x**

---

# 🏃 Training

Train the model (only once):

```bash
python train.py
```

The training pipeline will:

- Extract image embeddings
- Train the classifier
- Display training progress
- Evaluate validation accuracy
- Report misclassified images
- Save the trained model as:

```
model.pkl
```

---

# 🔍 Prediction

Run inference on any image:

```bash
python predict.py path/to/image.jpg
```

Example:

```bash
python predict.py sample.jpg
```

Example Output

```text
0.81
```

Interpretation:

| Probability | Result |
|------------:|--------|
| **< 0.5** | 📷 Real Photo |
| **≥ 0.5** | 🖥️ Screen Recapture |

---

# 🧠 Model Pipeline

```
Input Image
      │
      ▼
Resize (224 × 224)
      │
      ▼
MobileNetV2
(Frozen ImageNet Weights)
      │
      ▼
1280-D Feature Embedding
      │
      ▼
Logistic Regression
      │
      ▼
Probability (0 → 1)
```

---

# 🔬 Why MobileNetV2?

Instead of training a CNN from scratch, this project uses **transfer learning**.

### Feature Extraction

- Pretrained **MobileNetV2**
- ImageNet weights
- Frozen backbone
- Global Average Pooling
- 1280-dimensional feature vector

### Classification

The extracted embeddings are classified using **Logistic Regression**.

### Why not fine-tune?

The dataset contains only **~70 training images**.

Fine-tuning the CNN would likely **overfit**, whereas frozen embeddings provide significantly better generalization on small datasets.

---

# 📊 Results

| Metric | Value |
|---------|------:|
| Validation Accuracy | **95%** |
| Correct Predictions | **19 / 20** |
| Misclassified Images | **1** |

> **Note:** The validation set contains only **20 images**, so accuracy may vary on larger unseen datasets.

---

# ⚡ Performance

| Platform | Performance |
|-----------|-------------|
| Laptop CPU | **40–80 ms/image** |
| TFLite (Estimated) | **<50 ms/image** |
| Cloud CPU | **~$0.001–0.005 / 1000 images** |

---

# 🧪 Experiments

Before choosing transfer learning, several handcrafted computer vision features were evaluated:

- FFT-based moiré detection
- Local Binary Patterns (LBP)
- Color variance statistics

These approaches did **not** reliably separate real photographs from screen recaptures captured using smartphone cameras.

Transfer learning with **MobileNetV2** consistently achieved the best performance.

---

# ⚠️ Limitations

- Small training dataset (~80 images)
- Limited diversity of screen types
- Not yet converted to TensorFlow Lite (TFLite)
- Additional real-world data would improve robustness

---

# 📈 Future Improvements

- [ ] Larger dataset
- [ ] More screen types (OLED, LCD, TVs, tablets)
- [ ] TFLite conversion
- [ ] Quantized model for mobile deployment
- [ ] Web interface using Flask/FastAPI
- [ ] Confidence visualization

---


---

⭐ If you found this project useful, consider giving it a **star**!
