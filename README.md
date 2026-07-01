Spot the Fake Photo — Real vs. Screen Recapture Classifier

A binary image classification model that determines whether an input image is:

0 → Real photograph
1 → Photograph of a screen (screen recapture)

The model outputs a probability score between 0 and 1, where values closer to 1 indicate a higher likelihood that the image is a screen recapture.

Example
python predict.py some_image.jpg

Output:

0.93

Interpretation:

0.00 → Real photo
1.00 → Screen recapture
Requirements
Python 3.8+
TensorFlow 2.x

Install dependencies:

pip install tensorflow opencv-python scikit-learn joblib scipy scikit-image
Project Structure
Spot-the-Fake-Photo/
│
├── predict.py          # Submission script (run this)
├── train.py            # Model training pipeline
├── model.pkl           # Saved classifier (generated after training)
│
├── data/
│   ├── real/           # Training images (real photos)
│   ├── screen/         # Training images (screen recaptures)
│   ├── val/
│   │   ├── real/       # Validation images (real)
│   │   └── screen/     # Validation images (screen)
│
└── README.md
How to Run
Step 1 — Train the Model

Run once to generate the classifier.

python train.py

The training script will:

Extract image embeddings
Train the classifier
Print training progress
Evaluate validation accuracy
Display any misclassified images
Save the trained model as:
model.pkl
Step 2 — Predict

Run inference on any image.

python predict.py path/to/image.jpg

Example:

python predict.py sample.jpg

Output:

0.81

Interpretation:

Score	Prediction
< 0.5	Real photo
≥ 0.5	Screen recapture
Model Architecture

The project uses transfer learning with MobileNetV2.

Feature Extraction
Pretrained MobileNetV2 (ImageNet weights)
Weights remain frozen
Images resized to 224 × 224
Global Average Pooling produces a 1280-dimensional embedding
Classification

The extracted embeddings are fed into a Logistic Regression classifier that predicts:

Real Photo
Screen Recapture
Why Not Fine-Tune?

The training dataset contains only ~70 images.

Fine-tuning the CNN would likely overfit, while frozen embeddings provide much better generalization on small datasets.

Performance

Validation Results

Accuracy: 95%
Correct Predictions: 19 / 20
Misclassified Images: 1
Screen recapture predicted as a real photo

Note: The validation set contains only 20 images, so performance on larger unseen datasets may differ.

Inference Speed
Platform	Latency / Cost
Laptop CPU	~40–80 ms per image
Mid-range phone (TFLite)	<50 ms per image (estimated)
Cloud CPU	~$0.001–0.005 per 1,000 images
Experiments

Several classical computer vision features were evaluated before selecting the final approach.

Tested features included:

FFT-based moiré pattern detection
Local Binary Patterns (LBP)
Color variance statistics

These handcrafted features did not reliably distinguish screen recaptures from real photographs captured on modern smartphone cameras.

Transfer learning with MobileNetV2 consistently produced superior results.

Limitations
Small training dataset (~80 images).
Limited diversity of screen types (TVs, tablets, OLED displays).
Model has not yet been optimized for on-device deployment using TensorFlow Lite (TFLite).

Increasing the dataset size and diversity would likely improve robustness and generalization.
