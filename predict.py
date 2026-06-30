import sys
import numpy as np
import joblib
import cv2
import time

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
_clf = joblib.load("model.pkl")

def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    embedding = _model.predict(img, verbose=0)
    return embedding.flatten()

def predict(image_path):
    emb = get_embedding(image_path).reshape(1, -1)
    prob = _clf.predict_proba(emb)[0][1]  # probability of class 1 = screen
    return prob

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    score = predict(path)
    print(f"{score:.4f}")




if __name__ == "__main__":
    path = sys.argv[1]
    start = time.time()
    score = predict(path)
    elapsed_ms = (time.time() - start) * 1000
    print(f"{score:.4f}")
    print(f"Latency: {elapsed_ms:.1f} ms", file=sys.stderr)