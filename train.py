import os
import numpy as np
import joblib
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load pretrained model once
print("Loading MobileNetV2...")
_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

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

def load_folder(folder_path, label):
    """Loads all images in a folder, extracts embeddings, assigns label."""
    embeddings = []
    labels = []
    valid_ext = (".jpg", ".jpeg", ".png")
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(valid_ext):
            fpath = os.path.join(folder_path, fname)
            try:
                emb = get_embedding(fpath)
                embeddings.append(emb)
                labels.append(label)
                print(f"  Processed: {fname}")
            except Exception as e:
                print(f"  Skipped {fname}: {e}")
    return embeddings, labels

def main():
    base_dir = r"C:\Users\Asus\OneDrive\Desktop\data"

    print("\nLoading training data...")
    # Training folders sit directly in the 'data' root folder
    train_real_emb, train_real_lbl = load_folder(os.path.join(base_dir, "real"), 0)
    train_screen_emb, train_screen_lbl = load_folder(os.path.join(base_dir, "screen"), 1)

    X_train = np.array(train_real_emb + train_screen_emb)
    y_train = np.array(train_real_lbl + train_screen_lbl)

    print(f"\nTotal training samples: {len(X_train)}")

    print("\nTraining classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("\nLoading validation data...")
    # Validation folders sit inside the 'val' subfolder
    val_real_emb, val_real_lbl = load_folder(os.path.join(base_dir, "val", "real"), 0)
    val_screen_emb, val_screen_lbl = load_folder(os.path.join(base_dir, "val", "screen"), 1)

    X_val = np.array(val_real_emb + val_screen_emb)
    y_val = np.array(val_real_lbl + val_screen_lbl)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"\nValidation accuracy: {acc * 100:.2f}%")

    # Show which images were misclassified for debugging
    val_paths = (
        [f for f in os.listdir(os.path.join(base_dir, "val", "real")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        + [f for f in os.listdir(os.path.join(base_dir, "val", "screen")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    print("\nMisclassified images:")
    for path, true_lbl, pred_lbl in zip(val_paths, y_val, y_pred):
        if true_lbl != pred_lbl:
            print(f"  {path}: true={true_lbl}, predicted={pred_lbl}")

    # Save the trained classifier
    joblib.dump(clf, "model.pkl")
    print("\nSaved trained classifier to model.pkl")

# Direct execution entry point to guarantee the main pipeline kicks off
if __name__ == "__main__":
    main()