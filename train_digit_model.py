"""
Train a YOLOv8 digit detector on a labeled dataset.

Usage:
    1. Download a digit detection dataset from Roboflow in YOLOv8 format.
       Place it in: backend/datasets/digits/
       The folder should contain: data.yaml, train/, valid/, test/

    2. Run:
           cd backend && venv/Scripts/activate
           python train_digit_model.py

    3. Trained model saved to: backend/vision/digit_model.pt
"""

from ultralytics import YOLO
import os
import shutil

DATASET_YAML = "datasets/digits/data.yaml"
BASE_MODEL   = "yolov8n.pt"
EPOCHS       = 30
IMG_SIZE     = 416
BATCH_SIZE   = 8


def train():
    if not os.path.exists(DATASET_YAML):
        print(f"ERROR: Dataset not found at '{DATASET_YAML}'")
        print("Download a YOLOv8-format digit dataset from roboflow.com/universe")
        print("and place it in backend/datasets/digits/")
        return

    print(f"Starting training on: {DATASET_YAML}")
    print(f"Base model: {BASE_MODEL}, Epochs: {EPOCHS}, img_size: {IMG_SIZE}")

    model = YOLO(BASE_MODEL)
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project="runs/digit_training",
        name="digit_model",
        exist_ok=True,
        device="cpu",
        patience=10,
        verbose=True,
    )

    best_weights = "runs/digit_training/digit_model/weights/best.pt"
    if os.path.exists(best_weights):
        shutil.copy(best_weights, "vision/digit_model.pt")
        print("\nDone! Model saved to: vision/digit_model.pt")
    else:
        print("Check runs/digit_training/digit_model/weights/best.pt")


if __name__ == "__main__":
    train()
