# train_yolo.py

from ultralytics import YOLO

# --- 設定 ---
MODEL_NAME = "models/yolov8n.pt"
DATA_YAML = "dataset.yaml"
EPOCHS = 200
IMG_SIZE = 640
PROJECT = "runs/train_cpu"
EXPERIMENT_NAME = "cpu_test"

# --- 学習開始 ---
model = YOLO(MODEL_NAME)
model.train(data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            device='cpu',
            project=PROJECT,
            name=EXPERIMENT_NAME)
