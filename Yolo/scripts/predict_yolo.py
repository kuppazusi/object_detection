# predict_yolo.py

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from tqdm import tqdm

# --- 設定 ---
PROJECT = "runs/train_cpu"
EXPERIMENT_NAME = "cpu_test"
PREDICT_SOURCE = "images/val"      # 推論する画像フォルダ
OUTPUT_DIR = "output_annotated"    # 出力フォルダ

# --- モデルロード ---
best_model_path = Path(PROJECT) / EXPERIMENT_NAME / "weights" / "best.pt"
model = YOLO(str(best_model_path))

# --- 出力先フォルダ準備 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 推論とBBox付き画像保存 ---
image_paths = list(Path(PREDICT_SOURCE).glob("*.jpg"))

for img_path in tqdm(image_paths, desc="推論中"):
    img = cv2.imread(str(img_path))
    results = model(str(img_path), verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{model.names[cls_id]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = Path(OUTPUT_DIR) / img_path.name
    cv2.imwrite(str(out_path), img)

print(f"推論完了：出力画像 → {OUTPUT_DIR}/")
