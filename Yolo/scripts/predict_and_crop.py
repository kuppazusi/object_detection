#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
推論 + BBox 描画 + トリミング保存 (<=30KB, 最大解像度)
"""

import argparse, os, io
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image

# ────────── 画像を 30KB 以下で最大サイズ保存 ──────────
def save_under_30k(img_bgr, save_path, start_q=95, min_q=30, step=5):
    """
    img_bgr : OpenCV BGR image
    save_path : str / Path(.jpg)
    JPEG 品質を落としながら 30 KB 以下にする。品質が min_q まで下がったら
    それ以上は長辺 10% ずつ縮小しつつ再トライ。
    """
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    q = start_q
    scale = 1.0
    while True:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=q, optimize=True)
        if buf.tell() <= 30_000 or (q <= min_q and min(img_bgr.shape[:2]) < 64):
            with open(save_path, "wb") as f:
                f.write(buf.getvalue())
            return
        # サイズオーバー
        if q > min_q:
            q -= step
        else:
            # 画質これ以上下げたくない → リサイズ 0.9 倍して再試行
            scale *= 0.9
            new_w = int(pil.width * scale)
            new_h = int(pil.height * scale)
            pil = pil.resize((new_w, new_h), Image.LANCZOS)
            q = start_q  # 画質リセット

# ────────── 引数 ──────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",  required=True, help="学習済み .pt")
parser.add_argument("--source", required=True, help="推論画像フォルダ")
parser.add_argument("--out-full", default="out/full",  help="BBox 描画付き画像保存先")
parser.add_argument("--out-crop", default="out/crops", help="トリミング画像保存先")
parser.add_argument("--conf-thres", type=float, default=0.25, help="信頼度閾値")
args = parser.parse_args()

# ────────── 準備 ──────────
model = YOLO(args.model)
os.makedirs(args.out_full, exist_ok=True)
os.makedirs(args.out_crop, exist_ok=True)

image_paths = sorted(Path(args.source).glob("*.*g"))  # jpg/png
pbar = tqdm(image_paths, desc="推論")

for img_path in pbar:
    img_bgr = cv2.imread(str(img_path))
    results = model(str(img_path), verbose=False, conf=args.conf_thres)[0]

    # BBox 描画用コピー
    full_vis = img_bgr.copy()

    # クラス名ごとのカウント（+採番用）
    counter = {}

    # 検出がない場合 → 元画像をそのままコピー
    if not results.boxes:
        dst = Path(args.out_crop) / f"{img_path.stem}_.jpg"
        cv2.imwrite(str(dst), img_bgr)
    else:
        # ---- BBox ごとに処理 ----
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # クラス名
            cls_name = model.names[cls_id]
            counter.setdefault(cls_name, 0)
            idx = counter[cls_name]
            counter[cls_name] += 1

            # ########## 1) トリミング & 30KB 保存 ##########
            crop = img_bgr[y1:y2, x1:x2]
            suffix = "" if idx == 0 else f"+{idx}"
            crop_name = f"{img_path.stem}_{cls_name}{suffix}.jpg"
            crop_path = Path(args.out_crop) / crop_name
            save_under_30k(crop, crop_path)

            # ########## 2) Full 画像へ描画 ##########
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(full_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(full_vis, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ---- BBox 描画付きフル画像保存 ----
    full_out = Path(args.out_full) / img_path.name
    cv2.imwrite(str(full_out), full_vis)

pbar.close()
print(f"✅ 完了: 描画付き → {args.out_full} / トリミング → {args.out_crop}")
