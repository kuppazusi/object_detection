"""
patchcore_cpu_example.py
------------------------
1. MVTecAD の bottle クラスをダウンロード
2. PatchCore を CPU で 1 epoch だけ学習
3. 同じデータで推論し、画像レベル・ピクセルレベルの結果を取得
"""

from pathlib import Path

# ===== 1. ライブラリ =================================================================
from anomalib.data import MVTecAD, PredictDataset          8 データセット
from anomalib.models import Patchcore                      # モデル本体
from anomalib.engine import Engine                         # 学習／推論エンジン
from anomalib.data import Folder

# ===== 2. データモジュール ==========================================================
dm = Folder(
    root="datasets/product",
    normal_dir="train/good",
    abnormal_dir="test/defect_1",
    image_size=256,
    train_batch_size=8,
    eval_batch_size=4,
    num_workers=0,
)

# ===== 3. PatchCore モデル ===========================================================
# 手動で保存した重みファイルのパス
weights_path = "C:/models/resnet18-f37072fd.pth"

# ResNet18 をインスタンス化
resnet18 = models.resnet18(pretrained=False)
state_dict = torch.load(weights_path, map_location="cpu")
resnet18.load_state_dict(state_dict)

# classifier を削除（出力層は不要）
resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-2])  # 〜layer4 まで

model = Patchcore(
    backbone=resnet18,    # ImageNet 事前学習済み WRN-50-2
    layers=["layer2", "layer3"],   # 中間層の特徴を使用
    coreset_sampling_ratio=0.05,   # メモリ削減のため 5 %
)

# ===== 4. エンジン（CPU 強制）========================================================
engine = Engine(
    accelerator="cpu",             # GPU を探しに行かない
    devices=1,
    max_epochs=1,                  # お試しなので 1 epoch
    precision=32,                  # 16-bit にせず安全策
)

# ===== 5. 学習 =======================================================================
engine.fit(model=model, datamodule=dm)

# チェックポイントは engine.fit() 完了時に self.trainer.logger.log_dir 内に保存される。
ckpt_path = Path(engine.trainer.logger.log_dir) / "weights.ckpt"

# ===== 6. 推論（画像フォルダを直接指定しても OK）======================================
predict_ds = PredictDataset(
    path=Path("datasets/product/test/defect_1"),  # 例：テスト画像ディレクトリ
    image_size=(256, 256),
)

predictions = engine.predict(
    model=model,
    dataset=predict_ds,
    ckpt_path=ckpt_path,
    return_predictions=True,        # list[Prediction] が返る
)

# ===== 7. 結果の利用例 ================================================================
for p in predictions[:3]:           # 先頭 3 枚だけ表示
    print("Image :", p.image_path.name)
    print("label :", p.pred_label, "  score :", p.pred_score.item())
    # p.anomaly_map は (H, W) の heat-map ndarray

print("Done!")
