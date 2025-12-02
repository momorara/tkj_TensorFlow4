# -*- coding: utf-8 -*-
"""
解凍した元データdataのファル形式をjpgに統一する

現在のデータセットは
現在のデータセットは
data/test/
├── adenocarcinoma/
├── large.cell.carcinoma/
├── squamous.cell.carcinoma/
└── normal

data/train/
├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
├── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa
└── normal

data/valid/
├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib
├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa
├── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa
└── normal
です。

で、その中に画像ファィルが入っている
その画像をjpegに変換して、
次のようなデータ構造にして

data_j/test/
├── adeno/
├── largecell/
├── squamouscell/
└── normal

data_j/train/
├── adeno/
├── largecell/
├── squamouscell/
└── normal

data_j/valid/
├── adeno/
├── largecell/
├── squamouscell/
└── normal
とする
"""
import os
from PIL import Image
import shutil

# 元データと出力先
SRC_ROOT = "Data"
DST_ROOT = "data_j"

# クラス名マッピング
class_map = {
    "adenocarcinoma": "adeno",
    "large.cell.carcinoma": "largecell",
    "squamous.cell.carcinoma": "squamouscell",
    "normal": "normal"
}

# train, test, valid ごとに処理
splits = ["train", "test", "valid"]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_and_copy(src_file, dst_file):
    """画像を JPEG に変換して保存"""
    try:
        img = Image.open(src_file)

        # PNG 等の場合は RGB に変換
        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(dst_file, "JPEG")

    except Exception as e:
        print(f"変換エラー: {src_file}: {e}")


for split in splits:
    src_split_dir = os.path.join(SRC_ROOT, split)
    dst_split_dir = os.path.join(DST_ROOT, split)

    # 出力先のsplitフォルダ作成
    ensure_dir(dst_split_dir)

    # クラスフォルダ一覧取得
    for cls_name in os.listdir(src_split_dir):

        # 不要な項目を除外
        cls_path = os.path.join(src_split_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue

        # 下位フォルダ名をマッピング
        for key in class_map:
            if key in cls_name:
                new_cls = class_map[key]
                break
        else:
            print(f"警告: マッチするクラスがありません → {cls_name}")
            continue

        dst_cls_path = os.path.join(dst_split_dir, new_cls)
        ensure_dir(dst_cls_path)

        # クラスフォルダ内の画像を JPEG 変換して保存
        for fname in os.listdir(cls_path):
            src_file = os.path.join(cls_path, fname)

            # 拡張子を .jpg に統一
            base = os.path.splitext(fname)[0]
            dst_file = os.path.join(dst_cls_path, base + ".jpg")

            convert_and_copy(src_file, dst_file)

print("✔ データ変換が完了しました")
