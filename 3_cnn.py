# -*- coding: utf-8 -*-
"""
TensorFlowのライセンスは、Apache License 2.0です。
"""
"""
Raspberry Pi 上で動かす犬猫分類CNNサンプル
- データ：dataset/train, dataset/val に犬猫画像を用意済み
- 画像サイズ：128x128
- クラス：猫=0, 犬=1
"""


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

from PIL import Image, UnidentifiedImageError
import os
import  pandas as pd
import  json  

def remove_invalid_images(base_dir):
    """
    base_dir 以下のすべての画像ファイルをチェックし、
    壊れた画像は削除。
    削除数をディレクトリ単位で表示。
    """
    for root, dirs, files in os.walk(base_dir):
        deleted_count = 0
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                path = os.path.join(root, f)
                try:
                    img = Image.open(path)
                    img.verify()  # 破損チェック
                except (UnidentifiedImageError, IOError):
                    os.remove(path)
                    deleted_count += 1
        if deleted_count > 0:
            print(f"{root}: {deleted_count} 件削除しました")


# ディレクトリ指定
train_dir = 'dataset_sr/train'
val_dir   = 'dataset_sr/val'

# train/val 両方チェック
remove_invalid_images(train_dir)
remove_invalid_images(val_dir)


# -----------------------------------------
# 1. データジェネレータの作成
# -----------------------------------------
# 画像をリスケール（0-1に正規化）して、学習時に少し拡張
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # 画像をランダムに回転
    width_shift_range=0.1,  # 水平方向にランダムシフト
    height_shift_range=0.1, # 垂直方向にランダムシフト
    horizontal_flip=True    # 左右反転
)

val_datagen = ImageDataGenerator(rescale=1./255)  # 検証データは正規化のみ



# データジェネレータを作成
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),  # 画像サイズ
    batch_size=8,
    class_mode='binary'     # 犬 vs 猫の2クラス
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128,128),
    batch_size=8,
    class_mode='binary'
)

# -----------------------------------------
# 2. CNNモデルの定義
# -----------------------------------------
model = models.Sequential()

# 畳み込み層1
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(layers.MaxPooling2D((2,2)))

# 畳み込み層2
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# 畳み込み層3
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# Flatten
model.add(layers.Flatten())

# 全結合層
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # 過学習防止
model.add(layers.Dense(1, activation='sigmoid'))  # 出力層（0=猫, 1=犬）

# # 最近の書き方
# model = Sequential([
#     Input(shape=(128, 128, 3)),
#     Conv2D(32, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# -----------------------------------------
# 3. モデルのコンパイル
# -----------------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------------
# 4. 学習
# -----------------------------------------

epochs = 2  # Raspberry Piでは10~20程度が妥当

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# -----------------------------------------
# 5. 学習履歴の保存
# -----------------------------------------
# 学習履歴（history.history）をDataFrameに変換
history_df = pd.DataFrame(history.history)
# CSVファイルとして保存
history_df.to_csv("training_history.csv", index=False, encoding='utf-8-sig')
# history.history は dict形式
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

print("学習履歴を training_history.csv に保存しました。")

# -----------------------------------------
# 6. 学習済みモデルの保存
# -----------------------------------------
model.save('cats_vs_dogs_cnn.h5')
print("モデル保存完了: cats_vs_dogs_cnn.h5")
