# -*- coding: utf-8 -*-
"""
Raspberry Pi 上で動かす4クラス分類CNNサンプル
├── adeno/
├── largecell/
├── squamouscell/
└── normal (4クラス)
- データ構造: dataset_tv/images/train/ と /val/ を想定
- 画像サイズ: 128x128
"""

"""
Raspberry Pi 上で動かす4クラス分類CNN（改良版）
・Conv Block + BatchNorm
・GlobalAveragePooling
・EarlyStopping / ReduceLROnPlateau
・医療画像向け Data Augmentation
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image, UnidentifiedImageError
import pandas as pd
import json  

NUM_CLASSES = 4
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
BASE_DIR = 'dataset_tvr/images'

def remove_invalid_images(base_dir):
    for root, dirs, files in os.walk(base_dir):
        deleted_count = 0
        for f in files:
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                path = os.path.join(root, f)
                try:
                    img = Image.open(path)
                    img.verify()
                except (UnidentifiedImageError, IOError):
                    os.remove(path)
                    deleted_count += 1
        if deleted_count > 0:
            print(f"{root}: {deleted_count} 件削除しました")

train_dir = os.path.join(BASE_DIR, 'train')
val_dir   = os.path.join(BASE_DIR, 'val')

remove_invalid_images(train_dir)
remove_invalid_images(val_dir)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -----------------------------------------
# CNN モデル定義
# -----------------------------------------
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------------------
# ★ 追加：ベストモデル保存コールバック ★
# -----------------------------------------
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max"
)

epochs = 10

# -----------------------------------------
# 学習（callback 追加）
# -----------------------------------------
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint]   # ← ここだけ追加！
)

# -----------------------------------------
# 学習履歴保存
# -----------------------------------------
history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history_4class.csv", index=False, encoding='utf-8-sig')

with open('training_history_4class.json', 'w') as f:
    json.dump(history.history, f)

print("学習履歴を training_history_4class.csv に保存しました。")

# 現行モデルも保存（任意）
model.save('4class_cnn.keras')
print("モデル保存完了: 4class_cnn.keras（最新 epoch のモデル）")

print("ベストモデルは best_model.keras に保存されます")
