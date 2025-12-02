# -*- coding: utf-8 -*-
"""
dataset_j/
├── adeno/
├── largecell/
├── squamouscell/
└── normal
フォルダの画像データの名前を数字のみにする


"""
import os
import re # 正規表現モジュールを使用
import shutil

# 元データ
src_root = "data_j"
splits = ["train", "valid", "test"]

# 出力先
dst_root = "dataset_j"
classes = ["adeno", "largecell", "squamouscell", "normal"]

# 出力先フォルダを作成（なければ）
for cls in classes:
    os.makedirs(os.path.join(dst_root, cls), exist_ok=True)

# train / valid / test を順に処理
for split in splits:
    for cls in classes:
        src_dir = os.path.join(src_root, split, cls)
        dst_dir = os.path.join(dst_root, cls)

        if not os.path.exists(src_dir):
            print(f"Skipping missing folder: {src_dir}")
            continue

        # 画像をコピー
        for filename in os.listdir(src_dir):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)

            # 上書き防止のため、同名ファイルがある場合は名前を変更
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(filename)
                i = 1
                new_filename = f"{name}_{i}{ext}"
                new_dst_path = os.path.join(dst_dir, new_filename)
                while os.path.exists(new_dst_path):
                    i += 1
                    new_filename = f"{name}_{i}{ext}"
                    new_dst_path = os.path.join(dst_dir, new_filename)
                dst_path = new_dst_path

            shutil.copy2(src_path, dst_path)

        print(f"Copied: {src_dir} → {dst_dir}")

print("📦 完了しました！ data_j2 に統合されました。")

# --- 設定 ---
ROOT_DIR = "dataset_j" 

# 処理対象とするクラスフォルダ名
CLASSES = ['adeno', 'largecell', 'squamouscell', 'normal']

# 画像として処理する拡張子
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

# --- メイン処理 ---

def rename_files_to_numbers_only():
    """
    クラスフォルダ内のファイルを走査し、ファイル名の先頭にある英字プレフィックスを
    取り除いて数字のみの名前にリネームします。
    """
    print(f"ターゲットディレクトリ: {ROOT_DIR}")
    print("-" * 40)
    
    total_renamed_count = 0
    
    for class_name in CLASSES:
        target_dir = os.path.join(ROOT_DIR, class_name)
        
        if not os.path.exists(target_dir):
            print(f"⚠️ 警告: クラスフォルダ '{target_dir}' が見つかりません。スキップします。")
            continue
            
        print(f"\n--- クラス '{class_name}' の処理を開始 ---")
        renamed_count = 0

        for filename in os.listdir(target_dir):
            src_path = os.path.join(target_dir, filename)
            
            # ディレクトリや隠しファイルはスキップ
            if os.path.isdir(src_path) or filename.startswith('.'):
                continue
            
            name, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            # 画像ファイルであるかチェック
            if ext not in IMAGE_EXTENSIONS:
                continue
                
            # 1. プレフィックスの検出と除去
            
            # 正規表現: ファイル名の先頭にある英字とアンダースコア（_）を無視し、
            # その後に続く数字の連続を抽出する
            match = re.search(r'([a-zA-Z_]+)?(\d+)', name) 
            
            new_name = None
            if match:
                # グループ2が数字部分
                new_name = match.group(2) 
            
            # 2. リネームの実行
            if new_name and new_name != name:
                new_filename = new_name + ext
                dst_path = os.path.join(target_dir, new_filename)

                try:
                    # ファイル名を変更
                    os.rename(src_path, dst_path)
                    renamed_count += 1
                    # print(f"  リネーム: {filename} -> {new_filename}") 
                except Exception as e:
                    print(f"❌ リネーム失敗: {filename}。原因: {e}")
            
        print(f"  結果: {renamed_count} 個のファイルをリネームしました。")
        total_renamed_count += renamed_count
        
    print("-" * 40)
    print(f"🎉 全ての処理が完了しました。総リネーム数: {total_renamed_count} 個。")

if __name__ == "__main__":
    rename_files_to_numbers_only()