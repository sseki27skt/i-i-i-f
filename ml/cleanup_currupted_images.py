#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def verify_image(path):
    """画像が正しくロードできるか検証する。"""
    try:
        with Image.open(path) as img:
            img.load()  # JPEGの切断などはここでOSErrorを投げる
        return None  # 正常
    except Exception as e:
        return (path, str(e))

def main():
    parser = argparse.ArgumentParser(description="データセット内の破損・切断された画像をスキャンして削除します。")
    parser.add_argument("--dir", type=str, default="ml/dataset", help="スキャン対象ディレクトリ")
    parser.add_argument("--workers", type=int, default=16, help="並列スレッド数")
    parser.add_argument("--delete", action="store_true", help="破損ファイルを実際に削除する")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"Error: {args.dir} does not exist.")
        return

    print(f"🔍 スキャン開始: {root} (workers={args.workers})")
    
    # 画像ファイルの一覧を取得
    image_paths = list(root.rglob("*.jpg")) + list(root.rglob("*.jpeg")) + list(root.rglob("*.png"))
    total_files = len(image_paths)
    print(f"📸 合計画像数: {total_files}")

    corrupted = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(verify_image, p): p for p in image_paths}
        
        with tqdm(total=total_files, desc="Checking images", unit="img") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    path, error = result
                    corrupted.append(path)
                    tqdm.write(f"❌ CORRUPTED: {path} ({error})")
                pbar.update(1)

    print(f"\n📊 スキャン完了")
    print(f"❌ 検出された破損ファイル: {len(corrupted)}")

    if corrupted:
        if args.delete:
            print(f"🗑️  ファイルを削除しています...")
            for p in corrupted:
                try:
                    p.unlink()
                except Exception as e:
                    print(f"Failed to delete {p}: {e}")
            print(f"✅ 削除完了")
        else:
            print(f"💡 削除するには --delete フラグを付けて実行してください。")
            with open("corrupted_files.txt", "w") as f:
                for p in corrupted:
                    f.write(f"{p}\n")
            print(f"📎 破損ファイルリストを corrupted_files.txt に保存しました。")

if __name__ == "__main__":
    main()
