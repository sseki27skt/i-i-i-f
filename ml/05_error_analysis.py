#!/usr/bin/env python3
"""
05_error_analysis.py — MobileNet 用の定性的エラー分析 (FP/FN 画像抽出)

【目的】
1. テストセット等に対して推論を行い、誤判定（偽陽性 FP / 偽陰性 FN）を特定。
2. 誤判定画像をタイル状に並べた画像を生成（または個別保存）。
3. 信頼度 (Confidence) に基づき、モデルが「強く確信して間違えた」画像を優先。
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def analyze_errors_onnx(args):
    # 1. データの準備
    # ONNXの入力に合わせて手動で前処理 (torchvision不要で軽量に動く)
    def preprocess(pil_img):
        img = pil_img.resize((224, 224), Image.BILINEAR)
        img_data = np.array(img).astype(np.float32) / 255.0
        # Normalize (ImageNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        img_data = img_data.transpose(2, 0, 1) # HWC to CHW
        return np.expand_dims(img_data, axis=0) # CHW to NCHW

    # 2. ONNX セッションの開始
    print(f"Loading ONNX Model: {args.model_path}")
    session = ort.InferenceSession(args.model_path)
    input_name = session.get_inputs()[0].name
    
    # 3. データセットの走査
    fps = []
    fns = []
    
    data_path = Path(args.data_dir)
    # クラスフォルダ (Cover / Other) を想定
    classes = ["Other", "Cover"] # 国書DBのラベル順を想定 (0: Other, 1: Cover)
    
    # 手動でファイルをリストアップ
    all_files = []
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = data_path / cls_name
        if not cls_dir.exists():
            continue
        for f in cls_dir.glob("*.[jJ][pP][gG]"):
            all_files.append((f, cls_idx))
    
    if not all_files:
        print(f"No images found in {args.data_dir}")
        return

    # しきい値の取得 (メタデータにあれば使う、なければ0.5)
    threshold = 0.5
    try:
        # ONNXメタデータからの取得（あれば）
        meta = session.get_modelmeta().custom_metadata_map
        if "best_threshold" in meta:
            threshold = float(meta["best_threshold"])
            print(f"Using threshold from ONNX metadata: {threshold:.4f}")
    except:
        pass

    for img_path, label in tqdm(all_files, desc="Running Inference"):
        try:
            img = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(img)
            
            outputs = session.run(None, {input_name: input_tensor})
            logits = outputs[0]
            # Softmax
            exp_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
            
            prob_cover = probs[0, 1] # Index 1 is Cover
            pred = 1 if prob_cover >= threshold else 0
            
            if label == 0 and pred == 1: # FP
                fps.append({"path": str(img_path), "prob": prob_cover, "label": label})
            elif label == 1 and pred == 0: # FN
                fns.append({"path": str(img_path), "prob": prob_cover, "label": label})
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Results: FP={len(fps)}, FN={len(fns)}")

    # 4. 可視化
    model_name = Path(args.model_path).stem
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_tile_image(fps, "False_Positives", output_dir, max_imgs=48)
    save_tile_image(fns, "False_Negatives", output_dir, max_imgs=48)

    print(f"Error analysis completed. Check {output_dir}")

def save_tile_image(error_list, name, output_dir, max_imgs=48):
    if not error_list:
        print(f"No {name} found.")
        return

    # 信頼度が高い順（モデルが確信して間違えた順）にソート
    # FPの場合、probが1に近いほど確信、FNの場合、probが0に近いほど確信
    if "Positives" in name:
        error_list.sort(key=lambda x: x["prob"], reverse=True)
    else:
        error_list.sort(key=lambda x: x["prob"])
    
    selected = error_list[:max_imgs]
    
    # 224x224 のタイルを作成
    tile_size = 224
    cols = 8
    rows = (len(selected) + cols - 1) // cols
    
    combined_img = Image.new("RGB", (tile_size * cols, tile_size * rows), (255, 255, 255))
    
    for i, item in enumerate(selected):
        img = Image.open(item["path"]).convert("RGB")
        img.thumbnail((tile_size, tile_size))
        
        # 中央に配置
        x = (i % cols) * tile_size + (tile_size - img.width) // 2
        y = (i // cols) * tile_size + (tile_size - img.height) // 2
        
        combined_img.paste(img, (x, y))
        
        # テキスト追加 (信頼度)
        draw = ImageDraw.Draw(combined_img)
        text = f"P:{item['prob']:.3f}"
        draw.text(((i % cols) * tile_size + 5, (i // cols) * tile_size + 5), text, fill=(255, 0, 0))

    out_path = output_dir / f"{name}_tile.jpg"
    combined_img.save(out_path)
    print(f"Saved {name} tile to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="ml/dataset/test", help="Analysis target directory (test/val)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_xxx.pth")
    parser.add_argument("--output_dir", type=str, default="ml/reports/error_analysis")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    analyze_errors(args)
