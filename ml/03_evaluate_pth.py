#!/usr/bin/env python3
"""
03_evaluate_pth.py — .pth 形式のモデルを横断評価し比較レポートを生成する
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["content", "cover"]
COVER_IDX = 1

# ==========================================
# 1. モデルロード用ユーティリティ
# ==========================================
def get_model_structure(model_name, num_classes=2):
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def load_pth_model(pth_path, device):
    checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    model = get_model_structure(model_name)
    
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."): # torch.compile 対応
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    threshold = checkpoint.get("best_threshold", 0.5)
    return model, model_name, threshold

# ==========================================
# 2. 推論実行
# ==========================================
def run_evaluation(model, loader, device, threshold):
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, COVER_IDX]
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)
    y_pred = (y_probs >= threshold).astype(int)
    
    return y_true, y_probs, y_pred

# ==========================================
# 3. グラフ生成
# ==========================================
def plot_results(y_true, results_dict, output_dir):
    # ROC / PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10.colors
    
    for i, (name, probs) in enumerate(results_dict.items()):
        c = colors[i % len(colors)]
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true, probs)
        score_roc = roc_auc_score(y_true, probs)
        axes[0].plot(fpr, tpr, color=c, label=f"{name} (AUC={score_roc:.3f})")
        
        # PR
        prec, rec, _ = precision_recall_curve(y_true, probs)
        score_pr = auc(rec, prec)
        axes[1].plot(rec, prec, color=c, label=f"{name} (AUC={score_pr:.3f})")
        
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend()
    
    axes[1].set_title("PR Curve")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_curves.png", dpi=150)
    plt.close()

# ==========================================
# 4. メインルーチン
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="ml/models")
    parser.add_argument("--val_dir", type=str, default="ml/dataset/val")
    parser.add_argument("--output_dir", type=str, default="ml/reports")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データロード
    val_transforms = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = ImageFolder(root=args.val_dir, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)
    
    # モデル検索
    pth_files = list(models_dir.glob("baseline_*.pth")) + list(models_dir.glob("best_*.pth"))
    if not pth_files:
        logger.error("No benchmark .pth files found in ml/models.")
        return
    
    rows = []
    results_probs = {}
    y_true_final = None
    
    for pth_path in sorted(pth_files):
        display_name = pth_path.stem
        logger.info(f"Evaluating: {display_name}")
        
        model, model_name, threshold = load_pth_model(pth_path, device)
        y_true, y_probs, y_pred = run_evaluation(model, val_loader, device, threshold)
        
        if y_true_final is None: y_true_final = y_true
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_probs)
        
        rows.append({
            "Variant": display_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC-ROC": auc_roc,
            "Threshold": threshold
        })
        results_probs[display_name] = y_probs
        
    # レポート生成
    df = pd.DataFrame(rows)
    md_path = output_dir / "benchmark_report.md"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Benchmark Report (Baseline vs Optimized)\n\n")
        f.write("## Summary Metrics\n\n")
        f.write(df.to_markdown(index=False) + "\n\n")
        f.write("## Curves\n\n![Benchmark Curves](benchmark_curves.png)\n\n")
        f.write("## Notes\n\n")
        f.write("- **Baseline**: Constant LR, standard CrossEntropy, no custom sampler.\n")
        f.write("- **Best (Optimized)**: Layer-wise LR, Focal Loss, Weighted Sampler, OneCycleLR.\n")

    plot_results(y_true_final, results_probs, output_dir)
    logger.info(f"Benchmark completed. Report saved to {md_path}")

if __name__ == "__main__":
    main()
