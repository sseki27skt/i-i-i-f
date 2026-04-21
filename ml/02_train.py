#!/usr/bin/env python3


import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm

# ==========================================
# 1. 不均衡データ特化: Focal Loss (バランス調整版)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, label_smoothing: float = 0.0):
        """
        Args:
            alpha: クラス間の重み。Samplerで均衡化されているため 0.5 (等分) を推奨。
            gamma: 難判定サンプルへの集中度。
            label_smoothing: 過学習抑制のためのラベル平滑化。
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # targets == 1 (Cover) に alpha, targets == 0 (Content) に (1-alpha)
        at = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Label Smoothing を適用した CrossEntropy
        ce_loss = nn.functional.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        
        focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ==========================================
# 2. 動的しきい値探索アルゴリズム
# ==========================================
def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * (precisions * recalls) / (precisions + recalls),
            0.0,
        )
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1

# ==========================================
# 3. 超高速化: In-Memory Dataset
# ==========================================
class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, desc="Loading to RAM", max_workers=16):
        """
        初回エポックの前に全データをRAMに展開する。
        16コア並列処理により、Grace CPUの性能を活かしてロード時間を劇的に短縮する。
        """
        self.samples = []
        self.targets = []
        self.classes = dataset.classes
        
        indices = list(range(len(dataset)))
        num_imgs = len(indices)
        
        # 4コア分をシステム用に残し、16コア(またはCPU上限)で並列ロード
        actual_workers = min(max_workers, (os.cpu_count() or 1))
        print(f"📦 {desc} (Using {actual_workers} cores)...")

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # mapは順序を保持する
            results = list(tqdm(executor.map(lambda i: dataset[i], indices), 
                                total=num_imgs, desc=desc, leave=False))
        
        for sample, target in results:
            self.samples.append(sample)
            self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

# ==========================================
# 4. モデル定義ヘルパー
# ==========================================
def get_model(model_name, num_classes, dropout_p=0.2):
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        # Dropout を挟んだ改良版ヘッド
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = model.classifier[3].in_features
        # 既存の [0]:Linear, [1]:Hardswish, [2]:Dropout(0.2) を再利用し、
        # [3]:Linear(1280, 1000) を新設の Linear(1280, num_classes) に置換。
        # 余計な nn.Dropout は追加しない。
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

# ==========================================
# 4. 個別モデルの学習実行関数
# ==========================================
def run_training(args, model_name, device, train_loader, val_loader, class_names, train_gpu_transforms, val_gpu_transforms):
    print(f"\n🚀 学習開始: {model_name} (最大 {args.epochs}エポック, Patience: {args.patience})")
    
    model = get_model(model_name, len(class_names))
    model.to(device)
    if hasattr(torch, 'compile'):
        print(f"⏳ {model_name} をコンパイル中...")
        model = torch.compile(model, mode="reduce-overhead")

    log_name = f"{time.strftime('%Y%m%d-%H%M%S')}_{model_name}"
    log_dir = Path("ml/reports/logs") / log_name
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 1. 層別学習率 (Layer-wise Learning Rate) の設定
    # バックボーンは 0.1倍、新設Classifierは 1.0倍の学習率を適用
    if model_name == "mobilenet_v2":
        backbone_params = list(model.features.parameters())
        classifier_params = list(model.classifier.parameters())
    else: # v3_large
        # classifier[0] (Linear 960->1280) は事前学習済みなので、
        # バックボーンと同等の低い学習率を適用する
        backbone_params = list(model.features.parameters()) + list(model.classifier[0].parameters())
        classifier_params = list(model.classifier[3].parameters())

    # --- 段階的解凍 (Gradual Unfreezing) の設定 ---
    backbone_unfrozen = True
    if args.freeze_epochs > 0:
        print(f"❄️  Backbone frozen initially (Max {args.freeze_epochs} epochs, Min 5 epochs).")
        for param in backbone_params:
            param.requires_grad = False
        backbone_unfrozen = False

    optimizer = optim.AdamW([
        # {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": backbone_params, "lr": args.lr},
        {"params": classifier_params, "lr": args.lr}
    ], weight_decay=1e-4)

    # 2. OneCycleLR スケジューラの設定
    # バックボーンと Classifier それぞれに設定した学習率を最大値としてサイクル
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        # max_lr=[args.lr * 0.1, args.lr],
        max_lr=[args.lr, args.lr],
        steps_per_epoch=len(train_loader),
        epochs=args.epochs
    )
    
    # 二重掛けを避けつつ難判定に集中 (Label Smoothing は 0.1)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_val_ap = 0.0
    patience_counter = 0
    best_path = Path(args.model_dir) / f"best_{model_name}.pth"

    prev_val_ap = 0.0
    for epoch in range(1, args.epochs + 1):
        # (解凍判定は各エポックの最後で、次エポックのために行う)

        start_time = time.time()
        
        # --- Train ---
        model.train()
        train_running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"[{model_name}] Ep {epoch:02d} Train", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            images = train_gpu_transforms(images)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_running_loss += loss.item() * images.size(0)
            train_pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
        train_loss = train_running_loss / (len(train_loader.dataset))

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        all_val_targets, all_val_probs = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"[{model_name}] Ep {epoch:02d} Val", leave=False):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                images = val_gpu_transforms(images)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_val_targets.append(labels)
                all_val_probs.append(probs)

        # 全バッチ終了後に一括でCPUへ転送 (同期を最小化)
        val_targets = torch.cat(all_val_targets).cpu().numpy()
        val_probs = torch.cat(all_val_probs).cpu().numpy()

        val_loss = val_running_loss / (len(val_loader.dataset))
        
        val_ap = average_precision_score(val_targets, val_probs)
        best_threshold, val_f1 = find_best_threshold(val_targets, val_probs)
        
        # Cover (クラス1) の Precision / Recall を計算
        val_preds = (val_probs >= best_threshold).astype(int)
        tp = ((val_preds == 1) & (val_targets == 1)).sum()
        fp = ((val_preds == 1) & (val_targets == 0)).sum()
        fn = ((val_preds == 0) & (val_targets == 1)).sum()
        
        cover_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        cover_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        val_acc = (val_preds == val_targets).mean()
        
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[-1]['lr'] # Classifierの学習率を表示

        print(f"Epoch {epoch:02d}/{args.epochs} [{elapsed:.1f}s] LR: {current_lr:.2e} | Val AP: {val_ap:.4f} F1: {val_f1:.4f} (P:{cover_precision:.3f} R:{cover_recall:.3f} Thr:{best_threshold:.3f})")

        writer.add_scalar("Params/Learning_Rate", current_lr, epoch)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Metrics/Val_AP", val_ap, epoch)
        writer.add_scalar("Metrics/Val_F1_Score", val_f1, epoch)
        writer.add_scalar("Metrics/Cover_Precision", cover_precision, epoch)
        writer.add_scalar("Metrics/Cover_Recall", cover_recall, epoch)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            patience_counter = 0
            checkpoint = {
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": float(val_acc),
                "val_ap": float(val_ap),
                "cover_f1": float(val_f1),
                "cover_precision": float(cover_precision),
                "cover_recall": float(cover_recall),
                "best_threshold": float(best_threshold),
                "class_names": class_names
            }
            torch.save(checkpoint, best_path)
            print(f"  🌟 Best Model (AP) updated! -> {best_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"🛑 Early Stopping at epoch {epoch}")
                break

        # --- 段階的解凍の動的判定 (次エポックに適用) ---
        if not backbone_unfrozen:
            improvement = val_ap - prev_val_ap if epoch > 1 else val_ap
            should_unfreeze = False
            
            if epoch >= args.freeze_epochs:
                print(f"\n📢 Epoch {epoch}: Next epoch will unfreeze backbone (Hard limit of {args.freeze_epochs} reached).")
                should_unfreeze = True
            elif epoch >= 5 and improvement < 0.002:
                # 5エポック以上経過し、かつAPの改善が0.2%未満なら「停滞」とみなす
                print(f"\n🔥 Epoch {epoch}: Next epoch will unfreeze backbone! Head performance plateaued (Imp: {improvement:.4f} < 0.002).")
                should_unfreeze = True
            
            if should_unfreeze:
                for param in backbone_params:
                    param.requires_grad = True
                backbone_unfrozen = True
        
        prev_val_ap = val_ap
    writer.close()

# ==========================================
# 5. メインルーチン
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="MobileNet Training (Version Final)")
    parser.add_argument("--model", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "mobilenet_v3_large"])
    parser.add_argument("--all", action="store_true", help="全モデルを連続学習する")
    parser.add_argument("--train_dir", type=str, default="ml/dataset/train")
    parser.add_argument("--val_dir", type=str, default="ml/dataset/val")
    parser.add_argument("--model_dir", type=str, default="ml/models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for classifier (Backbone will be 0.1x)")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--freeze_epochs", type=int, default=40, help="Max number of epochs to freeze backbone (dynamic unfreeze may trigger earlier)")
    parser.add_argument("--focal_alpha", type=float, default=0.5, help="Alpha for Focal Loss (0.5 means balanced as sampler handles it)")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma for Focal Loss")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate before classifier")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.model_dir, exist_ok=True)

    # 共通データロード
    # 共通データロード (CPU側は最低限のサイズ調整のみ)
    cpu_transforms = v2.Compose([
        v2.Resize((256, 256), antialias=True), # 一旦少し大きめにリサイズ
        v2.ToImage(),
        # GPU側でまとめて float32 化するため、ここでは uint8 のまま保持
    ])
    train_dataset = ImageFolder(root=args.train_dir, transform=cpu_transforms)
    val_dataset = ImageFolder(root=args.val_dir, transform=cpu_transforms)
    
    # --- RAMキャッシュ化 (Blackwellの性能を最大化) ---
    print(f"🧠 RAMキャッシュ化を開始します (推定使用量: {len(train_dataset)+len(val_dataset)} 枚)")
    train_dataset = InMemoryDataset(train_dataset, desc="Train Dataset -> RAM")
    val_dataset = InMemoryDataset(val_dataset, desc="Val Dataset -> RAM")

    class_names = train_dataset.classes

    # サンプラー用重み計算 (DGX Spark/Grace 用にベクトル演算で高速化)
    # InMemoryDataset の targets (リスト) を使用
    targets = np.array(train_dataset.targets)
    counts = np.bincount(targets)
    class_weights = len(targets) / (len(class_names) * counts.astype(float))
    
    # Pythonループを避け、PyTorchのベクトルインデックス参照を使用
    class_weights_t = torch.from_numpy(class_weights).to(torch.float32)
    targets_t = torch.tensor(train_dataset.targets, dtype=torch.long)
    sample_weights = class_weights_t[targets_t]

    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(targets), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=10, pin_memory=True, persistent_workers=True, prefetch_factor=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True, persistent_workers=True, prefetch_factor=8)

    train_gpu_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]).to(device)
    val_gpu_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]).to(device)

    models_to_train = ["mobilenet_v2", "mobilenet_v3_large"] if args.all else [args.model]
    for m_name in models_to_train:
        run_training(args, m_name, device, train_loader, val_loader, class_names, train_gpu_transforms, val_gpu_transforms)

    print("\n✅ All training tasks completed (Final Version)!")

if __name__ == "__main__":
    main()
