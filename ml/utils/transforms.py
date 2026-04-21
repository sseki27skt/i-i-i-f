"""
transforms.py — データ拡張・前処理設定

MobileNet 学習用の ImageNet 標準正規化と拡張パイプラインを提供する。
"""

from torchvision import transforms

# ImageNet 統計量
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# MobileNet の標準入力サイズ
INPUT_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """学習用：データ拡張あり"""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),  # 少し大きめにリサイズ
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """検証・テスト用：拡張なし（中央クロップのみ）"""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
