#!/usr/bin/env python3
"""
05_evaluate_onnx.py — モデル横断比較レポート生成

使い方:
    python ml/05_evaluate_onnx.py
    python ml/05_evaluate_onnx.py --models_dir ml/models --dataset_dir ml/dataset

出力（ml/reports/ ディレクトリ）:
    comparison_report.md   — 全モデル比較 Markdown
    comparison_report.csv  — 数値のみ CSV（スプレッドシート用）
    confusion_{model}.png  — 混同行列
    confidence_dist.png    — Confidence 分布（ヒストグラム + 箱ひげ図）
    roc_pr_curves.png      — ROC / PR 曲線（3モデル重ね描き）
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境対応

import numpy as np
import onnxruntime as ort
import pandas as pd
import seaborn as sns
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
    precision_recall_curve, auc,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.transforms import get_eval_transforms, INPUT_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["content", "cover"]  # 0: content, 1: cover
COVER_IDX   = CLASS_NAMES.index("cover")


# ─────────────────────── データ読み込み ───────────────────────

def load_test_data(test_dir: Path) -> tuple[list[Path], list[int]]:
    """テストセットの画像パスとラベルを返す。"""
    img_paths: list[Path] = []
    labels: list[int]     = []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = test_dir / cls_name
        if not cls_dir.exists():
            logger.warning("Directory not found: %s", cls_dir)
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for p in sorted(cls_dir.glob(ext)):
                img_paths.append(p)
                labels.append(cls_idx)

    logger.info("Test set: %d images (cover=%d, content=%d)",
                len(img_paths),
                sum(1 for l in labels if l == COVER_IDX),
                sum(1 for l in labels if l != COVER_IDX))
    return img_paths, labels


def preprocess_image(img_path: Path) -> np.ndarray:
    """画像を ONNX 入力形式（float32 NCHW）に変換する。"""
    transform = get_eval_transforms()
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img)  # CHW float32
    return tensor.numpy()[np.newaxis]  # 1CHW


# ─────────────────────── ONNX 推論 ───────────────────────

def run_inference(
    ort_sess: ort.InferenceSession,
    img_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    全テスト画像に対して推論を実行する。

    Returns:
        probs      : (N, 2) Softmax 確率
        latencies  : (N,)   画像ごとのレイテンシ [ms]
        mean_lat   : 平均レイテンシ [ms]
        median_lat : 中央値レイテンシ [ms]
        p95_lat    : P95 レイテンシ [ms]
    """
    input_name = ort_sess.get_inputs()[0].name
    all_probs: list[np.ndarray] = []
    latencies: list[float] = []

    for img_path in tqdm(img_paths, desc="  Inference", unit="img", leave=False):
        inp = preprocess_image(img_path)

        t0 = time.perf_counter()
        logits = ort_sess.run(None, {input_name: inp})[0]  # (1, 2)
        latencies.append((time.perf_counter() - t0) * 1000)

        # Softmax
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        all_probs.append(probs[0])

    arr_probs = np.array(all_probs)         # (N, 2)
    arr_lat   = np.array(latencies)         # (N,)
    return (
        arr_probs,
        arr_lat,
        float(arr_lat.mean()),
        float(np.median(arr_lat)),
        float(np.percentile(arr_lat, 95)),
    )


# ─────────────────────── グラフ生成 ───────────────────────

def plot_confusion_matrix(
    y_true: list[int], y_pred: list[int],
    model_name: str, output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", output_path)


def plot_confidence_distribution(
    results_dict: dict[str, np.ndarray],  # model_name → cover probs
    output_path: Path,
) -> None:
    """Confidence 分布のヒストグラム + 箱ひげ図を重ね描き。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.Set2.colors

    for i, (model_name, probs) in enumerate(results_dict.items()):
        c = colors[i % len(colors)]
        axes[0].hist(probs, bins=40, alpha=0.6, label=model_name, color=c)

    axes[0].set_xlabel("Cover Probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence Distribution (cover class)")
    axes[0].legend()

    data_for_box = [p for p in results_dict.values()]
    axes[1].boxplot(data_for_box, labels=list(results_dict.keys()), patch_artist=True)
    axes[1].set_ylabel("Cover Probability")
    axes[1].set_title("Confidence Boxplot (cover class)")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confidence distribution → %s", output_path)


def plot_roc_pr_curves(
    y_true: list[int],
    probs_dict: dict[str, np.ndarray],  # model_name → cover probs
    output_path: Path,
) -> None:
    """ROC 曲線と PR 曲線を 1 図に重ね描き。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.Set1.colors

    for i, (model_name, probs) in enumerate(probs_dict.items()):
        c = colors[i % len(colors)]

        # ROC
        fpr, tpr, _ = roc_curve(y_true, probs, pos_label=COVER_IDX)
        auc_roc = roc_auc_score(y_true, probs)
        axes[0].plot(fpr, tpr, color=c, label=f"{model_name} (AUC={auc_roc:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_true, probs, pos_label=COVER_IDX)
        auc_pr = auc(rec, prec)
        axes[1].plot(rec, prec, color=c, label=f"{model_name} (AUC={auc_pr:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve"); axes[0].legend()

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC/PR curves → %s", output_path)


# ─────────────────────── レポート生成 ───────────────────────

def generate_report(
    rows: list[dict],
    reports_dir: Path,
) -> None:
    """Markdown + CSV レポートを生成する。"""
    df = pd.DataFrame(rows)

    # CSV
    csv_path = reports_dir / "comparison_report.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info("Saved CSV report → %s", csv_path)

    # Markdown
    md_path = reports_dir / "comparison_report.md"
    col_order = [
        "model", "onnx_size_mb",
        "best_threshold",
        "accuracy", "precision", "recall", "f1",
        "auc_roc", "auc_pr",
        "lat_mean_ms", "lat_median_ms", "lat_p95_ms",
    ]
    col_labels = {
        "model":          "Model",
        "onnx_size_mb":   "ONNX (MB)",
        "best_threshold": "Threshold",
        "accuracy":       "Accuracy",
        "precision":      "Precision",
        "recall":         "Recall",
        "f1":             "F1",
        "auc_roc":        "AUC-ROC",
        "auc_pr":         "AUC-PR",
        "lat_mean_ms":    "Lat Mean (ms)",
        "lat_median_ms":  "Lat Median (ms)",
        "lat_p95_ms":     "Lat P95 (ms)",
    }

    lines = [
        "# Model Comparison Report",
        "",
        "## Summary Table",
        "",
    ]
    header_cols = [col_labels[c] for c in col_order if c in df.columns]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for _, row in df.iterrows():
        vals = []
        for c in col_order:
            if c not in df.columns:
                continue
            v = row[c]
            if c == "model":
                vals.append(str(v))
            elif c == "onnx_size_mb":
                vals.append(f"{v:.2f}")
            elif c in ("lat_mean_ms", "lat_median_ms", "lat_p95_ms"):
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v:.4f}")
        lines.append("| " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Figures",
        "",
        "- `confusion_{model}.png` — 混同行列",
        "- `confidence_dist.png` — Confidence 分布（ヒストグラム・箱ひげ図）",
        "- `roc_pr_curves.png` — ROC / PR 曲線",
        "",
        "## Notes",
        "",
        "- Latency は CPU 推論（onnxruntime）の 1 画像あたり計測値。",
        "- F1, Precision, Recall は `cover` クラス正例として計算。",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved Markdown report → %s", md_path)


# ─────────────────────── メイン ───────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="全 ONNX モデルをテストセットで評価し比較レポートを生成する"
    )
    parser.add_argument(
        "--models_dir", type=Path, default=Path("ml/models"),
        help="ONNX モデルが格納されたディレクトリ (default: ml/models)",
    )
    parser.add_argument(
        "--dataset_dir", type=Path, default=Path("ml/dataset"),
        help="データセットルート（test/ サブディレクトリを使用）(default: ml/dataset)",
    )
    parser.add_argument(
        "--reports_dir", type=Path, default=Path("ml/reports"),
        help="レポート出力先 (default: ml/reports)",
    )
    parser.add_argument(
        "--providers", nargs="+", default=["CPUExecutionProvider"],
        help="onnxruntime 実行プロバイダ (default: CPUExecutionProvider)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # テストデータ読み込み
    test_dir = args.dataset_dir / "test"
    if not test_dir.exists():
        logger.error("Test directory not found: %s", test_dir)
        sys.exit(1)
    img_paths, y_true = load_test_data(test_dir)
    if not img_paths:
        logger.error("No test images found.")
        sys.exit(1)

    # 全 ONNX ファイルを検索
    onnx_files = sorted(args.models_dir.glob("*.onnx"))
    if not onnx_files:
        logger.error("No .onnx files found in %s", args.models_dir)
        sys.exit(1)
    logger.info("Found %d ONNX models: %s", len(onnx_files),
                [f.stem for f in onnx_files])

    rows: list[dict] = []
    cover_probs_dict: dict[str, np.ndarray] = {}

    for onnx_path in onnx_files:
        model_name = onnx_path.stem
        logger.info("=== Evaluating: %s ===", model_name)

        onnx_size_mb = onnx_path.stat().st_size / 1024 / 1024

        sess = ort.InferenceSession(
            str(onnx_path),
            providers=args.providers,
        )

        probs, latencies, lat_mean, lat_median, lat_p95 = run_inference(sess, img_paths)
        cover_probs = probs[:, COVER_IDX]  # cover クラスの確率

        # ONNX メタデータから最適しきい値を取得。なければ 0.5。
        meta = sess.get_modelmeta().custom_metadata_map
        best_threshold = float(meta.get("best_threshold", 0.5))
        logger.info("  Using threshold: %.4f (from ONNX metadata)" % best_threshold)

        y_pred = (cover_probs >= best_threshold).astype(int).tolist()

        # 精度指標
        acc   = accuracy_score(y_true, y_pred)
        prec  = precision_score(y_true, y_pred, pos_label=COVER_IDX, zero_division=0)
        rec   = recall_score(y_true, y_pred, pos_label=COVER_IDX, zero_division=0)
        f1    = f1_score(y_true, y_pred, pos_label=COVER_IDX, zero_division=0)
        auc_roc = roc_auc_score(y_true, cover_probs) if len(set(y_true)) > 1 else 0.0
        p, r, _ = precision_recall_curve(y_true, cover_probs, pos_label=COVER_IDX)
        auc_pr  = auc(r, p)

        logger.info("  Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f",
                    acc, prec, rec, f1)
        logger.info("  AUC-ROC=%.4f, AUC-PR=%.4f", auc_roc, auc_pr)
        logger.info("  Latency: mean=%.1fms, median=%.1fms, P95=%.1fms",
                    lat_mean, lat_median, lat_p95)
        logger.info("  ONNX size: %.2f MB", onnx_size_mb)

        # 混同行列
        plot_confusion_matrix(
            y_true, y_pred, model_name,
            args.reports_dir / f"confusion_{model_name}.png",
        )

        cover_probs_dict[model_name] = cover_probs
        rows.append({
            "model":          model_name,
            "onnx_size_mb":   round(onnx_size_mb, 2),
            "best_threshold": round(best_threshold, 4),
            "accuracy":       round(acc, 4),
            "precision":      round(prec, 4),
            "recall":         round(rec, 4),
            "f1":             round(f1, 4),
            "auc_roc":        round(auc_roc, 4),
            "auc_pr":         round(auc_pr, 4),
            "lat_mean_ms":    round(lat_mean, 2),
            "lat_median_ms":  round(lat_median, 2),
            "lat_p95_ms":     round(lat_p95, 2),
        })

    # 全モデル共通グラフ
    plot_confidence_distribution(
        cover_probs_dict,
        args.reports_dir / "confidence_dist.png",
    )
    plot_roc_pr_curves(
        y_true,
        cover_probs_dict,
        args.reports_dir / "roc_pr_curves.png",
    )

    # レポート生成
    generate_report(rows, args.reports_dir)
    logger.info("=== All done! Reports saved to %s ===", args.reports_dir)


if __name__ == "__main__":
    main()
