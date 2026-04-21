#!/usr/bin/env python3
"""
04_export_onnx.py — 学習済み PyTorch モデルを ONNX へ変換する

※ best_threshold がチェックポイントに含まれる場合、ONNX のメタデータに埋め込みます。
   推論時は onnx_model.metadata_props から取得できます。

使い方:
    python ml/04_export_onnx.py --model mobilenet_v2
    python ml/04_export_onnx.py --model mobilenet_v3_small
    python ml/04_export_onnx.py --model mobilenet_v3_large

    # 全モデルを一括変換
    python ml/04_export_onnx.py --all

出力:
    ml/models/{model_name}.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import onnx

sys.path.insert(0, str(Path(__file__).parent))

# utils/transforms が存在する場合はそちらを使用、なければフォールバック
try:
    from utils.transforms import INPUT_SIZE
except ImportError:
    INPUT_SIZE = 224

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]


def load_model(pth_path: Path, device: torch.device):
    """保存済み .pth チェックポイントからモデルを復元する。"""
    from torchvision import models
    import torch.nn as nn

    CLASS_NAMES = ["content", "cover"]
    num_classes = len(CLASS_NAMES)

    checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        # 02_train_new_final.py に合わせる
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "mobilenet_v3_small":
        # Note: 02_train_new_final.py では未定義だが、必要なら同様の変更を行う
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        # 02_train_new.py に合わせる (既存の [3]:Linear を置換)
        model.classifier[3] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict = checkpoint["state_dict"]
    # 🔄 [解決策] torch.compile で保存された場合、キーに "_orig_mod." プレフィックスが付くため、それを取り除く
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded %s (epoch=%d, val_acc=%.4f, cover_f1=%.4f)",
                model_name,
                checkpoint.get("epoch", -1),
                checkpoint.get("val_acc", 0),
                checkpoint.get("cover_f1", 0))
    threshold = checkpoint.get("best_threshold", 0.5)
    logger.info("  Best threshold (from checkpoint): %.4f", threshold)
    return model, model_name, threshold


def export_to_onnx(
    model: torch.nn.Module,
    model_name: str,
    output_path: Path,
    opset_version: int = 16,
    device: torch.device = torch.device("cpu"),
    best_threshold: float = 0.5,
) -> None:
    """モデルを ONNX 形式でエクスポートし、best_threshold をメタデータに埋め込む。"""
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    logger.info("Exporting Model: %s", model_name)

    model.to(device)
    model.eval()
    
    # 🔄 [重要] PyTorch 2.5+ の Dynamo エクスポーターの不具合を避けるため、
    # 確実に Legacy パスを通るように torch.jit.trace を使用します。
    try:
        logger.info("Tracing model with torch.jit.trace...")
        traced_model = torch.jit.trace(model, dummy_input)
    except Exception as e:
        logger.error("Tracing failed: %s", e)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = {
        "opset_version": opset_version,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        "export_params": True,
        "do_constant_folding": True,
    }

    try:
        # 🔄 [解決策] torch.onnx.export の代わりに 
        # torch.onnx.utils.export を直接叩くことで、新機能による
        # 強制的なシグネチャ検査とエラー(Failed to convert dynamic_axes)を回避します。
        from torch.onnx import utils
        utils.export(
            traced_model, 
            dummy_input, 
            str(output_path), 
            **export_kwargs
        )
        
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info("ONNX saved → %s (%.2f MB)", output_path, file_size_mb)

        if file_size_mb < 5.0:
            logger.warning("⚠ Output file size is unexpectedly small. Weights might be missing!")

        # ── best_threshold をメタデータとして ONNX に埋め込む ──────────────
        try:
            onnx_model = onnx.load(str(output_path))
            meta = onnx_model.metadata_props.add()
            meta.key   = "best_threshold"
            meta.value = str(best_threshold)
            meta_class = onnx_model.metadata_props.add()
            meta_class.key   = "class_names"
            meta_class.value = "content,cover"  # 0: content, 1: cover
            onnx.save(onnx_model, str(output_path))
            logger.info("  Embedded metadata: best_threshold=%.4f", best_threshold)
        except Exception as meta_err:
            logger.warning("  Could not embed metadata: %s", meta_err)

    except Exception as e:
        logger.error("ONNX export failed: %s", e)
        raise e

def verify_onnx(pth_path: Path, onnx_path: Path, device: torch.device) -> None:
    """
    PyTorch 出力と ONNX 出力を比較し、最大絶対誤差を確認する。
    差異が 1e-4 以上の場合は警告を出す。
    """
    import onnxruntime as ort

    # メタデータの確認
    try:
        onnx_model = onnx.load(str(onnx_path))
        for prop in onnx_model.metadata_props:
            logger.info("  ONNX metadata: %s = %s", prop.key, prop.value)
    except Exception:
        pass

    model, _, _ = load_model(pth_path, device)

    dummy_np = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_np).to(device)

    with torch.no_grad():
        pt_out = torch.softmax(model(dummy_tensor), dim=1).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    ort_out_raw = sess.run(None, {input_name: dummy_np})[0]
    # ONNX 出力は logits のため softmax を適用
    exp_out = np.exp(ort_out_raw - ort_out_raw.max(axis=1, keepdims=True))
    ort_out = exp_out / exp_out.sum(axis=1, keepdims=True)

    max_diff = float(np.abs(pt_out - ort_out).max())
    if max_diff < 1e-4:
        logger.info("✓ PyTorch vs ONNX max diff=%.2e (OK)", max_diff)
    else:
        logger.warning("⚠ PyTorch vs ONNX max diff=%.2e (large — check carefully)", max_diff)


def process_pth_file(pth_path: Path, device: torch.device) -> None:
    """.pth ファイルを指定して ONNX への変換・検証を行う。"""
    # 出力ファイル名の決定: best_ で始まる場合はそれを取り除く (従来互換)
    stem = pth_path.stem
    if stem.startswith("best_"):
        onnx_name = stem[5:]
    else:
        onnx_name = stem
    onnx_path = pth_path.parent / f"{onnx_name}.onnx"

    logger.info("--- Converting %s ---", pth_path.name)
    try:
        model, resolved_name, best_threshold = load_model(pth_path, device)
        export_to_onnx(model, resolved_name, onnx_path, device=device, best_threshold=best_threshold)
        verify_onnx(pth_path, onnx_path, device)
    except Exception as e:
        logger.error("Failed to process %s: %s", pth_path.name, e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch チェックポイントを ONNX へ変換する"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--model", choices=SUPPORTED_MODELS,
        help="変換する特定のモデル名 (例: mobilenet_v2)",
    )
    group.add_argument(
        "--all", action="store_true",
        help="models/ 内の全 .pth を変換する (デフォルト)",
    )
    parser.add_argument(
        "--models_dir", type=Path, default=Path("ml/models"),
        help="チェックポイント・ONNX の格納ディレクトリ (default: ml/models)",
    )
    parser.add_argument(
        "--opset", type=int, default=16, # 18でエラーが出る場合は16を推奨
        help="ONNX opset バージョン (default: 18)",
    )

    parser.add_argument(
        "--device", default="cpu",
        help="変換に使用するデバイス (default: cpu を推奨)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.model:
        # 特定のモデルが指定された場合
        pth_path = args.models_dir / f"best_{args.model}.pth"
        if not pth_path.exists():
            # best_ なしのパスも試行
            pth_path = args.models_dir / f"{args.model}.pth"
        
        if pth_path.exists():
            process_pth_file(pth_path, device)
        else:
            logger.error("Checkpoint not found for model: %s (checked best_%s.pth and %s.pth)", 
                         args.model, args.model, args.model)
            sys.exit(1)
    else:
        # デフォルトまたは --all: ディレクトリ内の全 .pth を対象にする
        pth_files = sorted(list(args.models_dir.glob("*.pth")))
        if not pth_files:
            logger.warning("No .pth files found in %s", args.models_dir)
            return
        
        logger.info("Found %d checkpoint(s) in %s", len(pth_files), args.models_dir)
        for pth_path in pth_files:
            process_pth_file(pth_path, device)


if __name__ == "__main__":
    main()
