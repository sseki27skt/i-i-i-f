#!/bin/bash
# ml/retry_failed_downloads.sh — 失敗した画像の再ダウンロードを安全に実行するスクリプト

# 設定（初回実行時と同じ値を維持することを推奨）
SEED=42
JSON_DIR="example_files"
OUTPUT_DIR="ml/dataset"
VAL_RATIO=0.1
TEST_RATIO=0.1

# 再試行向けの設定（強化版）
RETRIES=10      # リトライ回数を5から10に増加
TIMEOUT=60      # タイムアウトを30から60に延長
WORKERS=4        # サーバーへの負荷を最小限にするため、2ワーカー（逐次実行）に設定
DELAY=2.0       # リクエスト間の遅延を2.0秒に延長（+ランダムゆらぎ）

echo "--- Starting Retry of Failed Downloads ---"
echo "Seed: $SEED, Workers: $WORKERS, Retries: $RETRIES, Timeout: $TIMEOUT"

/home/seki/miniforge3/envs/pgx-seki/bin/python3 ml/01_download_dataset_v2.py \
    --json_dir "$JSON_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --val_ratio "$VAL_RATIO" \
    --test_ratio "$TEST_RATIO" \
    --workers "$WORKERS" \
    --retries "$RETRIES" \
    --timeout "$TIMEOUT" \
    --delay "$DELAY"

echo "--- Retry Process Completed ---"
