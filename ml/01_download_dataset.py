#!/usr/bin/env python3
"""
01_download_dataset_v2.py — IIIF画像ダウンロード & スマート同期 (8:1:1分割保証版)

主な強化内容:
1. URL単位での重複排除: 同一画像が複数のJSONに含まれていても1枚として扱います。
2. スマート同期: 分割の割り当てが変わった既存ファイルを「再ダウンロード」せず「ディスク内で移動」します。
3. クリーンアップ: 現在のデータセット構成に含まれない古いファイルを自動削除し、ディスク上の比率を8:1:1に保ちます。
4. サーバ負荷低減: User-Agent設定、リクエスト間遅延、指数バックオフ、429(Too Many Requests)対応。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import requests
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────── データ構造 ───────────────────────

class ImageRecord(NamedTuple):
    canvas_id: str
    image_url: str
    label: str          # "cover" or "content"
    source_file: str    # 元のJSONファイル名


# ─────────────────────── ユーティリティ ───────────────────────

def safe_filename(url: str) -> str:
    """URL から安全なファイル名を生成する。"""
    base_url = url.split("?")[0]
    name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", base_url)
    
    lower_name = name.lower()
    if lower_name.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
        return name[-200:]
    return name[-200:] + ".jpg"


def download_image(
    session: requests.Session,
    record: ImageRecord,
    dest_path: Path,
    timeout: int = 30,
    retries: int = 5,
    delay: float = 0.0,
) -> bool:
    """1枚の画像をダウンロードして保存する。強力なバックオフ付き。"""
    if dest_path.exists():
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    timeout_tuple = (10, timeout)
    for attempt in range(retries):
        if delay > 0:
            # 指定された遅延に 50% ~ 150% のゆらぎ（ジッター）を追加
            actual_delay = delay * random.uniform(0.5, 1.5)
            time.sleep(actual_delay)

        try:
            # ブラウザに近い詳細なヘッダー
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
                "Referer": "https://kokusho.nijl.ac.jp/",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Sec-Fetch-Dest": "image",
                "Sec-Fetch-Mode": "no-cors",
                "Sec-Fetch-Site": "same-site",
            }
            with session.get(record.image_url, timeout=timeout_tuple, stream=True, headers=headers) as resp:
                if resp.status_code == 429:
                    # レートリミット時は指数バックオフで長めに待機
                    wait = (2 ** attempt) * 30
                    logger.warning("Rate limited (429). Waiting %ds for %s", wait, record.image_url)
                    time.sleep(wait)
                    continue
                
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
            
            try:
                with Image.open(dest_path) as img:
                    img.load()  # verify() is too weak for truncated JPEG detection
                return True
            except Exception as e:
                logger.warning("Invalid image: %s — %s. Removing.", record.image_url, e)
                dest_path.unlink(missing_ok=True)
        except Exception as exc:
            # 一般的なエラー時も指数バックオフ
            wait = (2 ** attempt) * 2
            if attempt < retries - 1:
                logger.debug("Retry %d/%d: %s wait %ds (Error: %s)", attempt + 1, retries, record.image_url, wait, exc)
                time.sleep(wait)
            else:
                logger.error("Failed final attempt: %s — %s", record.image_url, exc)
    return False


# ─────────────────────── データ収集 ───────────────────────

def load_records(json_dir: Path) -> list[ImageRecord]:
    """JSON ディレクトリを全件読み込み、URL単位で重複排除したリストを返す。"""
    unique_records: dict[str, ImageRecord] = {}
    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")

    logger.info("Loading %d JSON files from %s...", len(json_files), json_dir)
    
    duplicate_count = 0
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                label = item.get("label", "").strip().lower()
                if label not in ("cover", "content"):
                    continue
                
                url = item.get("imageUrl", "")
                if not url:
                    continue
                
                if url in unique_records:
                    duplicate_count += 1
                    continue
                
                unique_records[url] = ImageRecord(
                    canvas_id=item.get("id", ""),
                    image_url=url,
                    label=label,
                    source_file=jf.name,
                )
        except Exception as exc:
            logger.error("Error reading %s: %s", jf.name, exc)

    if duplicate_count > 0:
        logger.info("Skipped %d duplicate URLs found across JSON files.", duplicate_count)
    
    records = list(unique_records.values())
    logger.info("Total unique records loaded: %d", len(records))
    return records


# ─────────────────────── Stratified Split ───────────────────────

def stratified_split(
    records: list[ImageRecord],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    """層化抽出を行い、Train/Val/Test に分割する。"""
    labels = [r.label for r in records]
    
    train_recs, rest_recs, _, rest_labels = train_test_split(
        records, labels, test_size=(val_ratio + test_ratio),
        stratify=labels, random_state=seed
    )
    val_recs, test_recs = train_test_split(
        rest_recs, test_size=0.5,
        stratify=rest_labels, random_state=seed
    )
    
    # 統計を表示
    for name, recs in [("Train", train_recs), ("Val", val_recs), ("Test", test_recs)]:
        counts = {"cover": 0, "content": 0}
        for r in recs:
            counts[r.label] += 1
        logger.info(f"Split {name:5}: Total={len(recs):6}, Cover={counts['cover']:5}, Content={counts['content']:6}")
        
    return train_recs, val_recs, test_recs


# ─────────────────────── スマート同期 & ダウンロード ───────────────────────
def sync_and_download(
    split_map: dict[str, list[ImageRecord]],
    output_dir: Path,
    workers: int,
    failed_log: Path,
    delay: float = 0.0,
    retries: int = 5,
    timeout: int = 30,
) -> None:
    """
    ディスク上の状態を理想の状態 (split_map) に同期する。
    1. 既存ファイルを検索
    2. 違う場所にある場合は移動
    3. 不要なファイルは削除
    4. 足りないファイルはダウンロード
    """
    expected_files: dict[str, tuple[Path, ImageRecord]] = {}
    for split, recs in split_map.items():
        for rec in recs:
            fname = safe_filename(rec.image_url)
            target_path = output_dir / split / rec.label / fname
            expected_files[str(target_path.relative_to(output_dir))] = (target_path, rec)

    logger.info("Syncing existing files in %s...", output_dir)
    
    current_files: list[Path] = []
    for split_dir in ["train", "val", "test"]:
        path = output_dir / split_dir
        if path.exists():
            current_files.extend(path.rglob("*.jpg"))
            current_files.extend(path.rglob("*.jpeg"))

    existing_pool: dict[str, list[Path]] = {}
    for p in current_files:
        existing_pool.setdefault(p.name, []).append(p)

    moved_count = 0
    deleted_count = 0
    kept_count = 0
    
    processed_paths: set[Path] = set()
    todo_downloads: list[tuple[ImageRecord, Path]] = []

    for rel_path_str, (target_path, rec) in expected_files.items():
        if target_path.exists():
            processed_paths.add(target_path)
            kept_count += 1
            continue
        
        fname = target_path.name
        found_elsewhere = False
        if fname in existing_pool:
            for source_path in existing_pool[fname]:
                if source_path.parent.name == rec.label and source_path not in processed_paths:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source_path), str(target_path))
                    processed_paths.add(target_path)
                    moved_count += 1
                    found_elsewhere = True
                    break
        
        if not found_elsewhere:
            todo_downloads.append((rec, target_path))

    for p in current_files:
        if p not in processed_paths:
            if p.exists():
                p.unlink()
                deleted_count += 1

    logger.info("Disk Sync Result: kept=%d, moved=%d, deleted=%d, to_download=%d", 
                kept_count, moved_count, deleted_count, len(todo_downloads))

    if not todo_downloads:
        logger.info("All files are synced. Nothing to download.")
        return

    # 4. ダウンロード実行
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": "https://kokusho.nijl.ac.jp/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-site",
    })
    
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=workers, 
        pool_maxsize=workers * 2,
        max_retries=1
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    logger.info("Starting download session (workers=%d, delay=%.1fs)...", workers, delay)
    t0 = time.time()
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_image, session, rec, dest, delay=delay, retries=retries, timeout=timeout): (rec, dest)
            for rec, dest in todo_downloads
        }
        
        # 進行状況をリアルタイムにログファイルと標準出力に書き出す
        with open(failed_log, "a", encoding="utf-8") as flog:
            with tqdm(total=len(todo_downloads) + kept_count + moved_count, 
                      desc="Downloading", unit="img", initial=kept_count + moved_count) as pbar:
                for future in as_completed(futures):
                    rec, dest = futures[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                            flog.write(f"{rec.image_url}\n")
                            flog.flush() # 即座にディスクに書き込み
                    except Exception as e:
                        fail_count += 1
                        logger.error("Unexpected error for %s: %s", rec.image_url, e)
                        flog.write(f"{rec.image_url}\n")
                        flog.flush()
                    pbar.update(1)

    elapsed = time.time() - t0
    logger.info("Download session done in %.1fs (Success=%d, Failed=%d)", elapsed, success_count, fail_count)


# ─────────────────────── メイン ───────────────────────

def main():
    p = argparse.ArgumentParser(description="IIIF Dataset Downloader & Smart Sync v2 (Improved)")
    p.add_argument("--json_dir", type=Path, default=Path("iiif_json"), help="Directory containing JSON annotation files")
    p.add_argument("--output_dir", type=Path, default=Path("ml/dataset"), help="Target directory for the dataset")
    p.add_argument("--workers", type=int, default=2, help="Number of concurrent download workers")
    p.add_argument("--delay", type=float, default=1.0, help="Introduct delay (seconds) between requests per worker")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Ratio for validation set")
    p.add_argument("--test_ratio", type=float, default=0.1, help="Ratio for test set")
    p.add_argument("--retries", type=int, default=5, help="Number of retries per image")
    p.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for download")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed_log = args.output_dir / "failed_downloads.log"
    
    # 既存の失敗ログがある場合はリネームしてバックアップ
    if failed_log.exists() and failed_log.stat().st_size > 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_log = failed_log.with_name(f"failed_downloads_{timestamp}.log")
        failed_log.rename(backup_log)
        logger.info("Backed up existing failed log to %s", backup_log.name)
    
    records = load_records(args.json_dir)
    
    if not records:
        logger.warning("No records to process.")
        return

    train, val, test = stratified_split(
        records, 
        val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio, 
        seed=args.seed
    )
    
    split_map = {"train": train, "val": val, "test": test}
    sync_and_download(split_map, args.output_dir, args.workers, failed_log, 
                      delay=args.delay, retries=args.retries, timeout=args.timeout)
    
    logger.info("Process completed successfully.")

if __name__ == "__main__":
    main()
