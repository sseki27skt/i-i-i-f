# IIIF Volume Segmenter

IIIFマニフェストから表紙の自動判別を行い、strucuturesプロパティを用いたIIIFマニフェストの分割を行います。

## 構成

1.  **Web Application (`src/`)**: React + TypeScript + Vite を使用したフロントエンド。ONNX Runtime Web を使用してブラウザ上で推論を実行します。
    -   デプロイ先: [https://sseki27skt.github.io/i-i-i-f/](https://sseki27skt.github.io/i-i-i-f/)
2.  **Machine Learning (`ml/`)**: モデルの学習、評価、およびONNX形式へのエクスポートを行うPythonスクリプト群（MobileNetのファインチューニングが主眼）。

## はじめに

### フロントエンド (Web App)

依存関係をインストールして開発サーバーを起動します。

```bash
npm install
npm run dev
```

### 機械学習パイプライン (ML)

Python 3.10以上を推奨します。

1.  依存関係のインストール:
    ```bash
    pip install -r ml/requirements.txt
    ```

2.  データの準備 (ダウンロード & 同期):
    `iiif_json/` にあるJSONアノテーションを元に、画像をダウンロードし、学習・検証・テスト用に分割（8:1:1）します。
    ```bash
    python ml/01_download_dataset.py
    ```

3.  モデルの学習:
    ```bash
    python ml/02_train.py
    ```

4.  推論テスト (PyTorch):
    ```bash
    python ml/03_evaluate_pth.py
    ```

5.  ONNXへのエクスポート:
    ```bash
    python ml/04_export_onnx.py
    ```

6.  推論テスト (ONNX):
    ```bash
    python ml/05_evaluate_onnx.py
    ```

## ライセンス

[MIT License](LICENSE)
