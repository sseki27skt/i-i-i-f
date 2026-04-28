# 📖 IIIF Volume Segmenter

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![React](https://img.shields.io/badge/React-19.2-blue?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue?logo=typescript)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![ONNX Runtime Web](https://img.shields.io/badge/ONNX%20Runtime%20Web-1.24-blue?logo=onnx)
![Vite](https://img.shields.io/badge/Vite-6.0-blue?logo=vite)

IIIFマニフェストから表紙の自動判別を行い、`structures` プロパティを用いたIIIFマニフェストの分割を行うツールです。

## ✨ 主な機能 (Features)

- **🖼️ 表紙の自動判別**: 機械学習モデルを用いて、IIIFマニフェスト内の画像から「表紙」となるページを自動で識別します。
- **📑 マニフェストの分割**: 識別された表紙情報を元に、`structures` プロパティを使用してIIIFマニフェストを論理的なボリュームごとに分割します。
- **🌐 ブラウザ完結の推論**: ONNX Runtime Webを利用し、サーバーサイドのGPUリソースを必要とせずブラウザ上で高速な推論を実現します。
- **🧠 独自の機械学習モデル**: MobileNetをベースとした軽量なカスタムモデルの学習・評価・エクスポートを一貫して行えるパイプラインを備えています。

## 📁 構成 (Project Structure)

プロジェクトは大きく2つのコンポーネントに分かれています。

1. **💻 フロントエンド (Web Application / `src/`)**
   - React + TypeScript + Vite を使用したモダンなフロントエンド。
   - ONNX Runtime Web を使用してブラウザ上で推論を実行します。
   - 🚀 **デプロイ先**: [https://sseki27skt.github.io/i-i-i-f/](https://sseki27skt.github.io/i-i-i-f/)

2. **🧠 機械学習パイプライン (Machine Learning / `ml/`)**
   - モデルの学習、評価、およびONNX形式へのエクスポートを行うPythonスクリプト群。
   - 主にMobileNetのファインチューニングを行っています。

## 🚀 はじめに (Getting Started)

### 💻 フロントエンド (Web App)

依存関係をインストールして開発サーバーを起動します。

```bash
npm install
npm run dev
```

### 🧠 機械学習パイプライン (ML)

Python 3.10以上の環境を推奨します。

#### 1. 📦 依存関係のインストール
```bash
pip install -r ml/requirements.txt
```

#### 2. 🛠️ スクリプトの実行順序と役割

MLディレクトリ（`ml/`）には、データ準備からモデルのデプロイまでをカバーする以下のスクリプトが含まれています。

- **📥 `01_download_dataset.py` (データの準備)**
  `iiif_json/` にあるJSONアノテーションを元に、画像をダウンロードし、学習・検証・テスト用に分割（8:1:1）します。

- **🏋️ `02_train.py` (モデルの学習)**
  ダウンロードしたデータセットを用いて、PyTorchによるモデル（MobileNetベース）の学習を行います。

- **🧪 `03_evaluate_pth.py` (推論テスト: PyTorch)**
  学習済みモデル（`.pth`）を用いて、テストデータに対する推論と精度評価を行います。

- **📦 `04_export_onnx.py` (ONNXへのエクスポート)**
  学習済みモデルをフロントエンド（ONNX Runtime Web）で利用可能なONNX形式（`.onnx`）に変換・エクスポートします。

- **🔍 `05_error_analysis.py` (エラー分析)**
  推論結果と正解データを比較し、モデルが誤認識した画像の詳細なエラー分析を行います。

- **🌐 `05_evaluate_onnx.py` (推論テスト: ONNX)**
  エクスポートされたONNXモデルを使用して推論テストを行い、PyTorchモデルと同等の精度が出ているか確認します。

## 📄 ライセンス (License)

このプロジェクトは [MIT License](LICENSE) の元に公開されています。
