/**
 * app.ts — アプリ内部の共通型定義
 *
 * IIIF V2/V3 を正規化した後の型と、UI 状態の型を定義する。
 */

// ─────────────── 正規化済みデータモデル ───────────────

/** マニフェスト内の 1 コマ（Canvas）の正規化表現 */
export interface NormalizedCanvas {
  /** Canvas の URI */
  id: string
  /** 0-based インデックス */
  index: number
  /** 表示ラベル */
  label: string
  /** サムネイル URL（IIIF Image API: /full/!150,150/0/default.jpg） */
  thumbnailUrl: string
  /** 推論用高解像度画像のベース URL（IIIF Image API ルート） */
  imageServiceUrl: string
  /** 推論用 224px 画像のURL */
  inferenceUrl: string
}

/** 正規化済みマニフェスト */
export interface NormalizedManifest {
  /** 元のマニフェスト URI */
  id: string
  /** マニフェストレベルのラベル */
  label: string
  /** 元のバージョン */
  version: 'v2' | 'v3'
  /** 元データ（エクスポート用） */
  raw: object
  canvases: NormalizedCanvas[]
}

// ─────────────── 推論・検出結果 ───────────────

/** 1 コマの推論結果 */
export interface InferenceResult {
  canvasIndex: number
  /** 表紙クラスの Softmax 確率 [0, 1] */
  confidence: number
}

/** 検出された表紙候補 */
export interface DetectedCover {
  /** 表紙コマのインデックス */
  canvasIndex: number
  confidence: number
  /**
   * ユーザー編集可能な冊ラベル。
   * デフォルト: "Book 1", "Book 2", ...
   */
  label: string
}

// ─────────────── モデル設定 ───────────────

export type ModelId = 'mobilenet_v2' | 'mobilenet_v3_small' | 'mobilenet_v3_large'

export interface ModelConfig {
  id: ModelId
  label: string
  filename: string
  inputSize: number
}

export const MODEL_CONFIGS: Record<ModelId, ModelConfig> = {
  mobilenet_v2: {
    id: 'mobilenet_v2',
    label: 'MobileNet V2',
    filename: 'mobilenet_v2.onnx',
    inputSize: 224,
  },
  mobilenet_v3_small: {
    id: 'mobilenet_v3_small',
    label: 'MobileNet V3 Small',
    filename: 'mobilenet_v3_small.onnx',
    inputSize: 224,
  },
  mobilenet_v3_large: {
    id: 'mobilenet_v3_large',
    label: 'MobileNet V3 Large',
    filename: 'mobilenet_v3_large.onnx',
    inputSize: 224,
  },
}

// ─────────────── UI 状態 ───────────────

export type ActiveView = 'thumbnail' | 'result'

export type InferenceStatus =
  | 'idle'
  | 'loading-model'
  | 'fetching-images'
  | 'inferring'
  | 'done'
  | 'error'

export type ExportVersion = 'v3' | 'v2'
export type ExportLanguage = string

export type InferenceMode = ModelId | 'ensemble'

export type NmsPriority = 'first' | 'last' | 'max'

// ─────────────── Worker メッセージ型 ───────────────

/** Main → Worker */
export type WorkerInMessage =
  | { type: 'LOAD_MODEL'; modelId: ModelId }
  | { type: 'RUN_INFERENCE'; canvasIndex: number; imageData?: ArrayBuffer; imageUrl?: string }
  | { type: 'RUN_BATCH_INFERENCE'; items: { canvasIndex: number; imageData?: ArrayBuffer; imageUrl?: string }[] }
  | { type: 'CANCEL' }

/** Worker → Main */
export type WorkerOutMessage =
  | { type: 'MODEL_LOADED'; modelId: ModelId }
  | { type: 'MODEL_ERROR'; error: string }
  | { type: 'INFERENCE_RESULT'; canvasIndex: number; confidence: number }
  | { type: 'INFERENCE_ERROR'; canvasIndex: number; error: string }
  | { type: 'BATCH_INFERENCE_RESULT'; results: InferenceResult[] }
  | { type: 'DONE' }
 
export interface ComparisonStatus {
  status: 'match' | 'mismatch' | 'none'
  diffCount: number
  /** V3 と V2 で検出結果が異なる（片方のみ検出されている）インデックス */
  mismatchedIndices: number[]
}
