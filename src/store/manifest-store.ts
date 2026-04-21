/**
 * manifest-store.ts — Zustand グローバルストア
 *
 * アプリ全体の状態を管理する。
 */

import { create } from 'zustand'
import type {
  NormalizedManifest,
  InferenceResult,
  DetectedCover,
  ActiveView,
  InferenceStatus,
  ExportVersion,
  ExportLanguage,
  ModelId,
  ComparisonStatus,
  InferenceMode,
  NmsPriority,
} from '@/types/app'
import { detectCovers } from '@/lib/cover-detection'

interface ManifestState {
  inferenceMode: InferenceMode
  setInferenceMode: (mode: InferenceMode) => void

  // ─── マニフェスト ───
  manifest: NormalizedManifest | null
  manifestUrl: string
  setManifestUrl: (url: string) => void
  setManifest: (manifest: NormalizedManifest | null) => void



  // ─── 推論結果 ───
  inferenceResults: Map<number, number>   // canvasIndex → confidence
  inferenceStatus: InferenceStatus
  inferenceProgress: number               // 0–100
  inferenceTotalCount: number
  inferenceCompletedCount: number
  inferenceError: string | null
  setInferenceResult: (result: InferenceResult) => void
  setInferenceResults: (results: InferenceResult[]) => void
  setInferenceStatus: (status: InferenceStatus) => void
  setInferenceProgress: (progress: number, completed?: number, total?: number) => void
  setInferenceError: (error: string | null) => void
  clearInferenceResults: () => void
  
  // ─── パフォーマンス計測 ───
  inferenceLatency: number | null         // 平均レイテンシ ms
  setInferenceLatency: (ms: number | null) => void
  
  // ─── モデル間比較 ───
  allModelInferenceResults: Map<ModelId, Map<number, number>>
  comparisonStatus: ComparisonStatus
  checkComparison: () => void

  // ─── 検出された表紙 ───
  detectedCovers: DetectedCover[]
  reDetectCovers: () => void
  setDetectedCovers: (covers: DetectedCover[]) => void
  updateCoverLabel: (canvasIndex: number, label: string) => void

  // ─── 選択状態 ───
  selectedCoverIndex: number | null   // detectedCovers 配列 index
  setSelectedCoverIndex: (i: number | null) => void

  // ─── UI 状態 ───
  activeView: ActiveView
  setActiveView: (view: ActiveView) => void
  resultViewMode: 'list' | 'single'
  setResultViewMode: (mode: 'list' | 'single') => void
  thumbnailSize: number               // サムネイル幅 px
  setThumbnailSize: (size: number) => void

  // ─── 検出設定 ───
  minPageDistance: number
  setMinPageDistance: (v: number) => void
  tailExclude: number
  setTailExclude: (v: number) => void
  detectionMethod: 'max-gap' | 'kmeans' | 'first-gap'
  setDetectionMethod: (m: 'max-gap' | 'kmeans' | 'first-gap') => void
  confidenceThreshold: number | null // null means automatic
  setConfidenceThreshold: (v: number | null) => void
  nmsPriority: NmsPriority
  setNmsPriority: (v: NmsPriority) => void
  calculatedThreshold: number | null
  setCalculatedThreshold: (v: number | null) => void

  // ─── エクスポート ───
  exportVersion: ExportVersion
  setExportVersion: (v: ExportVersion) => void
  exportLanguage: ExportLanguage
  setExportLanguage: (lang: ExportLanguage) => void
  
  // ─── 表示領域管理 ───
  visibleIndices: Set<number>
  setVisibleIndices: (indices: Set<number>) => void

  // ─── アクション ───
  toggleCover: (canvasIndex: number) => void
  
  // ─── 推論中断 ───
  inferenceSessionId: number
  stopInference: () => void

  // ─── テーマ ───
  theme: 'dark' | 'light'
  setTheme: (theme: 'dark' | 'light') => void
}

export const useManifestStore = create<ManifestState>((set, get) => ({
  inferenceMode: 'mobilenet_v2',
  setInferenceMode: (mode: InferenceMode) => {
    // もしモデル単体指定（V3やV2）なら履歴を復旧
    if (mode !== 'ensemble') {
      const state = get()
      const historical = state.allModelInferenceResults.get(mode as ModelId)
      const nextResults = historical ? new Map(historical) : new Map()
      
      set({
        inferenceMode: mode,
        inferenceResults: nextResults,
        inferenceStatus: historical ? 'done' : 'idle',
        inferenceProgress: historical ? 100 : 0,
        inferenceCompletedCount: historical ? (state.manifest?.canvases.length ?? 0) : 0,
        inferenceTotalCount: state.manifest?.canvases.length ?? 0,
        inferenceError: null,
        selectedCoverIndex: null,
      })
      
      if (historical) {
        get().reDetectCovers()
      } else {
        set({ detectedCovers: [], calculatedThreshold: null })
      }
    } else {
      // ensembleの場合は単に切り替えて再計算（必要なら両方の推論を行うのは呼び出し元で制御）
      set({ inferenceMode: mode })
      get().reDetectCovers()
    }
  },

  // マニフェスト
  manifest: null,
  manifestUrl: '',
  setManifestUrl: (url: string) => set({ manifestUrl: url }),
  setManifest: (manifest: NormalizedManifest | null) =>
    set((state) => ({
      manifest,
      inferenceResults: new Map(),
      inferenceStatus: 'idle',
      inferenceProgress: 0,
      inferenceCompletedCount: 0,
      inferenceTotalCount: 0,
      inferenceError: null,
      detectedCovers: [],
      selectedCoverIndex: null,
      allModelInferenceResults: new Map(),
      comparisonStatus: { status: 'none', diffCount: 0, mismatchedIndices: [] },
      inferenceSessionId: state.inferenceSessionId + 1,
    })),



  // 推論結果
  inferenceResults: new Map(),
  inferenceStatus: 'idle',
  inferenceProgress: 0,
  inferenceTotalCount: 0,
  inferenceCompletedCount: 0,
  inferenceError: null,
  
  setInferenceResult: ({ canvasIndex, confidence }: InferenceResult) =>
    set((state) => {
      const next = new Map(state.inferenceResults)
      next.set(canvasIndex, confidence)
      return { inferenceResults: next }
    }),

  setInferenceResults: (results: InferenceResult[], sourceModelId?: ModelId) => {
    set((state) => {
      const next = new Map<number, number>()
      for (const r of results) next.set(r.canvasIndex, r.confidence)
      const nextAll = new Map(state.allModelInferenceResults)
      
      // 保存先のmodelIdを明示するか、現在のinferenceMode(ensembleでないことを前提)を使う
      const targetModel = sourceModelId || (state.inferenceMode !== 'ensemble' ? state.inferenceMode : null)
      if (targetModel) {
        nextAll.set(targetModel as ModelId, next)
      }
      
      return { 
        inferenceResults: next,
        allModelInferenceResults: nextAll,
      }
    })
    get().reDetectCovers()
  },

  setInferenceStatus: (status: InferenceStatus) => {
    set({ inferenceStatus: status })
    if (status === 'done') {
      const state = get()
      // 表紙の再検出
      state.reDetectCovers()
      // 比較チェック
      state.checkComparison()
    }
  },

  setInferenceProgress: (progress: number, completed?: number, total?: number) =>
    set((state) => ({
      inferenceProgress: progress,
      inferenceCompletedCount: completed ?? state.inferenceCompletedCount,
      inferenceTotalCount: total ?? state.inferenceTotalCount,
    })),

  setInferenceError: (error: string | null) => set({ inferenceError: error }),
  clearInferenceResults: () =>
    set({
      inferenceResults: new Map(),
      inferenceProgress: 0,
      inferenceCompletedCount: 0,
      inferenceTotalCount: 0,
      inferenceError: null,
      inferenceLatency: null,
      comparisonStatus: { status: 'none', diffCount: 0, mismatchedIndices: [] },
    }),

  // パフォーマンス計測
  inferenceLatency: null,
  setInferenceLatency: (ms: number | null) => set({ inferenceLatency: ms }),

  // モデル間比較
  allModelInferenceResults: new Map(),
  comparisonStatus: { status: 'none', diffCount: 0, mismatchedIndices: [] },

  checkComparison: () => {
    const { allModelInferenceResults, manifest, detectionMethod, minPageDistance, confidenceThreshold, nmsPriority } = get()
    if (!manifest) return

    const v3Results = allModelInferenceResults.get('mobilenet_v3_large')
    const v2Results = allModelInferenceResults.get('mobilenet_v2')

    if (!v3Results || !v2Results) {
      set({ comparisonStatus: { status: 'none', diffCount: 0, mismatchedIndices: [] } })
      return
    }

    const commonOpts = {
      method: detectionMethod,
      minPageDistance,
      nmsPriority,
      fixedThreshold: confidenceThreshold ?? undefined,
    }

    const { covers: v3Covers } = detectCovers(
      Array.from(v3Results.entries()).map(([k, v]) => ({ canvasIndex: k, confidence: v })),
      manifest.canvases.length,
      commonOpts
    )
    const { covers: v2Covers } = detectCovers(
      Array.from(v2Results.entries()).map(([k, v]) => ({ canvasIndex: k, confidence: v })),
      manifest.canvases.length,
      commonOpts
    )

    const v3Indices = v3Covers.map(c => c.canvasIndex).sort((a, b) => a - b)
    const v2Indices = v2Covers.map(c => c.canvasIndex).sort((a, b) => a - b)

    const mismatchedIndices = [
      ...v3Indices.filter(idx => !v2Indices.includes(idx)),
      ...v2Indices.filter(idx => !v3Indices.includes(idx)),
    ].sort((a, b) => a - b)

    if (mismatchedIndices.length === 0) {
      set({ comparisonStatus: { status: 'match', diffCount: 0, mismatchedIndices: [] } })
    } else {
      set({ 
        comparisonStatus: { 
          status: 'mismatch', 
          diffCount: mismatchedIndices.length,
          mismatchedIndices 
        } 
      })
    }
  },

  // 検出された表紙
  detectedCovers: [],
  reDetectCovers: () => {
    const { 
      allModelInferenceResults, 
      manifest, 
      detectionMethod, 
      minPageDistance, 
      confidenceThreshold,
      nmsPriority,
      inferenceMode
    } = get()
    
    if (!manifest) return

    let resultsToUse: Map<number, number> | undefined

    if (inferenceMode === 'ensemble') {
      const v3 = allModelInferenceResults.get('mobilenet_v3_large')
      const v2 = allModelInferenceResults.get('mobilenet_v2')
      if (v3 && v2) {
        const averaged = new Map<number, number>()
        for (const [idx, conf] of v3.entries()) {
          const v2Conf = v2.get(idx) ?? 0
          averaged.set(idx, (conf + v2Conf) / 2)
        }
        resultsToUse = averaged
        // 表示用データも更新（CenterPaneなどの表示用）
        set({ inferenceResults: averaged })
      } else {
        resultsToUse = get().inferenceResults
      }
    } else {
      // 単体推論時は履歴から復旧、なければ今の results
      const historical = allModelInferenceResults.get(inferenceMode)
      if (historical) {
        resultsToUse = new Map(historical)
        set({ inferenceResults: resultsToUse })
      } else {
        resultsToUse = get().inferenceResults
      }
    }

    if (!resultsToUse || resultsToUse.size === 0) return

    const results = Array.from(resultsToUse.entries()).map(([canvasIndex, confidence]) => ({ canvasIndex, confidence }))
    const { covers, threshold } = detectCovers(results, manifest.canvases.length, {
      method: detectionMethod,
      minPageDistance,
      nmsPriority,
      fixedThreshold: confidenceThreshold ?? undefined,
    })
    set({ detectedCovers: covers, calculatedThreshold: threshold })
  },
  setDetectedCovers: (covers: DetectedCover[]) => set({ detectedCovers: covers, selectedCoverIndex: null }),
  updateCoverLabel: (canvasIndex: number, label: string) =>
    set((state) => ({
      detectedCovers: state.detectedCovers.map((c) =>
        c.canvasIndex === canvasIndex ? { ...c, label } : c,
      ),
    })),

  // 選択状態
  selectedCoverIndex: null,
  setSelectedCoverIndex: (i: number | null) => set({ selectedCoverIndex: i }),

  // UI 状態
  activeView: 'thumbnail',
  setActiveView: (view: ActiveView) => set({ activeView: view }),
  resultViewMode: 'single',
  setResultViewMode: (mode: 'list' | 'single') => set({ resultViewMode: mode }),
  thumbnailSize: 80,
  setThumbnailSize: (size: number) => set({ thumbnailSize: size }),

  // 検出設定
  minPageDistance: 2,
  setMinPageDistance: (v: number) => {
    set({ minPageDistance: v })
    get().reDetectCovers()
  },
  tailExclude: 2,
  setTailExclude: (v: number) => set({ tailExclude: v }),
  detectionMethod: 'max-gap',
  setDetectionMethod: (m: 'max-gap' | 'kmeans' | 'first-gap') => {
    set({ detectionMethod: m })
    get().reDetectCovers()
  },
  confidenceThreshold: null,
  setConfidenceThreshold: (v: number | null) => {
    set({ confidenceThreshold: v })
    get().reDetectCovers()
  },
  nmsPriority: 'first',
  setNmsPriority: (v: NmsPriority) => {
    set({ nmsPriority: v })
    get().reDetectCovers()
  },
  calculatedThreshold: null,
  setCalculatedThreshold: (v: number | null) => set({ calculatedThreshold: v }),

  // エクスポート
  exportVersion: 'v3',
  setExportVersion: (v: ExportVersion) => set({ exportVersion: v }),
  exportLanguage: 'ja',
  setExportLanguage: (lang: ExportLanguage) => set({ exportLanguage: lang }),

  // 表示領域管理
  visibleIndices: new Set<number>(),
  setVisibleIndices: (indices: Set<number>) => set({ visibleIndices: indices }),

  // アクション
  toggleCover: (canvasIndex: number) =>
    set((state) => {
      const isExisting = state.detectedCovers.some((c) => c.canvasIndex === canvasIndex)
      let nextCovers
      if (isExisting) {
        nextCovers = state.detectedCovers.filter((c) => c.canvasIndex !== canvasIndex)
      } else {
        const confidence = state.inferenceResults.get(canvasIndex) ?? 0
        nextCovers = [...state.detectedCovers, { canvasIndex, confidence, label: '' }]
      }

      nextCovers.sort((a, b) => a.canvasIndex - b.canvasIndex)
      nextCovers = nextCovers.map((c, i) => ({
        ...c,
        label: `${i + 1}`,
      }))

      let nextSelectedIdx = state.selectedCoverIndex
      if (isExisting) {
        const removedIdx = state.detectedCovers.findIndex((c) => c.canvasIndex === canvasIndex)
        if (nextSelectedIdx === removedIdx) {
          nextSelectedIdx = null
        } else if (nextSelectedIdx !== null && nextSelectedIdx > removedIdx) {
          nextSelectedIdx--
        }
      } else {
        nextSelectedIdx = nextCovers.findIndex((c) => c.canvasIndex === canvasIndex)
      }

      return {
        detectedCovers: nextCovers,
        selectedCoverIndex: nextSelectedIdx,
      }
    }),

  // 推論中断
  inferenceSessionId: 0,
  stopInference: () => {
    set((state) => ({
      inferenceStatus: 'idle',
      inferenceSessionId: state.inferenceSessionId + 1,
    }))
  },

  // テーマ
  theme: (localStorage.getItem('theme') as 'dark' | 'light') || 
         (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'),
  setTheme: (theme: 'dark' | 'light') => {
    localStorage.setItem('theme', theme)
    set({ theme })
  },
}))
