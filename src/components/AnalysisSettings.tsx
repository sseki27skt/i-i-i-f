/**
 * AnalysisSettings.tsx — モデル選択・検出設定・分析開始ボタン
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { 
  Play, 
  Settings2, 
  Loader2, 
  AlertCircle 
} from 'lucide-react'
import { useManifestStore } from '@/store/manifest-store'
import { type ModelId, type WorkerInMessage, type WorkerOutMessage } from '@/types/app'
import { imageCacheMap } from '@/hooks/useImagePreloader'

export function AnalysisSettings() {
  const {
    manifest,
    inferenceMode, setInferenceMode,
    minPageDistance, setMinPageDistance,
    detectionMethod, setDetectionMethod,
    confidenceThreshold, setConfidenceThreshold,
    inferenceStatus, setInferenceStatus,
    inferenceProgress,
    inferenceTotalCount,
    inferenceCompletedCount,
    setInferenceProgress,
    inferenceError, setInferenceError,
    setInferenceResult,
    clearInferenceResults,
    setActiveView,
    exportLanguage, setExportLanguage,
    nmsPriority, setNmsPriority,
    calculatedThreshold,
  } = useManifestStore()

  const [showSettings, setShowSettings] = useState(false)

  const NUM_WORKERS = Math.min(3, navigator.hardwareConcurrency ?? 3)
  const workersRef = useRef<Worker[]>([])
  const [modelReadyCount, setModelReadyCount] = useState(0)

  // Worker 初期化（マウント時のみ）
  useEffect(() => {
    const workers: Worker[] = []
    for (let i = 0; i < NUM_WORKERS; i++) {
      const worker = new Worker(
        new URL('../workers/inference.worker.ts', import.meta.url),
        { type: 'module' },
      )
      
      worker.onmessage = (e: MessageEvent<WorkerOutMessage>) => {
        const msg = e.data
        if (msg.type === 'MODEL_LOADED') {
          setModelReadyCount(prev => prev + 1)
          if (i === 0) setInferenceStatus('idle')
        } else if (msg.type === 'MODEL_ERROR') {
          setInferenceError(msg.error)
          setInferenceStatus('error')
        } else if (msg.type === 'INFERENCE_RESULT') {
          setInferenceResult({ canvasIndex: msg.canvasIndex, confidence: msg.confidence })
        } else if (msg.type === 'INFERENCE_ERROR') {
          console.warn('Inference error for canvas', msg.canvasIndex, msg.error)
        }
      }

      // 初期モデルをバックグラウンドでプリロード
      const initialMode = useManifestStore.getState().inferenceMode
      const modelToLoad = initialMode === 'ensemble' ? 'mobilenet_v2' : initialMode as ModelId
      worker.postMessage({ type: 'LOAD_MODEL', modelId: modelToLoad } as WorkerInMessage)
      workers.push(worker)
    }
    workersRef.current = workers
    // 初期マウント時のモデルロードではステータスを変更せずバックグラウンドで行う（UIのフラッシュ防止）

    return () => workers.forEach(w => w.terminate())
  }, [NUM_WORKERS, setInferenceError, setInferenceResult, setInferenceStatus])

  // モデル切り替え
  const handleModeChange = useCallback((mode: typeof inferenceMode) => {
    setInferenceMode(mode)
    if (mode !== 'ensemble') {
      setModelReadyCount(0)
      setInferenceStatus('loading-model')
      workersRef.current.forEach(w => {
        w.postMessage({ type: 'LOAD_MODEL', modelId: mode } as WorkerInMessage)
      })
    } else {
      // Ensembleの場合はV3とV2の両方が必要だが単体実行時に順次ロードするため今は何もしないか両方呼ぶ
      // 後続のhandleStartで順次実行されるのでここでは状態だけ変更
    }
  }, [setInferenceMode, setInferenceStatus])

  const modelReady = modelReadyCount === NUM_WORKERS

  // 推論実行 (単一モデル)
  const runSingleInference = useCallback(async (targetModelId: ModelId): Promise<void> => {
    if (!manifest || workersRef.current.length === 0) return

    const startSessionId = useManifestStore.getState().inferenceSessionId
    const isAborted = () => {
      const { inferenceSessionId: currentId, inferenceStatus: currentStatus } = useManifestStore.getState()
      return currentId !== startSessionId || currentStatus === 'idle'
    }

    // モデルがロードされていない場合はロードを待つ
    setInferenceStatus('loading-model')
    setModelReadyCount(0)
    await Promise.all(workersRef.current.map(w => {
      return new Promise<void>((resolve) => {
        const originalOnMessage = w.onmessage
        w.onmessage = (e: MessageEvent<WorkerOutMessage>) => {
          if (e.data.type === 'MODEL_LOADED' && e.data.modelId === targetModelId) {
            w.onmessage = originalOnMessage
            resolve()
          }
        }
        w.postMessage({ type: 'LOAD_MODEL', modelId: targetModelId } as WorkerInMessage)
      })
    }))
    if (isAborted()) return
    // 対象モデルとして記録するためにセット
    setInferenceMode(targetModelId)

    setInferenceStatus('fetching-images')
    setInferenceProgress(0, 0, manifest.canvases.length)
    setInferenceError(null)

    const total = manifest.canvases.length
    let completed = 0
    const inferenceStartTime = performance.now()

    await new Promise<void>((resolve) => {
      const originalOnMessages = workersRef.current.map(w => w.onmessage)
      
      workersRef.current.forEach(worker => {
        worker.onmessage = (e: MessageEvent<WorkerOutMessage>) => {
          if (isAborted()) {
            workersRef.current.forEach((w, i) => { w.onmessage = originalOnMessages[i] })
            resolve()
            return
          }

          const msg = e.data
          if (msg.type === 'INFERENCE_RESULT') {
            setInferenceResult({ canvasIndex: msg.canvasIndex, confidence: msg.confidence })
            completed++
            setInferenceProgress(Math.round((completed / total) * 100), completed, total)
            if (completed >= total) {
              const duration = performance.now() - inferenceStartTime
              const avgLatency = duration / total
              console.info(`[Performance] Total Time: ${duration.toFixed(2)}ms, Avg Latency: ${avgLatency.toFixed(2)}ms / img`)
              useManifestStore.getState().setInferenceLatency(avgLatency)
              workersRef.current.forEach((w, i) => { w.onmessage = originalOnMessages[i] })
              resolve()
            }
          } else if (msg.type === 'INFERENCE_ERROR') {
            console.error('Inference error', msg.canvasIndex, msg.error)
            setInferenceResult({ canvasIndex: msg.canvasIndex, confidence: 0 })
            completed++
            setInferenceProgress(Math.round((completed / total) * 100), completed, total)
            if (completed >= total) {
              const duration = performance.now() - inferenceStartTime
              const avgLatency = duration / total
              console.info(`[Performance] Total Time: ${duration.toFixed(2)}ms, Avg Latency: ${avgLatency.toFixed(2)}ms / img`)
              useManifestStore.getState().setInferenceLatency(avgLatency)
              workersRef.current.forEach((w, i) => { w.onmessage = originalOnMessages[i] })
              resolve()
            }
          }
        }
      })

      const BATCH_SIZE = 8
      setInferenceStatus('inferring')
      
      for (let i = 0; i < manifest.canvases.length; i += BATCH_SIZE) {
        if (isAborted()) return
        const chunk = manifest.canvases.slice(i, i + BATCH_SIZE)
        const workerIndex = Math.floor(i / BATCH_SIZE) % workersRef.current.length
        const worker = workersRef.current[workerIndex]

        const items = chunk.map(c => ({
          canvasIndex: c.index,
          imageData: imageCacheMap.get(c.index),
          imageUrl: imageCacheMap.get(c.index) ? undefined : c.inferenceUrl,
        }))

        worker.postMessage({
          type: 'RUN_BATCH_INFERENCE',
          items
        } as WorkerInMessage)
      }
    })

    if (isAborted()) return

    // 全コマの推論完了後、allModelInferenceResults に履歴として保存する
    const results = Array.from(useManifestStore.getState().inferenceResults.entries()).map(
      ([idx, conf]) => ({ canvasIndex: idx, confidence: conf })
    )
    useManifestStore.getState().setInferenceResults(results)
  }, [manifest, setInferenceStatus, setInferenceProgress, setInferenceError, setInferenceResult, setInferenceMode])

  // 分析メインフロー
  const [analysisPhase, setAnalysisPhase] = useState<string>('')

  const handleStart = useCallback(async () => {
    if (!manifest) return
    const currentMode = useManifestStore.getState().inferenceMode
    
    clearInferenceResults()

    if (currentMode === 'ensemble') {
      // 両モデルを順に実行し、allModelInferenceResults に蓄積する
      setAnalysisPhase('V3 Large で分析中...')
      await runSingleInference('mobilenet_v3_large')
      if (useManifestStore.getState().inferenceStatus === 'idle') return

      setAnalysisPhase('V2 で比較検証中...')
      await runSingleInference('mobilenet_v2')
      if (useManifestStore.getState().inferenceStatus === 'idle') return
      
      setAnalysisPhase('')

      // ensembleのまま再計算
      setInferenceMode('ensemble')
    } else {
      // 個別推論
      setAnalysisPhase('')
      await runSingleInference(currentMode as ModelId)
      if (useManifestStore.getState().inferenceStatus === 'idle') return
      setInferenceMode(currentMode)
    }

    setInferenceStatus('done')
    setActiveView('result')
  }, [manifest, clearInferenceResults, runSingleInference, setInferenceStatus, setActiveView, setInferenceMode])

  const isRunning = inferenceStatus === 'inferring' || inferenceStatus === 'fetching-images' || inferenceStatus === 'loading-model'
  const canStart = manifest && !isRunning && (inferenceMode === 'ensemble' || modelReady)

  return (
    <div className="relative flex items-center gap-2">
      {/* 設定歯車ボタン */}
      <button 
        className={`p-2 rounded-lg transition-colors border ${showSettings ? 'bg-indigo-600/10 text-indigo-500 border-indigo-600/30' : 'bg-transparent border-transparent hover:bg-black/5 dark:hover:bg-white/5 text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'}`}
        onClick={() => setShowSettings(!showSettings)}
        title="分析設定"
      >
        <Settings2 className="w-5 h-5" />
      </button>

      {/* 分析開始ボタン */}
      <button
        id="start-analysis-btn"
        className="btn-primary min-w-[110px]"
        onClick={() => { setShowSettings(false); handleStart(); }}
        disabled={!canStart}
      >
        <Play className="w-4 h-4" />
        分析開始
      </button>

      {/* 設定ポップオーバー */}
      {showSettings && (
        <div 
          className="absolute top-full right-0 mt-3 p-5 rounded-xl border shadow-2xl z-50 flex flex-col gap-5 w-[420px] animate-slide-up"
          style={{ background: 'var(--color-surface)', borderColor: 'var(--color-border)' }}
        >
          <div className="flex items-center justify-between pb-3 border-b border-white/10">
            <h3 className="font-semibold text-sm">推論・検出設定</h3>
            <button onClick={() => setShowSettings(false)} className="text-slate-400 hover:text-slate-200 text-xs text-underline">閉じる</button>
          </div>
          
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">推論モデル / モード</label>
              <div className="flex bg-black/5 dark:bg-white/5 p-1 rounded-lg gap-1 border border-black/10 dark:border-white/10">
                <button
                  onClick={() => handleModeChange('mobilenet_v3_large')}
                  className={`flex-1 px-2 py-1.5 rounded-md text-[11px] font-semibold transition-all ${
                    inferenceMode === 'mobilenet_v3_large'
                      ? 'bg-indigo-600 text-white shadow'
                      : 'text-slate-600 dark:text-slate-400 hover:text-indigo-500 hover:bg-black/5 dark:hover:bg-white/5'
                  }`}
                >
                  V3 Large
                </button>
                <button
                  onClick={() => handleModeChange('mobilenet_v2')}
                  className={`flex-1 px-2 py-1.5 rounded-md text-[11px] font-semibold transition-all ${
                    inferenceMode === 'mobilenet_v2'
                      ? 'bg-indigo-600 text-white shadow'
                      : 'text-slate-600 dark:text-slate-400 hover:text-indigo-500 hover:bg-black/5 dark:hover:bg-white/5'
                  }`}
                >
                  V2
                </button>
                <button
                  onClick={() => handleModeChange('ensemble')}
                  className={`flex-1 px-2 py-1.5 rounded-md text-[11px] font-semibold transition-all ${
                    inferenceMode === 'ensemble'
                      ? 'bg-indigo-600 text-white shadow'
                      : 'text-slate-600 dark:text-slate-400 hover:text-indigo-500 hover:bg-black/5 dark:hover:bg-white/5'
                  }`}
                >
                  Ensemble
                </button>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex flex-col gap-1.5 flex-1">
                <label className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">アルゴリズム</label>
                <select
                  value={detectionMethod}
                  onChange={(e) => setDetectionMethod(e.target.value as 'max-gap' | 'kmeans' | 'first-gap')}
                  className="input text-xs py-2 w-full"
                >
                  <option value="max-gap">Max Gap</option>
                  <option value="first-gap">First Gap</option>
                  <option value="kmeans">K-means</option>
                  
                </select>
              </div>
              <div className="flex flex-col gap-1.5 w-24">
                <label className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">抑制間隔</label>
                <input
                  type="number" min={1} max={50} step={1}
                  value={minPageDistance}
                  onChange={(e) => setMinPageDistance(Number(e.target.value))}
                  className="input text-xs py-2 w-full"
                />
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">抑制優先度</label>
              <div className="flex bg-black/5 dark:bg-white/5 p-1 rounded-lg gap-1 border border-black/10 dark:border-white/10">
                <button
                  type="button"
                  onClick={() => setNmsPriority('first')}
                  className={`flex-1 px-2 py-1.5 rounded-md text-[10px] font-semibold transition-all ${
                    nmsPriority === 'first'
                      ? 'bg-indigo-600 text-white shadow'
                      : 'text-slate-600 dark:text-slate-400 hover:text-indigo-500 hover:bg-black/5 dark:hover:bg-white/5'
                  }`}
                >
                  前方優先
                </button>
                <button
                  type="button"
                  onClick={() => setNmsPriority('max')}
                  className={`flex-1 px-2 py-1.5 rounded-md text-[10px] font-semibold transition-all ${
                    nmsPriority === 'max'
                      ? 'bg-indigo-600 text-white shadow'
                      : 'text-slate-600 dark:text-slate-400 hover:text-indigo-500 hover:bg-black/5 dark:hover:bg-white/5'
                  }`}
                >
                  確信度最大
                </button>
                <button
                  type="button"
                  onClick={() => setNmsPriority('last')}
                  className={`flex-1 px-2 py-1.5 rounded-md text-[10px] font-semibold transition-all ${
                    nmsPriority === 'last'
                      ? 'bg-indigo-600 text-white shadow'
                      : 'text-slate-600 dark:text-slate-400 hover:text-indigo-500 hover:bg-black/5 dark:hover:bg-white/5'
                  }`}
                >
                  後方優先
                </button>
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between items-center">
                <label className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">閾値 (Threshold)</label>
                <button
                  onClick={() => setConfidenceThreshold(null)}
                  className={`text-[10px] px-2 py-1 rounded transition-colors ${
                    confidenceThreshold === null
                      ? 'bg-indigo-600/20 text-indigo-400 cursor-default font-medium'
                      : 'text-slate-500 hover:bg-black/5 dark:hover:bg-white/5 hover:text-indigo-400'
                  }`}
                >
                  Auto
                </button>
              </div>
              <div className="flex items-center gap-3 bg-black/5 dark:bg-white/5 p-3 rounded-lg border border-black/10 dark:border-white/10">
                <input
                  type="range" min={0} max={1} step={0.01}
                  value={confidenceThreshold ?? calculatedThreshold ?? 0.5}
                  onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                  className="slider flex-1"
                />
                <span className={`text-sm w-12 text-right tabular-nums font-bold ${confidenceThreshold === null ? 'text-indigo-400' : 'text-slate-200'}`}>
                  {((confidenceThreshold ?? calculatedThreshold ?? 0.5) * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            <div className="pt-3 border-t border-white/10 flex flex-col gap-3">
              <label className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">エクスポート設定</label>
              <div className="flex flex-col gap-1.5">
                <span className="text-[10px] text-slate-500">Language (V3 Language Map)</span>
                <select
                  value={exportLanguage}
                  onChange={(e) => setExportLanguage(e.target.value)}
                  className="input text-xs py-2 w-full"
                >
                  <option value="ja">日本語 (ja)</option>
                  <option value="en">英語 (en)</option>
                  <option value="none">指定なし (none)</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 分析ステータス（進行中・エラー時のみ、下にフロート表示させる） */}
      {(isRunning || inferenceStatus === 'error') && !showSettings && (
        <div 
          className="absolute top-full right-0 mt-3 flex flex-col gap-2 p-3 rounded-xl border shadow-xl z-40 w-[350px] animate-slide-up"
          style={{ background: 'var(--color-surface-raised)', borderColor: 'var(--color-border)' }}
        >
          {isRunning && (
            <div className="flex flex-col gap-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-300 font-medium flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
                  {analysisPhase || '分析中...'}
                </span>
                <span className="text-indigo-400 font-bold tabular-nums">
                  {inferenceProgress}% 
                  <span className="text-[10px] opacity-70 ml-1 font-normal">({inferenceCompletedCount}/{inferenceTotalCount})</span>
                </span>
              </div>
              {useManifestStore.getState().inferenceLatency && (
                <div className="text-[10px] text-slate-400 tabular-nums">
                  Mean Latency: {useManifestStore.getState().inferenceLatency?.toFixed(1)} ms / img
                </div>
              )}
              <div className="h-2 w-full bg-black/10 dark:bg-white/10 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-indigo-500 transition-all duration-300 w-0"
                  style={{ width: `${inferenceProgress}%` }}
                />
              </div>
            </div>
          )}
          
          {inferenceStatus === 'error' && (
            <div className="flex items-center gap-2 text-xs text-red-400 p-1">
              <AlertCircle className="w-4 h-4" />
              <span>{inferenceError}</span>
            </div>
          )}
        </div>
      )}

      {/* 比較アラートなどの通知も必要ならここに（任意） */}
    </div>
  )
}
