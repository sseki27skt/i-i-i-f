/**
 * RightPane.tsx — メタデータ編集・エクスポートパネル
 *
 * - 各表紙候補のラベル（"Book N"）をインライン編集
 * - Range の範囲（開始〜終了コマ番号）を表示
 * - V3/V2 バージョン切り替え + エクスポートボタン
 */

import { useState } from 'react'
import { Download, FileJson, Edit3, ChevronRight, Copy, Check, Database } from 'lucide-react'
import { useManifestStore } from '@/store/manifest-store'
import { downloadManifest, copyManifestToClipboard, downloadTrainingDataJson } from '@/lib/iiif-exporter'
import type { ExportVersion } from '@/types/app'

export function RightPane() {
  const {
    manifest,
    detectedCovers,
    updateCoverLabel,
    exportVersion,
    setExportVersion,
    exportLanguage,
  } = useManifestStore()

  const [editingIdx, setEditingIdx] = useState<number | null>(null)
  const [editValue, setEditValue]   = useState('')
  const [copied, setCopied] = useState(false)

  const handleEditStart = (idx: number, current: string) => {
    setEditingIdx(idx)
    setEditValue(current)
  }

  const handleEditCommit = (canvasIndex: number) => {
    if (editValue.trim()) {
      updateCoverLabel(canvasIndex, editValue.trim())
    }
    setEditingIdx(null)
  }

  const handleExport = () => {
    if (!manifest || detectedCovers.length === 0) return
    downloadManifest(manifest, detectedCovers, exportVersion, exportLanguage)
  }

  const handleExportTrainingData = () => {
    if (!manifest || detectedCovers.length === 0) return
    downloadTrainingDataJson(manifest, detectedCovers)
  }

  const handleCopy = async () => {
    if (!manifest || detectedCovers.length === 0) return
    try {
      await copyManifestToClipboard(manifest, detectedCovers, exportVersion, exportLanguage)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy to clipboard', err)
    }
  }

  if (!manifest || detectedCovers.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 p-4" style={{ color: 'var(--color-text-muted)' }}>
        <FileJson className="w-8 h-8" />
        <p className="text-xs text-center">
          分析完了後にエクスポートのオプションが表示されます
        </p>
      </div>
    )
  }

  const total = manifest.canvases.length

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="pane-header">
        <Edit3 className="w-3.5 h-3.5" />
        <span>メタデータ編集</span>
      </div>

      {/* 各冊のラベル編集リスト */}
      <div className="flex-1 overflow-y-auto py-2 px-3 space-y-2">
        {detectedCovers.map((cover, idx) => {
          const rangeStart = cover.canvasIndex + 1
          const rangeEnd   = detectedCovers[idx + 1]?.canvasIndex ?? total
          const isEditing  = editingIdx === idx

          return (
            <div
              key={cover.canvasIndex}
              className="rounded-lg border border-white/8 bg-white/2 p-3 space-y-2"
            >
              {/* 冊番号バッジ + ラベル編集 */}
              <div className="flex items-center gap-2">
                <span className="flex-none w-6 h-6 flex items-center justify-center rounded-full bg-primary-600/30 text-primary-300 text-xs font-bold">
                  {idx + 1}
                </span>

                {isEditing ? (
                  <input
                    autoFocus
                    className="input flex-1 text-sm py-1"
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onBlur={() => handleEditCommit(cover.canvasIndex)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleEditCommit(cover.canvasIndex)
                      if (e.key === 'Escape') setEditingIdx(null)
                    }}
                  />
                ) : (
                  <button
                    className="flex-1 text-left text-sm font-medium truncate group flex items-center gap-1"
                    style={{ color: 'var(--color-text-primary)' }}
                    onClick={() => handleEditStart(idx, cover.label)}
                  >
                    {cover.label}
                    <Edit3 className="w-3 h-3 text-slate-600 group-hover:text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </button>
                )}
              </div>

            {/* Range 情報 */}
            <div className="flex items-center gap-1 text-xs pl-8" style={{ color: 'var(--color-text-muted)' }}>
              {rangeStart === rangeEnd ? (
                <>
                  <span className="tabular-nums">コマ {rangeStart}</span>
                  <span className="opacity-70">(1 コマ)</span>
                </>
              ) : (
                <>
                  <span className="tabular-nums">コマ {rangeStart}</span>
                  <ChevronRight className="w-3 h-3" />
                  <span className="tabular-nums">コマ {rangeEnd}</span>
                  <span className="opacity-70">
                    ({rangeEnd - rangeStart + 1} コマ)
                  </span>
                </>
              )}
            </div>
            </div>
          )
        })}
      </div>

      {/* エクスポートセクション */}
      <div className="p-3 space-y-2 border-t border-white/8">
        {/* バージョン切り替え */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500 shrink-0">出力形式</span>
          <div className="flex rounded-lg border border-white/10 overflow-hidden text-xs">
            {(['v3', 'v2'] as ExportVersion[]).map((v) => (
              <button
                key={v}
                id={`export-version-${v}`}
                onClick={() => setExportVersion(v)}
                className={`px-3 py-1.5 font-medium transition-colors ${
                  exportVersion === v
                    ? 'bg-primary-600 text-white'
                    : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-black/5 dark:hover:bg-white/5'
                }`}
              >
                IIIF {v.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* ダウンロードとコピーボタン */}
        <div className="flex items-center gap-2">
          <button
            id="export-manifest-btn"
            className="btn-primary flex-1 py-1.5"
            onClick={handleExport}
          >
            <Download className="w-4 h-4" />
            Download
          </button>
          <button
            id="copy-manifest-btn"
            className="btn-outline flex-1 py-1.5"
            onClick={handleCopy}
            title="クリップボードにコピー"
          >
            {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>

        {/* 学習データエクスポート */}
        <button
          id="export-training-data-btn"
          className="btn-outline w-full py-1.5 border-dashed border-primary-500/50 hover:border-primary-500 text-primary-600 dark:text-primary-400"
          onClick={handleExportTrainingData}
        >
          <Database className="w-4 h-4" />
          学習用データ出力 (JSON)
        </button>
      </div>
    </div>
  )
}
