/**
 * CenterPane.tsx — 表紙コンテキスト表示（前後4コマ、計9コマ横並び）
 *
 * 選択された表紙ごとにブロックを縦並びで表示。
 * 左ペインで選択した表紙へ自動スクロール。
 */

import { useEffect, useRef } from 'react'
import { useManifestStore } from '@/store/manifest-store'
import type { NormalizedCanvas } from '@/types/app'

const CONTEXT_RADIUS = 4   // 前後コマ数

function ContextThumb({
  canvas,
  isTarget,
  onClick,
}: {
  canvas: NormalizedCanvas
  isTarget: boolean
  onClick?: () => void
}) {
  const imgRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    const img = imgRef.current
    if (!img || img.src) return
    const ob = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          img.src = canvas.thumbnailUrl
          ob.disconnect()
        }
      },
      { rootMargin: '100px' },
    )
    ob.observe(img)
    return () => ob.disconnect()
  }, [canvas.thumbnailUrl])

  return (
    <div
      className={`relative flex flex-col items-center gap-1 shrink-0 cursor-pointer group active:scale-95 transition-transform duration-200 ${
        isTarget ? 'scale-110 z-10' : 'opacity-60'
      }`}
      onClick={(e) => {
        e.stopPropagation()
        onClick?.()
      }}
    >
      <div
        className={`relative overflow-hidden rounded-md ${
          isTarget ? 'ring-2 ring-amber-400 shadow-lg shadow-amber-400/20' : 'ring-1 ring-white/10'
        }`}
        style={{ width: 80, height: 112 }}
      >
        {/* スケルトン */}
        <div className="skeleton absolute inset-0" />
        <img
          ref={imgRef}
          alt={canvas.label}
          className="absolute inset-0 w-full h-full object-cover"
          onLoad={(e) => {
            const parent = (e.target as HTMLElement).parentElement
            parent?.querySelector('.skeleton')?.remove()
          }}
        />
        {isTarget && (
          <div className="absolute top-1 left-1 text-[9px] font-bold px-1 py-0.5 rounded bg-amber-400 text-black">
            表紙
          </div>
        )}
      </div>
      <span className="text-[10px] tabular-nums" style={{ color: 'var(--color-text-muted)' }}>
        #{canvas.index + 1}
      </span>
    </div>
  )
}

function CoverBlock({
  coverIdx,
  isSelected,
  blockRef,
}: {
  coverIdx: number
  isSelected: boolean
  blockRef: (el: HTMLDivElement | null) => void
}) {
  const { manifest, detectedCovers, inferenceResults, toggleCover } = useManifestStore()

  if (!manifest) return null

  const cover = detectedCovers[coverIdx]
  if (!cover) return null

  const targetIdx = cover.canvasIndex
  const total = manifest.canvases.length

  const start = Math.max(0, targetIdx - CONTEXT_RADIUS)
  const end   = Math.min(total - 1, targetIdx + CONTEXT_RADIUS)
  const contextCanvases = manifest.canvases.slice(start, end + 1)
  const conf = inferenceResults.get(targetIdx) ?? cover.confidence

  return (
    <div
      ref={blockRef}
      className={`flex flex-col gap-3 p-4 rounded-xl border transition-all duration-200 ${
        isSelected
          ? 'border-indigo-500/30 bg-indigo-500/10'
          : 'border-white/10 bg-black/5 dark:bg-white/5'
      }`}
    >
      {/* ヘッダー */}
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary-600 text-white text-xs font-bold shrink-0">
          {coverIdx + 1}
        </div>
        <div>
          <div className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>{cover.label}</div>
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            コマ #{targetIdx + 1} &nbsp;·&nbsp; Confidence: {(conf * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* コンテキストサムネイル列 */}
      <div className="flex gap-2 justify-center overflow-x-auto pb-1 custom-scrollbar">
        {contextCanvases.map((canvas) => (
          <ContextThumb
            key={canvas.id}
            canvas={canvas}
            isTarget={canvas.index === targetIdx}
            onClick={() => toggleCover(canvas.index)}
          />
        ))}
      </div>
    </div>
  )
}

function SingleCoverView({ coverIdx }: { coverIdx: number }) {
  const { manifest, detectedCovers } = useManifestStore()
  if (!manifest) return null

  const cover = detectedCovers[coverIdx]
  if (!cover) return null

  const targetIdx = cover.canvasIndex
  const canvas = manifest.canvases[targetIdx]
  
  // 高解像度用のURL
  const imgSrc = canvas.imageServiceUrl ? `${canvas.imageServiceUrl}/full/800,/0/default.jpg` : (canvas.thumbnailUrl ?? '')

  return (
    <div className="flex flex-col h-full items-center justify-center bg-black/5 dark:bg-white/5 rounded-2xl border border-white/10 p-5 overflow-hidden">
      <div className="flex items-center gap-3 w-full mb-4 shrink-0">
        <div className="flex items-center justify-center w-7 h-7 rounded-full bg-indigo-600 text-white text-sm font-bold shrink-0">
          {coverIdx + 1}
        </div>
        <div>
          <div className="text-base font-semibold" style={{ color: 'var(--color-text-primary)' }}>{cover.label}</div>
          <div className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
            コマ #{targetIdx + 1}
          </div>
        </div>
      </div>
      <div className="flex-1 relative w-full overflow-hidden flex items-center justify-center bg-black/10 dark:bg-black/40 rounded-xl">
        <img src={imgSrc} alt={canvas.label} className="absolute inset-0 w-full h-full object-contain p-2 drop-shadow-2xl" />
      </div>
    </div>
  )
}

export function CenterPane() {
  const { detectedCovers, selectedCoverIndex, resultViewMode } = useManifestStore()

  const blockRefs = useRef<Map<number, HTMLDivElement>>(new Map())

  // 選択変更時に該当ブロックへスクロール
  useEffect(() => {
    if (resultViewMode !== 'list') return
    if (selectedCoverIndex === null) return
    const el = blockRefs.current.get(selectedCoverIndex)
    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [selectedCoverIndex, resultViewMode])

  if (detectedCovers.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-slate-600 text-sm">
        分析結果がここに表示されます
      </div>
    )
  }

  const displayIdx = selectedCoverIndex ?? 0

  return (
    <div className={`flex flex-col gap-4 h-full p-4 ${resultViewMode === 'list' ? 'overflow-y-auto' : 'overflow-hidden'}`}>
      {resultViewMode === 'single' ? (
        <SingleCoverView coverIdx={displayIdx} />
      ) : (
        detectedCovers.map((_, idx) => (
          <CoverBlock
            key={idx}
            coverIdx={idx}
            isSelected={selectedCoverIndex === idx}
            blockRef={(el) => {
              if (el) blockRefs.current.set(idx, el)
              else blockRefs.current.delete(idx)
            }}
          />
        ))
      )}
    </div>
  )
}
