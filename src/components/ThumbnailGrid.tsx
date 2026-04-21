/**
 * ThumbnailGrid.tsx — react-virtuoso VirtuosoGrid による仮想スクロールグリッド
 */

import { useRef, useCallback, useState, useMemo } from 'react'
import { useManifestStore } from '@/store/manifest-store'
import type { NormalizedCanvas } from '@/types/app'
import { VirtuosoGrid } from 'react-virtuoso'
import type { ListRange } from 'react-virtuoso'

// ─────────────── 個別サムネイルアイテム ───────────────

interface ThumbnailItemProps {
  canvas: NormalizedCanvas
  size: number
  isCover: boolean
  coverLabel?: string
  groupIdx: number
  confidence?: number
  isRunning: boolean
  onClick?: () => void
}

function ThumbnailItem({
  canvas,
  size,
  isCover,
  coverLabel,
  groupIdx,
  confidence,
  isRunning,
  onClick,
}: ThumbnailItemProps) {
  const [isLoaded, setIsLoaded] = useState(false)
  const [errored, setErrored] = useState(false)

  const height = Math.round(size * 1.4)

  // Confidence バッジの色計算
  const conf = confidence !== undefined ? Math.round(confidence * 100) : null
  const confColor =
    conf === null
      ? null
      : conf >= 80
        ? 'bg-emerald-500/90 text-white shadow-sm'
        : conf >= 40
          ? 'bg-amber-500/90 text-white shadow-sm'
          : 'bg-slate-800/80 dark:bg-slate-700/80 text-white shadow-sm'

  return (
    <div
      className={[
        'thumbnail-item',
        isCover ? 'is-cover' : '',
        'group cursor-pointer transition-transform active:scale-95',
      ]
        .filter(Boolean)
        .join(' ')}
      style={{ width: size, height }}
      title={`#${canvas.index + 1} ${canvas.label}`}
      onClick={onClick}
    >
      {/* グループ背景色 */}
      {groupIdx >= 0 && (
        <div
          className="absolute inset-x-1 inset-y-1 rounded-lg opacity-40 transition-colors duration-500"
          style={{
            background: `hsl(${(groupIdx * 137) % 360}, 70%, 30%)`,
            boxShadow: `inset 0 0 0 1px hsl(${(groupIdx * 137) % 360}, 70%, 50%)`,
          }}
        />
      )}

      {/* スケルトン */}
      {!isLoaded && <div className="skeleton absolute inset-0" />}

      {/* サムネイル画像 */}
      {!errored ? (
        <img
          src={(size <= 120 || isRunning || !canvas.imageServiceUrl) ? (canvas.thumbnailUrl || '') : `${canvas.imageServiceUrl}/full/${size <= 180 ? 240 : 400},/0/default.jpg`}
          alt={canvas.label}
          className={`absolute inset-0 w-full h-full object-contain p-1 transition-opacity duration-300 ${isLoaded ? 'opacity-100' : 'opacity-0'}`}
          loading="lazy"
          onLoad={() => setIsLoaded(true)}
          onError={() => {
            setIsLoaded(true)
            setErrored(true)
          }}
        />
      ) : (
        <span className="absolute inset-0 flex items-center justify-center text-xs" style={{ color: 'var(--color-text-muted)' }}>
          No Image
        </span>
      )}

      {/* コマ番号 */}
      <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-center text-[10px] text-white py-0.5 tabular-nums">
        {canvas.index + 1}
      </div>

      {/* 表紙マーカー */}
      {isCover && coverLabel && (
        <div className="cover-marker">{coverLabel}</div>
      )}

      {/* Confidence バッジ */}
      {conf !== null && confColor !== null && (
        <div
          className={`absolute top-1 left-1 text-[9px] font-bold px-1.5 py-0.5 rounded shadow-sm z-10 tabular-nums leading-none ${confColor}`}
        >
          {conf}%
        </div>
      )}

      {/* 差分アラートアイコンを削除 */}

      {/* ホバー時のオーバーレイ表示 */}
      {!isCover && (
        <div className="absolute inset-0 bg-black/10 dark:bg-white/10 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
          <span className="bg-black/60 text-white text-[10px] px-2 py-1 rounded-full">
            表紙に設定
          </span>
        </div>
      )}
    </div>
  )
}

// ─────────────── グリッド本体 ───────────────

export function ThumbnailGrid() {
  const {
    manifest,
    thumbnailSize,
    setThumbnailSize,
    detectedCovers,
    setVisibleIndices,
    toggleCover,
    inferenceResults,
    inferenceStatus,
  } = useManifestStore()

  const timerRef = useRef<number | null>(null)

  // rangeChanged コールバック: 表示中のインデックスを Set に変換してストアに通知（300ms デバウンス）
  const handleRangeChange = useCallback(
    (range: ListRange) => {
      if (timerRef.current) window.clearTimeout(timerRef.current)
      timerRef.current = window.setTimeout(() => {
        const indices = new Set<number>()
        for (let i = range.startIndex; i <= range.endIndex; i++) {
          indices.add(i)
        }
        setVisibleIndices(indices)
      }, 300)
    },
    [setVisibleIndices],
  )

  const coverMap = useMemo(
    () => new Map(detectedCovers.map((c) => [c.canvasIndex, c.label])),
    [detectedCovers],
  )

  // グループ分けの計算（メモ化）
  const canvasGroups = useMemo(() => {
    if (!manifest) return []
    const groups = new Array(manifest.canvases.length).fill(-1)
    let currentGroup = -1
    let coverIdx = 0
    for (let i = 0; i < manifest.canvases.length; i++) {
      if (
        coverIdx < detectedCovers.length &&
        detectedCovers[coverIdx].canvasIndex === i
      ) {
        currentGroup = coverIdx
        coverIdx++
      }
      groups[i] = currentGroup
    }
    return groups
  }, [manifest?.canvases.length, detectedCovers])

  if (!manifest) return null

  const minItemWidth = `${thumbnailSize}px`

  const virtuosoData = useMemo(() => {
    if (!manifest) return []
    const isRunning = inferenceStatus === 'fetching-images' || inferenceStatus === 'inferring'
    return manifest.canvases.map((canvas) => ({
      canvas,
      isCover: coverMap.has(canvas.index),
      coverLabel: coverMap.get(canvas.index),
      groupIdx: canvasGroups[canvas.index] ?? -1,
      confidence: inferenceResults.get(canvas.index),
      isRunning,
    }))
  }, [manifest, coverMap, canvasGroups, inferenceResults, inferenceStatus])

  return (
    <div className="flex flex-col gap-4 h-full overflow-hidden">
      {/* サイズスライダー */}
      <div className="flex items-center gap-3 px-1">
        <span className="text-xs shrink-0" style={{ color: 'var(--color-text-muted)' }}>
          サイズ
        </span>
        <input
          id="thumbnail-size-slider"
          type="range"
          min={80}
          max={240}
          step={10}
          value={thumbnailSize}
          onChange={(e) => setThumbnailSize(Number(e.target.value))}
          className="slider flex-1"
        />
        <span
          className="text-xs w-10 text-right tabular-nums shrink-0"
          style={{ color: 'var(--color-text-muted)' }}
        >
          {thumbnailSize}px
        </span>
      </div>

      {/* グリッド（react-virtuoso 仮想スクロール） */}
      <div id="thumbnail-grid" className="flex-1 overflow-hidden">
        <VirtuosoGrid
          key={thumbnailSize}
          style={{
            height: '100%',
            // @ts-ignore
            '--min-item-width': minItemWidth,
          }}
          data={virtuosoData}
          rangeChanged={handleRangeChange}
          listClassName="grid dynamic-grid gap-2 p-2"
          itemClassName="flex"
          overscan={200}
          itemContent={(_, item) => {
            return (
              <ThumbnailItem
                canvas={item.canvas}
                size={thumbnailSize}
                isCover={item.isCover}
                coverLabel={item.coverLabel}
                groupIdx={item.groupIdx}
                confidence={item.confidence}
                isRunning={item.isRunning}
                onClick={() => toggleCover(item.canvas.index)}
              />
            )
          }}
        />
      </div>

      {/* フッター統計 */}
      <div
        className="flex items-center justify-between px-2 py-1.5 text-xs border-t"
        style={{
          background: 'var(--color-surface)',
          borderColor: 'var(--color-border)',
          color: 'var(--color-text-muted)',
        }}
      >
        <div className="flex items-center gap-4">
          <span>{manifest.canvases.length} コマ</span>
          {detectedCovers.length > 0 && (
            <span className="font-medium" style={{ color: 'var(--color-accent)' }}>
              {detectedCovers.length} 冊検出
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
