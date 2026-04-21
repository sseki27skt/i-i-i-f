/**
 * LeftPane.tsx — 表紙候補リスト（クリックで中央ペインに連動）
 */

import { BookMarked } from 'lucide-react'
import { useManifestStore } from '@/store/manifest-store'

export function LeftPane() {
  const { 
    detectedCovers, 
    selectedCoverIndex, 
    setSelectedCoverIndex, 
    inferenceResults,
    inferenceMode,
  } = useManifestStore()

  if (detectedCovers.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3" style={{ color: 'var(--color-text-muted)' }}>
        <BookMarked className="w-8 h-8" />
        <p className="text-xs">表紙が検出されていません</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="pane-header flex justify-between items-center">
        <div className="flex items-center gap-2">
          <BookMarked className="w-3.5 h-3.5" />
          <span>表紙候補 ({detectedCovers.length})</span>
          {inferenceMode === 'ensemble' && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full text-[9px] bg-blue-500/20 text-blue-300 border border-blue-500/30">
              Ensemble
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto py-1">
        {/* モデル間差分セクション (アラートがある場合のみ) */}


        <div className="px-2 mb-2 text-[10px] uppercase tracking-widest font-semibold opacity-50" style={{ color: 'var(--color-text-muted)' }}>
          検出リスト
        </div>

        {detectedCovers.map((cover, listIdx) => {
          const isSelected = selectedCoverIndex === listIdx
          const conf = inferenceResults.get(cover.canvasIndex) ?? cover.confidence

          return (
            <button
              key={cover.canvasIndex}
              id={`cover-list-item-${listIdx}`}
              onClick={() => setSelectedCoverIndex(isSelected ? null : listIdx)}
              className={`
                w-full flex items-center gap-3 px-4 py-3 text-left
                transition-all duration-150 group relative
                ${isSelected
                  ? 'bg-primary-600/15 border-l-2 border-primary-500'
                  : 'border-l-2 border-transparent hover:bg-white/4'
                }
              `}
            >

              {/* 冊番号 */}
              <div className={`
                flex-none w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold
                ${isSelected ? 'bg-primary-600 text-white' : 'bg-white/8 text-slate-400 group-hover:bg-white/12'}
              `}>
                {listIdx + 1}
              </div>

              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate" style={{ color: 'var(--color-text-primary)' }}>
                  {cover.label}
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    コマ #{cover.canvasIndex + 1}
                  </span>
                  {/* Confidence バー */}
                  <div className="flex-1 h-1 rounded-full bg-white/8 overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${(conf * 100).toFixed(0)}%`,
                        background: `hsl(${conf * 120}, 70%, 55%)`,
                      }}
                    />
                  </div>
                  <span className="text-xs tabular-nums" style={{ color: `hsl(${conf * 120}, 70%, 65%)` }}>
                    {(conf * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
