/**
 * App.tsx — メインレイアウト
 *
 * ・ヘッダー（ManifestLoader + 設定）
 * ・サムネイルビュー / 分析結果ビュー（3ペイン）の切り替え
 */

import { LayoutGrid, BarChart2, Sun, Moon, List, Maximize, HelpCircle, FileJson, Sparkles, Library, ArrowRight } from 'lucide-react'
import { useManifestStore } from '@/store/manifest-store'
import { useEffect, useState } from 'react'
import { ManifestLoader } from '@/components/ManifestLoader'
import { ThumbnailGrid } from '@/components/ThumbnailGrid'
import { useImagePreloader } from '@/hooks/useImagePreloader'
import { AnalysisSettings } from '@/components/AnalysisSettings'
import { LeftPane } from '@/components/ResultView/LeftPane'
import { CenterPane } from '@/components/ResultView/CenterPane'
import { RightPane } from '@/components/ResultView/RightPane'
import { HelpModal } from '@/components/HelpModal'

export default function App() {
  const [isHelpOpen, setIsHelpOpen] = useState(false)
  const { 
    manifest, setManifest, setManifestUrl, 
    activeView, setActiveView, 
    resultViewMode, setResultViewMode,
    detectedCovers, inferenceStatus,
    theme, setTheme
  } = useManifestStore()
  
  useImagePreloader()

  // テーマの適用
  useEffect(() => {
    const root = window.document.documentElement
    root.classList.toggle('dark', theme === 'dark')
    root.setAttribute('data-theme', theme)
  }, [theme])

  return (
    <div className="flex flex-col h-full" style={{ background: 'var(--color-bg)' }}>
      {/* ──────── ヘッダー ──────── */}
      <header
        className="shrink-0 flex flex-col gap-3 px-4 py-3 border-b"
        style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}
      >
        <div className="flex items-center gap-3">
          <div 
            className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => {
              setManifest(null)
              setManifestUrl('')
              setActiveView('thumbnail')
            }}
          >
            <div className="w-7 h-7 rounded-lg bg-indigo-600/20 flex items-center justify-center overflow-hidden">
              <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="Logo" className="w-5 h-5" />
            </div>
            <h1 className="font-semibold text-sm leading-tight" style={{ color: 'var(--color-text-primary)' }}>
              IIIF Volume Segmenter
            </h1>
          </div>

          {/* テーマ切り替え & ビュー切り替えタブ */}
          <div className="ml-auto flex items-center gap-3">
            <button
              onClick={() => setIsHelpOpen(true)}
              className="p-2 rounded-lg hover:bg-white/5 transition-colors text-slate-400 hover:text-slate-100"
              title="ヘルプ"
            >
              <HelpCircle className="w-4 h-4" />
            </button>
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="p-2 rounded-lg hover:bg-white/5 transition-colors text-slate-400 hover:text-slate-100"
              title={theme === 'dark' ? 'ライトテーマに切り替え' : 'ダークテーマに切り替え'}
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>

            {manifest && (
              <div className="flex items-center gap-2">
                <div className="flex rounded-lg border border-white/10 overflow-hidden text-xs">
                  <button
                    id="view-tab-thumbnail"
                    onClick={() => setActiveView('thumbnail')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 transition-colors ${
                      activeView === 'thumbnail'
                        ? 'bg-indigo-600/20 text-indigo-700 dark:text-indigo-300'
                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-black/5 dark:hover:bg-white/5'
                    }`}
                  >
                    <LayoutGrid className="w-3 h-3" />
                    サムネイル
                  </button>
                  <button
                    id="view-tab-result"
                    onClick={() => setActiveView('result')}
                    disabled={detectedCovers.length === 0 && inferenceStatus !== 'done'}
                    className={`flex items-center gap-1.5 px-3 py-1.5 transition-colors ${
                      activeView === 'result'
                        ? 'bg-indigo-600/20 text-indigo-700 dark:text-indigo-300'
                        : (detectedCovers.length === 0 && inferenceStatus !== 'done')
                          ? 'text-slate-400 dark:text-slate-600 cursor-not-allowed'
                          : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-black/5 dark:hover:bg-white/5'
                    }`}
                  >
                    <BarChart2 className="w-3 h-3" />
                    分析結果
                    {detectedCovers.length > 0 && (
                      <span className="badge-cover ml-0.5">{detectedCovers.length}</span>
                    )}
                  </button>
                </div>

                {activeView === 'result' && (
                  <div className="flex rounded-lg border border-white/10 overflow-hidden text-xs">
                    <button
                      onClick={() => setResultViewMode('list')}
                      className={`flex items-center gap-1.5 px-3 py-1.5 transition-colors ${
                        resultViewMode === 'list'
                          ? 'bg-indigo-600/20 text-indigo-700 dark:text-indigo-300'
                          : 'text-slate-500 hover:bg-black/5 dark:hover:bg-white/5'
                      }`}
                    >
                      <List className="w-3 h-3" />
                      リスト表示
                    </button>
                    <button
                      onClick={() => setResultViewMode('single')}
                      className={`flex items-center gap-1.5 px-3 py-1.5 transition-colors ${
                        resultViewMode === 'single'
                          ? 'bg-indigo-600/20 text-indigo-700 dark:text-indigo-300'
                          : 'text-slate-500 hover:bg-black/5 dark:hover:bg-white/5'
                      }`}
                    >
                      <Maximize className="w-3 h-3" />
                      1枚表示
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ツールバー (URL入力, Load, 設定, 分析開始) */}
        <div className="flex items-center gap-2 w-full">
          <div className="flex-1">
            <ManifestLoader />
          </div>
          {manifest && activeView === 'thumbnail' && (
            <AnalysisSettings />
          )}
        </div>
      </header>

      {/* ──────── メインコンテンツ ──────── */}
      <main className="flex-1 overflow-hidden">
        {!manifest ? (
          /* ウェルカム画面 (Empty State) */
          <div className="flex flex-col items-center justify-center h-full gap-10 text-center p-8">
            <div className="w-20 h-20 rounded-2xl bg-indigo-600/10 flex items-center justify-center overflow-hidden mb-2 shadow-sm border border-indigo-500/10 animate-fade-in">
               <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="Logo" className="w-12 h-12" />
            </div>
            
            <div className="flex items-center justify-center gap-6 opacity-60 animate-slide-up" style={{ color: 'var(--color-text-muted)', animationDelay: '0.1s' }}>
              <div className="flex flex-col items-center gap-2">
                <FileJson className="w-12 h-12" />
              </div>
              <ArrowRight className="w-6 h-6 opacity-40" />
              <div className="flex flex-col items-center gap-2">
                <Sparkles className="w-12 h-12" />
              </div>
              <ArrowRight className="w-6 h-6 opacity-40" />
              <div className="relative">
                <Library className="w-12 h-12" />
                <Library className="w-12 h-12 absolute top-1.5 left-1.5 opacity-50" />
              </div>
            </div>

            <p className="text-sm max-w-sm tracking-wide animate-fade-in" style={{ color: 'var(--color-text-secondary)', animationDelay: '0.2s' }}>
              IIIF Manifest URLを入力して分析を開始します。
            </p>
          </div>
        ) : activeView === 'thumbnail' ? (
          /* サムネイルビュー */
          <div className="h-full">
            <ThumbnailGrid />
          </div>
        ) : (
          /* 分析結果ビュー（3ペイン） */
          <div className="flex h-full">
            {/* 左ペイン: 表紙候補リスト */}
            <div
              className="w-64 shrink-0 flex flex-col overflow-hidden border-r"
              style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}
            >
              <LeftPane />
            </div>

            {/* 中央ペイン: コンテキスト表示 */}
            <div className="flex-1 flex flex-col overflow-hidden">
              <CenterPane />
            </div>

            {/* 右ペイン: メタデータ編集・エクスポート */}
            <div
              className="w-72 shrink-0 flex flex-col overflow-hidden border-l"
              style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}
            >
              <RightPane />
            </div>
          </div>
        )}
      </main>

      {/* ヘルプモーダル */}
      <HelpModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
    </div>
  )
}
