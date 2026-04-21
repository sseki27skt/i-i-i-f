/**
 * ManifestLoader.tsx — マニフェスト URL 入力コンポーネント
 */

import { useState } from 'react'
import { Search, Loader2, BookOpen } from 'lucide-react'
import { useManifestStore } from '@/store/manifest-store'
import { fetchAndParseManifest } from '@/lib/iiif-parser'

export function ManifestLoader() {
  const { manifestUrl, setManifestUrl, setManifest, setActiveView, stopInference } = useManifestStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleLoad = async () => {
    if (!manifestUrl.trim()) return
    
    // 分析中の場合は中断する
    stopInference()
    
    setLoading(true)
    setError(null)
    try {
      const parsed = await fetchAndParseManifest(manifestUrl.trim())
      setManifest(parsed)
      setActiveView('thumbnail')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load manifest')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleLoad()
  }

  return (
    <div className="flex flex-col w-full relative">
      <div className="flex items-center gap-2 w-full">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
          <input
            id="manifest-url-input"
            className="input pl-9"
            type="url"
            placeholder="IIIF Manifest URL を入力... (V2 / V3)"
            value={manifestUrl}
            onChange={(e) => setManifestUrl(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
        </div>
        <button
          id="load-manifest-btn"
          className="btn-primary min-w-[80px]"
          onClick={handleLoad}
          disabled={loading || !manifestUrl.trim()}
        >
          {loading
            ? <Loader2 className="w-4 h-4 animate-spin" />
            : <><BookOpen className="w-4 h-4" /><span>Load</span></>
          }
        </button>
      </div>

      {error && (
        <div className="mt-3 flex gap-2 items-start p-3 rounded-lg border border-red-500/30 bg-red-500/10 text-red-400 animate-fade-in shadow-sm">
          <span className="text-sm font-medium pt-0.5">⚠</span>
          <p className="text-xs leading-relaxed font-medium">
            {error}
          </p>
        </div>
      )}
    </div>
  )
}
