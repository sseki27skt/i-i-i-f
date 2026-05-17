import { useEffect, useState } from 'react'
import { useManifestStore } from '@/store/manifest-store'
import { exportV3, exportV2 } from '@/lib/iiif-exporter'

export function IiifViewerPane() {
  const { manifest, detectedCovers, exportVersion, exportLanguage } = useManifestStore()
  const [blobUrl, setBlobUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!manifest) return

    // エクスポート版マニフェストオブジェクトの生成
    const exportedManifest = exportVersion === 'v3'
      ? exportV3(manifest, detectedCovers, exportLanguage)
      : exportV2(manifest, detectedCovers)

    // JSON を文字列化し Blob に変換
    const jsonStr = JSON.stringify(exportedManifest)
    const blob = new Blob([jsonStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    
    setBlobUrl(url)

    // メモリリーク防止のためのクリーンアップ
    return () => {
      URL.revokeObjectURL(url)
    }
  }, [manifest, detectedCovers, exportVersion, exportLanguage])

  if (!blobUrl) {
    return (
      <div className="flex items-center justify-center w-full h-full text-slate-400">
        プレビューを生成しています...
      </div>
    )
  }

  // 同梱する Universal Viewer のローカルパスに blobUrl をハッシュパラメータとして渡す
  const uvUrl = `${import.meta.env.BASE_URL}viewer/uv/uv.html#?manifest=${encodeURIComponent(blobUrl)}`

  return (
    <div className="w-full h-full bg-black/5 dark:bg-black/20">
      <iframe
        src={uvUrl}
        className="w-full h-full border-none"
        allowFullScreen
        title="Universal Viewer"
      />
    </div>
  )
}
