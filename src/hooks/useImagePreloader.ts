/**
 * useImagePreloader.ts — 分析用画像のバックグラウンド事前ロードホック
 *
 * imageCache を Zustand ストアから切り離し、モジュールスコープの Map に保持することで
 * 画像フェッチ完了のたびに React の再レンダーが起きるのを防ぐ。
 */

import { useEffect, useRef } from 'react'
import { useManifestStore } from '@/store/manifest-store'

const CONCURRENCY = 4 // 並列度

/**
 * 推論用 ArrayBuffer を保持するモジュールスコープのキャッシュ。
 * Zustand 管轄外なので更新しても React の再レンダーは発生しない。
 */
export const imageCacheMap = new Map<number, ArrayBuffer>()

export function useImagePreloader() {
  const { manifest, visibleIndices } = useManifestStore()
  const activeRequests = useRef<Set<number>>(new Set())
  const currentManifestId = useRef<string | null>(null)

  useEffect(() => {
    if (!manifest) return

    // マニフェストが切り替わったらキャッシュをクリア
    if (currentManifestId.current !== manifest.id) {
      imageCacheMap.clear()
      currentManifestId.current = manifest.id
    }

    const canvases = manifest.canvases
    let isCancelled = false

    const processQueue = async () => {
      // すでにキャッシュにある、または現在取得中のものを除外
      const pending = canvases.filter(
        (c) => !imageCacheMap.has(c.index) && !activeRequests.current.has(c.index),
      )

      if (pending.length === 0) return

      // 表示領域 (visibleIndices) にあるものを優先的にソート
      pending.sort((a, b) => {
        const aVisible = visibleIndices.has(a.index)
        const bVisible = visibleIndices.has(b.index)
        if (aVisible && !bVisible) return -1
        if (!aVisible && bVisible) return 1
        return 0
      })

      // 並列実行用のワーカー関数
      const worker = async () => {
        while (pending.length > 0 && !isCancelled) {
          const canvas = pending.shift()
          if (!canvas) break

          activeRequests.current.add(canvas.index)
          try {
            const resp = await fetch(canvas.inferenceUrl, { priority: 'low' } as any)
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
            
            const buffer = await resp.arrayBuffer()
            if (!isCancelled) {
              // Zustand ストアを更新せずに直接 Map に書き込む → 再レンダーなし
              imageCacheMap.set(canvas.index, buffer)
            }
          } catch (err) {
            console.warn(`Failed to preload image for canvas ${canvas.index}:`, err)
          } finally {
            activeRequests.current.delete(canvas.index)
          }
        }
      }

      // 指定された並列数で開始
      const workers = Array.from({ length: CONCURRENCY }, () => worker())
      await Promise.all(workers)
    }

    processQueue()

    return () => {
      isCancelled = true
    }
  }, [manifest, visibleIndices])
}
