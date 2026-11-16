/**
 * useImagePreloader.ts — 分析用画像のバックグラウンド事前ロードホック
 *
 * imageCache を Zustand ストアから切り離し、モジュールスコープの Map に保持することで
 * 画像フェッチ完了のたびに React の再レンダーが起きるのを防ぐ。
 */

import { useEffect, useRef } from 'react'
import { useManifestStore } from '@/store/manifest-store'

const CONCURRENCY = 2 // サーバーへの負荷およびサムネイル表示との競合を抑えるため並列数を制限
const INITIAL_DELAY_MS = 1200 // マニフェスト読み込み直後はサムネイル表示を優先させる
const REQUEST_INTERVAL_MS = 150 // リクエスト間の最小ウェイト
const MAX_RETRIES = 2

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

async function fetchImageWithRetry(url: string, fallbackUrl?: string): Promise<ArrayBuffer> {
  let lastError: unknown
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const resp = await fetch(url, { priority: 'low' } as any)
      if (resp.status === 429 || resp.status === 503) {
        // レート制限時は指数バックオフで待機
        if (attempt < MAX_RETRIES) {
          await sleep(1500 * Math.pow(2, attempt) + Math.random() * 500)
          continue
        }
      }
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      return await resp.arrayBuffer()
    } catch (err) {
      lastError = err
      if (attempt < MAX_RETRIES) {
        await sleep(800 * Math.pow(2, attempt))
      }
    }
  }

  // Image API URL が失敗し、別オリジン/別パスの thumbnailUrl がある場合はフォールバック
  if (fallbackUrl && fallbackUrl !== url) {
    const resp = await fetch(fallbackUrl, { priority: 'low' } as any)
    if (resp.ok) {
      return await resp.arrayBuffer()
    }
  }

  throw lastError
}

/**
 * 推論用 ArrayBuffer を保持するモジュールスコープのキャッシュ。
 * Zustand 管轄外なので更新しても React の再レンダーは発生しない。
 */
export const imageCacheMap = new Map<number, ArrayBuffer>()

export function useImagePreloader() {
  const { manifest, visibleIndices, inferenceStatus } = useManifestStore()
  const activeRequests = useRef<Set<number>>(new Set())
  const currentManifestId = useRef<string | null>(null)
  const manifestLoadedAt = useRef<number>(0)

  useEffect(() => {
    if (!manifest) return

    // マニフェストが切り替わったらキャッシュをクリア
    if (currentManifestId.current !== manifest.id) {
      imageCacheMap.clear()
      currentManifestId.current = manifest.id
      manifestLoadedAt.current = Date.now()
    }

    // 推論実行中は Worker 側でフェッチを行うため、バックグラウンドプリローダーは停止して帯域を譲る
    if (inferenceStatus === 'fetching-images' || inferenceStatus === 'inferring') {
      return
    }

    const canvases = manifest.canvases
    let isCancelled = false

    const processQueue = async () => {
      // マニフェスト読み込み直後は初期サムネイル群のロードを優先させるため少し待機
      const elapsed = Date.now() - manifestLoadedAt.current
      if (elapsed < INITIAL_DELAY_MS) {
        await sleep(INITIAL_DELAY_MS - elapsed)
      }
      if (isCancelled) return

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
            const buffer = await fetchImageWithRetry(canvas.inferenceUrl, canvas.thumbnailUrl)
            if (!isCancelled) {
              // Zustand ストアを更新せずに直接 Map に書き込む → 再レンダーなし
              imageCacheMap.set(canvas.index, buffer)
            }
          } catch (err) {
            console.warn(`Failed to preload image for canvas ${canvas.index}:`, err)
          } finally {
            activeRequests.current.delete(canvas.index)
          }

          if (!isCancelled && REQUEST_INTERVAL_MS > 0) {
            await sleep(REQUEST_INTERVAL_MS)
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
  }, [manifest, visibleIndices, inferenceStatus])
}
