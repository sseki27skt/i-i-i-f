/**
 * cover-detection.ts — 閾値自動決定と NMS ロジック
 *
 * 推論結果（index → confidence）から表紙候補を自動検出する。
 *
 * アルゴリズム:
 *   1. 末尾 tailExclude コマを除外（デフォルト: 2）
 *   2. 上位 topPercentile% の Confidence 値を候補母集団として抽出
 *   3. 候補をソートし以下のいずれかで閾値決定:
 *      - max-gap:   最大 Gap 法
 *      - kmeans:    1D K-means (k=2)
 *      - first-gap: 最初の有意ギャップ法（絶対値 0.05 以上）
 *      - otsu:      大津の二値化法
 *   4. 閾値超過コマに min-page-distance NMS を適用（近接する検知を抑制）
 */

import type { DetectedCover, InferenceResult } from '@/types/app'

// ─────────────── ヘルパー ───────────────

/** 昇順ソートした配列を返す（元配列は変更しない） */
function sortedAsc(arr: number[]): number[] {
  return [...arr].sort((a, b) => a - b)
}

/** パーセンタイル値を計算する（線形補間） */
function percentile(sortedArr: number[], p: number): number {
  if (sortedArr.length === 0) return 0
  const idx = (p / 100) * (sortedArr.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * (idx - lo)
}

/**
 * 最大 Gap 法: ソート済み配列の隣接差が最大になる点を閾値とする。
 * 返す閾値は「低いクラスタの最大値」（= 高いクラスタの最小値 - ε）。
 */
function maxGapThreshold(sortedArr: number[]): number {
  if (sortedArr.length < 2) return sortedArr[0] ?? 0.5
  let maxGap = -1
  let cutIdx = 0
  for (let i = 1; i < sortedArr.length; i++) {
    const gap = sortedArr[i] - sortedArr[i - 1]
    if (gap > maxGap) {
      maxGap = gap
      cutIdx = i
    }
  }
  // 切れ目の中点を閾値として返す
  return (sortedArr[cutIdx - 1] + sortedArr[cutIdx]) / 2
}

/**
 * 簡易 1D K-means (k=2): 収束するまで反復。
 * 2 クラスタの間の中点を閾値として返す。
 */
function kMeansThreshold(arr: number[]): number {
  if (arr.length < 2) return arr[0] ?? 0.5
  const sorted = sortedAsc(arr)
  let c0 = sorted[0]
  let c1 = sorted[sorted.length - 1]

  for (let iter = 0; iter < 100; iter++) {
    const group0: number[] = []
    const group1: number[] = []
    for (const v of sorted) {
      if (Math.abs(v - c0) <= Math.abs(v - c1)) group0.push(v)
      else group1.push(v)
    }
    const newC0 = group0.length > 0 ? group0.reduce((a, b) => a + b, 0) / group0.length : c0
    const newC1 = group1.length > 0 ? group1.reduce((a, b) => a + b, 0) / group1.length : c1
    if (Math.abs(newC0 - c0) < 1e-6 && Math.abs(newC1 - c1) < 1e-6) break
    c0 = newC0
    c1 = newC1
  }
  return (c0 + c1) / 2
}

/**
 * First Gap 法: ソート済み配列を低い方からスキャンし、
 * 絶対値で minGap 以上の最初のギャップの中点を閾値とする。
 * ギャップが見つからない場合は Max Gap にフォールバック。
 */
function firstGapThreshold(sortedArr: number[], minGap = 0.05): number {
  if (sortedArr.length < 2) return sortedArr[0] ?? 0.5
  for (let i = 1; i < sortedArr.length; i++) {
    const gap = sortedArr[i] - sortedArr[i - 1]
    if (gap >= minGap) {
      return (sortedArr[i - 1] + sortedArr[i]) / 2
    }
  }
  // 有意なギャップが見つからない場合は Max Gap にフォールバック
  return maxGapThreshold(sortedArr)
}

  // 最大ギャップ法またはその他
  // Otsu削除済み

// ─────────────── メイン関数 ───────────────

  export interface DetectionOptions {
    /** 末尾から除外するコマ数（先頭は必ず含める）デフォルト: 2 */
    tailExclude?: number
    /** 候補母集団として採用する上位パーセンタイル閾値（デフォルト: 90 = 上位10%） */
    topPercentile?: number
    /**
     * 閾値決定アルゴリズム。
     * 'max-gap':   最大 Gap 法
     * 'kmeans':    1D K-means
     * 'first-gap': 最初の有意ギャップ法（絶対値 0.05 以上）
     */
    method?: 'max-gap' | 'kmeans' | 'first-gap'
    /** 手動で指定する固定閾値。指定された場合、自動決定をスキップする */
    fixedThreshold?: number
    /** 連続検知の抑制距離（コマ数）デフォルト: 2 */
    minPageDistance?: number
    /** NMS で同等とみなす確信度の差（0.05 = 5%） デフォルト: 0.05 */
    nmsThreshold?: number
    /** 同等確信度内での優先順位。'first' (前方), 'last' (後方), 'max' (スコア最大) */
    nmsPriority?: 'first' | 'last' | 'max'
  }
  
  export interface DetectionResult {
    covers: DetectedCover[]
    threshold: number
  }
  
  /**
   * 推論結果から表紙候補を検出する。
   *
   * @param results - Worker から集めた推論結果
   * @param totalCanvases - マニフェストの全コマ数
   * @param opts - 検出オプション
   * @returns DetectionResult 
   */
  export function detectCovers(
    results: InferenceResult[],
    totalCanvases: number,
    opts: DetectionOptions = {},
  ): DetectionResult {
    const {
      tailExclude = 2,
      topPercentile = 90,
      method = 'max-gap',
      fixedThreshold,
      minPageDistance = 2,
      nmsThreshold = 0.05,
      nmsPriority = 'first',
    } = opts
  
    if (results.length === 0) return { covers: [], threshold: 0 }
  
    // Step 1: 末尾 tailExclude コマを除外（最初のコマは必ず対象に含める）
    const maxIndex = Math.max(0, totalCanvases - 1 - tailExclude)
    const eligible = results.filter((r) => r.canvasIndex <= maxIndex)
    if (eligible.length === 0) return { covers: [], threshold: 0 }

  // Step 2: 閾値の決定
  let dynamicThreshold = 0
  
  if (fixedThreshold !== undefined) {
    dynamicThreshold = fixedThreshold
  } else {
    // 自動閾値決定ロジック
    const allConfs = sortedAsc(eligible.map((r) => r.confidence))
    const threshold90 = percentile(allConfs, topPercentile)
    const topCandidates = eligible.filter((r) => r.confidence >= threshold90)

    if (topCandidates.length === 0) return { covers: [], threshold: 0 }
    if (topCandidates.length === 1) {
      return { 
        covers: [{ canvasIndex: topCandidates[0].canvasIndex, confidence: topCandidates[0].confidence, label: '1' }], 
        threshold: 0 
      }
    }

    const topConfs = sortedAsc(topCandidates.map((r) => r.confidence))
      if (method === 'kmeans') {
        dynamicThreshold = kMeansThreshold(topConfs)
      } else if (method === 'first-gap') {
        dynamicThreshold = firstGapThreshold(topConfs)
      } else {
        dynamicThreshold = maxGapThreshold(topConfs)
      }
    }

  const aboveThreshold = eligible
    .filter((r) => r.confidence >= dynamicThreshold)
    .sort((a, b) => a.canvasIndex - b.canvasIndex)


  if (aboveThreshold.length === 0) return { covers: [], threshold: dynamicThreshold }

  // Step 4: NMS (Non-Maximum Suppression) による重複除去
  // 近接する（minPageDistance 以内）候補のうち、確信度の差が nmsThreshold (5%) 未満なら、設定された優先度（nmsPriority）に従って勝者を決定する。
  const candidates = [...aboveThreshold]
  const selected: InferenceResult[] = []
  const processed = new Set<number>()

  while (true) {
    // 未処理の中で最高確信度を見つける
    const remaining = candidates.filter((c) => !processed.has(c.canvasIndex))
    if (remaining.length === 0) break

    const best = remaining.reduce((prev, curr) => (curr.confidence > prev.confidence ? curr : prev), remaining[0])

    // 最高確信度の周辺（minPageDistance 以内）にある候補をすべて取得
    const inRange = remaining.filter((c) => Math.abs(c.canvasIndex - best.canvasIndex) <= minPageDistance)

    // 最高確信度との差が nmsThreshold 未満の候補（自分自身を含む）を抽出
    const nearTies = inRange.filter((c) => best.confidence - c.confidence <= nmsThreshold)

    // 優先度設定に基づき「勝者」を決定
    let winner: InferenceResult = best
    if (nmsPriority === 'first') {
      winner = nearTies.reduce((prev, curr) => (curr.canvasIndex < prev.canvasIndex ? curr : prev), nearTies[0])
    } else if (nmsPriority === 'last') {
      winner = nearTies.reduce((prev, curr) => (curr.canvasIndex > prev.canvasIndex ? curr : prev), nearTies[0])
    } else {
      // 'max' または指定なしの場合は best (既に最高確信度) を採用
      winner = best
    }

    selected.push(winner)

    // 勝者の周辺（および元の最高確信度の周辺）の候補をすべて抑制対象とする
    const toRemove = remaining.filter(
      (c) =>
        Math.abs(c.canvasIndex - winner.canvasIndex) <= minPageDistance ||
        Math.abs(c.canvasIndex - best.canvasIndex) <= minPageDistance,
    )
    toRemove.forEach((c) => processed.add(c.canvasIndex))
  }

  // canvasIndex 昇順でソート
  selected.sort((a, b) => a.canvasIndex - b.canvasIndex)

  // 番号のみのラベルを付与
  const covers = selected.map((r, i) => ({
    canvasIndex: r.canvasIndex,
    confidence: r.confidence,
    label: `${i + 1}`,
  }))

  return { covers, threshold: dynamicThreshold }
}
