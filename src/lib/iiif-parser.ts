/**
 * iiif-parser.ts — IIIF V2 / V3 マニフェストを内部形式に正規化する
 *
 * 入力: fetch した JSON（型不明）
 * 出力: NormalizedManifest
 */

import type {
  IiifV2Canvas,
  IiifV2Manifest,
  IiifV3Canvas,
  IiifV3Manifest,
  LanguageMap,
} from '@/types/iiif'
import type { NormalizedCanvas, NormalizedManifest } from '@/types/app'

// ─────────────── ヘルパー ───────────────

/** LanguageMap から最初の値の文字列を取り出す */
function extractLabel(langMap: LanguageMap | undefined, fallback = ''): string {
  if (!langMap) return fallback
  const langs = ['ja', 'en', 'none']
  for (const lang of langs) {
    if (langMap[lang]?.[0]) return langMap[lang][0]
  }
  const first = Object.values(langMap)[0]
  return first?.[0] ?? fallback
}

/** IIIF Image API の URL からサムネイル URL を生成する */
function makeThumbnailUrl(serviceId: string, size = 150): string {
  const base = serviceId.replace(/\/$/, '')
  return `${base}/full/${size},/0/default.jpg`
}

/** 推論用 URL（フル解像度ではなく 224px 程度） */
function makeInferenceUrl(serviceId: string, size = 224): string {
  const base = serviceId.replace(/\/$/, '')
  return `${base}/full/${size},/0/default.jpg`
}

// ─────────────── V2 パーサー ───────────────

function parseV2Canvas(canvas: IiifV2Canvas, index: number): NormalizedCanvas {
  const id = canvas['@id']

  // ラベル
  const label = typeof canvas.label === 'string'
    ? canvas.label
    : String(index + 1)

  // Image Service URL の取得
  let imageServiceUrl = ''
  const resource = canvas.images?.[0]?.resource
  if (resource?.service?.['@id']) {
    imageServiceUrl = resource.service['@id']
  } else if (resource?.['@id']) {
    // Image API URL が直接指定されている場合
    imageServiceUrl = resource['@id'].replace(/\/full\/.+$/, '')
  }

  // サムネイル URL
  let thumbnailUrl = ''
  if (canvas.thumbnail) {
    thumbnailUrl = typeof canvas.thumbnail === 'string'
      ? canvas.thumbnail
      : canvas.thumbnail['@id'] ?? ''
  } else if (imageServiceUrl) {
    thumbnailUrl = makeThumbnailUrl(imageServiceUrl)
  }

  return {
    id,
    index,
    label,
    thumbnailUrl,
    imageServiceUrl: imageServiceUrl || '',
    inferenceUrl: imageServiceUrl
      ? makeInferenceUrl(imageServiceUrl)
      : thumbnailUrl,
  }
}

function parseV2(raw: IiifV2Manifest): NormalizedManifest {
  const label = typeof raw.label === 'string'
    ? raw.label
    : raw.label?.[0]?.['@value'] ?? 'Untitled'

  const canvases = raw.sequences?.[0]?.canvases?.map(parseV2Canvas) ?? []

  return { id: raw['@id'], label, version: 'v2', raw, canvases }
}

// ─────────────── V3 パーサー ───────────────

function parseV3Canvas(canvas: IiifV3Canvas, index: number): NormalizedCanvas {
  const id = canvas.id
  const label = extractLabel(canvas.label, String(index + 1))

  // Image Service URL
  let imageServiceUrl = ''
  const body = canvas.items?.[0]?.items?.[0]?.body
  const bodyItem = Array.isArray(body) ? body[0] : body
  if (bodyItem?.service?.[0]?.id) {
    imageServiceUrl = bodyItem.service[0].id
  } else if (bodyItem?.id) {
    imageServiceUrl = bodyItem.id.replace(/\/full\/.+$/, '')
  }

  // サムネイル
  let thumbnailUrl = ''
  const thumb = canvas.thumbnail?.[0]
  if (thumb?.id) {
    thumbnailUrl = thumb.id
  } else if (imageServiceUrl) {
    thumbnailUrl = makeThumbnailUrl(imageServiceUrl)
  }

  return {
    id,
    index,
    label,
    thumbnailUrl,
    imageServiceUrl: imageServiceUrl || '',
    inferenceUrl: imageServiceUrl
      ? makeInferenceUrl(imageServiceUrl)
      : thumbnailUrl,
  }
}

function parseV3(raw: IiifV3Manifest): NormalizedManifest {
  const label = extractLabel(raw.label, 'Untitled')
  const canvases = raw.items?.map(parseV3Canvas) ?? []
  return { id: raw.id, label, version: 'v3', raw, canvases }
}

// ─────────────── 公開 API ───────────────

/** バージョンを自動判定して正規化する */
export function parseManifest(raw: unknown): NormalizedManifest {
  if (typeof raw !== 'object' || raw === null) {
    throw new Error('Invalid manifest: not an object')
  }

  const obj = raw as Record<string, unknown>

  // V3 判定: "@context" が "http://iiif.io/api/presentation/3/context.json" を含む
  const ctx = obj['@context']
  const isV3 =
    (typeof ctx === 'string' && ctx.includes('/3/context.json')) ||
    (Array.isArray(ctx) && ctx.some((c) => typeof c === 'string' && c.includes('/3/context.json'))) ||
    obj['type'] === 'Manifest'

  if (isV3) {
    return parseV3(raw as IiifV3Manifest)
  } else {
    return parseV2(raw as IiifV2Manifest)
  }
}

/** URL からマニフェストを fetch して正規化する */
export async function fetchAndParseManifest(url: string): Promise<NormalizedManifest> {
  const resp = await fetch(url)
  if (!resp.ok) {
    throw new Error(`Failed to fetch manifest: ${resp.status} ${resp.statusText}`)
  }
  const json = await resp.json()
  return parseManifest(json)
}
