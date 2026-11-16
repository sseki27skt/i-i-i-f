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

/**
 * ImageリソースとService情報から、IIIF Image API の末尾ファイル名（"default.jpg" | "native.jpg" 等）を判定する
 */
function detectImageQualitySuffix(
  resourceId?: string,
  service?: { '@context'?: string; profile?: unknown; type?: string; '@type'?: string }
): string {
  // 1. resource['@id'] / body.id の末尾から実際の quality.format を抽出（例: .../full/0/native.jpg → native.jpg）
  if (resourceId) {
    const match = resourceId.match(/\/((?:default|native|color|grey|gray|bitonal)\.[a-zA-Z0-9]+)$/i)
    if (match) {
      return match[1]
    }
  }

  // 2. service の @context / profile / type から Image API v1.x かどうかを判定
  if (service) {
    const ctx = service['@context'] ?? ''
    const profileStr = typeof service.profile === 'string' ? service.profile : JSON.stringify(service.profile ?? '')
    const typeStr = service.type ?? service['@type'] ?? ''

    const isV1 =
      ctx.includes('/image/1/') ||
      ctx.includes('image-api/1.1') ||
      profileStr.includes('library.stanford.edu/iiif/image-api') ||
      profileStr.includes('/image/1/') ||
      typeStr === 'ImageService1'

    if (isV1) {
      return 'native.jpg'
    }
  }

  // 3. デフォルト（Image API v2 / v3）
  return 'default.jpg'
}

/** IIIF Image API の URL から指定サイズの画像 URL を生成する */
export function buildIiifImageUrl(
  serviceId: string,
  size: number | 'full' | 'max' = 150,
  suffix = 'default.jpg',
): string {
  const base = serviceId.replace(/\/$/, '')
  const sizeParam = typeof size === 'number' ? `${size},` : size
  return `${base}/full/${sizeParam}/0/${suffix}`
}

/** NormalizedCanvas から指定サイズの画像 URL を取得する（Image Service がなければ thumbnailUrl にフォールバック） */
export function getCanvasImageUrl(
  canvas: Pick<NormalizedCanvas, 'imageServiceUrl' | 'imageQualitySuffix' | 'thumbnailUrl'>,
  size: number | 'full' | 'max',
): string {
  if (!canvas.imageServiceUrl) {
    return canvas.thumbnailUrl || ''
  }
  return buildIiifImageUrl(canvas.imageServiceUrl, size, canvas.imageQualitySuffix || 'default.jpg')
}

/** IIIF Image API の URL からサムネイル URL を生成する */
function makeThumbnailUrl(serviceId: string, size = 150, suffix = 'default.jpg'): string {
  return buildIiifImageUrl(serviceId, size, suffix)
}

/** 推論用 URL（フル解像度ではなく 224px 程度） */
function makeInferenceUrl(serviceId: string, size = 224, suffix = 'default.jpg'): string {
  return buildIiifImageUrl(serviceId, size, suffix)
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
  const rawService = Array.isArray(resource?.service) ? resource.service[0] : resource?.service
  if (rawService?.['@id'] || (rawService as any)?.id) {
    imageServiceUrl = rawService['@id'] || (rawService as any).id
  } else if (resource?.['@id']) {
    // Image API URL が直接指定されている場合
    const stripped = resource['@id'].replace(/\/full\/.+$/, '')
    if (stripped !== resource['@id']) {
      imageServiceUrl = stripped
    }
  }

  const imageQualitySuffix = detectImageQualitySuffix(resource?.['@id'], rawService as any)

  // サムネイル URL
  let thumbnailUrl = ''
  if (canvas.thumbnail) {
    if (typeof canvas.thumbnail === 'string') {
      thumbnailUrl = canvas.thumbnail
    } else if (Array.isArray(canvas.thumbnail)) {
      const firstThumb = canvas.thumbnail[0]
      thumbnailUrl = typeof firstThumb === 'string' ? firstThumb : firstThumb?.['@id'] ?? ''
    } else {
      thumbnailUrl = canvas.thumbnail['@id'] ?? (canvas.thumbnail as any).id ?? ''
    }
  } else if (imageServiceUrl) {
    thumbnailUrl = makeThumbnailUrl(imageServiceUrl, 150, imageQualitySuffix)
  } else if (resource?.['@id']) {
    thumbnailUrl = resource['@id']
  }

  return {
    id,
    index,
    label,
    thumbnailUrl,
    imageServiceUrl: imageServiceUrl || '',
    imageQualitySuffix,
    inferenceUrl: imageServiceUrl
      ? makeInferenceUrl(imageServiceUrl, 224, imageQualitySuffix)
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
  const rawService = Array.isArray(bodyItem?.service) ? bodyItem.service[0] : (bodyItem?.service as any)
  if (rawService?.id || rawService?.['@id']) {
    imageServiceUrl = rawService.id || rawService['@id']
  } else if (bodyItem?.id) {
    const stripped = bodyItem.id.replace(/\/full\/.+$/, '')
    if (stripped !== bodyItem.id) {
      imageServiceUrl = stripped
    }
  }

  const imageQualitySuffix = detectImageQualitySuffix(bodyItem?.id, rawService)

  // サムネイル
  let thumbnailUrl = ''
  const thumb = canvas.thumbnail?.[0]
  if (thumb?.id) {
    thumbnailUrl = thumb.id
  } else if (imageServiceUrl) {
    thumbnailUrl = makeThumbnailUrl(imageServiceUrl, 150, imageQualitySuffix)
  } else if (bodyItem?.id) {
    thumbnailUrl = bodyItem.id
  }

  return {
    id,
    index,
    label,
    thumbnailUrl,
    imageServiceUrl: imageServiceUrl || '',
    imageQualitySuffix,
    inferenceUrl: imageServiceUrl
      ? makeInferenceUrl(imageServiceUrl, 224, imageQualitySuffix)
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
