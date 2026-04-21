/**
 * iiif-exporter.ts — 内部状態を IIIF V2 / V3 マニフェストとして出力する
 *
 * 表紙候補（DetectedCover）に基づき structures (Range) を付与したマニフェストを生成する。
 */

import type { NormalizedManifest, DetectedCover } from '@/types/app'
import type { ExportVersion, ExportLanguage } from '@/types/app'
import type {
  IiifV2Manifest, IiifV2Range,
  IiifV3Manifest, IiifV3Range,
  LanguageMap,
} from '@/types/iiif'

// ─────────────── 変換ヘルパー ───────────────

/**
 * 任意のオブジェクトを強制的に IIIF V3 形式（Language Map 等）に補正する。
 * 元が V2 の場合、@id -> id, @type -> type などの変換も行う。
 */
function ensureV3(raw: any, lang: string): IiifV3Manifest {
  const toLangMap = (val: any): LanguageMap => {
    if (typeof val === 'object' && val !== null && !Array.isArray(val)) {
      // すでにオブジェクトならそのまま（簡易判定）
      return val as LanguageMap
    }
    const str = typeof val === 'string' ? val : String(val ?? '')
    const key = lang === 'none' ? 'none' : lang
    return { [key]: [str] }
  }

  const result: any = {
    '@context': 'http://iiif.io/api/presentation/3/context.json',
    id: raw.id || raw['@id'] || 'http://example.org/iiif/manifest',
    type: 'Manifest',
    label: toLangMap(raw.label || 'Untitled'),
  }

  // メタデータ
  if (raw.metadata && Array.isArray(raw.metadata)) {
    result.metadata = raw.metadata.map((m: any) => ({
      label: toLangMap(m.label),
      value: toLangMap(m.value),
    }))
  }

  // 説明文
  if (raw.summary || raw.description) {
    result.summary = toLangMap(raw.summary || raw.description)
  }

  // 権利情報・帰属
  // V3 の rights は CC か RightsStatements の URI のみ許可される
  const attribution = raw.attribution || (raw.requiredStatement?.value);
  let rightsVal = raw.rights || raw.license;

  if (typeof rightsVal === 'string') {
    // Creative Commons の正規化
    if (rightsVal.includes('creativecommons.org/licenses/')) {
      // バリデータの期待する形式 (http 且つ /deed などのサフィックスなし) に補正
      rightsVal = rightsVal.replace('https://', 'http://');
      // /deed.ja, /deed.en, /legalcode 等のサフィックスを除去
      rightsVal = rightsVal.replace(/\/(deed|legalcode)(\.[a-z]{2})?(\/?)?$/, '/');
      // 末尾の / を保証
      if (!rightsVal.endsWith('/')) rightsVal += '/';
    }
  }

  const isLegalRights = typeof rightsVal === 'string' &&
    /http:\/\/(creativecommons\.org\/licenses\/|rightsstatements\.org\/vocab\/)/.test(rightsVal);

  if (attribution || (rightsVal && !isLegalRights)) {
    result.requiredStatement = {
      label: toLangMap('Attribution/Rights'),
      value: toLangMap(
        [
          typeof attribution === 'string' ? attribution : (Array.isArray(attribution) ? attribution.join(', ') : ''),
          !isLegalRights && typeof rightsVal === 'string' ? rightsVal : null
        ].filter(Boolean).join(' / ')
      )
    }
  }

  if (isLegalRights) {
    result.rights = rightsVal;
  }

  // サムネイル
  if (raw.thumbnail) {
    const thumbs = Array.isArray(raw.thumbnail) ? raw.thumbnail : [raw.thumbnail]
    result.thumbnail = thumbs.map((t: any) => {
      if (typeof t === 'string') return { id: t, type: 'Image' }
      if (t['@id']) return { id: t['@id'], type: 'Image' }
      return t
    })
  }

  // 関連リンク
  if (raw.homepage || raw.related) {
    const pages = Array.isArray(raw.homepage || raw.related) ? (raw.homepage || raw.related) : [raw.homepage || raw.related]
    result.homepage = pages.map((p: any) => {
      if (typeof p === 'string') return { id: p, type: 'Text', label: toLangMap('Homepage'), format: 'text/html' }
      return p
    })
  }

  // 外部参照
  if (raw.seeAlso) {
    const see = Array.isArray(raw.seeAlso) ? raw.seeAlso : [raw.seeAlso]
    result.seeAlso = see.map((s: any) => {
      if (typeof s === 'string') return { id: s, type: 'Dataset' }
      return s
    })
  }

  // アイテム (Canvas) リストの変換
  const rawItems = raw.items || raw.sequences?.[0]?.canvases || [];
  result.items = rawItems.map((c: any, cIdx: number) => {
    // すでに V3 構造（items を持つ）なら一部流用、そうでなければ構築
    const canvasId = c.id || c['@id'] || `canvas-${cIdx}`;
    const canvas: any = {
      id: canvasId,
      type: 'Canvas',
      label: toLangMap(c.label || String(cIdx + 1)),
      width: c.width,
      height: c.height,
    };

    if (c.items) {
      canvas.items = c.items;
    } else {
      // V2 images -> V3 items (AnnotationPage)
      const images = c.images || [];
      canvas.items = [
        {
          id: `${canvasId}/page`,
          type: 'AnnotationPage',
          items: images.map((img: any, iIdx: number) => ({
            id: img.id || img['@id'] || `${canvasId}/anno/${iIdx}`,
            type: 'Annotation',
            motivation: 'painting',
            body: {
              id: img.resource?.id || img.resource?.['@id'],
              type: 'Image',
              format: img.resource?.format || 'image/jpeg',
              width: img.resource?.width,
              height: img.resource?.height,
              service: img.resource?.service ? (
                Array.isArray(img.resource.service) 
                  ? img.resource.service.map((s: any) => ({
                      id: s.id || s['@id'],
                      type: s['@context']?.includes('/3/context.json') ? 'ImageService3' : 'ImageService2',
                      profile: s.profile
                    }))
                  : [{
                      id: img.resource.service.id || img.resource.service['@id'],
                      type: img.resource.service['@context']?.includes('/3/context.json') ? 'ImageService3' : 'ImageService2',
                      profile: img.resource.service.profile
                    }]
              ) : undefined
            },
            target: canvasId
          }))
        }
      ];
    }
    return canvas;
  });

  return result as IiifV3Manifest
}

// ─────────────── V3 エクスポーター ───────────────

/**
 * V3 形式でエクスポートする。
 */
export function exportV3(
  manifest: NormalizedManifest,
  covers: DetectedCover[],
  language: ExportLanguage,
): IiifV3Manifest {
  // 元の manifest.raw を V3 仕様にクリーンアップ/変換する
  const base = ensureV3(manifest.raw, language)

  // Range を構築
  const structures: IiifV3Range[] = covers.map((cover, i) => {
    const start = cover.canvasIndex
    const end = covers[i + 1]?.canvasIndex ?? manifest.canvases.length
    const canvasIds = manifest.canvases
      .slice(start, end)
      .map((c) => ({ id: c.id, type: 'Canvas' as const }))

    const key = language === 'none' ? 'none' : language
    return {
      id: `${manifest.id}/range/r${i + 1}`,
      type: 'Range',
      label: { [key]: [cover.label] },
      items: canvasIds,
    }
  })

  return {
    ...base,
    structures,
  }
}

// ─────────────── V2 エクスポーター ───────────────

export function exportV2(
  manifest: NormalizedManifest,
  covers: DetectedCover[],
): IiifV2Manifest {
  const raw = manifest.raw as IiifV2Manifest

  const structures: IiifV2Range[] = covers.map((cover, i) => {
    const start = cover.canvasIndex
    const end = covers[i + 1]?.canvasIndex ?? manifest.canvases.length
    const canvasIds = manifest.canvases
      .slice(start, end)
      .map((c) => c.id)

    return {
      '@id': `${manifest.id}/range/r${i + 1}`,
      '@type': 'sc:Range',
      label: cover.label,
      canvases: canvasIds,
    }
  })

  return {
    ...raw,
    structures,
  }
}

// ─────────────── 公開 API ───────────────

/**
 * 内部状態から IIIF マニフェスト JSON を生成し、ダウンロードを起動する。
 */
export function downloadManifest(
  manifest: NormalizedManifest,
  covers: DetectedCover[],
  version: ExportVersion,
  language: ExportLanguage,
): void {
  const exported = version === 'v3'
    ? exportV3(manifest, covers, language)
    : exportV2(manifest, covers)

  const json = JSON.stringify(exported, null, 2)
  const blob = new Blob([json], { type: 'application/json' })
  const url = URL.createObjectURL(blob)

  const a = document.createElement('a')
  a.href = url
  a.download = `manifest_segmented_${version}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

/**
 * 内部状態から IIIF マニフェスト JSON を生成し、クリップボードにコピーする。
 */
export async function copyManifestToClipboard(
  manifest: NormalizedManifest,
  covers: DetectedCover[],
  version: ExportVersion,
  language: ExportLanguage,
): Promise<void> {
  const exported = version === 'v3'
    ? exportV3(manifest, covers, language)
    : exportV2(manifest, covers)

  const json = JSON.stringify(exported, null, 2)
  await navigator.clipboard.writeText(json)
}
