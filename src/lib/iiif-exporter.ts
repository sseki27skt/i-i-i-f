/**
 * iiif-exporter.ts — 内部状態を IIIF V2 / V3 マニフェストとして出力する
 *
 * 表紙候補（DetectedCover）に基づき structures (Range) を付与したマニフェストを生成する。
 */

import type { NormalizedManifest, NormalizedCanvas, DetectedCover } from '@/types/app'
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
 * 確実に全コマのサムネイルが表示されるよう、正規化されたデータ (NormalizedCanvas) を優先的に使用する。
 */
function ensureV3(raw: any, lang: string, normalizedCanvases?: NormalizedCanvas[]): IiifV3Manifest {
  const toLangMap = (val: any): LanguageMap => {
    if (typeof val === 'object' && val !== null && !Array.isArray(val)) {
      return val as LanguageMap
    }
    const str = typeof val === 'string' ? val : String(val ?? '')
    const key = lang === 'none' ? 'none' : lang
    return { [key]: [str] }
  }

  // マニフェスト ID にサフィックスを付けて、ビューワーのキャッシュを回避する
  const baseId = raw.id || raw['@id'] || 'http://example.org/iiif/manifest';
  const manifestId = baseId.includes('#') ? baseId : `${baseId}#segmented-${Date.now()}`;

  const result: any = {
    '@context': 'http://iiif.io/api/presentation/3/context.json',
    id: manifestId,
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
  const attribution = raw.attribution || (raw.requiredStatement?.value);
  let rightsVal = raw.rights || raw.license;

  if (typeof rightsVal === 'string') {
    if (rightsVal.includes('creativecommons.org/licenses/')) {
      rightsVal = rightsVal.replace('https://', 'http://');
      rightsVal = rightsVal.replace(/\/(deed|legalcode)(\.[a-z]{2})?(\/?)?$/, '/');
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

  // サムネイル (マニフェストレベル)
  if (normalizedCanvases?.[0]?.thumbnailUrl) {
    result.thumbnail = [
      {
        id: normalizedCanvases[0].thumbnailUrl,
        type: 'Image',
        format: 'image/jpeg'
      }
    ];
  }

  // アイテム (Canvas) リストの構築
  // 元データの Canvas リストを取得
  const rawItems = raw.items || raw.sequences?.[0]?.canvases || [];
  
  // NormalizedCanvas を主としてループし、不足情報を rawItems から補完する
  result.items = (normalizedCanvases || []).map((nc, cIdx) => {
    const c = rawItems[cIdx] || {};
    const canvas: any = {
      id: nc.id,
      type: 'Canvas',
      label: toLangMap(nc.label),
      width: nc.imageServiceUrl ? c.width : undefined, // ImageService があれば width/height はオプション
      height: nc.imageServiceUrl ? c.height : undefined,
    };

    // サムネイルの確実な付与
    if (nc.thumbnailUrl) {
      canvas.thumbnail = [
        {
          id: nc.thumbnailUrl,
          type: 'Image',
          format: 'image/jpeg'
        }
      ];
      
      // Image Service 情報があれば付与（一部のビューワーでギャラリー表示に必要）
      if (nc.imageServiceUrl) {
        canvas.thumbnail[0].service = [
          {
            id: nc.imageServiceUrl,
            type: 'ImageService2',
            profile: 'level1'
          }
        ];
      }
    }

    // 画像アノテーションの構築
    if (c.items) {
      // 既に V3 形式ならそのまま流用
      canvas.items = c.items;
    } else {
      // V2 からの変換
      const images = c.images || [];
      canvas.items = [
        {
          id: `${nc.id}/page`,
          type: 'AnnotationPage',
          items: images.map((img: any, iIdx: number) => ({
            id: img.id || img['@id'] || `${nc.id}/anno/${iIdx}`,
            type: 'Annotation',
            motivation: 'painting',
            body: {
              id: img.resource?.id || img.resource?.['@id'] || nc.imageServiceUrl + '/full/max/0/default.jpg',
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
              ) : (nc.imageServiceUrl ? [
                {
                  id: nc.imageServiceUrl,
                  type: 'ImageService2',
                  profile: 'level1'
                }
              ] : undefined)
            },
            target: nc.id
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
  const base = ensureV3(manifest.raw, language, manifest.canvases)

  // Range を構築
  const ranges: IiifV3Range[] = covers.map((cover, i) => {
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

  // Range リストをそのまま structures として使用する（以前は Table of Contents でラップしていたが、表示上の問題があるため廃止）
  const structures: IiifV3Range[] = ranges

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

/**
 * 内部状態から学習データ用 JSON (iiif_ml_dataset 形式) を生成し、ダウンロードを起動する。
 */
export function downloadTrainingDataJson(
  manifest: NormalizedManifest,
  covers: DetectedCover[],
): void {
  const coverIndices = new Set(covers.map((c) => c.canvasIndex))

  const dataset = manifest.canvases.map((canvas) => {
    // 画像URLの構築 (Image Service がある場合はフルサイズ、ない場合はそのまま)
    const imageUrl = canvas.imageServiceUrl
      ? `${canvas.imageServiceUrl.replace(/\/$/, '')}/full/full/0/default.jpg`
      : canvas.thumbnailUrl

    return {
      id: canvas.id,
      imageUrl,
      label: coverIndices.has(canvas.index) ? 'cover' : 'content',
    }
  })

  const json = JSON.stringify(dataset, null, 2)
  const blob = new Blob([json], { type: 'application/json' })
  const url = URL.createObjectURL(blob)

  const a = document.createElement('a')
  a.href = url
  a.download = `i-i-i-f_ml_dataset.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
