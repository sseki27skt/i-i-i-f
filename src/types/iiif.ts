/**
 * iiif.ts — IIIF V2 / V3 の型定義
 *
 * アプリ内では NormalizedManifest / NormalizedCanvas に正規化して扱う。
 */

// ─────────────── IIIF V2 ───────────────

export interface IiifV2Manifest {
  '@context': string
  '@id': string
  '@type': 'sc:Manifest'
  label: string | { '@language': string; '@value': string }[]
  sequences: IiifV2Sequence[]
  structures?: IiifV2Range[]
}

export interface IiifV2Sequence {
  '@type': 'sc:Sequence'
  canvases: IiifV2Canvas[]
}

export interface IiifV2Canvas {
  '@id': string
  '@type': 'sc:Canvas'
  label: string
  width: number
  height: number
  images: IiifV2Annotation[]
  thumbnail?: string | { '@id': string }
}

export interface IiifV2Annotation {
  '@type': 'oa:Annotation'
  resource: {
    '@id': string
    service?: {
      '@id': string
      '@context'?: string
      profile?: string
    }
  }
}

export interface IiifV2Range {
  '@id': string
  '@type': 'sc:Range'
  label: string
  canvases: string[]
}

// ─────────────── IIIF V3 ───────────────

export interface IiifV3Manifest {
  '@context': string | string[]
  id: string
  type: 'Manifest'
  label: LanguageMap
  items: IiifV3Canvas[]
  structures?: IiifV3Range[]
}

export type LanguageMap = Record<string, string[]>

export interface IiifV3Canvas {
  id: string
  type: 'Canvas'
  label?: LanguageMap
  width?: number
  height?: number
  items: IiifV3AnnotationPage[]
  thumbnail?: IiifV3Resource[]
}

export interface IiifV3AnnotationPage {
  id: string
  type: 'AnnotationPage'
  items: IiifV3Annotation[]
}

export interface IiifV3Annotation {
  id: string
  type: 'Annotation'
  motivation: 'painting'
  body: IiifV3Resource | IiifV3Resource[]
  target: string
}

export interface IiifV3Resource {
  id: string
  type: string
  format?: string
  width?: number
  height?: number
  service?: IiifV3Service[]
}

export interface IiifV3Service {
  id: string
  type: string
  profile?: string
}

export interface IiifV3Range {
  id: string
  type: 'Range'
  label: LanguageMap
  items: Array<{ id: string; type: 'Canvas' } | { id: string; type: 'Range' }>
}
