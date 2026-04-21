/**
 * HelpModal.tsx — 使い方と技術的詳細を説明するモーダル
 */

import { X, HelpCircle, FileJson, Sparkles, Library } from 'lucide-react'

interface HelpModalProps {
  isOpen: boolean
  onClose: () => void
}

export function HelpModal({ isOpen, onClose }: HelpModalProps) {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fade-in p-4">
      <div 
        className="card w-full max-w-2xl max-h-[85vh] flex flex-col overflow-hidden shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'var(--color-border)' }}>
          <div className="flex items-center gap-2">
            <HelpCircle className="w-5 h-5" style={{ color: 'var(--color-accent)' }} />
            <h2 className="text-lg font-semibold" style={{ color: 'var(--color-text-primary)' }}>IIIF Volume Segmenter の使い方</h2>
          </div>
          <button 
            onClick={onClose}
            className="p-2 -mr-2 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
            style={{ color: 'var(--color-text-muted)' }}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8" style={{ color: 'var(--color-text-secondary)' }}>
          
          <section className="space-y-3">
            <h3 className="font-medium text-base" style={{ color: 'var(--color-text-primary)' }}>基本フロー</h3>
            <div className="flex items-center justify-center gap-4 py-4 bg-slate-50 dark:bg-slate-900/50 rounded-xl border border-black/5 dark:border-white/5">
              <div className="flex flex-col items-center gap-2">
                <div className="w-12 h-12 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-blue-600 dark:text-blue-400">
                  <FileJson className="w-6 h-6" />
                </div>
                <span className="text-xs font-medium">1. URL入力</span>
              </div>
              <div className="w-8 h-px bg-slate-200 dark:bg-slate-700"></div>
              <div className="flex flex-col items-center gap-2">
                <div className="w-12 h-12 rounded-full bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center text-indigo-600 dark:text-indigo-400">
                  <Sparkles className="w-6 h-6" />
                </div>
                <span className="text-xs font-medium">2. AI推論</span>
              </div>
              <div className="w-8 h-px bg-slate-200 dark:bg-slate-700"></div>
              <div className="flex flex-col items-center gap-2">
                <div className="w-12 h-12 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center text-amber-600 dark:text-amber-400">
                  <Library className="w-6 h-6" />
                </div>
                <span className="text-xs font-medium">3. 冊分割</span>
              </div>
            </div>
            <p className="text-sm leading-relaxed">
              IIIF ManifestのURLを入力すると、ブラウザ上のAIを用いて全コマの画像を解析します。表紙と思われる画像を自動検出し、1つのマニフェストを複数の「冊」に分割することが可能です。
            </p>
          </section>

          <section className="space-y-3">
            <h3 className="font-medium text-base" style={{ color: 'var(--color-text-primary)' }}>操作・出力</h3>
            <ul className="list-disc pl-5 text-sm space-y-2 leading-relaxed">
              <li> <strong>推論設定:</strong> ヘッダー右の設定アイコンから、推論モデルや表紙検出のアルゴリズムをカスタマイズできます。
                <ul className="list-circle pl-5 mt-1 space-y-1 opacity-90">
                  <li><strong>推論モデル:</strong> MobileNetV3 Large, MobileNetV2, または両方の結果を平均する Ensemble モードを選択できます。</li>
                  <li><strong>自動閾値決定ロジック:</strong>
                    <ul className="list-square pl-5 mt-1 space-y-1">
                      <li><strong>Max Gap:</strong> Confidenceの差が最も大きい箇所の中点を閾値とします。</li>
                      <li><strong>First Gap:</strong> Confidenceを低い方からスキャンし、最初に5%以上のギャップが見つかった箇所を閾値とします。</li>
                      <li><strong>K-means:</strong> Confidenceを「表紙」と「非表紙」の2クラスターに1次元K-means法で分類し、その境界を閾値とします。</li>
                    </ul>
                  <li><strong>閾値スライダー:</strong>閾値を調整し表紙検出の結果を更新できます。</li>
                  </li>
                </ul>
              </li>
              <li><strong>サムネイル表示:</strong> 各画像の左上に推論された表紙としての確信度（Confidence）が表示されます。黄色の枠で囲まれたコマが検出された表紙候補です。</li>
              <li><strong>手動補正:</strong> 誤検知や漏れがあった場合は、サムネイルをクリックすることで表紙の選択・非選択を切り替えられます。</li>
              <li><strong>メタデータ編集:</strong> 「分析結果」タブの右ペインから、各冊のラベルを直接編集できます。</li>
              <li><strong>エクスポート機能:</strong> IIIF V2 または V3 フォーマットを選択し、分割済みマニフェスト（JSON）をダウンロード、またはクリップボードにコピーできます。</li>
            </ul>
          </section>

          <section className="space-y-3">
            <h3 className="font-medium text-base" style={{ color: 'var(--color-text-primary)' }}>技術的な詳細</h3>
            <p className="text-sm leading-relaxed">
              すべての推論処理はフロントエンド（ユーザーのブラウザ内）で実行されます。画像データが外部サーバーに送信されることはありません。<br/>
              推論モデルには軽量な MobileNetV3 派生アーキテクチャを採用しており、画像は自動的に低解像度化・正規化（Normalization）されてからモデルに入力されます。
            </p>
          </section>

        </div>
      </div>
    </div>
  )
}
