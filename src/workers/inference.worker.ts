/**
 * inference.worker.ts — ONNX 推論 Web Worker
 *
 * Main スレッドとのメッセージ通信:
 *   受信: WorkerInMessage (LOAD_MODEL | RUN_INFERENCE | CANCEL)
 *   送信: WorkerOutMessage (MODEL_LOADED | MODEL_ERROR | INFERENCE_RESULT | INFERENCE_ERROR | DONE)
 */

import * as ort from 'onnxruntime-web'
import type { WorkerInMessage, WorkerOutMessage, ModelId, InferenceResult } from '@/types/app'
import { MODEL_CONFIGS } from '@/types/app'

// onnxruntime-web の WASM ファイルを CDN から読み込む
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/'

let session: ort.InferenceSession | null = null
let currentModelId: ModelId | null = null
let cancelled = false

// ─────────────── 先行フェッチ管理 ───────────────
/** 前処理済みデータの Promise を保持するマップ (canvasIndex -> Promise) */
const prefetchMap = new Map<number, Promise<Float32Array>>()
const MAX_PREFETCH_COUNT = 32 // 先読みする最大枚数

// ─────────────── WebGPU リソース ───────────────
let gpuDevice: GPUDevice | null = null
let gpuPipeline: GPUComputePipeline | null = null
let gpuSampler: GPUSampler | null = null

/**
 * 与えられた WebGPU Device を用いて前処理パイプラインを初期化する
 */
async function initWebGPUPipeline(device: GPUDevice) {
  if (gpuPipeline) return
  try {
    // 前処理用シェーダーの構築
    const shaderModule = device.createShaderModule({
      label: 'Preprocess Shader',
      code: `
        struct Params {
          width: f32,
          height: f32,
          mean_r: f32,
          mean_g: f32,
          mean_b: f32,
          std_r: f32,
          std_g: f32,
          std_b: f32,
        }

        @group(0) @binding(0) var mySampler: sampler;
        @group(0) @binding(1) var myTexture: texture_2d<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        @group(0) @binding(3) var<uniform> params: Params;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let x = id.x;
          let y = id.y;
          let w = u32(params.width);
          let h = u32(params.height);

          if (x >= w || y >= h) { return; }

          // テクスチャ座標 (0.0 to 1.0)
          let uv = vec2<f32>(
            (f32(x) + 0.5) / params.width,
            (f32(y) + 0.5) / params.height
          );

          // バイリニアサンプリング
          let color = textureSampleLevel(myTexture, mySampler, uv, 0.0);

          // 正規化: (color - mean) / std
          let norm_r = (color.r - params.mean_r) / params.std_r;
          let norm_g = (color.g - params.mean_g) / params.std_g;
          let norm_b = (color.b - params.mean_b) / params.std_b;

          // NCHW レイアウトへの書き込み
          let pixels = w * h;
          let idx = y * w + x;
          
          output[idx]              = norm_r; // R 平面
          output[pixels + idx]     = norm_g; // G 平面
          output[pixels * 2 + idx] = norm_b; // B 平面
        }
      `
    })

    gpuPipeline = device.createComputePipeline({
      label: 'Preprocess Pipeline',
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      }
    })

    gpuSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    })
  } catch (err) {
    console.warn('WebGPU pipeline initialization failed:', err)
  }
}

// ─────────────── 前処理用リソース (CPU Fallback用) ───────────────
let offscreenCanvas: OffscreenCanvas | null = null
let offscreenCtx: OffscreenCanvasRenderingContext2D | null = null

// ─────────────── 前処理 ───────────────

const IMAGENET_MEAN = [0.485, 0.456, 0.406]
const IMAGENET_STD  = [0.229, 0.224, 0.225]

/**
 * ArrayBuffer (JPEG/PNG バイナリ) → 正規化済み Float32Array (NCHW)
 * createImageBitmap → OffscreenCanvas でリサイズ → ピクセル値抽出
 */
async function preprocessImage(
  buffer: ArrayBuffer,
  inputSize: number,
): Promise<Float32Array> {
  const blob = new Blob([buffer])
  const bitmap = await createImageBitmap(blob)

  if (!offscreenCanvas || offscreenCanvas.width !== inputSize) {
    offscreenCanvas = new OffscreenCanvas(inputSize, inputSize)
    offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true })
  }
  
  const ctx = offscreenCtx!
  ctx.drawImage(bitmap, 0, 0, inputSize, inputSize)
  bitmap.close()

  const { data } = ctx.getImageData(0, 0, inputSize, inputSize)
  const pixels = inputSize * inputSize
  const tensor = new Float32Array(3 * pixels)

  // ループ内の計算を最小化するための定数
  const rNorm = 1 / (255 * IMAGENET_STD[0])
  const gNorm = 1 / (255 * IMAGENET_STD[1])
  const bNorm = 1 / (255 * IMAGENET_STD[2])
  const rOffset = -IMAGENET_MEAN[0] / IMAGENET_STD[0]
  const gOffset = -IMAGENET_MEAN[1] / IMAGENET_STD[1]
  const bOffset = -IMAGENET_MEAN[2] / IMAGENET_STD[2]

  for (let i = 0; i < pixels; i++) {
    const i4 = i * 4
    tensor[i]              = data[i4]     * rNorm + rOffset
    tensor[pixels + i]     = data[i4 + 1] * gNorm + gOffset
    tensor[pixels * 2 + i] = data[i4 + 2] * bNorm + bOffset
  }
  return tensor
}

/**
 * WebGPU を使用して画像の前処理を行う
 * 1枚の画像を GPU 側のバッファの指定位置に書き込む
 */
async function preprocessImageGPU(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  sampler: GPUSampler,
  buffer: ArrayBuffer,
  outputBuffer: GPUBuffer,
  outputOffset: number, // バイト単位
  inputSize: number,
) {
  const blob = new Blob([buffer])
  const bitmap = await createImageBitmap(blob)
  
  const texture = device.createTexture({
    size: [bitmap.width, bitmap.height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  })
  device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height])
  bitmap.close()

  const paramsBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([
    inputSize, inputSize, 
    IMAGENET_MEAN[0], IMAGENET_MEAN[1], IMAGENET_MEAN[2],
    IMAGENET_STD[0], IMAGENET_STD[1], IMAGENET_STD[2]
  ]))

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: texture.createView() },
      { binding: 2, resource: { buffer: outputBuffer, offset: outputOffset, size: inputSize * inputSize * 3 * 4 } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ]
  })

  const commandEncoder = device.createCommandEncoder()
  const passEncoder = commandEncoder.beginComputePass()
  passEncoder.setPipeline(pipeline)
  passEncoder.setBindGroup(0, bindGroup)
  const workgroupCount = Math.ceil(inputSize / 8)
  passEncoder.dispatchWorkgroups(workgroupCount, workgroupCount)
  passEncoder.end()
  
  device.queue.submit([commandEncoder.finish()])

  // 画像テクスチャは即時破棄してVRAMを節約
  texture.destroy()
  // paramsBuffer も本来は再利用すべきだが、一旦実行ごとに作成
}

// ─────────────── Softmax ───────────────

function softmax(logits: Float32Array): Float32Array {
  const max = Math.max(...Array.from(logits))
  const exp = logits.map((v) => Math.exp(v - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map((v) => v / sum) as Float32Array
}

// ─────────────── メッセージキュー管理 ───────────────
let messageQueue: WorkerInMessage[] = []
let isProcessing = false

self.onmessage = (e: MessageEvent<WorkerInMessage>) => {
  const msg = e.data
  
  if (msg.type === 'CANCEL') {
    cancelled = true
    messageQueue = []
    prefetchMap.clear()
    return
  }

  messageQueue.push(msg)
  
  // 先行フェッチをトリガー
  if (msg.type === 'RUN_BATCH_INFERENCE' || msg.type === 'RUN_INFERENCE') {
    triggerPrefetch()
  }

  if (!isProcessing) {
    processQueue()
  }
}

/**
 * メッセージキューを走査し、未処理のアイテムがあれば先行フェッチを開始する
 */
function triggerPrefetch() {
  if (cancelled) return

  // キュー全体から先行フェッチすべきアイテムを抽出
  const itemsToPrefetch: { canvasIndex: number; imageData?: ArrayBuffer; imageUrl?: string }[] = []
  
  for (const msg of messageQueue) {
    if (msg.type === 'RUN_BATCH_INFERENCE') {
      itemsToPrefetch.push(...msg.items)
    } else if (msg.type === 'RUN_INFERENCE') {
      itemsToPrefetch.push({ canvasIndex: msg.canvasIndex, imageData: msg.imageData, imageUrl: msg.imageUrl })
    }
  }

  // 設定された制限数まで先行フェッチを開始
  let startedCount = 0
  for (const item of itemsToPrefetch) {
    if (prefetchMap.size >= MAX_PREFETCH_COUNT) break
    if (!prefetchMap.has(item.canvasIndex)) {
      prefetchMap.set(item.canvasIndex, prefetchItem(item))
      startedCount++
    }
  }
}

/**
 * 個別の画像をフェッチ・前処理して Promise を返す
 */
async function prefetchItem(item: { canvasIndex: number; imageData?: ArrayBuffer; imageUrl?: string }): Promise<Float32Array> {
  const config = currentModelId ? MODEL_CONFIGS[currentModelId] : MODEL_CONFIGS['mobilenet_v2']
  const inputSize = config.inputSize

  let data = item.imageData
  if (!data && item.imageUrl) {
    // 解析用のフェッチは優先度を高く設定
    const resp = await fetch(item.imageUrl, { priority: 'high' } as any)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    data = await resp.arrayBuffer()
  }
  
  if (!data) throw new Error(`No image data for canvas ${item.canvasIndex}`)
  return preprocessImage(data, inputSize)
}

async function processQueue() {
  isProcessing = true
  while (messageQueue.length > 0) {
    const msg = messageQueue.shift()
    if (!msg) break
    await handleMessage(msg)
  }
  isProcessing = false
}

async function handleMessage(msg: WorkerInMessage) {
  if (msg.type === 'LOAD_MODEL') {
    try {
      const config = MODEL_CONFIGS[msg.modelId]
      if (currentModelId === msg.modelId && session) {
        // すでに同じモデルがロード済み
        const out: WorkerOutMessage = { type: 'MODEL_LOADED', modelId: msg.modelId }
        self.postMessage(out)
        return
      }
      session?.release?.()
      session = null

      const modelUrl = `${import.meta.env.BASE_URL}models/${config.filename}`
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu', 'wasm'],
        graphOptimizationLevel: 'all',
      })

      // ORT が作成した Device を取得して前処理でも使用する
      // @ts-ignore
      const internalDevice = (ort.env.webgpu as any).device
      if (internalDevice) {
        gpuDevice = internalDevice
        await initWebGPUPipeline(gpuDevice!)
      }

      currentModelId = msg.modelId
      const out: WorkerOutMessage = { type: 'MODEL_LOADED', modelId: msg.modelId }
      self.postMessage(out)
    } catch (err) {
      const out: WorkerOutMessage = { type: 'MODEL_ERROR', error: err instanceof Error ? err.message : String(err) }
      self.postMessage(out)
    }
  }

  if (msg.type === 'RUN_INFERENCE') {
    await handleInferenceBatch([{ 
      canvasIndex: msg.canvasIndex, 
      imageData: msg.imageData, 
      imageUrl: msg.imageUrl 
    }])
  }

  if (msg.type === 'RUN_BATCH_INFERENCE') {
    await handleInferenceBatch(msg.items)
  }
}

/**
 * 複数枚の画像を一度に前処理し、バッチ推論として実行する
 */
async function handleInferenceBatch(items: { canvasIndex: number; imageData?: ArrayBuffer; imageUrl?: string }[]) {
  if (!session || !currentModelId) {
    items.forEach(item => {
      const out: WorkerOutMessage = {
        type: 'INFERENCE_ERROR',
        canvasIndex: item.canvasIndex,
        error: 'Model not loaded',
      }
      self.postMessage(out)
    })
    return
  }

  if (cancelled) return

  try {
    const config = MODEL_CONFIGS[currentModelId]
    const batchSize = items.length
    const inputSize = config.inputSize
    const pixels = inputSize * inputSize
    let outputData: Float32Array

    // --- GPU パス ---
    if (gpuDevice && gpuPipeline && gpuSampler) {
      const batchBuffer = gpuDevice.createBuffer({
        size: batchSize * 3 * pixels * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: false,
      })

      // 各画像の前処理を GPU 上で並列/順次実行
      await Promise.all(items.map(async (item, i) => {
        // ... (画像データの取得)
        let imageData = item.imageData
        if (!imageData && item.imageUrl) {
          const resp = await fetch(item.imageUrl, { priority: 'high' } as any)
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
          imageData = await resp.arrayBuffer()
        }
        if (!imageData) throw new Error(`No data for ${item.canvasIndex}`)
        
        await preprocessImageGPU(
          gpuDevice!, 
          gpuPipeline!, 
          gpuSampler!, 
          imageData, 
          batchBuffer, 
          i * 3 * pixels * 4, 
          inputSize
        )
      }))

      // IO Binding (Zero-copy) の試行
      let tensor: ort.Tensor
      try {
        // @ts-ignore
        tensor = await (ort.Tensor as any).fromGpuBuffer(batchBuffer, {
          dataType: 'float32',
          dims: [batchSize, 3, inputSize, inputSize],
        })
      } catch (e) {
        console.warn('IO Binding failed, falling back to CPU copy:', e)
        const readBuffer = gpuDevice.createBuffer({
          size: batchSize * 3 * pixels * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
        const copyEncoder = gpuDevice.createCommandEncoder()
        copyEncoder.copyBufferToBuffer(batchBuffer, 0, readBuffer, 0, batchSize * 3 * pixels * 4)
        gpuDevice.queue.submit([copyEncoder.finish()])

        await readBuffer.mapAsync(GPUMapMode.READ)
        const combinedData = new Float32Array(readBuffer.getMappedRange().slice(0))
        readBuffer.unmap()
        readBuffer.destroy()
        
        tensor = new ort.Tensor('float32', combinedData, [batchSize, 3, inputSize, inputSize])
      }

      // 3. 推論実行
      const inputName = session.inputNames[0]
      const feeds = { [inputName]: tensor }
      const results = await session.run(feeds)
      
      outputData = results[session.outputNames[0]].data as Float32Array

      // バッチバッファの解放
      batchBuffer.destroy()
    } else {
      // --- 従来の CPU パス ---
      const tensorDataList = await Promise.all(items.map(async (item) => {
        let promise = prefetchMap.get(item.canvasIndex)
        if (!promise) {
          promise = prefetchItem(item)
          prefetchMap.set(item.canvasIndex, promise)
        }
        return promise
      }))
      
      items.forEach(item => prefetchMap.delete(item.canvasIndex))
      triggerPrefetch()

      const combinedData = new Float32Array(batchSize * 3 * pixels)
      for (let i = 0; i < batchSize; i++) {
        combinedData.set(tensorDataList[i], i * 3 * pixels)
      }

      const tensor = new ort.Tensor('float32', combinedData, [batchSize, 3, inputSize, inputSize])
      const inputName = session.inputNames[0]
      const feeds = { [inputName]: tensor }
      const results = await session.run(feeds)
      outputData = results[session.outputNames[0]].data as Float32Array
    }

    // 4. 結果のパースと返却
    // outputData の形状は [batchSize, numClasses] (MobileNetなら 通常 [N, 1000] または転移学習後なら [N, 2])
    const numClasses = outputData.length / batchSize

    const batchResults: InferenceResult[] = []

    for (let i = 0; i < batchSize; i++) {
      const start = i * numClasses
      const logits = outputData.slice(start, start + numClasses)
      const probs = softmax(logits)
      const confidence = probs[1] // 表紙クラスの確率

      batchResults.push({
        canvasIndex: items[i].canvasIndex,
        confidence
      })

      // 互換性のため、個別の結果も送信しておく（Progress更新などで使いやすいため）
      self.postMessage({
        type: 'INFERENCE_RESULT',
        canvasIndex: items[i].canvasIndex,
        confidence
      } as WorkerOutMessage)
    }

    // まとめた結果も送信
    self.postMessage({
      type: 'BATCH_INFERENCE_RESULT',
      results: batchResults
    } as WorkerOutMessage)

  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err)
    items.forEach(item => {
      self.postMessage({
        type: 'INFERENCE_ERROR',
        canvasIndex: item.canvasIndex,
        error: errorMsg,
      } as WorkerOutMessage)
    })
  }
}
