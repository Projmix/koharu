'use client'

import type {
  AgentStatus,
  AnalysisRegion,
  CanvasState,
  Download,
  FontFamily,
  Frame,
  GeometryUpdate,
  InpaintingModelChoice,
  Job,
  Layer,
  Model,
  ModelResources,
  Page,
  PageSummary,
  Point,
  Preferences,
  ProjectInfo,
  ProjectSummary,
  ProjectVersion,
  StartupState,
  TransformFrame,
  TypographyUpdate,
} from '@koharu/bridge/protocol'

interface ChannelLike<T> {
  onmessage?: (payload: T) => void
}

function rectPoints(x: number, y: number, width: number, height: number): Point[] {
  return [
    { x, y },
    { x: x + width, y },
    { x: x + width, y: y + height },
    { x, y: y + height },
  ]
}

function boundsFromPoints(points: Point[]): Frame {
  if (points.length === 0) {
    return { x: 0, y: 0, width: 100, height: 60, angle_degrees: 0 }
  }
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const minX = Math.min(...xs)
  const minY = Math.min(...ys)
  const maxX = Math.max(...xs)
  const maxY = Math.max(...ys)
  return {
    x: minX,
    y: minY,
    width: Math.max(1, maxX - minX),
    height: Math.max(1, maxY - minY),
    angle_degrees: 0,
  }
}

function createSamplePages(): Page[] {
  return [
    {
      id: 'page-1',
      label: '001.png',
      size: { width: 900, height: 1280 },
      regions: [
        {
          id: 'region-1',
          parent: null,
          geometry: { points: rectPoints(565, 110, 175, 170) },
          kind: 'text_bubble',
          label: 'speech_bubble',
        },
        {
          id: 'region-2',
          parent: null,
          geometry: { points: rectPoints(94, 780, 160, 130) },
          kind: 'text_bubble',
          label: 'speech_bubble',
        },
        {
          id: 'region-3',
          parent: null,
          geometry: { points: rectPoints(608, 795, 154, 138) },
          kind: 'text_bubble',
          label: 'speech_bubble',
        },
      ],
      layers: [
        {
          type: 'artwork',
          id: 'layer-artwork-1',
          parent: null,
          geometry: { points: rectPoints(0, 0, 900, 1280) },
          visibility: { visible: true, opacity: 1 },
          image: 'artwork-1',
        },
        {
          type: 'raster',
          id: 'layer-cleanup-1',
          parent: null,
          visibility: { visible: true, opacity: 1 },
          image: 'cleanup-1',
          name: 'Inpainted Cleanup',
          kind: 'cleanup',
        },
        {
          type: 'text',
          id: 'layer-text-1',
          parent: null,
          geometry: { points: rectPoints(565, 110, 175, 170) },
          angle_degrees: null,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'text-content-1',
            source: { text: '先生、準備はいいですか？', language: 'ja-JP' },
            translation: { text: 'Sensei, are you ready to begin?', language: 'en-US' },
            role: 'dialogue',
            source_region: 'region-1',
          },
          typography: {
            preferred_font: 'CCWildWords',
            font_weight: 600,
            font_style: 'normal',
            size: 20,
            auto_fit: true,
            color: [24, 24, 27, 255],
            stroke_color: [255, 255, 255, 255],
            stroke_width: 0,
            alignment: 'Center',
            writing_mode: 'Horizontal',
          },
          layout: 'paragraph',
          automatic_region: 'region-1',
        },
        {
          type: 'text',
          id: 'layer-text-2',
          parent: null,
          geometry: { points: rectPoints(94, 780, 160, 130) },
          angle_degrees: null,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'text-content-2',
            source: { text: '翻訳パイプラインを起動します！', language: 'ja-JP' },
            translation: { text: 'Starting the translation pipeline!', language: 'en-US' },
            role: 'dialogue',
            source_region: 'region-2',
          },
          typography: {
            preferred_font: 'CCWildWords',
            font_weight: 600,
            font_style: 'normal',
            size: 19,
            auto_fit: true,
            color: [24, 24, 27, 255],
            stroke_color: [255, 255, 255, 255],
            stroke_width: 0,
            alignment: 'Center',
            writing_mode: 'Horizontal',
          },
          layout: 'paragraph',
          automatic_region: 'region-2',
        },
        {
          type: 'text',
          id: 'layer-text-3',
          parent: null,
          geometry: { points: rectPoints(608, 795, 154, 138) },
          angle_degrees: null,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'text-content-3',
            source: { text: 'よし、行こう。', language: 'ja-JP' },
            translation: { text: "Alright, let's go.", language: 'en-US' },
            role: 'dialogue',
            source_region: 'region-3',
          },
          typography: {
            preferred_font: 'CCWildWords',
            font_weight: 600,
            font_style: 'normal',
            size: 20,
            auto_fit: true,
            color: [24, 24, 27, 255],
            stroke_color: [255, 255, 255, 255],
            stroke_width: 0,
            alignment: 'Center',
            writing_mode: 'Horizontal',
          },
          layout: 'paragraph',
          automatic_region: 'region-3',
        },
      ],
    },
    {
      id: 'page-2',
      label: '002.png',
      size: { width: 900, height: 1280 },
      regions: [
        {
          id: 'region-2-1',
          parent: null,
          geometry: { points: rectPoints(565, 110, 175, 170) },
          kind: 'text_bubble',
          label: 'speech_bubble',
        },
      ],
      layers: [
        {
          type: 'artwork',
          id: 'layer-artwork-2',
          parent: null,
          geometry: { points: rectPoints(0, 0, 900, 1280) },
          visibility: { visible: true, opacity: 1 },
          image: 'artwork-2',
        },
        {
          type: 'text',
          id: 'layer-text-2-1',
          parent: null,
          geometry: { points: rectPoints(565, 110, 175, 170) },
          angle_degrees: null,
          visibility: { visible: true, opacity: 1 },
          content: {
            id: 'text-content-2-1',
            source: { text: '次のページのレイアウトも完璧です。', language: 'ja-JP' },
            translation: { text: 'The layout on the next page is ready too.', language: 'en-US' },
            role: 'dialogue',
            source_region: 'region-2-1',
          },
          typography: {
            preferred_font: 'CCWildWords',
            font_weight: 600,
            font_style: 'normal',
            size: 20,
            auto_fit: true,
            color: [24, 24, 27, 255],
            stroke_color: [255, 255, 255, 255],
            stroke_width: 0,
            alignment: 'Center',
            writing_mode: 'Horizontal',
          },
          layout: 'paragraph',
          automatic_region: 'region-2-1',
        },
      ],
    },
  ]
}

const defaultPreferences: Preferences = {
  pipeline: {
    detection: { model: 'koharu-layout-rfdetr-seg-2xl' },
    ocr: {
      method: 'local',
      local_model: { model: 'paddleocr-vl-1.6' },
      api: {
        model: {
          provider: 'openrouter',
          model: 'qwen/qwen3.8-27b',
          quantization: null,
          vision: true,
          reasoning: true,
          reasoning_required: false,
        },
        generation: {
          temperature: 0,
          max_tokens: 1024,
          vision: true,
          reasoning: false,
        },
        instructions: null,
      },
    },
    translation: {
      source_language: 'ja-JP',
      target_language: 'en-US',
      page: {
        model: {
          provider: 'local',
          model: 'lfm2.5-1.2b-instruct',
          quantization: null,
          vision: false,
          reasoning: false,
          reasoning_required: false,
        },
        generation: {
          temperature: 0.3,
          max_tokens: 1024,
          vision: false,
          reasoning: false,
        },
        instructions: null,
      },
      chapter: {
        model: {
          provider: 'openrouter',
          model: 'google/gemini-2.5-flash',
          quantization: null,
          vision: true,
          reasoning: true,
          reasoning_required: false,
        },
        generation: {
          temperature: 0.2,
          max_tokens: 4096,
          vision: true,
          reasoning: true,
        },
        instructions: null,
      },
    },
    inpainting: {
      method: 'local',
      local_model: { model: 'lama' },
      manual_model: { model: 'lama' },
      api: {
        provider: 'fal',
        model: 'microsoft/mai-image-2.5/edit',
        prompt: 'Remove all text and reconstruct the original manga artwork.',
        apply_mode: 'full-page',
      },
    },
    processor: {
      'koharu-layout-rfdetr-seg-2xl': {
        text_threshold: 0.35,
        bubble_threshold: 0.4,
        panel_threshold: 0.5,
      },
    },
  },
  providers: {
    fal: { configured: false, value: null, clear: false },
    entries: [
      {
        name: 'Local GGUF',
        config: { provider: 'local', settings: {} },
        credential: null,
      },
      {
        name: 'Gemini',
        config: { provider: 'gemini', settings: {} },
        credential: { configured: false, value: null, clear: false },
      },
      {
        name: 'OpenRouter',
        config: { provider: 'openrouter', settings: {} },
        credential: { configured: false, value: null, clear: false },
      },
      {
        name: 'OpenAI',
        config: { provider: 'openai', settings: {} },
        credential: { configured: false, value: null, clear: false },
      },
      {
        name: 'Claude',
        config: { provider: 'claude', settings: {} },
        credential: { configured: false, value: null, clear: false },
      },
      {
        name: 'DeepL',
        config: { provider: 'deepl', settings: { base_url: null } },
        credential: { configured: false, value: null, clear: false },
      },
    ],
  },
  typesetting: {
    font_families: ['CCWildWords', 'Noto Sans JP', 'Inter'],
  },
  languages: [
    { tag: 'ja-JP', name: 'Japanese' },
    { tag: 'en-US', name: 'English (US)' },
    { tag: 'zh-CN', name: 'Chinese (Simplified)' },
    { tag: 'zh-TW', name: 'Chinese (Traditional)' },
    { tag: 'ko-KR', name: 'Korean' },
    { tag: 'es-ES', name: 'Spanish' },
  ],
}

const sampleTranslationModels: Model[] = [
  {
    provider: 'local',
    model: 'lfm2.5-1.2b-instruct',
    name: 'LFM 2.5 1.2B Instruct (Local)',
    quantizations: [{ id: 'Q4_K_M', name: 'Q4_K_M', downloaded: true }],
    vision: false,
    reasoning: false,
    reasoning_required: false,
  },
  {
    provider: 'gemini',
    model: 'gemini-2.5-flash',
    name: 'Gemini 2.5 Flash',
    quantizations: [],
    vision: true,
    reasoning: true,
    reasoning_required: false,
  },
  {
    provider: 'openrouter',
    model: 'google/gemini-2.5-flash',
    name: 'Gemini 2.5 Flash (OpenRouter)',
    quantizations: [],
    vision: true,
    reasoning: true,
    reasoning_required: false,
  },
  {
    provider: 'openrouter',
    model: 'qwen/qwen3.8-27b',
    name: 'Qwen 3.8 27B',
    quantizations: [],
    vision: true,
    reasoning: true,
    reasoning_required: false,
  },
]

const sampleInpaintingModels: InpaintingModelChoice[] = [
  {
    provider: 'fal',
    model: 'microsoft/mai-image-2.5/edit',
    name: 'MAI Image 2.5 Edit',
  },
]

const sampleFonts: FontFamily[] = [
  {
    name: 'CCWildWords',
    metadata: {
      primary_script: 'Latn',
      scripts: ['Latn'],
      languages: ['en'],
      category: 'comic',
      classifications: ['sans-serif'],
      use_cases: ['dialogue'],
    },
    sources: ['bundled'],
    faces: [
      {
        postscript_name: 'CCWildWords-Regular',
        weight: 400,
        weight_range: null,
        style: 'normal',
      },
      {
        postscript_name: 'CCWildWords-Bold',
        weight: 700,
        weight_range: null,
        style: 'normal',
      },
    ],
  },
  {
    name: 'Noto Sans JP',
    metadata: {
      primary_script: 'Jpan',
      scripts: ['Jpan', 'Latn'],
      languages: ['ja', 'en'],
      category: 'sans-serif',
      classifications: ['sans-serif'],
      use_cases: ['dialogue', 'sfx'],
    },
    sources: ['system'],
    faces: [
      {
        postscript_name: 'NotoSansJP-Regular',
        weight: 400,
        weight_range: null,
        style: 'normal',
      },
    ],
  },
]

// 1x1 transparent PNG bytes for thumbnail/font preview requests
const TINY_PNG_BYTES: number[] = [
  137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0,
  0, 31, 21, 196, 137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 248, 255, 255, 63, 0, 5, 254, 2,
  254, 167, 53, 129, 132, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
]

interface StoredProject {
  info: ProjectInfo
  pages: Page[]
  versions: ProjectVersion[]
}

export function installWebTauriShim(): void {
  if (typeof window === 'undefined') return
  const win = window as unknown as Record<string, unknown>
  if ('__TAURI_INTERNALS__' in win) return
  if (typeof process !== 'undefined' && process.env?.VITEST) return

  let preferences: Preferences = structuredClone(defaultPreferences)
  const projects = new Map<string, StoredProject>()

  const initialPages = createSamplePages()
  projects.set('Blue Archive — Chapter 1', {
    info: {
      name: 'Blue Archive — Chapter 1',
      revision: 1,
      active_page: initialPages[0].id,
      can_undo: false,
      can_redo: false,
    },
    pages: initialPages,
    versions: [
      {
        id: 'v1',
        name: 'Initial OCR & Translation',
        created_at_ms: Date.now() - 3600_000,
        revision: 1,
      },
    ],
  })

  let activeProjectName: string | null = 'Blue Archive — Chapter 1'
  let canvasGeneration = 1

  let onCanvasChannel: ChannelLike<CanvasState> | null = null
  let onJobChannel: ChannelLike<Job> | null = null
  let _onDownloadChannel: ChannelLike<Download> | null = null
  let onResourcesChannel: ChannelLike<ModelResources> | null = null
  let onProjectChannel: ChannelLike<ProjectInfo | null> | null = null

  let agentStatus: AgentStatus = {
    account: null,
    models: [
      {
        id: 'gpt-5-codex',
        name: 'GPT-5 Codex',
        reasoning: ['low', 'medium', 'high'],
      },
    ],
    config: {
      model: null,
      reasoning: 'medium',
    },
    running: null,
  }

  function currentStoredProject(): StoredProject | null {
    if (!activeProjectName) return null
    return projects.get(activeProjectName) ?? null
  }

  function currentActivePage(): Page | null {
    const proj = currentStoredProject()
    if (!proj || !proj.info.active_page) return null
    return proj.pages.find((p) => p.id === proj.info.active_page) ?? proj.pages[0] ?? null
  }

  function buildCanvasState(): CanvasState {
    const proj = currentStoredProject()
    const page = currentActivePage()
    if (!proj || !page) {
      return {
        page: null,
        revision: null,
        generation: canvasGeneration,
        size: [0, 0],
        element_frames: [],
      }
    }
    const elementFrames: TransformFrame[] = []
    for (const layer of page.layers) {
      if ('geometry' in layer && layer.geometry?.points?.length) {
        elementFrames.push({
          element: layer.id,
          frame: boundsFromPoints(layer.geometry.points),
        })
      }
    }
    return {
      page: page.id,
      revision: proj.info.revision,
      generation: canvasGeneration,
      size: [page.size.width, page.size.height],
      element_frames: elementFrames,
    }
  }

  function emitCanvasAndProject() {
    canvasGeneration += 1
    const proj = currentStoredProject()
    onProjectChannel?.onmessage?.(proj ? structuredClone(proj.info) : null)
    onCanvasChannel?.onmessage?.(buildCanvasState())
  }

  function encodePageManifest(page: Page | null, revision: number): number[] {
    const proj = currentStoredProject()
    const pageIndex = proj && page ? Math.max(0, proj.pages.findIndex((p) => p.id === page.id)) : 0
    const payload = JSON.stringify({
      pageId: page?.id ?? 'none',
      pageIndex,
      revision,
      width: page?.size.width ?? 900,
      height: page?.size.height ?? 1280,
      layers: page?.layers ?? [],
    })
    return Array.from(new TextEncoder().encode(payload))
  }

  const callbacks = new Map<number, { fn: (data: unknown) => void; once: boolean }>()
  let nextCallbackId = 1

  win.__TAURI_INTERNALS__ = {
    metadata: {
      currentWindow: { label: 'main' },
      currentWebview: { windowLabel: 'main', label: 'main' },
    },
    transformCallback(callback?: (data: unknown) => void, once = false) {
      const id = nextCallbackId++
      if (callback) callbacks.set(id, { fn: callback, once })
      return id
    },
    unregisterCallback(id: number) {
      callbacks.delete(id)
    },
    convertFileSrc(filePath: string) {
      return filePath
    },
    async invoke(cmd: string, args: Record<string, unknown> = {}): Promise<unknown> {
      // Handle Tauri window/app/plugin commands cleanly
      if (cmd.startsWith('plugin:window|')) {
        if (cmd.endsWith('is_maximized')) return false
        return null
      }
      if (cmd.startsWith('plugin:event|')) {
        return 1
      }
      if (cmd.startsWith('plugin:app|')) {
        if (cmd.endsWith('version')) return '0.63.0-web'
        return 'Koharu'
      }
      if (cmd.startsWith('plugin:updater|')) {
        return null
      }
      if (cmd.startsWith('plugin:opener|')) {
        return null
      }

      switch (cmd) {
        case 'subscribe': {
          onCanvasChannel = (args.onCanvas as ChannelLike<CanvasState>) ?? null
          onJobChannel = (args.onJob as ChannelLike<Job>) ?? null
          _onDownloadChannel = (args.onDownload as ChannelLike<Download>) ?? null
          onResourcesChannel = (args.onResources as ChannelLike<ModelResources>) ?? null
          onProjectChannel = (args.onProject as ChannelLike<ProjectInfo | null>) ?? null

          queueMicrotask(() => {
            onResourcesChannel?.onmessage?.({
              process_memory: 420 * 1024 * 1024,
              system_memory: 16 * 1024 * 1024 * 1024,
              process_cpu: 2.4,
              devices: [
                {
                  name: 'WebGPU / Browser Canvas',
                  selected: true,
                  memory_budget: 8 * 1024 * 1024 * 1024,
                  memory_used: 512 * 1024 * 1024,
                  utilization: 12,
                },
              ],
            })
          })

          const state: StartupState = {
            preferences: structuredClone(preferences),
            jobs: [],
            canvas: buildCanvasState(),
          }
          return state
        }

        case 'get_preferences':
          return structuredClone(preferences)

        case 'save_preferences': {
          preferences = {
            ...preferences,
            pipeline: structuredClone(args.pipeline as Preferences['pipeline']),
            providers: structuredClone(args.providers as Preferences['providers']),
            typesetting: structuredClone(args.typesetting as Preferences['typesetting']),
          }
          return structuredClone(preferences)
        }

        case 'get_translation_models':
          return structuredClone(sampleTranslationModels)

        case 'get_inpainting_models':
          return structuredClone(sampleInpaintingModels)

        case 'get_fonts':
          return structuredClone(sampleFonts)

        case 'get_font_preview':
        case 'get_thumbnail':
          return TINY_PNG_BYTES

        case 'list_projects': {
          const list: ProjectSummary[] = Array.from(projects.keys()).map((name) => ({ name }))
          return list
        }

        case 'get_project': {
          const proj = currentStoredProject()
          return proj ? structuredClone(proj.info) : null
        }

        case 'create_project': {
          const name = String(args.name || 'Untitled Project')
          const pages = createSamplePages()
          projects.set(name, {
            info: {
              name,
              revision: 1,
              active_page: pages[0].id,
              can_undo: false,
              can_redo: false,
            },
            pages,
            versions: [],
          })
          activeProjectName = name
          emitCanvasAndProject()
          return null
        }

        case 'open_project': {
          const name = String(args.name)
          if (projects.has(name)) {
            activeProjectName = name
            emitCanvasAndProject()
          }
          return null
        }

        case 'delete_project': {
          const name = String(args.name)
          projects.delete(name)
          if (activeProjectName === name) {
            activeProjectName = null
            emitCanvasAndProject()
          }
          return null
        }

        case 'close_project': {
          activeProjectName = null
          emitCanvasAndProject()
          return null
        }

        case 'get_pages': {
          const proj = currentStoredProject()
          if (!proj) return []
          const summaries: PageSummary[] = proj.pages.map((p) => ({
            id: p.id,
            label: p.label,
            size: p.size,
            source_asset: p.id,
            layer_count: p.layers.length,
            warning: null,
          }))
          return summaries
        }

        case 'get_page': {
          const page = currentActivePage()
          return page ? structuredClone(page) : null
        }

        case 'select_page': {
          const proj = currentStoredProject()
          const pageId = String(args.page)
          if (!proj) throw new Error('No open project')
          const found = proj.pages.find((p) => p.id === pageId) ?? proj.pages[0]
          proj.info.active_page = found.id
          emitCanvasAndProject()
          return {
            project: structuredClone(proj.info),
            page: structuredClone(found),
          }
        }

        case 'import':
        case 'import_pages': {
          const proj = currentStoredProject()
          if (proj) {
            const nextIndex = proj.pages.length + 1
            const newPage: Page = {
              id: `page-${nextIndex}`,
              label: `${String(nextIndex).padStart(3, '0')}.png`,
              size: { width: 900, height: 1280 },
              regions: [],
              layers: [
                {
                  type: 'artwork',
                  id: `layer-artwork-${nextIndex}`,
                  parent: null,
                  geometry: { points: rectPoints(0, 0, 900, 1280) },
                  visibility: { visible: true, opacity: 1 },
                  image: `artwork-${nextIndex}`,
                },
              ],
            }
            proj.pages.push(newPage)
            proj.info.active_page = newPage.id
            proj.info.revision += 1
            emitCanvasAndProject()
          }
          return null
        }

        case 'get_canvas_manifest': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          return encodePageManifest(page, proj?.info.revision ?? 1)
        }

        case 'get_canvas_resource':
        case 'get_canvas_page_resource':
          return []

        case 'prepare_canvas_page': {
          const proj = currentStoredProject()
          const pageId = String(args.page)
          const page = proj?.pages.find((p) => p.id === pageId)
          if (!proj || !page) return null
          return {
            revision: proj.info.revision,
            page: structuredClone(page),
          }
        }

        case 'get_canvas_page_manifest': {
          const proj = currentStoredProject()
          const pageId = String(args.page)
          const page = proj?.pages.find((p) => p.id === pageId) ?? null
          return encodePageManifest(page, Number(args.revision ?? proj?.info.revision ?? 1))
        }

        case 'set_source_text': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          if (proj && page) {
            const layer = page.layers.find((l) => l.id === args.layer && l.type === 'text')
            if (layer && layer.type === 'text') {
              layer.content.source = {
                text: String(args.text ?? ''),
                language: preferences.pipeline.translation.source_language,
              }
              proj.info.revision += 1
              emitCanvasAndProject()
            }
          }
          return null
        }

        case 'set_translation': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          if (proj && page) {
            const layer = page.layers.find((l) => l.id === args.layer && l.type === 'text')
            if (layer && layer.type === 'text') {
              layer.content.translation =
                args.text === null
                  ? null
                  : {
                      text: String(args.text ?? ''),
                      language: preferences.pipeline.translation.target_language,
                    }
              proj.info.revision += 1
              emitCanvasAndProject()
            }
          }
          return null
        }

        case 'set_text_role': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          if (proj && page) {
            const layer = page.layers.find((l) => l.id === args.layer && l.type === 'text')
            if (layer && layer.type === 'text') {
              layer.content.role = String(args.role ?? 'dialogue')
              proj.info.revision += 1
              emitCanvasAndProject()
            }
          }
          return null
        }

        case 'set_typography': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          const updates = (args.updates as TypographyUpdate[]) ?? []
          if (proj && page) {
            for (const update of updates) {
              const layer = page.layers.find((l) => l.id === update.layer && l.type === 'text')
              if (layer && layer.type === 'text') {
                layer.typography = structuredClone(update.typography)
              }
            }
            proj.info.revision += 1
            emitCanvasAndProject()
          }
          return null
        }

        case 'set_geometry': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          const updates = (args.updates as GeometryUpdate[]) ?? []
          if (proj && page) {
            for (const update of updates) {
              const layer = page.layers.find((l) => l.id === update.layer)
              if (layer && 'geometry' in layer && update.points) {
                layer.geometry = { points: structuredClone(update.points) }
              }
              const region = page.regions.find((r) => r.id === update.layer)
              if (region && update.points) {
                region.geometry = { points: structuredClone(update.points) }
              }
            }
            proj.info.revision += 1
            emitCanvasAndProject()
          }
          return null
        }

        case 'set_visibility': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          const targetIds = (args.layers as string[]) ?? []
          if (proj && page) {
            for (const layer of page.layers) {
              if (targetIds.includes(layer.id)) {
                if (typeof args.visible === 'boolean') layer.visibility.visible = args.visible
                if (typeof args.opacity === 'number') layer.visibility.opacity = args.opacity
              }
            }
            proj.info.revision += 1
            emitCanvasAndProject()
          }
          return null
        }

        case 'add_point_text': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          const pt = (args.point as Point) ?? { x: 200, y: 200 }
          const id = `layer-text-${Date.now()}`
          if (proj && page) {
            const newLayer: Layer = {
              type: 'text',
              id,
              parent: null,
              geometry: { points: rectPoints(pt.x, pt.y, 160, 60) },
              angle_degrees: null,
              visibility: { visible: true, opacity: 1 },
              content: {
                id: `content-${id}`,
                source: { text: '', language: 'ja-JP' },
                translation: { text: 'New text', language: 'en-US' },
                role: 'dialogue',
                source_region: null,
              },
              typography: {
                preferred_font: 'CCWildWords',
                font_weight: 600,
                font_style: 'normal',
                size: 20,
                auto_fit: true,
                color: [24, 24, 27, 255],
                stroke_color: null,
                stroke_width: 0,
                alignment: 'Center',
                writing_mode: 'Horizontal',
              },
              layout: 'point',
              automatic_region: null,
            }
            page.layers.push(newLayer)
            proj.info.revision += 1
            emitCanvasAndProject()
          }
          return { revision: proj?.info.revision ?? 1, layer: id }
        }

        case 'add_text_box': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          const frame = (args.frame as Frame) ?? {
            x: 200,
            y: 200,
            width: 160,
            height: 80,
            angle_degrees: 0,
          }
          const id = `layer-text-${Date.now()}`
          if (proj && page) {
            const newLayer: Layer = {
              type: 'text',
              id,
              parent: null,
              geometry: { points: rectPoints(frame.x, frame.y, frame.width, frame.height) },
              angle_degrees: frame.angle_degrees || null,
              visibility: { visible: true, opacity: 1 },
              content: {
                id: `content-${id}`,
                source: { text: '', language: 'ja-JP' },
                translation: { text: 'New text box', language: 'en-US' },
                role: 'dialogue',
                source_region: null,
              },
              typography: {
                preferred_font: 'CCWildWords',
                font_weight: 600,
                font_style: 'normal',
                size: 20,
                auto_fit: true,
                color: [24, 24, 27, 255],
                stroke_color: null,
                stroke_width: 0,
                alignment: 'Center',
                writing_mode: 'Horizontal',
              },
              layout: 'paragraph',
              automatic_region: null,
            }
            page.layers.push(newLayer)
            proj.info.revision += 1
            emitCanvasAndProject()
          }
          return { revision: proj?.info.revision ?? 1, layer: id }
        }

        case 'add_ocr_region': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          const frame = (args.frame as Frame) ?? {
            x: 200,
            y: 200,
            width: 160,
            height: 80,
            angle_degrees: 0,
          }
          const regionId = `region-${Date.now()}`
          const layerId = `layer-text-${Date.now()}`
          if (proj && page) {
            const region: AnalysisRegion = {
              id: regionId,
              parent: null,
              geometry: { points: rectPoints(frame.x, frame.y, frame.width, frame.height) },
              kind: 'text_bubble',
              label: 'speech_bubble',
            }
            page.regions.push(region)
            proj.info.revision += 1
            emitCanvasAndProject()
          }
          return { revision: proj?.info.revision ?? 1, layer: layerId, region: regionId }
        }

        case 'delete_layers': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          const ids = (args.layers as string[]) ?? []
          if (proj && page) {
            page.layers = page.layers.filter((l) => !ids.includes(l.id))
            proj.info.revision += 1
            emitCanvasAndProject()
          }
          return null
        }

        case 'delete_pages': {
          const proj = currentStoredProject()
          const ids = (args.pages as string[]) ?? []
          if (proj) {
            proj.pages = proj.pages.filter((p) => !ids.includes(p.id))
            proj.info.active_page = proj.pages[0]?.id ?? null
            proj.info.revision += 1
            emitCanvasAndProject()
          }
          return null
        }

        case 'rename_page': {
          const proj = currentStoredProject()
          if (proj) {
            const page = proj.pages.find((p) => p.id === args.page)
            if (page) {
              page.label = String(args.label ?? page.label)
              proj.info.revision += 1
              emitCanvasAndProject()
            }
          }
          return null
        }

        case 'commit_transform': {
          const proj = currentStoredProject()
          const page = currentActivePage()
          const elements = (args.elements as TransformFrame[]) ?? []
          if (proj && page) {
            for (const item of elements) {
              const layer = page.layers.find((l) => l.id === item.element)
              if (layer && 'geometry' in layer) {
                layer.geometry = {
                  points: rectPoints(
                    item.frame.x,
                    item.frame.y,
                    item.frame.width,
                    item.frame.height,
                  ),
                }
              }
            }
            proj.info.revision += 1
            emitCanvasAndProject()
            return proj.info.revision
          }
          return null
        }

        case 'commit_paint':
        case 'commit_erase': {
          const proj = currentStoredProject()
          if (proj) {
            proj.info.revision += 1
            emitCanvasAndProject()
          }
          return {
            revision: proj?.info.revision ?? 1,
            layer: String(args.layer ?? 'layer-cleanup-1'),
          }
        }

        case 'commit_inpaint': {
          const jobId = `job-${Date.now()}`
          return jobId
        }

        case 'process': {
          const jobId = `job-${Date.now()}`
          const page = currentActivePage()
          queueMicrotask(() => {
            onJobChannel?.onmessage?.({
              id: jobId,
              state: 'finished',
              completed: 1,
              total: 1,
              target: page ? { target: 'page', value: page.id } : { target: 'chapter' },
              stage: 'translation',
              model: 'lfm2.5-1.2b-instruct',
              error: null,
            })
          })
          return jobId
        }

        case 'list_project_versions': {
          const proj = currentStoredProject()
          return proj ? structuredClone(proj.versions) : []
        }

        case 'save_project_version': {
          const proj = currentStoredProject()
          const version: ProjectVersion = {
            id: `v-${Date.now()}`,
            name: String(args.name || 'Snapshot'),
            created_at_ms: Date.now(),
            revision: proj?.info.revision ?? 1,
          }
          proj?.versions.unshift(version)
          return version
        }

        case 'restore_project_version': {
          const proj = currentStoredProject()
          if (!proj) throw new Error('No open project')
          return structuredClone(proj.info)
        }

        case 'delete_project_version': {
          const proj = currentStoredProject()
          if (proj) {
            proj.versions = proj.versions.filter((v) => v.id !== args.id)
          }
          return null
        }

        case 'get_agent_status':
          return structuredClone(agentStatus)

        case 'save_agent_config': {
          agentStatus = {
            ...agentStatus,
            config: structuredClone(args.config as AgentStatus['config']),
          }
          return structuredClone(agentStatus.config)
        }

        case 'undo':
        case 'redo':
        case 'stop_job':
        case 'export':
        case 'export_pages':
        case 'export_chapter_translation':
        case 'import_chapter_translation':
          return null

        default:
          return null
      }
    },
  }
}

installWebTauriShim()
