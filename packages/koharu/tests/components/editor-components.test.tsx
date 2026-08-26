import { QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render as testingRender, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ThemeProvider } from 'next-themes'
import { StrictMode, type ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { TitleBar } from '@/components/app/TitleBar'
import { WindowControls } from '@/components/app/WindowChrome'
import { ModelPicker } from '@/components/controls/ModelPicker'
import { ActivityCenter } from '@/components/editor/ActivityCenter'
import { CanvasCommandBar } from '@/components/editor/CanvasCommandBar'
import { Inspector } from '@/components/editor/Inspector'
import { PageRail } from '@/components/editor/PageRail'
import { ResourceMonitor } from '@/components/editor/ResourceMonitor'
import { StatusBar } from '@/components/editor/StatusBar'
import { ToolBar } from '@/components/editor/ToolBar'
import { ProviderPreferences } from '@/components/preferences/ProviderPreferences'
import { SettingsPage } from '@/components/preferences/SettingsPage'
import { inpaintingPromptForPreset } from '@/lib/inpaintingPrompts'
import {
  fontsKey,
  pageKey,
  pagesKey,
  preparedPageKey,
  projectKey,
  queryClient,
} from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import * as canvasRuntime from '@koharu/bridge/canvas'
import {
  commands,
  type Layer,
  type Model,
  type Page,
  type PageSummary,
  type Preferences,
  type ProjectInfo,
} from '@koharu/bridge/protocol'
import { TooltipProvider } from '@koharu/ui/components/tooltip'

const nativeWindow = vi.hoisted(() => ({
  close: vi.fn(async () => undefined),
  isMaximized: vi.fn(async () => false),
  minimize: vi.fn(async () => undefined),
  onResized: vi.fn(async () => () => undefined),
  startResizeDragging: vi.fn(async () => undefined),
  toggleMaximize: vi.fn(async () => undefined),
}))
const nativeOpenUrl = vi.hoisted(() => vi.fn(async () => undefined))
const nativeGetVersion = vi.hoisted(() => vi.fn(async () => '0.62.0'))

vi.mock('@tauri-apps/api/window', () => ({ getCurrentWindow: () => nativeWindow }))
vi.mock('@tauri-apps/api/app', () => ({ getVersion: nativeGetVersion }))
vi.mock('@tauri-apps/plugin-opener', () => ({ openUrl: nativeOpenUrl }))

const emptyCredential = () => ({ configured: false, value: null, clear: false })
const translationProfile = () => ({
  model: {
    provider: 'local' as const,
    model: 'gemma4-e2b-it',
    quantization: null,
    vision: true,
    reasoning: true,
  },
  generation: { vision: true, reasoning: false },
  instructions: null,
})

const textLayer: Layer = {
  type: 'text',
  id: 'element',
  parent: 'page',
  geometry: {
    points: [
      { x: 10, y: 20 },
      { x: 110, y: 20 },
      { x: 110, y: 70 },
      { x: 10, y: 70 },
    ],
  },
  visibility: { visible: true, opacity: 1 },
  content: {
    id: 'content',
    source: { text: 'こんにちは', language: 'ja' },
    translation: { text: 'Hello', language: null },
    role: null,
    source_region: null,
  },
  typography: {
    preferred_font: 'Noto Sans',
    font_weight: 400,
    font_style: 'normal',
    size: null,
    auto_fit: true,
    color: [0, 0, 0, 255],
    stroke_color: [255, 255, 255, 255],
    stroke_width: 0,
    alignment: 'Center',
    writing_mode: null,
  },
  layout: 'paragraph',
  automatic_region: null,
}

const preferences: Preferences = {
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
      page: translationProfile(),
      chapter: translationProfile(),
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
    processor: {},
  },
  providers: {
    fal: emptyCredential(),
    entries: [
      {
        name: 'Local',
        config: { provider: 'local', settings: {} },
        credential: null,
      },
      {
        name: 'OpenAI-compatible',
        config: {
          provider: 'openai-compatible',
          settings: { base_url: 'http://localhost:11434/v1' },
        },
        credential: emptyCredential(),
      },
      {
        name: 'LM Studio',
        config: { provider: 'lm-studio', settings: { base_url: 'http://localhost:1234' } },
        credential: emptyCredential(),
      },
      {
        name: 'DeepL',
        config: { provider: 'deepl', settings: { base_url: null } },
        credential: emptyCredential(),
      },
    ],
  },
  typesetting: {
    font_families: ['Noto Sans'],
  },
  languages: [
    { tag: 'en-US', name: 'English' },
    { tag: 'ja-JP', name: 'Japanese' },
  ],
}

function installProject() {
  const page = {
    id: 'page',
    label: 'Page 1',
    size: { width: 1000, height: 1500 },
    layers: [textLayer],
    regions: [],
  }
  queryClient.setQueryData(projectKey, {
    name: 'Book',
    revision: 1,
    active_page: 'page',
    can_undo: true,
    can_redo: false,
  })
  queryClient.setQueryData(pagesKey, [
    {
      id: 'page',
      label: 'Page 1',
      size: { width: 1000, height: 1500 },
      source_asset: 'source',
      layer_count: 1,
    },
  ])
  queryClient.setQueryData(pageKey, page)
  useKoharuStore.setState({
    preferences,
    translationModels: [
      {
        provider: 'local',
        model: 'gemma4-e2b-it',
        name: 'Gemma 4 E2B Instruct',
        quantizations: [],
        vision: true,
        reasoning: true,
        reasoning_required: false,
      },
    ],
    inpaintingModels: [
      {
        provider: 'fal',
        model: 'microsoft/mai-image-2.5/edit',
        name: 'Microsoft MAI Image 2.5 Edit',
      },
      {
        provider: 'fal',
        model: 'microsoft/mai-image-2.5-pro/edit',
        name: 'Microsoft MAI Image 2.5 Pro Edit',
      },
      {
        provider: 'openrouter',
        model: 'microsoft/mai-image-2.5',
        name: 'Microsoft MAI Image 2.5',
      },
    ],
    selectedPages: ['page'],
    selectedLayers: ['element'],
    layerFrames: {
      element: { x: 10, y: 20, width: 100, height: 50, angle_degrees: 0 },
    },
  })
  vi.spyOn(commands, 'getPreferences').mockImplementation(
    async () => useKoharuStore.getState().preferences!,
  )
  vi.spyOn(commands, 'getTranslationModels').mockImplementation(async () => [
    ...useKoharuStore.getState().translationModels,
  ])
  vi.spyOn(commands, 'getInpaintingModels').mockImplementation(async () => [
    ...useKoharuStore.getState().inpaintingModels,
  ])
}

function render(ui: ReactNode) {
  return testingRender(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>)
}

function useCompactProcessingControls() {
  fireEvent.click(screen.getByRole('button', { name: 'Use compact processing controls' }))
}

function SettingsNavigationHarness() {
  const settingsOpen = useKoharuStore((state) => state.settingsOpen)
  return settingsOpen ? <SettingsPage /> : <CanvasCommandBar />
}

describe('greenfield editor', () => {
  it('registers one native resize listener in React strict mode', async () => {
    const unlisten = vi.fn()
    nativeWindow.onResized.mockClear()
    nativeWindow.onResized.mockResolvedValueOnce(unlisten)

    const view = render(
      <StrictMode>
        <WindowControls />
      </StrictMode>,
    )

    await waitFor(() => expect(nativeWindow.onResized).toHaveBeenCalledTimes(1))
    view.unmount()
    await waitFor(() => expect(unlisten).toHaveBeenCalledTimes(1))
  })

  it('starts native resize dragging from every frameless window edge', () => {
    nativeWindow.startResizeDragging.mockClear()
    const view = render(<WindowControls />)
    const directions = [
      'North',
      'South',
      'East',
      'West',
      'NorthEast',
      'NorthWest',
      'SouthEast',
      'SouthWest',
    ]

    for (const direction of directions) {
      fireEvent.pointerDown(
        view.container.querySelector(`[data-window-resize-handle="${direction}"]`)!,
        { button: 0 },
      )
    }

    expect(nativeWindow.startResizeDragging.mock.calls).toEqual(
      directions.map((direction) => [direction]),
    )
  })

  it('removes resize handles while the window is maximized', async () => {
    nativeWindow.isMaximized.mockResolvedValueOnce(true)
    const view = render(<WindowControls />)

    await waitFor(() =>
      expect(view.container.querySelector('[data-window-resize-handle]')).not.toBeInTheDocument(),
    )
  })

  it('shows import activity and prevents duplicate imports', async () => {
    const user = userEvent.setup()
    installProject()
    let finishImport: (() => void) | undefined
    const importPages = vi.spyOn(commands, 'importPages').mockImplementation(
      () =>
        new Promise<null>((resolve) => {
          finishImport = () => resolve(null)
        }),
    )
    render(
      <>
        <TitleBar />
        <PageRail />
      </>,
    )

    expect(screen.getByText('/')).toHaveClass('mx-2')
    expect(screen.queryByRole('button', { name: 'Import pages' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('menuitem', { name: 'File' }))
    await user.hover(await screen.findByRole('menuitem', { name: 'Import Pages…' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Files…' }))

    expect(await screen.findByRole('status')).toHaveTextContent('Importing pages…')
    expect(importPages).toHaveBeenCalledTimes(1)

    await user.click(screen.getByRole('menuitem', { name: 'File' }))
    expect(await screen.findByRole('menuitem', { name: 'Importing pages…' })).toHaveAttribute(
      'aria-disabled',
      'true',
    )

    finishImport?.()
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())
  })

  it('confirms before deleting every selected page', async () => {
    installProject()
    queryClient.setQueryData(pagesKey, [
      ...(queryClient.getQueryData<PageSummary[]>(pagesKey) ?? []),
      {
        id: 'page-2',
        label: 'Page 2',
        size: { width: 1000, height: 1500 },
        source_asset: null,
        layer_count: 0,
      },
    ])
    useKoharuStore.setState({ selectedPages: ['page', 'page-2'], selectedLayers: ['element'] })
    const remove = vi.spyOn(commands, 'deletePages').mockResolvedValue(null)
    render(<PageRail />)

    fireEvent.keyDown(window, { key: 'Delete' })
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()

    useKoharuStore.setState({ selectedLayers: [] })
    fireEvent.keyDown(window, { key: 'Delete' })

    expect(screen.getByRole('alertdialog')).toHaveTextContent('Delete all pages?')
    expect(remove).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Delete all pages' }))

    await waitFor(() => expect(remove).toHaveBeenCalledWith(['page', 'page-2']))
    expect(useKoharuStore.getState().selectedPages).toEqual([])
  })

  it('clears every page from the File menu after confirmation', async () => {
    const user = userEvent.setup()
    installProject()
    const remove = vi.spyOn(commands, 'deletePages').mockResolvedValue(null)
    render(<TitleBar />)

    await user.click(screen.getByRole('menuitem', { name: 'File' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Clear Project' }))

    expect(screen.getByRole('alertdialog')).toHaveTextContent('Delete all pages?')
    expect(remove).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Delete all pages' }))

    await waitFor(() => expect(remove).toHaveBeenCalledWith(['page']))
    expect(useKoharuStore.getState().selectedPages).toEqual([])
    expect(useKoharuStore.getState().selectedLayers).toEqual([])
  })

  it('opens community links through the Tauri opener plugin', async () => {
    nativeOpenUrl.mockClear()
    const user = userEvent.setup()
    render(<TitleBar />)

    await user.click(screen.getByRole('menuitem', { name: 'Help' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Discord' }))
    expect(nativeOpenUrl).toHaveBeenLastCalledWith('https://discord.gg/mHvHkxGnUY')

    await user.click(screen.getByRole('menuitem', { name: 'Help' }))
    fireEvent.click(await screen.findByRole('menuitem', { name: 'GitHub' }))
    expect(nativeOpenUrl).toHaveBeenLastCalledWith('https://github.com/mayocream/koharu')
  })

  it('shows the current version and author in About', async () => {
    const user = userEvent.setup()
    render(<TitleBar />)

    await user.click(screen.getByRole('menuitem', { name: 'Help' }))
    await user.click(await screen.findByRole('menuitem', { name: 'About' }))

    expect(await screen.findByRole('heading', { name: 'Koharu' })).toBeInTheDocument()
    expect(await screen.findByText('0.62.0')).toBeInTheDocument()
    expect(screen.getByText('Mayo Takanashi')).toBeInTheDocument()
    expect(nativeGetVersion).toHaveBeenCalledTimes(1)
  })

  it('loads page thumbnails into the filmstrip', async () => {
    installProject()
    const thumbnail = vi.spyOn(commands, 'getThumbnail').mockResolvedValue([1])
    render(<PageRail />)
    await waitFor(() => expect(thumbnail).toHaveBeenCalledWith('page'))
    expect(await screen.findByRole('img', { name: 'Page 1' })).toHaveAttribute(
      'src',
      'blob:koharu-thumbnail',
    )
    expect(screen.queryByText('01')).not.toBeInTheDocument()
  })

  it('shows an inpainting warning with its tooltip in the filmstrip', async () => {
    const user = userEvent.setup()
    installProject()
    queryClient.setQueryData(pagesKey, [
      {
        id: 'page',
        label: 'Page 1',
        size: { width: 1000, height: 1500 },
        source_asset: 'source',
        layer_count: 1,
        warning: {
          stage: 'inpainting',
          model: 'fal-model',
          message: 'Fal.ai content policy rejected this page',
        },
      },
    ])

    render(
      <TooltipProvider>
        <PageRail />
      </TooltipProvider>,
    )

    const warning = screen.getByLabelText('Fal.ai content policy rejected this page')
    expect(warning).toBeInTheDocument()
    await user.hover(warning)
    expect(await screen.findByText('Fal.ai content policy rejected this page')).toBeInTheDocument()
  })

  it('keeps rapid page switches on the latest native selection', async () => {
    installProject()
    const pages = [
      { id: 'page', label: 'Page 1', size: { width: 1000, height: 1500 }, layers: [], regions: [] },
      {
        id: 'page-2',
        label: 'Page 2',
        size: { width: 1000, height: 1500 },
        layers: [],
        regions: [],
      },
      {
        id: 'page-3',
        label: 'Page 3',
        size: { width: 1000, height: 1500 },
        layers: [],
        regions: [],
      },
    ]
    queryClient.setQueryData(
      pagesKey,
      pages.map((page) => ({
        id: page.id,
        label: page.label,
        size: page.size,
        source_asset: null,
        layer_count: 0,
      })),
    )
    vi.spyOn(canvasRuntime, 'showCanvasPage').mockReturnValue(false)
    const pending = new Map<
      string,
      (selection: Awaited<ReturnType<typeof commands.selectPage>>) => void
    >()
    const selectPage = vi.spyOn(commands, 'selectPage').mockImplementation(
      (page) =>
        new Promise((resolve) => {
          pending.set(page, resolve)
        }),
    )
    render(<PageRail />)

    fireEvent.click(screen.getByText('Page 2').closest('article')!)
    expect(selectPage).toHaveBeenLastCalledWith('page-2')

    fireEvent.click(screen.getByText('Page 3').closest('article')!)
    expect(selectPage).toHaveBeenLastCalledWith('page-3')

    await act(async () => {
      pending.get('page-3')!({
        project: {
          name: 'Book',
          revision: 1,
          active_page: 'page-3',
          can_undo: true,
          can_redo: false,
        },
        page: pages[2]!,
      })
      await Promise.resolve()
    })
    await act(async () => {
      pending.get('page-2')!({
        project: {
          name: 'Book',
          revision: 1,
          active_page: 'page-2',
          can_undo: true,
          can_redo: false,
        },
        page: pages[1]!,
      })
      await Promise.resolve()
    })

    expect(queryClient.getQueryData<{ active_page: string }>(projectKey)?.active_page).toBe(
      'page-3',
    )
    expect(queryClient.getQueryData<{ id: string }>(pageKey)?.id).toBe('page-3')
  })

  it('lets a cached canvas frame paint before synchronizing native page state', async () => {
    installProject()
    queryClient.setQueryData(pagesKey, [
      ...(queryClient.getQueryData<PageSummary[]>(pagesKey) ?? []),
      {
        id: 'page-2',
        label: 'Page 2',
        size: { width: 1000, height: 1500 },
        source_asset: null,
        layer_count: 0,
      },
    ])
    vi.spyOn(canvasRuntime, 'showCanvasPage').mockReturnValue(true)
    const selectPage = vi.spyOn(commands, 'selectPage')
    let paint: FrameRequestCallback | undefined
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => {
      paint = callback
      return 1
    })
    render(<PageRail />)

    fireEvent.click(screen.getByText('Page 2').closest('article')!)

    expect(selectPage).not.toHaveBeenCalled()
    await act(async () => {
      paint?.(performance.now())
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(selectPage).toHaveBeenCalledWith('page-2')
  })

  it('prefetches an inactive page on pointer intent and caches its preparation', async () => {
    installProject()
    const page = {
      id: 'page-2',
      label: 'Page 2',
      size: { width: 1000, height: 1500 },
      layers: [],
      regions: [],
    }
    queryClient.setQueryData(pagesKey, [
      ...(queryClient.getQueryData<PageSummary[]>(pagesKey) ?? []),
      {
        id: page.id,
        label: page.label,
        size: page.size,
        source_asset: null,
        layer_count: 0,
      },
    ])
    const prepared = { revision: 1, page }
    const prefetch = vi.spyOn(canvasRuntime, 'prefetchCanvasPages').mockResolvedValue([prepared])
    const selectPage = vi.spyOn(commands, 'selectPage')
    render(<PageRail />)

    fireEvent.pointerEnter(screen.getByText('Page 2').closest('article')!)

    await waitFor(() => expect(prefetch).toHaveBeenCalledWith(['page-2']))
    await waitFor(() =>
      expect(queryClient.getQueryData(preparedPageKey('page-2'))).toEqual(prepared),
    )
    expect(selectPage).not.toHaveBeenCalled()
  })

  it('deduplicates page intent within a project revision and retries on a newer revision', async () => {
    installProject()
    const page = {
      id: 'page-2',
      label: 'Page 2',
      size: { width: 1000, height: 1500 },
      layers: [],
      regions: [],
    }
    queryClient.setQueryData(pagesKey, [
      ...(queryClient.getQueryData<PageSummary[]>(pagesKey) ?? []),
      {
        id: page.id,
        label: page.label,
        size: page.size,
        source_asset: null,
        layer_count: 0,
      },
    ])
    const prefetch = vi
      .spyOn(canvasRuntime, 'prefetchCanvasPages')
      .mockResolvedValueOnce([{ revision: 1, page }])
      .mockResolvedValueOnce([{ revision: 2, page }])
    render(<PageRail />)
    const item = screen.getByText('Page 2').closest('article')!

    fireEvent.pointerEnter(item)
    fireEvent.focus(screen.getByRole('button', { name: 'Actions for Page 2' }))
    await waitFor(() => expect(prefetch).toHaveBeenCalledTimes(1))
    fireEvent.pointerEnter(item)
    expect(prefetch).toHaveBeenCalledTimes(1)

    queryClient.setQueryData(projectKey, {
      ...queryClient.getQueryData<ProjectInfo>(projectKey)!,
      revision: 2,
    })
    fireEvent.focus(item)

    await waitFor(() => expect(prefetch).toHaveBeenCalledTimes(2))
    expect(prefetch).toHaveBeenLastCalledWith(['page-2'])
  })

  it('switches tools and applies typography from the contextual inspector', async () => {
    const user = userEvent.setup()
    installProject()
    const setTypography = vi.spyOn(commands, 'setTypography').mockResolvedValue(null)
    render(
      <TooltipProvider>
        <ToolBar />
        <Inspector />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Brush' }))
    expect(useKoharuStore.getState().tool).toBe('draw')
    const ocr = screen.getByRole('button', { name: 'OCR region' })
    fireEvent.click(ocr)
    expect(useKoharuStore.getState().tool).toBe('ocr')
    await user.hover(ocr)
    expect(await screen.findByText('Ctrl+O')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Text' }))
    expect(screen.getByTestId('type-inspector')).toBeInTheDocument()
    expect(screen.getByTestId('type-font-picker')).toHaveTextContent('Noto Sans')
    expect(screen.getByTestId('type-size')).toHaveValue('')
    expect(screen.getByTestId('type-size')).toHaveAttribute('placeholder', 'Auto')
    expect(screen.getByRole('combobox', { name: 'Text direction' })).toHaveTextContent('Auto')
    await user.clear(screen.getByTestId('type-size'))
    await user.type(screen.getByTestId('type-size'), '18')
    await user.tab()
    await waitFor(() =>
      expect(setTypography).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            layer: 'element',
            typography: expect.objectContaining({ size: 18, writing_mode: null }),
          }),
        ]),
      ),
    )
  })

  it('defaults vertical text alignment to top and maps end to bottom', async () => {
    const user = userEvent.setup()
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: page.layers.map((layer) =>
        layer.type === 'text'
          ? {
              ...layer,
              typography: { ...layer.typography, alignment: null, writing_mode: 'Vertical' },
            }
          : layer,
      ),
    }))
    const setTypography = vi.spyOn(commands, 'setTypography').mockResolvedValue(null)
    render(<Inspector />)

    expect(screen.getByRole('button', { name: 'Align top' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    await user.click(screen.getByRole('button', { name: 'Align bottom' }))
    await waitFor(() =>
      expect(setTypography).toHaveBeenCalledWith([
        expect.objectContaining({
          layer: 'element',
          typography: expect.objectContaining({ alignment: 'End', writing_mode: 'Vertical' }),
        }),
      ]),
    )
  })

  it('adjusts brush size from the toolbar popover', async () => {
    const user = userEvent.setup()
    installProject()
    useKoharuStore.setState({ tool: 'draw' })
    render(
      <TooltipProvider>
        <ToolBar />
      </TooltipProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Brush size: 48 pixels' }))
    expect(screen.getByRole('textbox', { name: 'Brush size' })).toHaveValue('48')

    await user.click(screen.getByRole('button', { name: 'Increase brush size' }))
    expect(useKoharuStore.getState().brush.diameter).toBe(49)
  })

  it('uses the border color well to enable and disable the border', async () => {
    const user = userEvent.setup()
    installProject()
    const setTypography = vi.spyOn(commands, 'setTypography').mockResolvedValue(null)
    render(<Inspector />)

    expect(screen.queryByRole('button', { name: 'Enable text border' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Border color' }))
    await user.click(screen.getByRole('button', { name: 'Transparent' }))
    await waitFor(() =>
      expect(setTypography).toHaveBeenCalledWith([
        expect.objectContaining({
          layer: 'element',
          typography: expect.objectContaining({
            stroke_color: [255, 255, 255, 0],
            stroke_width: 1.5,
          }),
        }),
      ]),
    )

    setTypography.mockClear()
    fireEvent.change(screen.getByRole('textbox', { name: 'Hex color code' }), {
      target: { value: '#FF0000' },
    })
    await waitFor(() =>
      expect(setTypography).toHaveBeenCalledWith([
        expect.objectContaining({
          layer: 'element',
          typography: expect.objectContaining({
            stroke_color: [255, 0, 0, 255],
            stroke_width: 1.5,
          }),
        }),
      ]),
    )
  })

  it('only offers styles and weights available for the selected font family', async () => {
    const user = userEvent.setup()
    installProject()
    const setTypography = vi.spyOn(commands, 'setTypography').mockResolvedValue(null)
    queryClient.setQueryData(fontsKey, [
      {
        name: 'Noto Sans',
        metadata: {
          primary_script: 'latn',
          scripts: ['latn'],
          languages: ['en'],
          category: 'SANS_SERIF',
          classifications: ['sans-serif'],
          use_cases: ['body-text'],
        },
        sources: ['system'],
        faces: [
          {
            postscript_name: 'NotoSans-Regular',
            weight: 400,
            weight_range: null,
            style: 'normal',
          },
          {
            postscript_name: 'NotoSans-Bold',
            weight: 700,
            weight_range: null,
            style: 'normal',
          },
          {
            postscript_name: 'NotoSans-Italic',
            weight: 400,
            weight_range: null,
            style: 'italic',
          },
        ],
      },
    ])
    const { unmount } = render(<Inspector />)

    fireEvent.click(screen.getByRole('combobox', { name: 'Font weight' }))

    expect(screen.getByRole('option', { name: '400' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: '700' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: '100' })).not.toBeInTheDocument()

    unmount()
    render(<Inspector />)
    fireEvent.click(screen.getByRole('combobox', { name: 'Font style' }))
    expect(screen.getByRole('option', { name: 'Regular' })).toBeInTheDocument()
    await user.click(screen.getByRole('option', { name: 'Italic' }))
    await waitFor(() =>
      expect(setTypography).toHaveBeenCalledWith([
        expect.objectContaining({
          layer: 'element',
          typography: expect.objectContaining({ font_style: 'italic', font_weight: 400 }),
        }),
      ]),
    )
  })

  it('debounces layer text editing and flushes it when focus leaves the field', async () => {
    installProject()
    const save = vi.spyOn(commands, 'setSourceText').mockResolvedValue(null)
    render(<Inspector />)
    const layer = screen.getByRole('button', { name: 'Edit Hello' })
    expect(screen.getByTestId('edit-source-element')).toBeInTheDocument()
    fireEvent.click(layer)
    expect(screen.queryByTestId('edit-source-element')).not.toBeInTheDocument()
    fireEvent.click(layer)
    const source = screen.getByTestId('edit-source-element')
    fireEvent.change(source, { target: { value: 'corrected OCR' } })
    fireEvent.blur(source)
    await waitFor(() => expect(save).toHaveBeenCalledWith('element', 'corrected OCR'))
  })

  it('shows actual layers with only the useful text-role distinction', () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        ...page.layers.map((layer) =>
          layer.type === 'text'
            ? { ...layer, content: { ...layer.content, role: 'dev.koharu.text.onomatopoeia' } }
            : layer,
        ),
        {
          ...textLayer,
          id: 'dialogue',
          content: {
            ...textLayer.content,
            id: 'dialogue-content',
            translation: { text: 'Dialogue line', language: null },
            role: 'dev.koharu.text.dialogue',
          },
        },
        {
          ...textLayer,
          id: 'free-text',
          content: {
            ...textLayer.content,
            id: 'free-text-content',
            translation: { text: 'Caption', language: null },
            role: 'dev.koharu.text.free-text',
          },
        },
      ],
    }))
    render(<Inspector />)

    expect(screen.queryByRole('button', { name: /Filter layers by type/ })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Edit Hello' })).toHaveTextContent('Text')
    expect(screen.getByRole('button', { name: 'Edit Dialogue line' })).toHaveTextContent('Dialogue')
    expect(screen.getByRole('button', { name: 'Edit Caption' })).toHaveTextContent('Free text')
    expect(screen.queryByText('Onomatopoeia')).not.toBeInTheDocument()
  })

  it('numbers only OCR layers in their canonical layer order', () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: [
        {
          ...textLayer,
          content: { ...textLayer.content, source_region: 'region-1' },
        },
        {
          ...textLayer,
          id: 'second',
          content: { ...textLayer.content, id: 'second-content', source_region: 'region-2' },
        },
        {
          ...textLayer,
          id: 'manual',
          content: { ...textLayer.content, id: 'manual-content', source_region: null },
        },
      ],
    }))
    render(<Inspector />)

    expect(screen.getByTestId('ocr-layer-number-element')).toHaveValue(1)
    expect(screen.getByTestId('ocr-layer-number-second')).toHaveValue(2)
    expect(screen.queryByTestId('ocr-layer-number-manual')).not.toBeInTheDocument()
  })

  it('changes an OCR layer role from compact role buttons', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: page.layers.map((layer) =>
        layer.type === 'text'
          ? {
              ...layer,
              content: {
                ...layer.content,
                role: 'dev.koharu.text.dialogue',
                source_region: 'region-1',
              },
            }
          : layer,
      ),
    }))
    const setRole = vi.spyOn(commands, 'setTextRole').mockResolvedValue(null)
    const user = userEvent.setup()
    render(<Inspector />)

    const selector = screen.getByTestId('ocr-role-element')
    expect(selector).toHaveAttribute('role', 'group')
    expect(screen.getByTestId('ocr-role-element-dialogue')).toHaveTextContent('D')
    expect(screen.getByTestId('ocr-role-element-onomatopoeia')).toHaveTextContent('S')
    expect(screen.getByTestId('ocr-role-element-free-text')).toHaveTextContent('F')
    await user.click(screen.getByTestId('ocr-role-element-onomatopoeia'))

    await waitFor(() =>
      expect(setRole).toHaveBeenCalledWith('element', 'dev.koharu.text.onomatopoeia'),
    )
  })

  it('reorders OCR layers by editing a number or dragging a row', async () => {
    installProject()
    const textGroup = {
      type: 'group',
      id: 'text-group',
      parent: 'page',
      visibility: { visible: true, opacity: 1 },
      name: 'Text',
      role: 'text',
    } satisfies Layer
    const ocrLayers = Array.from({ length: 12 }, (_, index) => ({
      ...textLayer,
      id: `text-${index + 1}`,
      parent: textGroup.id,
      content: {
        ...textLayer.content,
        id: `content-${index + 1}`,
        source_region: `region-${index + 1}`,
      },
    })) satisfies Layer[]
    const page = {
      ...(queryClient.getQueryData(pageKey) as Page),
      layers: [textGroup, ...ocrLayers],
    } satisfies Page
    queryClient.setQueryData(pageKey, page)
    const move = vi.spyOn(commands, 'moveLayer').mockResolvedValue(page)
    const reorder = vi.spyOn(commands, 'reorderLayers').mockResolvedValue(page)
    render(<Inspector />)

    const secondNumber = screen.getByTestId('ocr-layer-number-text-2')
    fireEvent.change(secondNumber, { target: { value: '1' } })
    fireEvent.blur(secondNumber)
    await waitFor(() => expect(move).toHaveBeenCalledWith('text-2', 'text-group', 0))

    move.mockClear()
    fireEvent.dragStart(screen.getByTestId('layer-row-text-6'))
    fireEvent.dragOver(screen.getByTestId('layer-row-text-11'))
    expect(screen.getByTestId('ocr-layer-number-text-6')).toHaveValue(11)
    expect(screen.getByTestId('ocr-layer-number-text-11')).toHaveValue(10)
    const rows = screen.getAllByTestId(/layer-row-/)
    expect(rows.indexOf(screen.getByTestId('layer-row-text-6'))).toBeGreaterThan(
      rows.indexOf(screen.getByTestId('layer-row-text-11')),
    )
    // Chromium may cancel drop when the live preview moves the dragged DOM
    // node. dragend still fires and must persist the last previewed position.
    fireEvent.dragEnd(screen.getByTestId('layer-row-text-6'))
    await waitFor(() =>
      expect(reorder).toHaveBeenCalledWith('text-group', [
        'text-1',
        'text-2',
        'text-3',
        'text-4',
        'text-5',
        'text-7',
        'text-8',
        'text-9',
        'text-10',
        'text-11',
        'text-6',
        'text-12',
      ]),
    )
  })

  it('shift-selects OCR rows and reorders them as one stable group', async () => {
    installProject()
    const textGroup = {
      type: 'group',
      id: 'text-group',
      parent: 'page',
      visibility: { visible: true, opacity: 1 },
      name: 'Text',
      role: 'text',
    } satisfies Layer
    const ocrLayers = Array.from({ length: 12 }, (_, index) => ({
      ...textLayer,
      id: `text-${index + 1}`,
      parent: textGroup.id,
      content: {
        ...textLayer.content,
        id: `content-${index + 1}`,
        source_region: `region-${index + 1}`,
      },
    })) satisfies Layer[]
    const page = {
      ...(queryClient.getQueryData(pageKey) as Page),
      layers: [textGroup, ...ocrLayers],
    } satisfies Page
    queryClient.setQueryData(pageKey, page)
    useKoharuStore.setState({ selectedLayers: [] })
    const reorder = vi.spyOn(commands, 'reorderLayers').mockResolvedValue(page)
    render(<Inspector />)

    const fifth = screen.getByTestId('layer-row-text-5')
    const sixth = screen.getByTestId('layer-row-text-6')
    fireEvent.click(fifth.querySelector('button[aria-label]')!)
    fireEvent.click(sixth.querySelector('button[aria-label]')!, { shiftKey: true })
    expect(useKoharuStore.getState().selectedLayers).toEqual(['text-5', 'text-6'])

    fireEvent.dragStart(fifth)
    fireEvent.dragOver(screen.getByTestId('layer-row-text-11'))
    expect(screen.getByTestId('ocr-layer-number-text-5')).toHaveValue(10)
    expect(screen.getByTestId('ocr-layer-number-text-6')).toHaveValue(11)
    fireEvent.dragEnd(fifth)

    await waitFor(() =>
      expect(reorder).toHaveBeenCalledWith('text-group', [
        'text-1',
        'text-2',
        'text-3',
        'text-4',
        'text-7',
        'text-8',
        'text-9',
        'text-10',
        'text-11',
        'text-5',
        'text-6',
        'text-12',
      ]),
    )
    expect(useKoharuStore.getState().selectedLayers).toEqual(['text-5', 'text-6'])
  })

  it('resets a custom text frame to its automatic region', async () => {
    installProject()
    queryClient.setQueryData(pageKey, (page: { layers: Layer[] }) => ({
      ...page,
      layers: page.layers.map((layer) =>
        layer.type === 'text' ? { ...layer, automatic_region: 'bubble' } : layer,
      ),
    }))
    const reset = vi.spyOn(commands, 'setGeometry').mockResolvedValue(null)
    render(<Inspector />)

    expect(screen.getByText('Custom frame')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Reset to auto fit' }))

    await waitFor(() => expect(reset).toHaveBeenCalledWith([{ layer: 'element', points: null }]))
  })

  it('shows zoom before page size without a fit control', () => {
    installProject()
    useKoharuStore.setState({ camera: { zoom: 1.25, translation: [0, 0], fitted: false } })
    render(<StatusBar onZoomChange={vi.fn()} />)

    const zoom = screen.getByText('125%')
    const size = screen.getByText('1000 × 1500 px')
    expect(zoom.compareDocumentPosition(size) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0)
    expect(screen.queryByRole('button', { name: 'Fit window' })).not.toBeInTheDocument()
  })

  it('changes the pipeline scope and selected stages from the runtime selector', async () => {
    installProject()
    const run = vi.spyOn(commands, 'process').mockResolvedValue('job')
    const exportChapter = vi.spyOn(commands, 'exportChapterTranslation').mockResolvedValue(null)
    const importChapter = vi.spyOn(commands, 'importChapterTranslation').mockResolvedValue(null)
    render(<CanvasCommandBar />)
    useCompactProcessingControls()

    fireEvent.click(screen.getByRole('button', { name: 'Processing settings' }))
    fireEvent.click(screen.getByRole('button', { name: /Scope Page/ }))
    fireEvent.click(screen.getByRole('button', { name: /Entire project/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Export chapter translation request' }))
    fireEvent.click(screen.getByRole('button', { name: 'Import chapter translation response' }))
    await waitFor(() => {
      expect(exportChapter).toHaveBeenCalledOnce()
      expect(importChapter).toHaveBeenCalledOnce()
    })
    fireEvent.click(screen.getByRole('button', { name: 'Processing settings' }))
    fireEvent.click(screen.getByRole('button', { name: /Stages 4 stages/ }))
    fireEvent.click(screen.getByRole('button', { name: /Translation/ }))
    fireEvent.click(screen.getByRole('button', { name: /Inpainting/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Run processing' }))
    await waitFor(() =>
      expect(run).toHaveBeenLastCalledWith(
        { scope: 'project' },
        { operation: 'stages', stages: ['detection', 'ocr'] },
      ),
    )

    fireEvent.click(screen.getByRole('button', { name: 'Processing settings' }))
    fireEvent.click(screen.getByRole('button', { name: /Scope Project/ }))
    fireEvent.click(screen.getByRole('button', { name: /Selected pages/ }))
    expect(
      screen.queryByRole('button', { name: 'Export chapter translation request' }),
    ).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Stages 2 stages/ }))
    fireEvent.click(screen.getByRole('button', { name: /Translation/ }))
    fireEvent.click(screen.getByRole('button', { name: /Inpainting/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Run processing' }))
    await waitFor(() =>
      expect(run).toHaveBeenLastCalledWith(
        { scope: 'pages', value: ['page'] },
        { operation: 'stages', stages: ['detection', 'ocr', 'translation', 'inpainting'] },
      ),
    )
  })

  it('runs the current page and exposes the runtime shortcuts', async () => {
    installProject()
    const run = vi.spyOn(commands, 'process').mockResolvedValue('job')
    render(<CanvasCommandBar />)
    useCompactProcessingControls()

    fireEvent.click(screen.getByRole('button', { name: 'Run processing' }))
    await waitFor(() =>
      expect(run).toHaveBeenLastCalledWith(
        { scope: 'pages', value: ['page'] },
        { operation: 'stages', stages: ['detection', 'ocr', 'translation', 'inpainting'] },
      ),
    )

    const selector = screen.getByRole('button', { name: 'Processing settings' })
    const runButton = screen.getByRole('button', { name: 'Run processing' })
    expect(selector).toHaveClass('h-7')
    expect(runButton).toHaveClass('h-7', 'bg-primary/80', 'hover:bg-primary/90')
    fireEvent.click(selector)
    await waitFor(() => expect(commands.getTranslationModels).toHaveBeenCalled())
    expect(screen.getByRole('button', { name: /Model Gemma 4 E2B Instruct/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Scope Page/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Stages 4 stages/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Output English/ })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }))
    expect(useKoharuStore.getState().settingsOpen).toBe(true)
  })

  it('allows an empty stage selection and reruns the explicit stages', async () => {
    installProject()
    const user = userEvent.setup()
    const run = vi.spyOn(commands, 'process').mockResolvedValue('job')
    render(<CanvasCommandBar />)

    expect(screen.getByTestId('inference-expanded')).toBeInTheDocument()
    for (const stage of ['Detection', 'OCR', 'Translation', 'Inpainting']) {
      await user.click(screen.getByRole('button', { name: stage }))
    }

    expect(useKoharuStore.getState().processingStages).toEqual([])
    const runButton = screen.getByRole('button', { name: 'Run processing' })
    expect(runButton).toBeDisabled()

    await user.click(screen.getByRole('button', { name: 'OCR' }))
    expect(runButton).not.toBeDisabled()
    await user.click(runButton)
    await user.click(runButton)
    expect(run).toHaveBeenNthCalledWith(
      1,
      { scope: 'pages', value: ['page'] },
      { operation: 'stages', stages: ['ocr'] },
    )
    expect(run).toHaveBeenNthCalledWith(
      2,
      { scope: 'pages', value: ['page'] },
      { operation: 'stages', stages: ['ocr'] },
    )
  })

  it('scrolls the selected translation model into view when the picker opens', async () => {
    const selected: Model = {
      provider: 'local',
      model: 'gemma4-e2b-it',
      name: 'Gemma 4 E2B Instruct',
      quantizations: [],
      vision: true,
      reasoning: true,
      reasoning_required: false,
    }
    const models: Model[] = [
      ...Array.from({ length: 30 }, (_, index) => ({
        provider: 'local' as const,
        model: `dummy-${index}`,
        name: `Dummy model ${index}`,
        quantizations: [],
        vision: true,
        reasoning: false,
        reasoning_required: false,
      })),
      selected,
    ]
    const scroll = vi.spyOn(Element.prototype, 'scrollIntoView')
    render(
      <ModelPicker
        value={selected}
        models={models}
        providers={preferences.providers.entries}
        onSelect={vi.fn()}
      />,
    )

    const selectedButton = screen.getByRole('button', {
      name: 'Use Gemma 4 E2B Instruct from Local',
    })
    await waitFor(() => expect(scroll).toHaveBeenCalledWith({ block: 'center' }))
    expect(scroll.mock.instances[scroll.mock.instances.length - 1]).toBe(selectedButton)
  })

  it('exposes every pipeline method in expanded processing controls', async () => {
    installProject()
    const user = userEvent.setup()
    const configured: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          chapter: {
            ...preferences.pipeline.translation.chapter,
            model: {
              provider: 'deepseek',
              model: 'deepseek-chat',
              quantization: null,
              vision: false,
              reasoning: true,
            },
          },
        },
      },
    }
    useKoharuStore.setState({
      preferences: configured,
      translationModels: [
        ...useKoharuStore.getState().translationModels,
        {
          provider: 'deepseek',
          model: 'deepseek-chat',
          name: 'DeepSeek Chat',
          quantizations: [],
          vision: false,
          reasoning: true,
          reasoning_required: false,
        },
      ],
    })
    const notoSans = {
      name: 'Noto Sans',
      metadata: {
        primary_script: 'latn',
        scripts: ['latn'],
        languages: ['en'],
        category: 'SANS_SERIF',
        classifications: ['sans-serif'],
        use_cases: ['body-text'],
      },
      sources: ['system' as const],
      faces: [
        {
          postscript_name: 'NotoSans-Regular',
          weight: 400,
          weight_range: null,
          style: 'normal' as const,
        },
      ],
    }
    queryClient.setQueryData(fontsKey, [
      notoSans,
      {
        ...notoSans,
        name: 'Arial',
        faces: [{ ...notoSans.faces[0]!, postscript_name: 'ArialMT' }],
      },
    ])
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...configured,
        pipeline,
        providers,
        typesetting,
      }))
    const run = vi.spyOn(commands, 'process').mockResolvedValue('job')
    render(<CanvasCommandBar />)

    expect(screen.getByTestId('inference-expanded')).toBeInTheDocument()
    const detectionChip = screen.getByRole('button', { name: /^Detection:/ })
    expect(detectionChip).toHaveTextContent('Koharu Layout RF-DETR Seg 2XL')
    expect(detectionChip.querySelector('[data-slot="expanded-stage-label"]')).toHaveTextContent(
      'Detect',
    )
    expect(detectionChip.querySelector('[data-slot="expanded-stage-value"]')).toHaveClass('sr-only')
    expect(detectionChip.querySelector('[data-slot="expanded-stage-value"]')).toHaveAttribute(
      'title',
      'Koharu Layout RF-DETR Seg 2XL',
    )
    expect(screen.getByRole('button', { name: /^OCR:/ })).toHaveTextContent('PaddleOCR-VL 1.6')
    expect(screen.getByRole('button', { name: /^Translation:/ })).toHaveTextContent(
      'Gemma 4 E2B Instruct',
    )
    expect(screen.getByRole('button', { name: /^Inpainting:/ })).toHaveTextContent('LaMa')
    expect(screen.getByRole('button', { name: /^Typing:/ })).toHaveTextContent('Noto Sans')
    for (const stage of ['Detection', 'OCR', 'Translation', 'Inpainting']) {
      expect(screen.getByRole('button', { name: stage })).toHaveAttribute('aria-pressed', 'true')
    }

    await user.click(screen.getByRole('button', { name: /^Translation:/ }))
    expect(document.querySelector('[data-slot="current-translation-model"]')).toHaveTextContent(
      'Gemma 4 E2B Instruct',
    )
    await user.click(screen.getByRole('button', { name: /^Translation:/ }))

    await user.click(screen.getByRole('button', { name: 'Scope: Page' }))
    await user.click(screen.getByRole('button', { name: /Entire project/ }))
    expect(screen.getByRole('button', { name: /^Translation: DeepSeek Chat/ })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /^OCR:/ }))
    await user.click(screen.getByRole('combobox', { name: 'OCR method' }))
    await user.click(await screen.findByRole('option', { name: 'API / cloud' }))
    await waitFor(() =>
      expect(save).toHaveBeenLastCalledWith(
        expect.objectContaining({ ocr: expect.objectContaining({ method: 'api' }) }),
        configured.providers,
        configured.typesetting,
      ),
    )

    await user.click(screen.getByRole('button', { name: /^Inpainting:/ }))
    await user.click(screen.getByRole('combobox', { name: 'Restoration method' }))
    await user.click(await screen.findByRole('option', { name: 'API / cloud' }))
    await waitFor(() =>
      expect(save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          inpainting: expect.objectContaining({ method: 'api' }),
        }),
        configured.providers,
        configured.typesetting,
      ),
    )
    expect(screen.getByRole('button', { name: /^Inpainting:/ })).toHaveTextContent(
      'Microsoft MAI Image 2.5 Edit',
    )

    await user.click(screen.getByRole('button', { name: 'Detection' }))
    const stagesBeforeTyping = useKoharuStore.getState().processingStages
    await user.click(screen.getByRole('button', { name: /^Typing:/ }))
    await user.click(screen.getByRole('button', { name: 'Default font family 1' }))
    await user.click(await screen.findByRole('option', { name: 'Arial, System' }))
    await waitFor(() =>
      expect(save).toHaveBeenLastCalledWith(
        expect.objectContaining({ ocr: expect.objectContaining({ method: 'api' }) }),
        configured.providers,
        { font_families: ['Arial'] },
      ),
    )
    expect(useKoharuStore.getState().processingStages).toEqual(stagesBeforeTyping)

    await user.click(screen.getByRole('button', { name: 'Run processing' }))
    await waitFor(() =>
      expect(run).toHaveBeenLastCalledWith(
        { scope: 'project' },
        { operation: 'stages', stages: ['ocr', 'translation', 'inpainting'] },
      ),
    )

    await user.click(screen.getByRole('button', { name: 'Use compact processing controls' }))
    expect(screen.getByRole('button', { name: 'Processing settings' })).toBeInTheDocument()
  }, 15_000)

  it('configures translation output from the runtime selector', async () => {
    installProject()
    const user = userEvent.setup()
    const nextPreferences: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          target_language: 'ja-JP',
          page: {
            ...preferences.pipeline.translation.page,
            instructions: 'Keep character names unchanged.',
          },
        },
      },
    }
    let finishSave!: (saved: Preferences) => void
    const pendingSave = new Promise<Preferences>((resolve) => {
      finishSave = resolve
    })
    const save = vi.spyOn(commands, 'savePreferences').mockImplementation(() => pendingSave)
    render(<CanvasCommandBar />)
    useCompactProcessingControls()

    await user.click(screen.getByRole('button', { name: 'Processing settings' }))
    await waitFor(() => expect(commands.getTranslationModels).toHaveBeenCalled())
    await user.click(screen.getByRole('button', { name: /Output English/ }))
    const language = screen.getByRole('combobox', { name: 'Target language' })
    expect(language).toHaveTextContent('English')
    expect(language).not.toHaveTextContent('en-US')
    await user.click(language)
    await user.click(await screen.findByRole('option', { name: 'Japanese' }))
    const instructions = screen.getByRole('textbox', { name: 'Translation instructions' })
    expect(instructions).toHaveClass('max-h-20', 'overflow-y-auto')
    await user.type(instructions, 'Keep character names unchanged.')
    expect(screen.queryByRole('button', { name: 'Apply output' })).not.toBeInTheDocument()

    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        nextPreferences.pipeline,
        preferences.providers,
        preferences.typesetting,
      ),
    )
    expect(
      screen.queryByLabelText(/Saving output settings|outputPicker\.saving/),
    ).not.toBeInTheDocument()
    expect(language).not.toBeDisabled()
    expect(instructions).not.toBeDisabled()
    expect(instructions).toHaveFocus()
    await act(async () => finishSave(nextPreferences))
    await waitFor(() =>
      expect(useKoharuStore.getState().preferences?.pipeline.translation).toEqual(
        nextPreferences.pipeline.translation,
      ),
    )
    expect(instructions).toBeInTheDocument()
    expect(instructions).toHaveFocus()
  })

  it('preserves the runtime shortcuts while visiting settings', async () => {
    installProject()
    const user = userEvent.setup()
    useKoharuStore.setState({
      preferences: {
        ...preferences,
        pipeline: {
          ...preferences.pipeline,
          translation: {
            ...preferences.pipeline.translation,
            page: { ...preferences.pipeline.translation.page, instructions: 'Page guidance' },
            chapter: {
              ...preferences.pipeline.translation.chapter,
              instructions: 'Chapter guidance',
            },
          },
        },
      },
    })
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...preferences,
        pipeline,
        providers,
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsNavigationHarness />
      </ThemeProvider>,
    )
    useCompactProcessingControls()

    await user.click(screen.getByRole('button', { name: 'Processing settings' }))
    await user.click(screen.getByRole('button', { name: /Output English/ }))
    expect(screen.getByRole('textbox', { name: 'Translation instructions' })).toHaveValue(
      'Page guidance',
    )
    await user.click(screen.getByRole('button', { name: 'Back' }))
    await user.click(screen.getByRole('button', { name: /Scope Page/ }))
    await user.click(screen.getByRole('button', { name: /Entire project/ }))
    await user.click(screen.getByRole('button', { name: /Stages 4 stages/ }))
    await user.click(screen.getByRole('button', { name: /Translation/ }))
    await user.click(screen.getByRole('button', { name: /Inpainting/ }))
    await user.click(screen.getByRole('button', { name: 'Back' }))
    await user.click(screen.getByRole('button', { name: /Output English/ }))
    expect(screen.getByRole('textbox', { name: 'Translation instructions' })).toHaveValue(
      'Chapter guidance',
    )
    await user.click(screen.getByRole('combobox', { name: 'Target language' }))
    await user.click(await screen.findByRole('option', { name: 'Japanese' }))

    act(() => useKoharuStore.getState().setSettingsOpen(true))
    await waitFor(() => expect(save).toHaveBeenCalled())
    await user.click(screen.getByRole('button', { name: 'Back to editor' }))
    useCompactProcessingControls()
    await user.click(screen.getByRole('button', { name: 'Processing settings' }))

    expect(screen.getByRole('button', { name: /Scope Project/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Stages 2 stages/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Output Japanese/ })).toBeInTheDocument()
  })

  it('changes the translation model without re-enabling vision or reasoning', async () => {
    installProject()
    const user = userEvent.setup()
    const currentPreferences: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          page: {
            ...preferences.pipeline.translation.page,
            generation: {
              ...preferences.pipeline.translation.page.generation,
              vision: false,
              reasoning: false,
            },
          },
        },
      },
    }
    useKoharuStore.setState({ preferences: currentPreferences })
    const nextPreferences: Preferences = {
      ...currentPreferences,
      pipeline: {
        ...currentPreferences.pipeline,
        translation: {
          ...currentPreferences.pipeline.translation,
          page: {
            ...currentPreferences.pipeline.translation.page,
            model: {
              provider: 'local',
              model: 'gemma4-12b-it',
              quantization: null,
              vision: true,
              reasoning: true,
            },
          },
        },
      },
    }
    const save = vi.spyOn(commands, 'savePreferences').mockResolvedValue(nextPreferences)
    useKoharuStore.setState({
      translationModels: [
        ...useKoharuStore.getState().translationModels,
        {
          provider: 'local',
          model: 'gemma4-12b-it',
          name: 'Gemma 4 12B',
          quantizations: [],
          vision: true,
          reasoning: true,
          reasoning_required: false,
        },
      ],
    })
    render(<CanvasCommandBar />)
    useCompactProcessingControls()

    fireEvent.click(screen.getByRole('button', { name: 'Processing settings' }))
    fireEvent.click(screen.getByRole('button', { name: /Model Gemma 4 E2B Instruct/ }))
    const search = screen.getByRole('textbox', { name: 'Search models' })
    expect(search).toHaveFocus()
    await user.type(search, '12b')
    expect(
      screen.queryByRole('button', { name: 'Use Gemma 4 E2B Instruct from Local' }),
    ).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Use Gemma 4 12B from Local' }))

    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        nextPreferences.pipeline,
        currentPreferences.providers,
        currentPreferences.typesetting,
      ),
    )
    expect(useKoharuStore.getState().preferences?.pipeline.translation).toEqual(
      nextPreferences.pipeline.translation,
    )
  })

  it('repairs saved mandatory reasoning when the runtime selector reselects its model', async () => {
    installProject()
    const user = userEvent.setup()
    const requiredModel: Model = {
      provider: 'openrouter',
      model: 'google/gemini-3.5-flash-lite',
      name: 'Gemini 3.5 Flash Lite',
      quantizations: [],
      vision: true,
      reasoning: true,
      reasoning_required: true,
    }
    const configured: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          page: {
            ...preferences.pipeline.translation.page,
            model: {
              provider: requiredModel.provider,
              model: requiredModel.model,
              quantization: null,
              vision: true,
              reasoning: true,
            },
            generation: {
              ...preferences.pipeline.translation.page.generation,
              reasoning: false,
            },
          },
        },
      },
    }
    useKoharuStore.setState({
      preferences: configured,
      translationModels: [requiredModel],
    })
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...configured,
        pipeline,
        providers,
        typesetting,
      }))
    render(<CanvasCommandBar />)
    useCompactProcessingControls()

    await user.click(screen.getByRole('button', { name: 'Processing settings' }))
    await user.click(screen.getByRole('button', { name: /Model Gemini 3.5 Flash Lite/ }))
    await user.click(
      screen.getByRole('button', { name: 'Use Gemini 3.5 Flash Lite from openrouter' }),
    )

    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          translation: expect.objectContaining({
            page: expect.objectContaining({
              model: expect.objectContaining({ reasoning_required: true }),
              generation: expect.objectContaining({ reasoning: true }),
            }),
          }),
        }),
        configured.providers,
        configured.typesetting,
      ),
    )
  })

  it('preserves vision when the runtime selector chooses a text-only model', async () => {
    installProject()
    const nextPreferences: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          page: {
            ...preferences.pipeline.translation.page,
            model: {
              provider: 'deepseek',
              model: 'deepseek-chat',
              quantization: null,
              vision: false,
              reasoning: true,
            },
            generation: {
              ...preferences.pipeline.translation.page.generation,
              vision: true,
              reasoning: false,
            },
          },
        },
      },
    }
    const save = vi.spyOn(commands, 'savePreferences').mockResolvedValue(nextPreferences)
    useKoharuStore.setState({
      translationModels: [
        ...useKoharuStore.getState().translationModels,
        {
          provider: 'deepseek',
          model: 'deepseek-chat',
          name: 'DeepSeek Chat',
          quantizations: [],
          vision: false,
          reasoning: true,
          reasoning_required: false,
        },
      ],
    })
    render(<CanvasCommandBar />)
    useCompactProcessingControls()

    fireEvent.click(screen.getByRole('button', { name: 'Processing settings' }))
    fireEvent.click(screen.getByRole('button', { name: /Model Gemma 4 E2B Instruct/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Use DeepSeek Chat from deepseek' }))

    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        nextPreferences.pipeline,
        preferences.providers,
        preferences.typesetting,
      ),
    )
    expect(useKoharuStore.getState().preferences?.pipeline.translation).toEqual(
      nextPreferences.pipeline.translation,
    )
  })

  it('constrains long model names inside the runtime selector', async () => {
    installProject()
    const longName = 'Llama 3.2 8x3b Moe Dark Champion Instruct Uncensored Abliterated 18.4b'
    useKoharuStore.setState({
      translationModels: [
        ...useKoharuStore.getState().translationModels,
        {
          provider: 'local',
          model: 'long-model',
          name: longName,
          quantizations: [],
          vision: true,
          reasoning: false,
          reasoning_required: false,
        },
      ],
    })
    render(<CanvasCommandBar />)
    useCompactProcessingControls()

    fireEvent.click(screen.getByRole('button', { name: 'Processing settings' }))
    fireEvent.click(screen.getByRole('button', { name: /Model Gemma 4 E2B Instruct/ }))

    const label = await screen.findByText(longName)
    await waitFor(() => expect(commands.getTranslationModels).toHaveBeenCalled())
    expect(label).toHaveClass('truncate')
    expect(label.parentElement).toHaveClass('overflow-hidden')
    expect(label.closest('button')).toHaveClass('max-w-full', 'min-w-0', 'overflow-hidden')
    const content = label.closest('[data-slot="scroll-area-content"]')
    expect(content).toHaveStyle({ width: '100%' })
    expect(content?.closest('[data-slot="scroll-area-viewport"]')).toHaveClass('h-auto', 'max-h-64')
  })

  it('edits and persists pipeline and translation preferences from the settings page', async () => {
    installProject()
    const user = userEvent.setup()
    useKoharuStore.setState({ settingsOpen: true })
    const save = vi.spyOn(commands, 'savePreferences').mockResolvedValue(preferences)
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Pipeline' }))
    expect(screen.getByRole('heading', { level: 2, name: 'Pipeline' })).toBeInTheDocument()
    expect(screen.getAllByRole('combobox')).toHaveLength(6)
    const manualCleanup = screen.getByRole('combobox', {
      name: 'Remove: Local restoration model',
    })
    expect(manualCleanup).toHaveTextContent('LaMa')
    await user.click(manualCleanup)
    await user.click(await screen.findByRole('option', { name: 'AOT Inpainting' }))
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          inpainting: expect.objectContaining({ manual_model: { model: 'aot-inpainting' } }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
    save.mockClear()
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Text threshold' }), {
      target: { value: '0.42' },
    })
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          processor: expect.objectContaining({
            'koharu-layout-rfdetr-seg-2xl': expect.objectContaining({ text_threshold: 0.42 }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Providers' }))
    expect(screen.getByRole('heading', { level: 2, name: 'Providers' })).toBeInTheDocument()
    expect(screen.getByLabelText('DeepL credential')).toBeInTheDocument()
    expect(screen.getAllByLabelText('Base URL')).toHaveLength(3)
    expect(screen.queryByRole('switch', { name: 'Vision' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Translation' }))
    expect(screen.getByRole('heading', { level: 2, name: 'Translation' })).toBeInTheDocument()
    expect(screen.getByText('Choose the model used to translate text.')).toBeInTheDocument()
    expect(
      screen.getByText('Control how the model selects and varies generated text.'),
    ).toBeInTheDocument()
    expect(screen.getByText('Use model reasoning during translation.')).toBeInTheDocument()
    const reasoning = screen.getByRole('switch', { name: 'Enable reasoning' })
    expect(reasoning).not.toHaveAttribute('aria-disabled', 'true')
    expect(screen.queryByText('Enable reasoning')).not.toBeInTheDocument()
    save.mockClear()
    await user.click(reasoning)
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          translation: expect.objectContaining({
            page: expect.objectContaining({
              generation: expect.objectContaining({ reasoning: true }),
            }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
    const vision = screen.getByRole('switch', { name: 'Vision' })
    expect(screen.getByText('Feed page images to the LLM during translation.')).toBeInTheDocument()
    expect(vision).toBeChecked()
    save.mockClear()
    await user.click(vision)
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          translation: expect.objectContaining({
            page: expect.objectContaining({
              generation: expect.objectContaining({ vision: false }),
            }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
    expect(screen.getByLabelText('Translation model')).toHaveTextContent('Gemma 4 E2B Instruct')
    await user.click(screen.getByLabelText('Translation model'))
    const modelSearch = screen.getByRole('textbox', { name: 'Search models' })
    expect(modelSearch).toHaveFocus()
    await user.type(modelSearch, 'missing model')
    expect(screen.getByText('No models match this search.')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Clear search' }))
    expect(
      screen.getByRole('button', { name: 'Use Gemma 4 E2B Instruct from Local' }),
    ).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back' }))
    expect(screen.getByLabelText('Target language')).toHaveTextContent('English')
    expect(screen.getByText('Choose the language to translate text into.')).toBeInTheDocument()
    expect(screen.getByLabelText('Translation instructions')).toHaveClass(
      'field-sizing-fixed',
      'overflow-y-auto',
    )
  })

  it('switches OCR between local CPU and a selected vision API model', async () => {
    installProject()
    const user = userEvent.setup()
    useKoharuStore.setState((state) => ({
      settingsOpen: true,
      translationModels: [
        ...state.translationModels,
        {
          provider: 'openrouter',
          model: 'qwen/qwen3.8-27b',
          name: 'Qwen 3.8 27B',
          quantizations: [],
          vision: true,
          reasoning: true,
          reasoning_required: false,
        },
        {
          provider: 'openrouter',
          model: 'text-only',
          name: 'Text Only',
          quantizations: [],
          vision: false,
          reasoning: false,
          reasoning_required: false,
        },
      ],
    }))
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...preferences,
        pipeline,
        providers,
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Pipeline' }))
    expect(screen.getByRole('combobox', { name: 'Local OCR model' })).toHaveTextContent(
      'PaddleOCR-VL 1.6',
    )
    await user.click(screen.getByRole('combobox', { name: 'OCR method' }))
    await user.click(await screen.findByRole('option', { name: 'API / cloud' }))
    expect(screen.queryByRole('combobox', { name: 'Local OCR model' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Vision API model' }))
    expect(screen.queryByText('Text Only')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Use Qwen 3\.8 27B from openrouter/i }))
    fireEvent.change(screen.getByLabelText('OCR instructions'), {
      target: { value: 'Preserve furigana and line breaks.' },
    })
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Temperature' }), {
      target: { value: '0.2' },
    })

    await waitFor(() =>
      expect(save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          ocr: expect.objectContaining({
            method: 'api',
            api: expect.objectContaining({
              model: expect.objectContaining({
                provider: 'openrouter',
                model: 'qwen/qwen3.8-27b',
                vision: true,
              }),
              generation: expect.objectContaining({ temperature: 0.2 }),
              instructions: 'Preserve furigana and line breaks.',
            }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
  })

  it('keeps page and chapter translation profiles independent', async () => {
    installProject()
    const user = userEvent.setup()
    const configured: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          page: { ...preferences.pipeline.translation.page, instructions: 'Page guidance' },
          chapter: {
            ...preferences.pipeline.translation.chapter,
            model: {
              provider: 'deepseek',
              model: 'deepseek-chat',
              quantization: null,
              vision: false,
              reasoning: true,
            },
            instructions: 'Chapter guidance',
          },
        },
      },
    }
    useKoharuStore.setState({
      settingsOpen: true,
      preferences: configured,
      translationModels: [
        ...useKoharuStore.getState().translationModels,
        {
          provider: 'deepseek',
          model: 'deepseek-chat',
          name: 'DeepSeek Chat',
          quantizations: [],
          vision: false,
          reasoning: true,
          reasoning_required: false,
        },
      ],
    })
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...configured,
        pipeline,
        providers,
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Translation' }))
    expect(screen.getByRole('tab', { name: 'Page' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByLabelText('Translation model')).toHaveTextContent('Gemma 4 E2B Instruct')
    expect(screen.getByLabelText('Translation instructions')).toHaveValue('Page guidance')
    expect(screen.getByLabelText('Source language')).toHaveTextContent('Japanese')

    await user.click(screen.getByLabelText('Source language'))
    await user.click(await screen.findByRole('option', { name: 'English' }))

    await user.click(screen.getByRole('tab', { name: 'Chapter' }))
    expect(screen.getByLabelText('Translation model')).toHaveTextContent('DeepSeek Chat')
    const instructions = screen.getByLabelText('Translation instructions')
    expect(instructions).toHaveValue('Chapter guidance')
    fireEvent.change(instructions, { target: { value: 'Updated chapter guidance' } })

    await waitFor(() =>
      expect(save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          translation: expect.objectContaining({
            source_language: 'en-US',
            page: expect.objectContaining({ instructions: 'Page guidance' }),
            chapter: expect.objectContaining({ instructions: 'Updated chapter guidance' }),
          }),
        }),
        configured.providers,
        configured.typesetting,
      ),
    )
  })

  it('persists independent Fal prompts and apply modes', async () => {
    installProject()
    const user = userEvent.setup()
    useKoharuStore.setState({ settingsOpen: true })
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...preferences,
        pipeline,
        providers,
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Pipeline' }))
    await user.click(screen.getByRole('combobox', { name: 'Restoration method' }))
    await user.click(await screen.findByRole('option', { name: 'API / cloud' }))
    const inpainting = screen.getByRole('button', { name: 'Image editing model' })
    await user.click(inpainting)
    const falChoices = await screen.findAllByRole('button', { name: /from Fal\.ai/ })
    expect(falChoices.map((choice) => choice.getAttribute('aria-label'))).toEqual([
      'Use Microsoft MAI Image 2.5 Edit from Fal.ai',
      'Use Microsoft MAI Image 2.5 Pro Edit from Fal.ai',
    ])
    await user.click(
      await screen.findByRole('button', {
        name: 'Use Microsoft MAI Image 2.5 Edit from Fal.ai',
      }),
    )
    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    fireEvent.change(prompt, { target: { value: 'Standard Fal prompt' } })
    await user.click(screen.getByRole('combobox', { name: 'Apply mode' }))
    await user.click(await screen.findByRole('option', { name: 'Mask' }))

    await user.click(inpainting)
    await user.click(
      await screen.findByRole('button', {
        name: 'Use Microsoft MAI Image 2.5 Pro Edit from Fal.ai',
      }),
    )
    fireEvent.change(screen.getByRole('textbox', { name: 'Prompt' }), {
      target: { value: 'Pro Fal prompt' },
    })

    await user.click(inpainting)
    await user.click(
      await screen.findByRole('button', {
        name: 'Use Microsoft MAI Image 2.5 Edit from Fal.ai',
      }),
    )
    expect(screen.getByRole('textbox', { name: 'Prompt' })).toHaveValue('Standard Fal prompt')
    expect(screen.getByRole('combobox', { name: 'Apply mode' })).toHaveTextContent(/mask/i)
    await waitFor(() =>
      expect(save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          processor: expect.objectContaining({
            'microsoft/mai-image-2.5/edit': {
              prompt: 'Standard Fal prompt',
              apply_mode: 'mask',
            },
            'microsoft/mai-image-2.5-pro/edit': expect.objectContaining({
              prompt: 'Pro Fal prompt',
            }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
  })

  it('selects restoration prompt presets and keeps manual edits as custom', async () => {
    installProject()
    const user = userEvent.setup()
    useKoharuStore.setState({ settingsOpen: true })
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...preferences,
        pipeline,
        providers,
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Pipeline' }))
    await user.click(screen.getByRole('combobox', { name: 'Restoration method' }))
    await user.click(await screen.findByRole('option', { name: 'API / cloud' }))

    const preset = screen.getByRole('combobox', { name: 'Prompt preset' })
    expect(preset).toHaveTextContent('Custom prompt')
    await user.click(preset)
    await user.click(await screen.findByRole('option', { name: 'Remove text only' }))

    const cleanupText = inpaintingPromptForPreset('cleanup-text')!
    expect(screen.getByRole('textbox', { name: 'Prompt' })).toHaveValue(cleanupText)
    await waitFor(() =>
      expect(save).toHaveBeenLastCalledWith(
        expect.objectContaining({
          inpainting: expect.objectContaining({
            api: expect.objectContaining({ prompt: cleanupText }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )

    await user.click(screen.getByRole('combobox', { name: 'Prompt preset' }))
    await user.click(await screen.findByRole('option', { name: 'Replace lettering with Russian' }))
    const translateRussian = inpaintingPromptForPreset('translate-russian')!
    expect(screen.getByRole('textbox', { name: 'Prompt' })).toHaveValue(translateRussian)

    fireEvent.change(screen.getByRole('textbox', { name: 'Prompt' }), {
      target: { value: 'My custom image-editing prompt' },
    })
    expect(screen.getByRole('combobox', { name: 'Prompt preset' })).toHaveTextContent(
      'Custom prompt',
    )
  })

  it('selects only image-edit models from the chosen restoration provider', async () => {
    installProject()
    const user = userEvent.setup()
    useKoharuStore.setState({ settingsOpen: true })
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...preferences,
        pipeline,
        providers,
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Pipeline' }))
    await user.click(screen.getByRole('combobox', { name: 'Restoration method' }))
    await user.click(await screen.findByRole('option', { name: 'API / cloud' }))
    await user.click(screen.getByRole('combobox', { name: 'Image provider' }))
    await user.click(await screen.findByRole('option', { name: 'OpenRouter' }))
    await user.click(screen.getByRole('button', { name: 'Image editing model' }))

    expect(
      await screen.findByRole('button', {
        name: 'Use Microsoft MAI Image 2.5 from OpenRouter',
      }),
    ).toBeInTheDocument()
    expect(screen.queryByText('Gemma 4 E2B Instruct')).not.toBeInTheDocument()
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          inpainting: expect.objectContaining({
            method: 'api',
            api: expect.objectContaining({
              provider: 'openrouter',
              model: 'microsoft/mai-image-2.5',
            }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
  })

  it('keeps generation controls independent from model capabilities', async () => {
    installProject()
    const user = userEvent.setup()
    const configured: Preferences = {
      ...preferences,
      pipeline: {
        ...preferences.pipeline,
        translation: {
          ...preferences.pipeline.translation,
          page: {
            ...preferences.pipeline.translation.page,
            model: {
              provider: 'deepseek',
              model: 'deepseek-chat',
              quantization: null,
              vision: false,
              reasoning: true,
            },
            generation: {
              ...preferences.pipeline.translation.page.generation,
              reasoning: false,
            },
          },
        },
      },
    }
    useKoharuStore.setState({
      settingsOpen: true,
      preferences: configured,
      translationModels: [
        {
          provider: 'deepseek',
          model: 'deepseek-chat',
          name: 'DeepSeek Chat',
          quantizations: [],
          vision: false,
          reasoning: true,
          reasoning_required: false,
        },
      ],
    })
    const save = vi.spyOn(commands, 'savePreferences').mockResolvedValue(configured)
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Translation' }))
    const reasoning = screen.getByRole('switch', { name: 'Enable reasoning' })
    expect(reasoning).not.toHaveAttribute('aria-disabled', 'true')
    expect(screen.getByRole('switch', { name: 'Vision' })).not.toHaveAttribute(
      'aria-disabled',
      'true',
    )
    await user.click(reasoning)
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          translation: expect.objectContaining({
            page: expect.objectContaining({
              generation: expect.objectContaining({ reasoning: true }),
            }),
          }),
        }),
        configured.providers,
        configured.typesetting,
      ),
    )
  })

  it('locks reasoning on after selecting a model that requires it', async () => {
    installProject()
    const user = userEvent.setup()
    const requiredModel: Model = {
      provider: 'openrouter',
      model: 'google/gemini-3.5-flash-lite',
      name: 'Gemini 3.5 Flash Lite',
      quantizations: [],
      vision: true,
      reasoning: true,
      reasoning_required: true,
    }
    useKoharuStore.setState({
      settingsOpen: true,
      translationModels: [requiredModel],
    })
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...preferences,
        pipeline,
        providers,
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Translation' }))
    await user.click(screen.getByLabelText('Translation model'))
    await user.click(
      screen.getByRole('button', { name: 'Use Gemini 3.5 Flash Lite from openrouter' }),
    )

    const reasoning = screen.getByRole('switch', { name: 'Enable reasoning' })
    expect(reasoning).toBeChecked()
    expect(reasoning).toHaveAttribute('aria-disabled', 'true')
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          translation: expect.objectContaining({
            page: expect.objectContaining({
              model: expect.objectContaining({ reasoning_required: true }),
              generation: expect.objectContaining({ reasoning: true }),
            }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
  })

  it('shows a reloaded Fal credential as configured without exposing it', () => {
    const saved = render(
      <ProviderPreferences
        value={{
          fal: { configured: false, value: 'fal-key', clear: false },
          entries: [],
        }}
        onChange={vi.fn()}
      />,
    )
    saved.unmount()

    render(
      <ProviderPreferences
        value={{
          fal: { configured: true, value: null, clear: false },
          entries: [],
        }}
        onChange={vi.fn()}
      />,
    )

    const input = screen.getByLabelText('Fal.ai credential')
    expect(input).toHaveValue('')
    expect(input).toHaveAttribute('placeholder', 'Configured')
    expect(screen.getByRole('status')).toHaveTextContent('Configured')
  })

  it('keeps a configured provider credential private and allows clearing it', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    const providers = {
      fal: emptyCredential(),
      entries: [
        {
          name: 'DeepL',
          config: { provider: 'deepl' as const, settings: { base_url: null } },
          credential: { configured: true, value: null, clear: false },
        },
      ],
    }
    render(<ProviderPreferences value={providers} onChange={onChange} />)

    const fal = screen.getByLabelText('Fal.ai credential')
    await user.type(fal, 'fal-key')
    expect(onChange).toHaveBeenLastCalledWith({
      ...providers,
      fal: { configured: false, value: 'fal-key', clear: false },
    })

    const input = screen.getByLabelText('DeepL credential')
    expect(document.querySelector(`label[for="${input.id}"]`)).toHaveTextContent('Credential')
    expect(input).toHaveAttribute('type', 'text')
    expect(input).toHaveAttribute('autocomplete', 'off')
    expect(input).toHaveAttribute('autocapitalize', 'none')
    expect(input).toHaveAttribute('spellcheck', 'false')
    expect(input).toHaveClass(
      '[-webkit-text-security:disc]',
      '[&::placeholder]:[-webkit-text-security:none]',
    )
    expect(input).toHaveValue('')
    expect(input).toHaveAttribute('placeholder', 'Configured')
    expect(
      screen.queryByRole('button', { name: 'Reveal DeepL credential' }),
    ).not.toBeInTheDocument()

    const clear = screen.getByRole('button', { name: 'Clear DeepL credential' })
    expect(clear.querySelector('.lucide-eraser')).toBeInTheDocument()
    expect(clear).not.toHaveClass('text-destructive')
    await user.click(clear)
    expect(onChange).toHaveBeenCalledWith({
      fal: emptyCredential(),
      entries: [
        expect.objectContaining({
          credential: { configured: false, value: null, clear: true },
        }),
      ],
    })
  })

  it('preserves a credential draft and focus when autosave finishes', async () => {
    installProject()
    const user = userEvent.setup()
    useKoharuStore.setState({ settingsOpen: true })
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...preferences,
        pipeline,
        providers: {
          ...providers,
          entries: providers.entries.map((entry) => ({
            ...entry,
            credential: entry.credential?.value
              ? { configured: true, value: null, clear: false }
              : entry.credential,
          })),
        },
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Providers' }))
    const input = screen.getByLabelText('DeepL credential')
    await user.type(input, 's')
    await waitFor(() => expect(save).toHaveBeenCalled())

    await waitFor(() => {
      expect(screen.getByLabelText('DeepL credential')).toBe(input)
      expect(input).toHaveFocus()
      expect(input).toHaveValue('s')
    })
  })

  it('clamps a threshold typed past its bounds before saving it', async () => {
    installProject()
    useKoharuStore.setState({ settingsOpen: true })
    const save = vi.spyOn(commands, 'savePreferences').mockResolvedValue(preferences)
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Pipeline' }))
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Text threshold' }), {
      target: { value: '15' },
    })
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        expect.objectContaining({
          processor: expect.objectContaining({
            'koharu-layout-rfdetr-seg-2xl': expect.objectContaining({ text_threshold: 1 }),
          }),
        }),
        preferences.providers,
        preferences.typesetting,
      ),
    )
  })

  it('saves a newer OpenRouter selection after an in-flight provider save', async () => {
    installProject()
    const user = userEvent.setup()
    const configured: Preferences = {
      ...preferences,
      providers: {
        ...preferences.providers,
        entries: [
          ...preferences.providers.entries,
          {
            name: 'OpenRouter',
            config: { provider: 'openrouter', settings: {} },
            credential: emptyCredential(),
          },
        ],
      },
    }
    useKoharuStore.setState({
      settingsOpen: true,
      preferences: configured,
      translationModels: [
        ...useKoharuStore.getState().translationModels,
        {
          provider: 'openrouter',
          model: 'openrouter/auto',
          name: 'OpenRouter Auto',
          quantizations: [],
          vision: true,
          reasoning: true,
          reasoning_required: false,
        },
      ],
    })
    let resolveFirst: ((value: Preferences) => void) | undefined
    const first = new Promise<Preferences>((resolve) => {
      resolveFirst = resolve
    })
    let firstResult: Preferences | undefined
    let invocation = 0
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => {
        const saved = { ...configured, pipeline, providers, typesetting }
        if (invocation++ === 0) {
          firstResult = saved
          return first
        }
        return saved
      })
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Providers' }))
    await user.type(screen.getByLabelText('OpenRouter credential'), 'secret')
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1))
    await user.click(screen.getByRole('button', { name: 'Translation' }))
    await user.click(screen.getByLabelText('Translation model'))
    await user.click(screen.getByRole('button', { name: 'Use OpenRouter Auto from OpenRouter' }))
    await user.click(screen.getByRole('button', { name: 'Back to editor' }))

    expect(save).toHaveBeenCalledTimes(1)
    await act(async () => {
      resolveFirst?.(firstResult!)
      await first
    })
    await waitFor(() => expect(save).toHaveBeenCalledTimes(2))
    await waitFor(() => {
      expect(useKoharuStore.getState().settingsOpen).toBe(false)
      expect(useKoharuStore.getState().preferences?.pipeline.translation.page.model).toMatchObject({
        provider: 'openrouter',
        model: 'openrouter/auto',
      })
    })
  })

  it('adds and removes default font families from typesetting settings', async () => {
    installProject()
    const user = userEvent.setup()
    useKoharuStore.setState({ settingsOpen: true })
    queryClient.setQueryData(fontsKey, [
      {
        name: 'Noto Sans',
        metadata: {
          primary_script: 'latn',
          scripts: ['latn'],
          languages: ['en'],
          category: 'SANS_SERIF',
          classifications: ['sans-serif'],
          use_cases: ['body-text'],
        },
        sources: ['system'],
        faces: [
          {
            postscript_name: 'NotoSans-Regular',
            weight: 400,
            weight_range: null,
            style: 'normal',
          },
        ],
      },
      {
        name: 'Arial',
        metadata: {
          primary_script: 'latn',
          scripts: ['latn'],
          languages: ['en'],
          category: 'SANS_SERIF',
          classifications: ['sans-serif'],
          use_cases: ['body-text'],
        },
        sources: ['system'],
        faces: [
          {
            postscript_name: 'ArialMT',
            weight: 400,
            weight_range: null,
            style: 'normal',
          },
        ],
      },
    ])
    const save = vi
      .spyOn(commands, 'savePreferences')
      .mockImplementation(async (pipeline, providers, typesetting) => ({
        ...preferences,
        pipeline,
        providers,
        typesetting,
      }))
    render(
      <ThemeProvider attribute='class'>
        <SettingsPage />
      </ThemeProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Typesetting' }))
    expect(screen.getByRole('heading', { level: 2, name: 'Typesetting' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Add font family' }))
    await user.click(await screen.findByRole('option', { name: 'Arial, System' }))

    await waitFor(() =>
      expect(save).toHaveBeenLastCalledWith(preferences.pipeline, preferences.providers, {
        font_families: ['Noto Sans', 'Arial'],
      }),
    )
    await user.click(screen.getByRole('button', { name: 'Remove Noto Sans' }))
    await waitFor(() =>
      expect(save).toHaveBeenLastCalledWith(preferences.pipeline, preferences.providers, {
        font_families: ['Arial'],
      }),
    )
  })

  it('shows model resources in the left sidebar footer', () => {
    useKoharuStore.setState({
      resources: {
        process_memory: 2 * 1024 ** 3,
        system_memory: 64 * 1024 ** 3,
        process_cpu: 8,
        devices: [
          {
            name: 'GPU',
            selected: true,
            memory_budget: 16 * 1024 ** 3,
            memory_used: 6 * 1024 ** 3,
            utilization: 40,
          },
        ],
      },
    })
    render(<ResourceMonitor />)
    expect(screen.getByText('8%')).toBeInTheDocument()
    expect(screen.getByText('3%')).toBeInTheDocument()
  })

  it('keeps running work visible and stoppable', async () => {
    installProject()
    useKoharuStore.setState({
      jobs: {
        job: {
          state: 'running',
          id: 'job',
          completed: 1,
          total: 4,
          target: { target: 'page', value: 'page' },
          stage: 'ocr',
          model: 'manga-ocr',
          error: null,
        },
      },
    })
    const stop = vi.spyOn(commands, 'stopJob').mockResolvedValue(null)
    render(<ActivityCenter />)
    expect(screen.getByText('25%')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    await waitFor(() => expect(stop).toHaveBeenCalledWith('job'))
  })

  it('combines concurrent downloads into one progress bar', () => {
    useKoharuStore.setState({
      downloads: {
        one: {
          id: 1,
          state: 'running',
          name: 'one.ttf',
          completed: 25,
          total: 100,
          error: null,
        },
        two: {
          id: 2,
          state: 'running',
          name: 'two.ttf',
          completed: 75,
          total: 100,
          error: null,
        },
      },
    })

    render(<ActivityCenter />)

    expect(screen.getByText('Downloading 2 files')).toBeInTheDocument()
    expect(screen.getByText('50%')).toBeInTheDocument()
    expect(screen.getAllByRole('progressbar')).toHaveLength(1)
    expect(screen.queryByText('one.ttf')).not.toBeInTheDocument()
    expect(screen.queryByText('two.ttf')).not.toBeInTheDocument()
  })
})
